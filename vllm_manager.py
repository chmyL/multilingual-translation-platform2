import subprocess
import time
import os
import signal
import glob
import requests
from typing import Optional, Dict, Any
from config import (
    MODELS,
    DEFAULT_MODEL,
    VLLM_HOST,
    VLLM_PORT,
    VLLM_BASE_URL,
    VLLM_START_CMD_TEMPLATE,
    VLLM_DEFAULT_GPU_MEMORY,
    VLLM_DEFAULT_MAX_MODEL_LEN,
    VLLM_STARTUP_TIMEOUT,
    VLLM_HEALTH_CHECK_INTERVAL,
    _VLLM_BIN,
    _VLLM_ENV_BIN,
)

# 当前运行的 vLLM 进程
_current_process: Optional[subprocess.Popen] = None
_current_model: Optional[str] = None


# 返回当前选中的模型名称
def get_current_model() -> Optional[str]:
    return _current_model


# 查询 vLLM 服务实际运行的模型，匹配 MODELS 配置返回模型 key。
# 如果 vLLM 未运行或无法匹配，返回 None。
def get_actual_running_model() -> Optional[str]:
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data:
            return None
        vllm_model_id = data[0]["id"]  # vLLM 返回的模型 ID
        # 按路径匹配 MODELS 配置
        for key, info in MODELS.items():
            if info["path"] == vllm_model_id or vllm_model_id.startswith(info["path"]):
                return key
            # 也尝试用模型名匹配（有些 vLLM 返回名称而非路径）
            if info.get("name", "") in vllm_model_id or vllm_model_id in info.get("name", ""):
                return key
        # 无法匹配，返回原始 ID 作为 fallback
        return vllm_model_id
    except Exception:
        return None

# 检查 vLLM 服务是否存活
# 通过调用 /v1/models 端点确认服务就绪
def check_health() -> bool:
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False

# 等待 vLLM 服务就绪，每隔 VLLM_HEALTH_CHECK_INTERVAL 秒检查一次
# 返回 True 表示服务已就绪，False 表示超时
def wait_until_ready(timeout: int = VLLM_STARTUP_TIMEOUT) -> bool:
    elapsed = 0
    while elapsed < timeout:
        if check_health():
            return True
        time.sleep(VLLM_HEALTH_CHECK_INTERVAL)
        elapsed += VLLM_HEALTH_CHECK_INTERVAL
    return False

# GPU 进程强制清理 —— 解决 EngineCore 僵尸进程问题
# 通过 nvidia-smi 获取所有占用 GPU 的 PID 列表
def _get_gpu_pids() -> list:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        pids = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line and line.isdigit():
                pids.append(int(line))
        return pids
    except Exception:
        return []


# 通过 /proc 扫描所有 VLLM::EngineCore 进程（即使被 init 收养也能找到）
def _get_vllm_engine_pids() -> list:
    pids = []
    try:
        for comm_path in glob.glob("/proc/*/comm"):
            try:
                with open(comm_path, "r") as f:
                    comm = f.read().strip()
                if "VLLM" in comm or "vllm" in comm:
                    pid = int(comm_path.split("/")[2])
                    pids.append(pid)
            except (OSError, ValueError, IndexError):
                continue
    except Exception:
        pass
    return pids


# 强制杀死所有占用 GPU 的 vLLM 相关进程。
def _force_kill_all_gpu_vllm(verbose: bool = True):
    killed = set()
    # server.py 里的 Uvicorn 主进程
    my_pid = os.getpid()  # 绝不能杀自己（transformers 卸模型后本进程仍在 nvidia-smi 列表中）

    # 策略 1: nvidia-smi GPU 进程
    for pid in _get_gpu_pids():
        if pid == my_pid:
            continue  # 跳过自己
        if pid not in killed:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.add(pid)
                if verbose:
                    print(f"[vllm_manager] 🔪 杀死 GPU 进程 PID={pid}")
            except Exception:
                pass

    # 策略 2: /proc 扫描 VLLM::EngineCore（可能在策略 1 之前没被 nvidia-smi 列出）
    for pid in _get_vllm_engine_pids():
        if pid not in killed:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.add(pid)
                if verbose:
                    print(f"[vllm_manager] 🔪 杀死 VLLM 引擎进程 PID={pid}")
            except Exception:
                pass

    # 策略 3: pkill 兜底（进程名匹配）
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "VLLM::EngineCore"],
            capture_output=True, timeout=5,
        )
        subprocess.run(
            ["pkill", "-9", "-f", "vllm serve"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass

    # 策略 4: 端口清理
    try:
        subprocess.run(
            ["fuser", "-k", f"{VLLM_PORT}/tcp"],
            capture_output=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except Exception:
        pass

    if killed and verbose:
        print(f"[vllm_manager] 已强制清理 {len(killed)} 个 GPU/vLLM 进程")

    return len(killed)

# 等待 GPU 显存释放到至少 min_free_mb 可用。
# 返回 True 表示显存足够，False 表示超时。
def _wait_gpu_memory_free(min_free_mb: int = 4500, timeout: int = 30) -> bool:
    elapsed = 0
    interval = 2
    while elapsed < timeout:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            free_mb = int(result.stdout.strip().split("\n")[0].strip())
            if free_mb >= min_free_mb:
                print(f"[vllm_manager] GPU 显存充足: {free_mb} MiB 可用（需要 ≥{min_free_mb} MiB）")
                return True
            print(f"[vllm_manager] 等待 GPU 显存释放: {free_mb} MiB / 需要 {min_free_mb} MiB ...")
        except Exception:
            pass
        time.sleep(interval)
        elapsed += interval

    # 最后一次查询
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        free_mb = int(result.stdout.strip().split("\n")[0].strip())
        print(f"[vllm_manager] 超时后 GPU 可用: {free_mb} MiB")
    except Exception:
        pass

    return False

# Hy-MT2 等需要走 transformers 的模型
_TRANSFORMERS_ONLY = {"Hy-MT2-1.8B"}

# 判断模型是否只能走 transformers 后端
def _is_transformers_model(model_name: str) -> bool:
    return model_name in _TRANSFORMERS_ONLY

# 停止 / 启动 / 切换
# 停止当前运行的服务（vLLM 或 transformers），彻底清理 GPU 显存。返回 True 表示成功停止。
def stop_vllm() -> bool:
    global _current_process, _current_model
    try:
        # 停止正在运行的实验
        try:
            from experiment import _experiment_progress
            _experiment_progress["running"] = False
            _experiment_progress["message"] = "模型已停止，实验中断"
        except ImportError:
            pass

        # 卸载 transformers 模型
        if _current_model and _is_transformers_model(_current_model):
            print(f"[vllm_manager] 卸载 transformers 模型: {_current_model}")
            from llm_adapter import _unload_transformers
            _unload_transformers()
            _current_model = None
            _current_process = None
            return True

        # 终止 vLLM 进程（先尝试 SIGTERM，再 SIGKILL）
        if _current_process is not None:
            print(f"[vllm_manager] 停止 vLLM 进程: PID={_current_process.pid}")
            try:
                os.killpg(os.getpgid(_current_process.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                _current_process.terminate()
                _current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"[vllm_manager] 进程未响应 SIGTERM，发送 SIGKILL...")
                try:
                    _current_process.kill()
                    _current_process.wait(timeout=5)
                except Exception:
                    pass
            except Exception:
                pass

        # 强制清理：杀死所有 GPU/vLLM 相关进程
        # 这是关键——解决 EngineCore 被 WSL init 收养后无法清理的问题
        print(f"[vllm_manager] 强制清理所有 GPU 上的 vLLM 进程...")
        _force_kill_all_gpu_vllm()

        # 等待 GPU 显存实际释放（CUDA 上下文 ~1GB 常驻，所以目标 4.5GB）
        _wait_gpu_memory_free(min_free_mb=4500, timeout=15)

        _current_process = None
        _current_model = None
        time.sleep(2)
        return True
    except Exception as e:
        print(f"[vllm_manager] 停止 vLLM 时出错: {e}")
        # 即使出错也要尝试强制清理
        try:
            _force_kill_all_gpu_vllm(verbose=False)
        except Exception:
            pass
        _current_process = None
        _current_model = None
        return False

# 清理启动失败的 vLLM 子进程并释放所有 GPU 资源
def _cleanup_failed_process(proc: subprocess.Popen):
    if proc is not None:
        try:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        except Exception:
            pass

    # 强制清理所有 GPU/vLLM 进程
    _force_kill_all_gpu_vllm()
    _wait_gpu_memory_free(min_free_mb=4500, timeout=10)

# 启动指定模型的服务（vLLM 或 transformers）
def start_vllm(
    model_name: str,
    gpu_memory: float = VLLM_DEFAULT_GPU_MEMORY,
    max_model_len: int = VLLM_DEFAULT_MAX_MODEL_LEN,
) -> Dict[str, Any]:
    global _current_process, _current_model

    if model_name not in MODELS:
        return {"success": False, "message": f"未知模型: {model_name}", "model": None}

    model_info = MODELS[model_name]
    # 优先用模型自己配置的 max_model_len，否则用全局默认值
    max_model_len = model_info.get("max_model_len", max_model_len)
    model_path = model_info["path"]

    if not os.path.isdir(model_path):
        return {"success": False, "message": f"模型路径不存在: {model_path}", "model": None}

    # ── 步骤 1: 停止旧服务 ──
    if _current_process is not None or (_current_model and not _is_transformers_model(_current_model)):
        print(f"[vllm_manager] 停止旧服务: {_current_model}")
        stop_vllm()
    elif _current_model and _is_transformers_model(_current_model):
        print(f"[vllm_manager] 卸载 transformers 模型: {_current_model}")
        from llm_adapter import _unload_transformers
        _unload_transformers()
        _current_model = None

    # 步骤 2: 兜底——确保 GPU 上没有残留的 vLLM 进程
    _force_kill_all_gpu_vllm(verbose=False)

    # 步骤 3: 检查 GPU 显存是否够用（需要 ~4.5GB 给 vLLM 0.80 利用率）
    if not _wait_gpu_memory_free(min_free_mb=4500, timeout=30):
        # 再试一轮强制清理
        print(f"[vllm_manager] GPU 显存不足，再次强制清理...")
        _force_kill_all_gpu_vllm()
        time.sleep(3)
        if not _wait_gpu_memory_free(min_free_mb=4000, timeout=20):
            # 最后查一次还剩多少
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                free_mb = int(result.stdout.strip().split("\n")[0].strip())
            except Exception:
                free_mb = 0
            return {
                "success": False,
                "message": (
                    f"GPU 显存不足: {free_mb} MiB 可用，需要至少 4000 MiB。"
                    f"\n请尝试手动重启 WSL 或运行: pkill -9 -f vllm"
                ),
                "model": None,
            }

    # transformers 模型：不启动 vLLM
    if _is_transformers_model(model_name):
        print(f"[vllm_manager] 使用 transformers 后端: {model_info['name']}")
        _current_process = None
        _current_model = model_name
        return {
            "success": True,
            "message": f"模型 {model_info['name']} 已就绪（transformers 后端）",
            "model": model_name,
        }

    # 步骤 4: 启动 vLLM
    cmd = VLLM_START_CMD_TEMPLATE.format(
        env_bin=_VLLM_ENV_BIN,
        vllm_bin=_VLLM_BIN,
        model_path=model_path,
        port=VLLM_PORT,
        gpu_memory=gpu_memory,
        max_model_len=max_model_len,
    )

    print(f"[vllm_manager] 启动模型: {model_info['name']}")
    print(f"[vllm_manager] 命令: {cmd}")

    try:
        log_file = open("/tmp/vllm_output.log", "a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid, # 让子进程在一个新的会话（session）中启动
        )
        _current_process = proc
        _current_log_file = log_file

        # 等待服务就绪
        print(f"[vllm_manager] 等待 vLLM 服务就绪（最多 {VLLM_STARTUP_TIMEOUT} 秒）...")
        if wait_until_ready():
            _current_model = model_name  # 只有健康检查通过才设置
            print(f"[vllm_manager] ✅ vLLM 服务就绪！模型: {model_info['name']}")
            return {
                "success": True,
                "message": f"模型 {model_info['name']} 启动成功",
                "model": model_name,
            }
        else:
            # 超时：清理失败的进程，回滚状态
            print(f"[vllm_manager] ❌ 启动超时，清理失败的进程...")
            _cleanup_failed_process(proc)
            _current_process = None

            # 尝试读取部分日志
            log_preview = ""
            try:
                if proc.stdout:
                    import select
                    lines = []
                    for _ in range(20):
                        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
                        if ready:
                            line = proc.stdout.readline()
                            if line:
                                lines.append(line.strip())
            except Exception:
                pass
            log_preview = "\n".join(lines[-10:]) if lines else "(无日志)"

            return {
                "success": False,
                "message": f"vLLM 启动超时（{VLLM_STARTUP_TIMEOUT} 秒）。最后日志:\n{log_preview}",
                "model": None,
            }

    except Exception as e:
        print(f"[vllm_manager] ❌ 启动异常: {e}")
        if _current_process is not None:
            _cleanup_failed_process(_current_process)
        _current_process = None
        return {
            "success": False,
            "message": f"启动 vLLM 失败: {str(e)}",
            "model": None,
        }

# 切换到另一个模型：停止当前服务，启动新模型
def switch_model(model_name: str) -> Dict[str, Any]:
    if model_name == _current_model:
        if check_health():
            return {
                "success": True,
                "message": f"模型 {MODELS[model_name]['name']} 已在运行",
                "model": model_name,
            }
        else:
            # 服务挂了，重新启动
            return start_vllm(model_name)

    print(f"[vllm_manager] 切换模型: {_current_model} → {model_name}")
    return start_vllm(model_name)

# 获取当前服务状态
def get_status() -> Dict[str, Any]:
    vllm_alive = check_health()

    backend = "vLLM"
    if not vllm_alive and _current_model and _is_transformers_model(_current_model):
        backend = "transformers"
        vllm_alive = True

    return {
        "running": vllm_alive,
        "current_model": _current_model,
        "current_model_name": MODELS.get(_current_model, {}).get("name", "无"),
        "backend": backend,
        "port": VLLM_PORT,
        "base_url": VLLM_BASE_URL,
    }

# 读取当前 vLLM 进程的最近日志
def read_vllm_logs(max_lines: int = 50) -> str:
    global _current_process
    if _current_process is None or _current_process.stdout is None:
        return "(无运行中的 vLLM 进程)"
    try:
        import select
        lines = []
        for _ in range(max_lines):
            ready, _, _ = select.select(
                [_current_process.stdout], [], [], 0.5
            )
            if ready:
                line = _current_process.stdout.readline()
                if line:
                    lines.append(line.strip())
            else:
                break
        return "\n".join(lines) if lines else "(暂无新日志)"
    except Exception:
        return "(无法读取日志)"
