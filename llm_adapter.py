# 统一大模型推理适配器
import os
import time
import warnings
from typing import List, Dict, Optional, Any

import requests
from openai import OpenAI

from config import VLLM_BASE_URL, MODELS

# vLLM OpenAI兼容客户端（timeout=60s，超时快触发自恢复）
_vllm_client = OpenAI(base_url=f"{VLLM_BASE_URL}/v1", api_key="not-needed", timeout=60.0, max_retries=0)

# transformers本地模型缓存
_tf_model = None
_tf_tokenizer = None
_tf_loaded_model_path: Optional[str] = None

# 需要走transformers后端的模型列表
_TRANSFORMERS_MODELS = {"Hy-MT2-1.8B"}

# 查询vLLM服务当前运行的模型路径
def _get_running_vllm_model() -> Optional[str]:
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=3)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                return data[0]["id"]
    except Exception:
        pass
    return None

# 卸载transformers模型，释放显存
def _unload_transformers():
    global _tf_model, _tf_tokenizer, _tf_loaded_model_path
    if _tf_model is not None:
        del _tf_model
        _tf_model = None
    if _tf_tokenizer is not None:
        del _tf_tokenizer
        _tf_tokenizer = None
    _tf_loaded_model_path = None
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 按需加载transformers模型到显存
def _load_transformers(model_path: str):
    global _tf_model, _tf_tokenizer, _tf_loaded_model_path

    if _tf_loaded_model_path == model_path and _tf_model is not None:
        return  # 已加载

    # 先卸载旧模型
    _unload_transformers()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[llm_adapter] 正在加载 transformers 模型: {model_path} ...")
    t0 = time.time()

    _tf_tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    _tf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 移到GPU（单卡直接.to 即可，避开device_map对accelerate的依赖）
    if torch.cuda.is_available():
        _tf_model = _tf_model.to("cuda")
    _tf_model.eval()
    _tf_loaded_model_path = model_path

    print(f"[llm_adapter] 模型加载完成，耗时 {time.time() - t0:.1f}s，"
          f"显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else
          f"耗时 {time.time() - t0:.1f}s (CPU)")

# 通过vLLM OpenAI API推理
def _chat_vllm(messages: List[Dict[str, str]], model_name: str,
               temperature: float, max_tokens: int) -> str:
    response = _vllm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

# 通过transformers本地推理
def _chat_transformers(messages: List[Dict[str, str]], model_path: str,
                       temperature: float, max_tokens: int) -> str:
    import torch

    _load_transformers(model_path)

    # 使用tokenizer的chat_template (Jinja2)
    if hasattr(_tf_tokenizer, 'apply_chat_template') and _tf_tokenizer.chat_template:
        try:
            # tokenize=False→返回字符串，再手动encode
            prompt_text = _tf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = _tf_tokenizer.encode(prompt_text, return_tensors="pt")
        except Exception:
            input_ids = None
    else:
        input_ids = None

    if input_ids is None:
        # 兜底：用tokenizer的特殊token手动构建
        # 模型文本起始标记
        bos = _tf_tokenizer.bos_token or ""
        # 模型终止输出标记
        eos = _tf_tokenizer.eos_token or ""
        text_parts = [bos] if bos else []
        for msg in messages:
            text_parts.append(f"{msg['content']}")
        text_parts.append("")  # generation prompt trigger
        text = "\n".join(text_parts)
        input_ids = _tf_tokenizer.encode(text, return_tensors="pt")

    input_ids = input_ids.to(_tf_model.device)

    generation_config = {
        "max_new_tokens": min(max_tokens, 512),
        "temperature": temperature if temperature > 0 else 0.1,
        "do_sample": temperature > 0,
        "pad_token_id": _tf_tokenizer.pad_token_id or _tf_tokenizer.eos_token_id,
        "eos_token_id": _tf_tokenizer.eos_token_id,
    }

    with torch.no_grad():
        # 调用模型自回归生成文本
        output_ids = _tf_model.generate(input_ids, **generation_config)

    # 只取生成的新token
    new_tokens = output_ids[0][input_ids.shape[1]:]
    result = _tf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return result

# 统一推理接口 — 自动路由到vLLM或transformers
def chat(messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 512) -> str:
    # 1.优先检查当前选中模型是否应该走transformers
    from vllm_manager import get_current_model, get_actual_running_model
    current = get_current_model()

    if current and current in _TRANSFORMERS_MODELS:
        model_path = MODELS[current]["path"]
        print(f"[llm_adapter] 使用 transformers: {current}")
        return _chat_transformers(messages, model_path, temperature, max_tokens)

    # 2.查询vLLM实际运行的服务（不信任 _current_model，直接查/v1/models）
    vllm_model = _get_running_vllm_model()
    if vllm_model:
        # 检测模型不一致：警告但不阻止（converter/translator已用实际运行模型适配了prompt）
        actual_key = get_actual_running_model()
        if actual_key and actual_key != current:
            selected_name = MODELS.get(current, {}).get("name", current or "无")
            running_name = MODELS.get(actual_key, {}).get("name", actual_key)
            print(f"[llm_adapter] ⚠️ 模型不一致警告: 侧边栏选择={selected_name}，vLLM 实际运行={running_name}")
        print(f"[llm_adapter] 使用 vLLM: {vllm_model}")
        return _chat_vllm(messages, vllm_model, temperature, max_tokens)

    # 3.vLLM没运行 — 报错。给出明确的诊断信息
    msg = "vLLM 服务未启动，请在侧边栏选择模型后点击「启动/切换」。"
    if current and current not in _TRANSFORMERS_MODELS:
        msg += (
            f"\n当前选中模型: {MODELS.get(current, {}).get('name', current)}，但 vLLM 服务未检测到运行中的模型。"
            f"\n可能原因: 模型启动失败（显存不足？），请检查服务端日志后重试。"
        )
    else:
        msg += f"\n当前选中模型: {current or '无'}。"
    raise RuntimeError(msg)

# 返回当前推理后端名称。给前端/api/status提供后端类型字符串
def get_backend() -> str:
    if _get_running_vllm_model():
        return "vLLM"
    from vllm_manager import get_current_model
    current = get_current_model()
    if current and current in _TRANSFORMERS_MODELS:
        return "transformers"
    return "unknown"

# 去除DeepSeek R1的思考链，只保留最终输出
def strip_deepseek_thinking(text: str) -> str:
    if not text:
        return text

    tag_start = "<think>"
    tag_end = "</think>"

    # 1.如果有闭合标签，取最后闭合标签之后的内容
    if tag_end in text:
        text = text.rsplit(tag_end, 1)[-1].strip()
    else:
        # 2.如果只有开标签没有闭合，简单清理开标签
        if tag_start in text:
            text = text.replace(tag_start, "").strip()

    # 3.去掉所有残留标签
    text = text.replace(tag_start, "").replace(tag_end, "").strip()

    return text

# 清理资源（进程退出时调用）
def cleanup():
    _unload_transformers()
