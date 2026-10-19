#!/bin/bash
# 一键启动脚本—清进程+启vLLM+启FastAPI
# vLLM参数从config.py读取，无需手动同步

echo "=== 清理旧进程 ==="
pkill -9 python 2>/dev/null
pkill -9 -f vllm 2>/dev/null
pkill -9 -f VLLM 2>/dev/null
sleep 2
for port in 8000 8501; do
    fuser -k ${port}/tcp 2>/dev/null
done
# 清理 WSL2 僵尸 GPU 进程
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null
done
sleep 2

echo "=== GPU 状态 ==="
nvidia-smi | grep -E "MiB|Processes" -A3

source /home/dell/miniconda3/etc/profile.d/conda.sh
conda activate vllm-env

# 从config.py读取参数
eval $(python3 -c "
import sys; sys.path.insert(0, '/mnt/f/PythonProject3')
from config import MODELS, VLLM_DEFAULT_GPU_MEMORY, VLLM_DEFAULT_MAX_MODEL_LEN, VLLM_PORT
model = list(MODELS.keys())[0]
print(f'MODEL_KEY={model}')
print(f'MODEL_PATH={MODELS[model][\"path\"]}')
print(f'GPU_MEM={VLLM_DEFAULT_GPU_MEMORY}')
print(f'MAX_LEN={VLLM_DEFAULT_MAX_MODEL_LEN}')
print(f'PORT={VLLM_PORT}')
")

echo "=== 启动 vLLM: ${MODEL_KEY} ==="
echo "  gpu_memory_utilization=${GPU_MEM}"
echo "  max_model_len=${MAX_LEN}"

CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 vllm serve "${MODEL_PATH}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs 1 \
  --enforce-eager &

echo "等待 vLLM 就绪..."
for i in $(seq 1 20); do
    sleep 5
    if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "vLLM 就绪"
        break
    fi
    echo "  等待中... ($((i*5))s)"
done

echo "=== 启动 FastAPI ==="
python /mnt/f/PythonProject3/server.py &

sleep 3
echo ""
echo "=== 启动完成 ==="
echo "  前端: http://localhost:8501"
echo "  vLLM: http://localhost:${PORT}"
