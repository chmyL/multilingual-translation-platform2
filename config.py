# 模型配置
MODELS = {
    "Hy-MT2-1.8B": {
        "name": "Hy-MT2-1.8B（混元翻译专用 v2）",
        "path": "/mnt/f/wsl_share/models/Hy-MT2-1.8B/Tencent-Hunyuan/Hy-MT2-1___8B",
        "description": "混元翻译专用模型 v2，翻译引擎对比",
        "type": "translation",
        "architecture": "hunyuan_v1_dense",
        "max_model_len": 512,
    },
    "Qwen2.5-1.5B-Instruct": {
        "name": "Qwen2.5-1.5B-Instruct（通义千问）",
        "path": "/mnt/f/wsl_share/models/Qwen2.5-1.5B-Instruct",
        "description": "通义千问指令模型，用于润色、转换与翻译对比",
        "type": "general",
        "architecture": "qwen2",
        "max_model_len": 512,
    },
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        "name": "DeepSeek-R1-Distill-Qwen-1.5B（推理模型）",
        "path": "/mnt/f/wsl_share/models/DeepSeek-R1-Distill-Qwen-1.5B",
        "description": "DeepSeek推理蒸馏模型，用于质量分析与翻译对比",
        "type": "reasoning",
        "architecture": "qwen2",
        "max_model_len": 1024,  # 推理链需要额外 token，给长上下文
    },
}

# 推理参数配置
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 128  # 翻译默认最大生成 token 数
DEEPSEEK_MAX_TOKENS = 512  # DeepSeek-R1 推理链需更多 token

# 默认模型（程序启动时使用）
DEFAULT_MODEL = "Hy-MT2-1.8B"

# vLLM 服务配置
VLLM_PORT = 8000
VLLM_HOST = "localhost"
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"

# vLLM 安装路径（vllm-env conda 环境中）
_VLLM_BIN = "/home/dell/miniconda3/envs/vllm-env/bin/vllm"
_VLLM_ENV_BIN = "/home/dell/miniconda3/envs/vllm-env/bin"

# vLLM 启动参数（必须把 conda env bin 加到 PATH，否则 flashinfer 找不到 ninja）
VLLM_START_CMD_TEMPLATE = (
    "PATH={env_bin}:$PATH "
    "CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 "
    "{vllm_bin} serve {model_path} "
    "--port {port} "
    "--gpu-memory-utilization {gpu_memory} "
    "--max-model-len {max_model_len} "
    "--no-enable-prefix-caching "
    "--enforce-eager"
)

VLLM_DEFAULT_GPU_MEMORY = 0.78  # RTX 4050 6GB
VLLM_DEFAULT_MAX_MODEL_LEN = 512

# vLLM 启动超时（秒）
VLLM_STARTUP_TIMEOUT = 600  # WSL 共享目录 IO 慢，首次加载可能需要 5-10 分钟
# 健康检查间隔（秒）
VLLM_HEALTH_CHECK_INTERVAL = 2

# 语言对配置
LANGUAGE_PAIRS = {
    "zh-en": {"source": "中文", "target": "英文", "label": "中 → 英"},
    "en-zh": {"source": "英文", "target": "中文", "label": "英 → 中"},
    "zh-ja": {"source": "中文", "target": "日文", "label": "中 → 日"},
    "ja-zh": {"source": "日文", "target": "中文", "label": "日 → 中"},
    "zh-ko": {"source": "中文", "target": "韩文", "label": "中 → 韩"},
    "ko-zh": {"source": "韩文", "target": "中文", "label": "韩 → 中"},
}

LANGUAGE_NAMES = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "ko": "韩文",
}

# 翻译策略配置
TRANSLATION_STRATEGIES = {
    "literal": {
        "name": "直译",
        "description": "尽可能字面翻译，保留原文句子结构",
    },
    "semantic": {
        "name": "意译",
        "description": "传达核心语义，允许调整句子结构",
    },
    "academic": {
        "name": "学术润色翻译",
        "description": "翻译同时进行学术风格优化，使用正式学术表达",
    },
}

# 润色级别配置
POLISH_LEVELS = {
    "light": {"name": "轻度润色", "description": "仅修正明显口语化表达"},
    "moderate": {"name": "中度润色", "description": "修正口语化并提升正式度"},
    "heavy": {"name": "重度润色", "description": "完全重构为学术论文风格"},
}

# 领域配置（实训部分可扩展）
DOMAINS = ["academic", "business", "daily", "law", "medicine", "cs"]

DOMAIN_NAMES = {
    "academic": "学术",
    "business": "商务",
    "daily": "日常对话",
    "law": "法律",
    "medicine": "医学",
    "cs": "计算机科学",
}

# 文件路径配置
DATA_DIR = "/mnt/f/PythonProject3/data"
CORPUS_DIR = "/mnt/f/PythonProject3/data/corpus"
RESULTS_DIR = "/mnt/f/PythonProject3/data/results"
OUTPUT_DIR = "/mnt/f/PythonProject3/outputs"

# 评估配置
EVALUATION_METRICS = ["BLEU", "METEOR"]
MANUAL_SCORE_CRITERIA = {
    "accuracy": {"name": "准确度", "range": (1, 5), "description": "翻译是否准确传达原文含义"},
    "fluency": {"name": "流畅度", "range": (1, 5), "description": "译文是否自然流畅"},
    "adequacy": {"name": "充分性", "range": (1, 5), "description": "术语和关键信息是否完整传达"},
}

# 实验配置
EXPERIMENT_SAMPLE_SIZE = 80  # 实验测试样本数（从数据集中抽取）
EXPERIMENT_REPEAT_TIMES = 1  # 每个文本翻译重复次数（测量稳定性）
