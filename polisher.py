# 学术文本润色服务
# 对翻译后的文本进行学术风格润色：去口语化、提升正式度、调整句式结构。
# 支持三个润色级别：轻度、中度、重度。
from typing import Dict, Any
from config import POLISH_LEVELS, DEEPSEEK_MAX_TOKENS
from llm_adapter import chat, strip_deepseek_thinking
from vllm_manager import get_actual_running_model

# 根据润色级别和目标语言构建system prompt
def _build_polish_prompt(text: str, level: str, target_lang: str) -> str:
    level_info = POLISH_LEVELS.get(level, POLISH_LEVELS["moderate"])

    lang_name = {"zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文"}.get(target_lang, "中文")

    lang_force = {"zh":"【重要】你必须输出中文。","en":"【重要】你必须输出英文（English）。","ja":"【重要】你必须输出日文。","ko":"【重要】你必须输出韩文。"}
    base = (
        f"你是一个{lang_name}学术写作润色专家。"
        f"请对以下{lang_name}文本进行学术风格润色。"
        f"\n{lang_force.get(target_lang, '')}"
    )

    if level == "light":
        instructions = (
            "要求（轻度润色）：\n"
            "1. 仅修正明显的口语化表达和口语词汇\n"
            "2. 保持原句式和段落结构不变\n"
            "3. 保持内容完整性，不增删信息\n"
            "4. 只输出润色后的文本，不要解释"
        )
    elif level == "heavy":
        instructions = (
            "要求（重度润色）：\n"
            "1. 完全重构为正式学术论文写作风格\n"
            "2. 去除所有口语化表达、俚语、缩写\n"
            "3. 使用高级学术词汇和复杂句式\n"
            "4. 优化段落结构和逻辑连贯性\n"
            "5. 适当添加学术衔接词\n"
            "6. 保持原意不变，不增删核心信息\n"
            "7. 只输出润色后的文本，不要解释"
        )
    else:  # moderate
        instructions = (
            "要求（中度润色）：\n"
            "1. 修正口语化和非正式表达\n"
            "2. 提升用词正式度，替换为学术词汇\n"
            "3. 适当调整句式使其更规范\n"
            "4. 保持原意和主要内容完整\n"
            "5. 只输出润色后的文本，不要解释"
        )

    return base + "\n" + instructions

# 对文本进行学术风格润色
def polish(text: str, level: str = "moderate", target_lang: str = "zh", model: str = "", temperature: float = 0.3,
           max_tokens: int = 256,) -> Dict[str, Any]:
    if not text or not text.strip():
        return {
            "success": False,
            "polished_text": "",
            "level": level,
            "level_name": POLISH_LEVELS.get(level, {}).get("name", level),
            "error": "输入文本为空",
        }

    system_prompt = _build_polish_prompt(text, level, target_lang)
    level_name = POLISH_LEVELS.get(level, {}).get("name", level)

    try:
        current = get_actual_running_model()
        is_deepseek = current and "DeepSeek" in str(current)
        if is_deepseek:
            # DeepSeek 不理解 system prompt，直接用简化 user 消息
            level_label = {"light":"fix obvious colloquialisms","moderate":"improve formality and academic tone","heavy":"fully restructure into formal academic style"}.get(level, "improve academic style")
            lang_label = {"zh":"Chinese","en":"English","ja":"Japanese","ko":"Korean"}.get(target_lang, target_lang)
            polished = chat(
                [{"role": "user", "content": f"Polish this {lang_label} text ({level_label}). Output only the polished text, nothing else: {text}"}],
                temperature=temperature, max_tokens=DEEPSEEK_MAX_TOKENS,
            )
        else:
            polished = chat(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
                temperature=temperature, max_tokens=max_tokens,
            )
        if is_deepseek:
            polished = strip_deepseek_thinking(polished)

        return {
            "success": True,
            "original_text": text,
            "polished_text": polished,
            "level": level,
            "level_name": level_name,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "original_text": text,
            "polished_text": "",
            "level": level,
            "level_name": level_name,
            "error": str(e),
        }

# 用所有润色级别处理同一文本，用于效果对比
def polish_all_levels(text: str, target_lang: str = "zh", model: str = "",) -> Dict[str, Any]:
    results = {}
    for level_key in POLISH_LEVELS:
        results[level_key] = polish(text, level_key, target_lang, model)
    return {
        "original_text": text,
        "target_lang": target_lang,
        "results": results,
    }
