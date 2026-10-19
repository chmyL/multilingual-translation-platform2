# 口语/书面语转换模块
# 支持口语化表达转为书面语，以及书面语转为通俗易懂的口语版本。
from typing import Dict, Any
from config import DEEPSEEK_MAX_TOKENS
from llm_adapter import chat, strip_deepseek_thinking
from vllm_manager import get_actual_running_model

# 将口语化表达转换为书面语
def spoken_to_written(text: str, lang: str = "zh", model: str = "",
                      temperature: float = 0.3, max_tokens: int = 256,) -> Dict[str, Any]:
    if not text or not text.strip():
        return {
            "success": False,
            "converted_text": "",
            "direction": "spoken_to_written",
            "error": "输入文本为空",
        }

    lang_full = {"zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文"}.get(lang, lang)

    lang_force = {"zh":"【重要】你必须输出中文。","en":"【重要】你必须输出英文（English）。","ja":"【重要】你必须输出日文。","ko":"【重要】你必须输出韩文。"}
    system_prompt = (
        f"你是一个{lang_full}写作助手。请将以下口语化的{lang_full}文本转换为正式书面语。\n"
        f"要求：\n"
        f"1. 移除口语填充词（如'嗯'、'那个'、'就是'、'然后'等）\n"
        f"2. 将口语词汇替换为书面语词汇\n"
        f"3. 优化语法结构，使句子完整规范\n"
        f"4. 保持原意和语气基调不变\n"
        f"5. 只输出转换后的文本，不要解释\n"
        f"{lang_force.get(lang, '')}"
    )

    try:
        current = get_actual_running_model()
        is_deepseek = current and "DeepSeek" in current
        if is_deepseek:
            lang_label = {"zh":"Chinese","en":"English","ja":"Japanese","ko":"Korean"}.get(lang, lang)
            converted = chat(
                [{"role": "user", "content": f"Convert this spoken {lang_label} text into formal written {lang_label}. Remove filler words, replace casual terms with formal vocabulary. Output only the converted text: {text}"}],
                temperature=temperature, max_tokens=DEEPSEEK_MAX_TOKENS,
            )
        else:
            converted = chat(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
                temperature=temperature, max_tokens=max_tokens,
            )
        if is_deepseek:
            converted = strip_deepseek_thinking(converted)

        return {
            "success": True,
            "original_text": text,
            "converted_text": converted,
            "direction": "spoken_to_written",
            "direction_name": "口语 → 书面语",
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "original_text": text,
            "converted_text": "",
            "direction": "spoken_to_written",
            "direction_name": "口语 → 书面语",
            "error": str(e),
        }

# 将书面语转换为通俗口语
def written_to_spoken(text: str, lang: str = "zh", model: str = "", temperature: float = 0.3,
                      max_tokens: int = 256,) -> Dict[str, Any]:
    if not text or not text.strip():
        return {
            "success": False,
            "converted_text": "",
            "direction": "written_to_spoken",
            "error": "输入文本为空",
        }

    lang_full = {"zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文"}.get(lang, lang)

    lang_force = {"zh":"【重要】你必须输出中文。","en":"【重要】你必须输出英文（English）。","ja":"【重要】你必须输出日文。","ko":"【重要】你必须输出韩文。"}
    system_prompt = (
        f"你是一个{lang_full}写作助手。请将以下正式的{lang_full}书面语文本转换为通俗易懂的口语表达。\n"
        f"要求：\n"
        f"1. 简化复杂句式，使用短句和日常表达\n"
        f"2. 用常见词汇替代生僻词汇和术语\n"
        f"3. 增加口语自然的语气和流畅度\n"
        f"4. 保持原意和核心信息不变\n"
        f"5. 只输出转换后的文本，不要解释\n"
        f"{lang_force.get(lang, '')}"
    )

    try:
        current = get_actual_running_model()
        is_deepseek = current and "DeepSeek" in current
        if is_deepseek:
            lang_label = {"zh":"Chinese","en":"English","ja":"Japanese","ko":"Korean"}.get(lang, lang)
            converted = chat(
                [{"role": "user", "content": f"Convert this formal written {lang_label} text into casual spoken {lang_label}. Use shorter sentences, simpler words, and natural conversational tone. Output only the converted text: {text}"}],
                temperature=temperature, max_tokens=DEEPSEEK_MAX_TOKENS,
            )
        else:
            converted = chat(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
                temperature=temperature, max_tokens=max_tokens,
            )
        if is_deepseek:
            converted = strip_deepseek_thinking(converted)

        return {
            "success": True,
            "original_text": text,
            "converted_text": converted,
            "direction": "written_to_spoken",
            "direction_name": "书面语 → 口语",
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "original_text": text,
            "converted_text": "",
            "direction": "written_to_spoken",
            "direction_name": "书面语 → 口语",
            "error": str(e),
        }
