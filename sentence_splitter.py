# 长文本分句翻译 + 特殊格式保护 + 上下文关联
# 纯正则实现，无外部依赖。
import re
from typing import List, Tuple, Dict
# 特殊格式提取

# 反引号代码块
_CODE_FENCE = re.compile(r'```[\s\S]*?```', re.MULTILINE)
# 代码行（无 fence 的裸代码）
_CODE_LINE = re.compile(
    r'(?:^|\n)[ \t]*(?:'
    r'import\s+\w+|from\s+\w+\s+import|'
    r'def\s+\w+\s*\(|class\s+\w+|'
    r'model\s*=\s*\w+\.\w+\(|transform\s*=\s*\w+\.\w+\(|'
    r'optimizer\s*=\s*\w+\.\w+\(|trainer\s*=\s*\w+\.\w+\(|'
    r'\w+\s*=\s*\w+\.\w+\.\w+|\w+\s*=\s*\w+\.\w+\(|'
    r'if\s+__name__\s*=='
    r')(?:[^\n]*\n(?:\s{2,}[^\n]*\n)*)',
    re.MULTILINE
)
_URL = re.compile(r'https?://[^\s)\]>]+')
# 邮箱
_EMAIL = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')


def extract_special_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    placeholders = {}
    c = [0]

    def _ph(match):
        c[0] += 1
        k = f"▣{c[0]}▣"
        placeholders[k] = match.group(0)
        return k

    # 0. 预处理：把跨行 Markdown 链接压成单行
    text = re.sub(r'\[([^\]]*)\]\s*\(([^)]*)\)', lambda m: f'[{m.group(1)}]({m.group(2)})', text, flags=re.DOTALL)

    # 优先级：代码块 > 代码行 > Markdown链接 > URL > 邮箱
    text = _CODE_FENCE.sub(_ph, text)
    text = _CODE_LINE.sub(_ph, text)
    text = _URL.sub(_ph, text)
    text = _EMAIL.sub(_ph, text)
    return text, placeholders


def restore_special_tokens(text: str, placeholders: Dict[str, str]) -> str:
    for k, v in placeholders.items():
        text = text.replace(k, v)
    return text

# 分句
def _split_sentences(text: str, min_len: int = 8) -> List[str]:
    terminators = set('。！？；?!;…')
    pattern = r'([' + ''.join(terminators) + r']\s*)'
    parts = re.split(pattern, text)

    sentences, buf = [], ""
    for p in parts:
        if not p: continue
        buf += p
        if any(buf.rstrip().endswith(t) for t in terminators):
            s = buf.rstrip()
            if len(s) >= min_len:
                sentences.append(s)
                buf = ""
    if buf.rstrip():
        sentences.append(buf.rstrip())

    merged = []
    for s in sentences:
        if merged and len(s.rstrip()) < min_len and not s.rstrip().startswith('#'):
            merged[-1] = merged[-1] + ' ' + s.rstrip()
        else:
            merged.append(s.rstrip())
    return [s for s in merged if s]


# 清理模型输出
_BAD_PREFIXES = (
    "(Previous:", "Context:", "Sentence to translate:", "Now translate:", "Translate:",
    "Previous:", "[上下文]", "[待翻译]", "[上文]", "[当前]",
    "术语映射", "术语对应", "使用正式", "使用自然", "使用严谨",
    "【学术", "【商务", "【技术", "【日常", "【法律", "【医学", "【计算机",
    "Now translate this:", "Now translate:",
)

def _clean_output(raw: str, original_len: int) -> str:
    if not raw or not raw.strip():
        return raw

    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    clean = []
    for l in lines:
        stripped = l
        for b in _BAD_PREFIXES:
            while b in stripped:
                idx = stripped.rfind(b)
                rest = stripped[idx+len(b):].lstrip("：:」」「『\"' ")
                if len(rest) > 3:
                    stripped = rest
                else:
                    break
        if stripped != l and len(stripped) < 3:
            continue
        clean.append(stripped)

    result = "\n".join(clean)
    if len(result) > original_len * 4:
        short = [l for l in clean if len(l) < original_len * 2 and len(l) > 2]
        result = short[-1] if short else clean[-1]
    return result

# 主函数
def translate_long_text(translate_fn, text: str, source_lang: str = "zh",
                        target_lang: str = "en", strategy: str = "semantic", domain: str = "",
                        context_window: int = 2, **kw,) -> Tuple[str, List[Dict]]:
    # 1. 提取特殊格式
    clean_text, placeholders = extract_special_tokens(text)

    # 2. 分句
    raw_sentences = _split_sentences(clean_text)
    sentences = []
    for s in raw_sentences:
        if s.count("|") >= 4:
            for line in s.split("\n"):
                line = line.strip()
                if line and len(line) >= 4:
                    sentences.append(line)
        elif s and len(s.strip()) >= 4:
            sentences.append(s)
    sentences = sentences[:80]

    if not sentences:
        return "", []

    details = []
    translated = []
    ctx_buf = []

    for i, s in enumerate(sentences):
        # 含占位符的行：用 PH 标记替换后翻译
        if "▣" in s:
            local_ph = {k: placeholders[k] for k in placeholders if k in s}
            simple_map = {}
            inp = s
            for j, k in enumerate(sorted(local_ph.keys(), key=lambda x: -len(x))):
                label = f"{{PH{j+1}}}"
                simple_map[label] = k
                inp = inp.replace(k, label)

            try:
                result = translate_fn(inp, source_lang=source_lang, target_lang=target_lang,
                                     strategy=strategy, domain=domain)
            except TypeError:
                result = translate_fn(inp, source_lang=source_lang, target_lang=target_lang,
                                     strategy=strategy)
            t = ""
            if isinstance(result, dict) and result.get("success"):
                raw = result.get("translated_text", "").strip()
                if raw.startswith("(") and "Previous:" in raw[:30]:
                    idx = raw.find(")")
                    if idx > 0: raw = raw[idx+1:].strip()
                t = _clean_output(raw, len(inp))
            if not t or len(t) < 2:
                t = inp

            for label, k in simple_map.items():
                t = t.replace(label, k)

            details.append({"index": i, "original": s, "translated": t, "has_context": False, "elapsed_s": 0})
            translated.append(t)
            ctx_clean = re.sub(r'▣\d+▣', '', t).strip()
            if ctx_clean: ctx_buf.append(ctx_clean)
            continue

        use_ctx = context_window > 0 and len(translated) > 0
        if use_ctx:
            ctx_part = " ".join(ctx_buf[-context_window:])
            prompt = f"(Previous: {ctx_part})  {s}"
        else:
            prompt = s

        import time as _time
        _t0 = _time.time()
        try:
            result = translate_fn(prompt, source_lang=source_lang, target_lang=target_lang,
                                 strategy=strategy, domain=domain)
        except TypeError:
            result = translate_fn(prompt, source_lang=source_lang, target_lang=target_lang,
                                 strategy=strategy)
        _elapsed = round(_time.time() - _t0, 3)

        t = ""
        if isinstance(result, dict) and result.get("success"):
            raw = result.get("translated_text", "").strip()
            # 去上下文回声：Qwen2.5 有时在译文前复述 "(Previous: ...)"
            if raw.startswith("(") and "Previous:" in raw[:30]:
                idx = raw.find(")")
                if idx > 0:
                    raw = raw[idx+1:].strip()
            t = _clean_output(raw, len(s))

        if not t or len(t) < 2:
            t = s

        details.append({"index": i, "original": s, "translated": t, "has_context": use_ctx, "elapsed_s": _elapsed})
        translated.append(t)
        if t and t != s:
            ctx_buf.append(t)

    # 4. 合并 + 恢复特殊格式
    merged = " ".join(t for t in translated if t)
    merged = restore_special_tokens(merged, placeholders)
    return merged, details
