# 学术风格自动分析器（支持中英双语）
# 无外部依赖，纯正则+统计。
import re
from typing import Dict

# 中文风格分析
def _cn_style(text: str, word_count: int, sent_count: int) -> Dict:
    cn_words = re.findall(r'[\w一-鿿]+', text)
    all_tokens = cn_words if cn_words else ['x']

    # 中文正式连接词
    cn_formal = len(re.findall(r'(?:因此|然而|此外|综上|值得|进一步|显著|本文|实验|研究|分析|证明|验证|评估)', text))
    # 中文学术特征词
    cn_academic = len(re.findall(r'(?:模型|算法|数据|方法|框架|架构|指标|比率|参数|阈值|机制|范式|尺度)', text))
    # 中文口语词
    cn_colloq = len(re.findall(r'(?:搞|贼|挺|蛮|啥|咋|嘛|呗|吧|啊|啦|超|简直|特别|比较|有点|稍微)', text))
    # 第一人称
    cn_first = len(re.findall(r'(?:我|我们|咱|俺)', text))
    # 句均长度
    avg_sent_len = word_count / max(sent_count, 1)
    # 平均词长
    avg_word_len = sum(len(t) for t in all_tokens) / max(len(all_tokens), 1)

    passive_ratio = 0  # 中文无形态被动
    # 连接词密度 = (正式词 + 学术词) / 句数 × 10
    conn_density = (cn_formal + cn_academic) / max(sent_count, 1) * 10
    # 第一人称占比 = 第一人称数 / 总词数 × 100
    first_ratio = cn_first / max(word_count, 1) * 100
    # 口语词占比 = 口语词数 / 总词数 × 100
    collo_ratio = cn_colloq / max(word_count, 1) * 100
    raw = conn_density / 2 + avg_word_len / 2 - collo_ratio * 2 - first_ratio
    norm = max(0.0, min(1.0, (raw + 10) / 25))

    return {
        "word_count": word_count,
        "sentence_count": sent_count,
        "avg_sentence_length": round(avg_sent_len, 1),
        "avg_word_length": round(avg_word_len, 1),
        "passive_count": 0,
        "passive_ratio_per_100w": 0.0,
        "connector_count": cn_formal + cn_academic,
        "connector_density_10s": round(conn_density, 1),
        "first_person_count": cn_first,
        "first_person_ratio": round(first_ratio, 2),
        "colloquial_count": cn_colloq,
        "formality_score_0_to_1": round(norm, 2),
    }

# 英文风格分析
def _en_style(text: str, word_count: int, sent_count: int) -> Dict:
    words = [w for w in text.split() if w.strip()]
    # 被动
    passive = len(re.findall(r'\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?(?:\w+ed|built|made|taken|given|known|seen|shown|done|found)\b', text, re.I))
    # 连接词
    conns = len(re.findall(r'\b(?:therefore|however|moreover|furthermore|consequently|nevertheless|nonetheless|accordingly|hence|thus|whereas|additionally|in addition|in contrast|on the other hand|as a result|for instance|specifically)\b', text, re.I))
    # 第一人称
    first = len(re.findall(r'\b(?:I\b|my\b|we\b|our\b|us\b)', text, re.I))
    # 口语词
    colloq = len(re.findall(r'\b(?:kind of|sort of|a lot|stuff|thing|really|very|actually|basically|just|like|you know|anyway|ok|okay|great|nice)\b', text, re.I))

    avg_sent = word_count / max(sent_count, 1)
    avg_word = sum(len(w) for w in words) / max(len(words), 1)
    pass_ratio = passive / max(word_count, 1) * 100
    conn_dens = conns / max(sent_count, 1) * 10
    first_ratio = first / max(word_count, 1) * 100
    collo_ratio = colloq / max(word_count, 1) * 100
    raw = pass_ratio + conn_dens / 2 + avg_word / 2 - collo_ratio * 2 - first_ratio * 3
    norm = max(0.0, min(1.0, (raw + 10) / 25))

    return {
        "word_count": word_count,
        "sentence_count": sent_count,
        "avg_sentence_length": round(avg_sent, 1),
        "avg_word_length": round(avg_word, 1),
        "passive_count": passive,
        "passive_ratio_per_100w": round(pass_ratio, 2),
        "connector_count": conns,
        "connector_density_10s": round(conn_dens, 1),
        "first_person_count": first,
        "first_person_ratio": round(first_ratio, 2),
        "colloquial_count": colloq,
        "formality_score_0_to_1": round(norm, 2),
    }


def analyze(text: str) -> Dict:
    if not text or not text.strip():
        return {"error": "empty text"}

    cmap = {'。': '. ', '！': '! ', '？': '? ', '；': '; '}
    simple = text
    for k, v in cmap.items():
        simple = simple.replace(k, v)
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', simple) if s.strip() and len(s) >= 2]
    sent_count = max(len(sentences), 1)
    cn_chars = len(re.findall(r'[一-鿿]', text))
    en_tokens = len([w for w in text.split() if re.match(r'^[a-zA-Z]', w)])
    word_count = max(cn_chars + en_tokens, 1)

    if cn_chars > en_tokens * 2:
        return _cn_style(text, word_count, sent_count)
    else:
        return _en_style(text, word_count, sent_count)


def compare(original: str, polished: str) -> Dict:
    a = analyze(original)
    b = analyze(polished)
    if "error" in a or "error" in b:
        return {"error": a.get("error") or b.get("error")}
    def delta(key):
        return round(b.get(key, 0) - a.get(key, 0), 2)
    return {
        "original": a, "polished": b,
        "delta": {k: delta(k) for k in [
            "avg_sentence_length", "passive_ratio_per_100w", "connector_density_10s",
            "first_person_ratio", "colloquial_count", "formality_score_0_to_1"
        ]}
    }
