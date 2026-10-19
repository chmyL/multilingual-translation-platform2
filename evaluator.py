# 翻译质量评估模块
# 基于 BLEU 指标对翻译结果进行自动评估，支持人工打分。
# (BLEU: sacrebleu 纯 Python 无额外依赖; METEOR: 用字符级近似代替 NLTK)
import json
import os
import math
from typing import Dict, Any, List, Optional
from datetime import datetime
from sacrebleu.metrics import BLEU
import re
from config import EVALUATION_METRICS, MANUAL_SCORE_CRITERIA

# BLEU 计算器
bleu_calc = BLEU(effective_order=True)

# 按语言分词。中文/日文用字符级，英文/韩文用空格（韩文也适用）。
def _tokenize(text: str, lang: str = "en") -> list:
    text = text.lower().strip()
    if lang in ("zh", "ja"):
        # 中文/日文：去掉空格，按字符切分（保留汉字、假名、数字、字母）
        chars = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af a-z0-9]", text)
        return [c for c in chars if c.strip()]
    else:
        # 英文/韩文：空格分词 + 去标点
        text = re.sub(r"[^\w\s]", " ", text)
        return [w for w in text.split() if w]

# 计算 BLEU 评分 (sacrebleu)
def compute_bleu(reference: str, candidate: str, lang: str = "en") -> Dict[str, float]:
    if not reference or not candidate:
        return {"bleu": 0.0}

    # 如果是中文或日文，用字符级切分并加空格，骗过 sacrebleu 的按空格分词
    if lang in ("zh", "ja"):
        reference = " ".join(_tokenize(reference, lang))
        candidate = " ".join(_tokenize(candidate, lang))

    try:
        result = bleu_calc.sentence_score(candidate, [reference])
        return {"bleu": round(result.score, 2)}
    except Exception:
        try:
            result = bleu_calc.corpus_score([candidate], [[reference]])
            return {"bleu": round(result.score, 2)}
        except Exception:
            return {"bleu": 0.0}

# 简化 METEOR 近似 (纯 Python，无需 NLTK)
# 基于 unigram 匹配 + 罚分
def compute_meteor_simple(reference: str, candidate: str, lang: str = "en") -> Dict[str, float]:
    if not reference or not candidate:
        return {"meteor": 0.0}

    ref_words = _tokenize(reference, lang)
    cand_words = _tokenize(candidate, lang)

    if not ref_words or not cand_words:
        return {"meteor": 0.0}

    ref_set = set(ref_words)
    cand_set = set(cand_words)
    matches = len(ref_set & cand_set)

    precision = matches / len(cand_words) if cand_words else 0
    recall = matches / len(ref_words) if ref_words else 0

    alpha = 0.9
    if precision + recall > 0:
        f_mean = precision * recall / ((1 - alpha) * recall + alpha * precision)
    else:
        f_mean = 0.0

    chunks = 0
    in_chunk = False
    for w in cand_words:
        if w in ref_set:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False

    penalty = 0.5 * ((chunks / max(matches, 1)) ** 3) if matches > 0 else 0.5
    score = f_mean * (1 - penalty)
    return {"meteor": round(max(0, score), 2)}

# 对翻译结果进行多指标自动评估
def evaluate(reference: str, candidate: str, lang: str = "en",
             metrics: Optional[List[str]] = None,) -> Dict[str, Any]:
    if metrics is None:
        metrics = EVALUATION_METRICS

    results = {
        "reference": reference,
        "candidate": candidate,
        "metrics": {},
        "timestamp": datetime.now().isoformat(),
    }

    if "BLEU" in metrics:
        results["metrics"]["BLEU"] = compute_bleu(reference, candidate, lang)

    if "METEOR" in metrics:
        results["metrics"]["METEOR"] = compute_meteor_simple(reference, candidate, lang)

    return results

# 批量评估翻译结果
def batch_evaluate(references: List[str], candidates: List[str],
                   lang: str = "en", metrics: Optional[List[str]] = None,) -> List[Dict[str, Any]]:
    results = []
    for i, (ref, cand) in enumerate(zip(references, candidates)):
        eval_result = evaluate(ref, cand, lang, metrics)
        eval_result["index"] = i
        results.append(eval_result)
    return results

# 批量评估汇总统计
def batch_evaluate_summary(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    bleu_scores = []
    meteor_scores = []
    for r in batch_results:
        m = r.get("metrics", {})
        if "bleu" in m.get("BLEU", {}):
            bleu_scores.append(m["BLEU"]["bleu"])
        if "meteor" in m.get("METEOR", {}):
            meteor_scores.append(m["METEOR"]["meteor"])

    return {
        "count": len(batch_results),
        "BLEU": {
            "avg": round(sum(bleu_scores) / len(bleu_scores), 2) if bleu_scores else 0,
            "max": max(bleu_scores) if bleu_scores else 0,
            "min": min(bleu_scores) if bleu_scores else 0,
        },
        "METEOR": {
            "avg": round(sum(meteor_scores) / len(meteor_scores), 2) if meteor_scores else 0,
            "max": max(meteor_scores) if meteor_scores else 0,
            "min": min(meteor_scores) if meteor_scores else 0,
        },
    }

# 人工打分
class ManualScoring:
    def __init__(self, scores_file: str = ""):
        self.scores: List[Dict[str, Any]] = []
        self.scores_file = scores_file
        if scores_file and os.path.exists(scores_file):
            self.load(scores_file)

    def add_score(
        self, source_text: str, translated_text: str, reference_text: str,
        accuracy: int, fluency: int, adequacy: int,
        engine: str = "", notes: str = "",
        error_categories: list = None, suggested_translation: str = "",
    ) -> Dict[str, Any]:
        record = {
            "id": len(self.scores) + 1,
            "source_text": source_text,
            "translated_text": translated_text,
            "reference_text": reference_text,
            "scores": {
                "accuracy": max(1, min(5, accuracy)),
                "fluency": max(1, min(5, fluency)),
                "adequacy": max(1, min(5, adequacy)),
            },
            "total_score": accuracy + fluency + adequacy,
            "engine": engine, "notes": notes,
            "error_categories": error_categories or [],
            "suggested_translation": suggested_translation,
            "timestamp": datetime.now().isoformat(),
        }
        self.scores.append(record)
        return record

    def get_summary(self) -> Dict[str, Any]:
        if not self.scores:
            return {"count": 0, "avg_scores": {}, "total_avg": 0}
        acc_list = [s["scores"]["accuracy"] for s in self.scores]
        flu_list = [s["scores"]["fluency"] for s in self.scores]
        ade_list = [s["scores"]["adequacy"] for s in self.scores]
        total_list = [s["total_score"] for s in self.scores]
        return {
            "count": len(self.scores),
            "avg_scores": {
                "accuracy": round(sum(acc_list) / len(acc_list), 2),
                "fluency": round(sum(flu_list) / len(flu_list), 2),
                "adequacy": round(sum(ade_list) / len(ade_list), 2),
            },
            "total_avg": round(sum(total_list) / len(total_list), 2),
        }

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.scores, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.scores = json.load(f)

    # 聚合反馈数据，返回错误分布和建议译文对
    def get_feedback_summary(self) -> Dict[str, Any]:
        if not self.scores:
            return {"count": 0, "error_distribution": {}, "suggestions": [], "insights": []}
        error_counts = {}
        suggestions = []
        for s in self.scores:
            for cat in s.get("error_categories", []):
                error_counts[cat] = error_counts.get(cat, 0) + 1
            if s.get("suggested_translation"):
                suggestions.append({
                    "source": s.get("source_text", "")[:100],
                    "original": s.get("translated_text", "")[:100],
                    "suggested": s["suggested_translation"][:100],
                })
        insights = []
        total = len(self.scores) # 总反馈条数
        for cat, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            # 计算该错误占总条数百分比，保留1位小数
            pct = round(count / total * 100, 1)
            hints = {
                "术语错误": f"{pct}% 反馈涉及术语错误，建议加强领域术语表。",
                "语法问题": f"{pct}% 反馈存在语法问题，建议强化语法规范提示。",
                "风格不当": f"{pct}% 反馈风格不当，建议明确风格要求。",
                "漏译": f"{pct}% 反馈存在漏译，建议强调完整性检查。",
                "冗余": f"{pct}% 反馈表达冗余，建议增加简洁性约束。",
            }
            insights.append({
                "category": cat, "count": count, "percentage": pct,
                "suggestion": hints.get(cat, f"{cat}类问题占比{pct}%，建议优化相关提示。"),
            })
        return {
            "count": total,
            "error_distribution": error_counts,
            "suggestion_count": len(suggestions),
            "recent_suggestions": suggestions[-5:],
            "insights": insights,
        }

    def get_all_scores(self) -> List[Dict[str, Any]]:
        return self.scores
