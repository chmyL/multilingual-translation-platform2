# domain_classifier.py
# 基于多语言sentence embedding+原型句相似度，替代纯关键词匹配。
import os
import threading
from typing import Dict, Tuple, Optional, List
import numpy as np

# 领域列表
DOMAINS = ["academic", "business", "technical", "daily", "law", "medicine", "cs"]

# 每个领域的原型句（覆盖中英日韩，加得越典型越准）
PROTOTYPES: Dict[str, List[str]] = {
    "academic": [
        "本文提出了一种新的算法模型，实验结果表明该方法显著优于基线。",
        "研究采用定量分析方法，对收集的数据进行了统计检验。",
        "This paper proposes a novel algorithm and the experimental results demonstrate its superiority.",
        "本研究では新しいアルゴリズムを提案し、実験結果はその有効性を示している。",
    ],
    "business": [
        "本季度公司营收同比增长百分之十五，利润也有明显提升。",
        "双方就合同条款进行了多轮谈判，最终达成了合作协议。",
        "The company's revenue grew by 15% year-over-year this quarter.",
        "当四半期の売上高は前年同期比で15%増加し、利益も改善した。",
    ],
    "technical": [
        "请将服务器端口配置为8080，并确保防火墙允许该端口的入站连接。",
        "该接口在高并发场景下的响应延迟需要控制在五十毫秒以内。",
        "Please configure the server port to 8080 and ensure the firewall allows inbound connections.",
        "서버 포트를 8080으로 설정하고 방화벽에서 인바운드 연결을 허용하십시오.",
    ],
    "daily": [
        "今天天气不错，我们出去走走吧。",
        "昨天晚上看了一部很好看的电影，推荐你也看看。",
        "The weather is nice today; let's go for a walk.",
        "今日はいい天気ですね、散歩でも行きましょうか。",
    ],
    "law": [
        "双方需严格履行本协议约定的违约责任与保密义务。",
        "原告向法院提起诉讼，主张被告侵犯其知识产权。",
        "Both parties shall strictly perform the breach of contract and NDA obligations.",
    ],
    "medicine": [
        "该临床试验采用双盲随机对照设计，评估新药的适应症与不良反应。",
        "患者的预后情况取决于早期诊断及手术切除的彻底程度。",
        "The clinical trial uses a double-blind randomized controlled design.",
    ],
    "cs": [
        "该微服务架构采用容器化部署，通过消息队列解耦各个节点。",
        "算法的空间复杂度为 O(n)，需使用哈希表进行优化。",
        "The microservice architecture uses containerized deployment and a message queue.",
    ],
}

# 关键词兜底
_FALLBACK_KEYWORDS = {
    "academic": ["研究", "实验", "算法", "模型", "论文", "假设", "定理", "统计",
                 "research", "paper", "experiment"],
    "business": ["公司", "合同", "营收", "利润", "客户", "销售", "市场", "投资",
                 "company", "revenue", "contract"],
    "technical": ["服务器", "接口", "配置", "部署", "端口", "数据库", "防火墙",
                  "server", "deploy", "config"],
    "daily": ["今天", "昨天", "周末", "吃饭", "朋友", "家人", "天气", "电影",
              "today", "weekend", "weather"],
    "law": ["侵权", "仲裁", "诉讼", "判决", "法院", "原告", "被告", "知识产权",
            "court", "litigation", "tort"],
    "medicine": ["临床", "诊断", "治疗", "患者", "病理", "药物", "手术", "双盲",
                 "clinical", "diagnosis", "surgery"],
    "cs": ["算法", "编程", "软件", "编译", "并发", "容器", "微服务", "数据结构",
           "algorithm", "software", "microservice"],
}


class DomainClassifier:
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.MODEL_NAME
        self._model = None
        self._centers = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            # 强制离线，避免网络卡顿
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            print(f"[DomainClassifier] 加载模型: {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)

            self._centers = {}
            for domain, sentences in PROTOTYPES.items():
                embs = self._model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
                center = embs.mean(axis=0)
                norm = np.linalg.norm(center)
                if norm > 0:
                    center = center / norm
                self._centers[domain] = center
            print(f"[DomainClassifier] 就绪，覆盖 {len(self._centers)} 个领域")

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        if not text or not text.strip():
            return "academic", 0.0, {d: 0.0 for d in DOMAINS}

        self._ensure_loaded()
        emb = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]

        sims = {d: float(np.dot(emb, center)) for d, center in self._centers.items()}
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        best_domain, best_sim = ranked[0]
        second_sim = ranked[1][1] if len(ranked) > 1 else 0.0
        return best_domain, best_sim - second_sim, sims

    @staticmethod
    def keyword_fallback(text: str) -> str:
        text_lower = text.lower()
        scores = {d: sum(1 for kw in kws if kw.lower() in text_lower) for d, kws in _FALLBACK_KEYWORDS.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "academic"


# 全局单例
_classifier: Optional[DomainClassifier] = None
_classifier_lock = threading.Lock()


def _get_classifier() -> DomainClassifier:
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = DomainClassifier()
    return _classifier


def classify_domain(text: str, confidence_threshold: float = 0.08) -> Tuple[str, float, str]:
    try:
        clf = _get_classifier()
        domain, conf, _ = clf.predict(text)
        if conf >= confidence_threshold:
            return domain, conf, "embedding"
        return DomainClassifier.keyword_fallback(text), conf, "keyword"
    except Exception as e:
        print(f"[DomainClassifier] 分类失败: {e}")
        return DomainClassifier.keyword_fallback(text), 0.0, "keyword"


def get_similarity_distribution(text: str) -> Dict[str, float]:
    try:
        clf = _get_classifier()
        _, _, sims = clf.predict(text)
        return sims
    except Exception:
        return {}