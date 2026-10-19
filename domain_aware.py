# 领域感知模块（实训）
# 提供：场景自动识别、领域提示词注入、消融实验开关。
# 独立于 translator.py，方便做启用/关闭对比实验。
from typing import Dict, Optional


# 自动场景识别（含七大领域关键词）
DOMAIN_KEYWORDS = {
    "academic": [
        "神经网络", "深度学习", "机器学习", "损失函数", "优化器",
        "论文", "本文", "数据集", "训练集", "卷积", "消融", "基准模型",
        "neural network", "deep learning", "loss function", "optimizer",
        "paper", "dataset", "hypothesis", "ablation", "benchmark",
        "研究", "実験", "アルゴリズム", "モデル", "論文",
        "연구", "실험", "알고리즘", "모델", "논문"
    ],
    "business": [
        "合同", "协议", "报价", "订单", "客户", "销售", "营收", "利润",
        "季度", "财报", "审计", "预算", "投标", "招标", "采购",
        "contract", "agreement", "quotation", "order", "revenue", "profit",
        "quarterly", "audit", "budget", "bid", "supply chain",
        "会社", "契約", "売上", "利益", "顧客",
        "회사", "계약", "매출", "이익", "고객"
    ],
    "technical": [
        "API", "SDK", "配置", "部署", "接口", "代码", "编译", "调试",
        "服务器", "数据库", "缓存", "线程", "进程", "内存", "CPU", "GPU",
        "config", "deploy", "interface", "compile", "debug", "server",
        "database", "cache", "thread", "memory", "protocol",
        "サーバ", "インターフェース", "設定", "デプロイ",
        "서버", "인터페이스", "설정", "배포"
    ],
    "daily": [
        "今天", "昨天", "明天", "周末", "吃饭", "睡觉", "逛街", "电影",
        "咖啡", "奶茶", "天气", "上班", "下班", "地铁", "公交", "手机",
        "today", "yesterday", "tomorrow", "weekend", "eat", "friend",
        "family", "weather", "movie", "coffee",
        "今日", "昨日", "明日", "週末", "ご飯", "友達", "天気", "映画",
        "오늘", "어제", "내일", "주말", "밥", "친구", "날씨", "영화"
    ],
    "law": [
        "侵权", "仲裁", "诉讼", "判决", "法院", "法官", "原告", "被告",
        "知识产权", "违约金", "保密", "版权", "专利", "法人", "抵押", "担保",
        "tort", "arbitration", "litigation", "judgment", "court",
        "intellectual property", "breach of contract", "NDA",
        "特許", "契約", "訴訟", "判決", "裁判所",
        "특허", "계약", "소송", "판결", "법원"
    ],
    "medicine": [
        "临床", "诊断", "治疗", "患者", "病理", "预后", "药物", "剂量",
        "手术", "双盲", "安慰剂", "副作用", "适应症", "并发症",
        "clinical", "diagnosis", "prognosis", "pathology", "drug",
        "surgery", "placebo", "side effect", "complication",
        "臨床", "診断", "治療", "患者", "病理",
        "임상", "진단", "치료", "환자", "병리"
    ],
    "cs": [
        "算法", "编程", "软件", "硬件", "编译", "调试", "部署", "架构",
        "并发", "容器", "虚拟化", "数据库", "缓存", "微服务",
        "algorithm", "programming", "software", "hardware", "compile",
        "deploy", "architecture", "concurrency", "container", "microservice",
        "アルゴリズム", "プログラミング", "ソフトウェア", "アーキテクチャ",
        "알고리즘", "프로그래밍", "소프트웨어", "아키텍처"
    ],
}

# 基于关键词匹配自动识别文本所属领域。专业领域加权，无匹配不注入。
def detect_domain(text: str) -> str:
    text_lower = text.lower()
    domain_weight = {"law": 2.5, "medicine": 2.5, "cs": 2.5,
                     "academic": 0.6, "business": 1.0, "technical": 1.2, "daily": 1.0}
    scores = {}
    for domain, kws in DOMAIN_KEYWORDS.items():
        weight = domain_weight.get(domain, 1.0)
        scores[domain] = sum(weight for kw in kws if kw.lower() in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else ""


# 领域专属提示词注入
DOMAIN_INSTRUCTIONS = {
    "academic": "【学术论文】使用正式学术英语（formal academic English），被动语态优先，避免第一人称。术语如'实验结果表明'→'experimental results demonstrate'、'本文提出'→'this paper proposes'。",
    "business": "【商务文本】使用正式商务英语（business formal）。术语必须精确：'合同'→'contract'（不用agreement）、'营收'→'revenue'（不用income）、'季度'→'quarterly'（不用every three months）、'投标'→'bid/tender'。",
    "technical": "【技术文档】使用精准技术英语。代码和API名保持原文不翻译。'部署'→'deploy'（不用install）、'低延迟'→'low latency'（不用low delay）、'接口'→'interface/API endpoint'。",
    "daily": "【日常对话】使用自然口语化英语（colloquial English）。'吃饭'→'grab a meal/eat'、'逛街'→'go shopping'、语气轻松自然，可适当使用缩略形式（I'll, don't）。",
    "law": "【法律文本】使用法律英语（legal English），术语严格对应。'违约责任'→'breach of contract'、'不可抗力'→'force majeure'、'知识产权'→'intellectual property'、'仲裁'→'arbitration'。保留条款结构。",
    "medicine": "【医学文本】使用医学术语。'诊断'→'diagnosis'、'预后'→'prognosis'、'双盲随机对照试验'→'double-blind randomized controlled trial (RCT)'、'不良事件'→'adverse event'、'适应症'→'indication'。",
    "cs": "【计算机科学】使用CS专业英语。'算法复杂度'→'algorithmic complexity'、'微服务架构'→'microservice architecture'、'分布式系统'→'distributed system'、'并发'→'concurrency'。框架名/变量名保留原文。",
}

# 返回指定领域的提示词指令。
def get_domain_instruction(domain: str) -> str:
    return DOMAIN_INSTRUCTIONS.get(domain, "")


# ============================================================
# 消融实验控制
# ============================================================
# domain_detection / domain_instruction: 由实验 wrapper (server.py) 控制
# glossary_injection: 由 translator._build_prompt 检查
_ABLATION_FLAGS: Dict[str, bool] = {
    "domain_detection": True,
    "domain_instruction": True,
    "glossary_injection": True,
}

def set_ablation(feature: str, enabled: bool):
    if feature in _ABLATION_FLAGS:
        _ABLATION_FLAGS[feature] = enabled

# 查询消融实验开关状态。
def get_ablation(feature: str) -> bool:
    return _ABLATION_FLAGS.get(feature, True)

# 返回所有消融开关状态。
def get_ablation_state() -> Dict[str, bool]:
    return dict(_ABLATION_FLAGS)
