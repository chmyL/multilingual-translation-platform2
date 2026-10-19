# 多语言翻译服务
# 支持中英、中日、中韩双向互译，三种翻译策略（直译、意译、学术润色翻译）。
from typing import List, Optional, Dict, Any
from config import (
    LANGUAGE_PAIRS,
    LANGUAGE_NAMES,
    TRANSLATION_STRATEGIES,
)
from llm_adapter import chat
from domain_classifier import classify_domain

# 场景检测 + 领域模板
DOMAIN_KEYWORDS = {
    "academic": [
        "神经网络", "深度学习", "机器学习", "损失函数", "优化器",
        "论文", "本文", "数据集", "训练集", "卷积",
        "序列", "回归", "聚类", "解码", "embedding",
        "neural network", "deep learning", "loss function", "optimizer",
        "paper", "dataset", "hypothesis",
        "实验组", "对照组", "消融", "基准模型", "state-of-the-art",
        "accuracy", "F1-score", "benchmark",
    ],
    "business": [
        "合同", "协议", "报价", "订单", "客户", "销售", "营收", "利润",
        "季度", "财报", "审计", "预算", "投标", "招标", "采购", "供应链",
        "营销", "市场", "品牌", "竞标", "尽职调查", "贴现", "结算",
        "contract", "agreement", "quotation", "order", "revenue", "profit",
        "quarterly", "audit", "budget", "bid", "supply chain", "marketing",
    ],
    "technical": [
        "API", "SDK", "配置", "部署", "接口", "代码", "编译", "调试",
        "服务器", "数据库", "缓存", "线程", "进程", "内存", "CPU", "GPU",
        "协议", "带宽", "延迟", "吞吐", "终端", "容器", "镜像",
        "config", "deploy", "interface", "compile", "debug", "server",
        "database", "cache", "thread", "memory", "protocol", "bandwidth",
    ],
    "daily": [
        "今天", "昨天", "明天", "周末", "吃饭", "睡觉", "逛街", "电影",
        "咖啡", "奶茶", "天气", "上班", "下班", "地铁", "公交", "手机",
        "朋友", "家人", "开心", "难过", "有时间", "最近", "怎么样",
    ],
    "law": [
        "合同", "协议", "侵权", "仲裁", "诉讼", "判决", "法院", "法官",
        "原告", "被告", "知识产权", "违约金", "保密", "版权", "专利",
        "法人", "抵押", "担保", "破产", "清算", "损害赔偿",
        "证据", "证人", "管辖", "法条", "条款", "调解", "免责", "终止",
    ],
    "medicine": [
        "临床", "诊断", "治疗", "患者", "病理", "预后", "药物", "剂量",
        "手术", "试验", "双盲", "安慰剂", "副作用", "适应症", "并发症",
        "影像", "检测", "筛查", "发病率", "死亡率", "切除", "恶性", "转移",
        "化疗", "免疫", "靶向", "炎症", "糖尿病", "高血压", "疫苗",
    ],
    "cs": [
        "算法", "编程", "软件", "硬件", "编译", "调试", "部署", "架构",
        "并发", "容器", "虚拟化", "负载", "数据库", "缓存", "微服务",
        "API", "RESTful", "分布式", "排序", "搜索", "递归", "二叉树",
        "哈希", "时间复杂度", "设计模式", "持续集成", "容器编排",
    ],
}

# 根据关键词自动检测场景（优先用 embedding 模型）
def detect_domain(text: str) -> str:
    try:
        domain, conf, method = classify_domain(text)
        if conf > 0.1:  # 如果有一定置信度，用 embedding 结果
            return domain
    except Exception:
        pass
    # 兜底：用旧的关键词匹配
    text_lower = text.lower()
    scores = {}
    domain_weight = {"law": 2.5, "medicine": 2.5, "cs": 2.5,
                     "academic": 0.6, "business": 1.0, "technical": 1.2, "daily": 1.0}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        weight = domain_weight.get(domain, 1.0)
        scores[domain] = sum(weight for kw in keywords if kw.lower() in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else ""

# 返回场景翻译提示。风格指引，保持等长。
def _get_domain_instructions(domain: str, src_name: str, tgt_name: str) -> str:
    if tgt_name == "英文":
        guides = {
            "academic": "使用正式学术英语，优先采用被动语态。",
            "business": "使用正式商务英语，杜绝口语化表达。",
            "technical": "使用严谨技术英语，代码/API/配置项保持原文不翻译。",
            "daily": "使用自然口语化英语，可正常使用缩写（如 don't、won't）。",
            "law": "使用法律专业术语，采用正式书面英语，保留条款结构。",
            "medicine": "使用医学专业术语，采用标准规范英语。",
            "cs": "使用计算机专业术语，框架名/变量名保留原文。",
        }
    elif tgt_name == "中文":
        guides = {
            "academic": "使用正式学术中文，句式规范严谨。",
            "business": "使用正式商务中文，敬语得体、格式规范。",
            "technical": "使用精准技术中文，代码/API/配置项保持原文不翻译。",
            "daily": "使用自然口语化中文，表达地道流畅。",
            "law": "使用法律专业术语，条款结构严谨。",
            "medicine": "使用医学专业术语，表达准确规范。",
            "cs": "使用计算机专业术语，框架名/变量名保留原文。",
        }
    elif tgt_name == "日文":
        guides = {
            "academic": "使用正式学术日文，句式严谨规范。",
            "business": "使用正式商务日文，敬语得体。",
            "technical": "使用精准技术日文，代码/API/配置项保持原文。",
            "daily": "使用自然口语化日文。",
            "law": "使用法律专业术语，条款结构严谨。",
            "medicine": "使用医学专业术语，表达准确规范。",
            "cs": "使用计算机专业术语，框架名/变量名保留原文。",
        }
    elif tgt_name == "韩文":
        guides = {
            "academic": "使用正式学术韩文，句式严谨规范。",
            "business": "使用正式商务韩文，敬语得体。",
            "technical": "使用精准技术韩文，代码/API/配置项保持原文。",
            "daily": "使用自然口语化韩文。",
            "law": "使用法律专业术语，条款结构严谨。",
            "medicine": "使用医学专业术语，表达准确规范。",
            "cs": "使用计算机专业术语，框架名/变量名保留原文。",
        }
    else:
        guides = {}
    return f"[{domain}] {guides.get(domain, '')}" if guides.get(domain) else ""

# 根据语言对和策略构建 messages
def _build_prompt(text: str, source_lang: str, target_lang: str,
                  strategy: str = "semantic", domain: str = "",) -> List[Dict[str, str]]:
    src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    strategy_info = TRANSLATION_STRATEGIES.get(strategy, TRANSLATION_STRATEGIES["semantic"])

    # 针对不同策略的 system prompt
    if strategy == "literal":
        system_prompt = (
            f"你是一个专业翻译助手。请将以下{src_name}文本直译为{tgt_name}。"
            f"要求：尽可能保持原文的词序、句式和结构，逐句字面翻译。"
            f"只输出翻译结果，不要添加任何解释或注释。"
        )
    elif strategy == "academic":
        system_prompt = (
            f"你是一个学术翻译专家。请将以下{src_name}文本翻译为{tgt_name}，"
            f"并采用学术论文的正式写作风格。"
            f"要求：使用正式、严谨的学术表达；优化句式使其符合学术规范；"
            f"专业术语准确；避免口语化表达。"
            f"只输出翻译结果，不要添加任何解释或注释。"
        )
    else:  # semantic（默认意译）
        system_prompt = (
            f"你是一个专业翻译助手。请将以下{src_name}文本翻译为{tgt_name}。"
            f"要求：准确传达原文语义，译文自然流畅，可适当调整句式结构以确保地道表达。"
            f"只输出翻译结果，不要添加任何解释或注释。"
        )

    # 针对特定语言对的额外说明 — 强制目标语言输出
    lang_specific = ""
    if target_lang == "zh":
        lang_specific = "\n【重要】你必须输出中文（简体中文）。不要输出英文，不要输出其他语言。只输出纯中文翻译结果。"
    elif target_lang == "en":
        lang_specific = "\n【重要】你必须输出英文（English）。不要输出中文，不要输出其他语言。只输出纯英文翻译结果。"
    elif target_lang == "ja":
        lang_specific = "\n【重要】你必须输出日文。使用适当的敬语和日文书面表达习惯。只输出纯日文翻译结果。"
    elif target_lang == "ko":
        lang_specific = "\n【重要】你必须输出韩文。使用适当的韩文敬语和书面表达习惯。只输出纯韩文翻译结果。"

    system_prompt += lang_specific

    # 场景特定提示
    if domain:
        system_prompt += "\n" + _get_domain_instructions(domain, src_name, tgt_name)

    # 术语注入：从输入文本匹配专业术语，选最重要的3个注入（避免上下文过载）
    if domain in ("law", "medicine", "cs") and target_lang == "en":
        from domain_aware import get_ablation as _abl_check
        if _abl_check("glossary_injection"):
            from domain_glossary import get_relevant_terms
            matched = get_relevant_terms(text, domain)
            if matched:
                # 优先注入可能有歧义的关键术语（出现在文本最前面/最长的）
                items = sorted(matched.items(), key=lambda x: text.find(x[0]))
                items = items[:3]  # 最多注入3个术语
                term_str = ", ".join(f"{cn}→{en.split('/')[0].strip()}" for cn, en in items)
                system_prompt += f"\n术语参考：{term_str}"
                print(f"[translator] 术语注入: {domain} matched={len(matched)}, injected={len(items)} terms")

    # DeepSeek-R1会把system prompt当文本翻译，改用user消息前缀方式
    from vllm_manager import get_actual_running_model
    _running = get_actual_running_model()
    if _running and "DeepSeek" in str(_running):
        return [
            {"role": "user", "content": f"Translate this {src_name} text to {tgt_name}. Only output the translation, nothing else: {text}"},
        ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

# 翻译单条文本
def translate(text: str, source_lang: str = "zh", target_lang: str = "en", strategy: str = "semantic",
              domain: str = "", source_domain: str = "", model: str = "", temperature: float = 0.3, max_tokens: int = 128,) -> Dict[str, Any]:
    if not text or not text.strip():
        return {
            "success": False,
            "translated_text": "",
            "strategy": strategy,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "error": "输入文本为空",
        }

    # 兼容实验传过来的 source_domain
    actual_domain = source_domain or domain

    # 场景自动检测
    if actual_domain == "auto":
        actual_domain = detect_domain(text)

    messages = _build_prompt(text, source_lang, target_lang, strategy, actual_domain)
    strategy_name = TRANSLATION_STRATEGIES.get(strategy, {}).get("name", strategy)

    # DeepSeek-R1 需要更多 token 完成推理链，同时取最后一句真实翻译
    from vllm_manager import get_actual_running_model
    current_model = get_actual_running_model()
    is_deepseek = current_model and "DeepSeek" in str(current_model)

    try:
        translated = chat(messages, temperature=temperature,
                         max_tokens=512 if is_deepseek else max_tokens)

        # DeepSeek 推理链提取
        if is_deepseek and translated:
            from llm_adapter import strip_deepseek_thinking
            translated = strip_deepseek_thinking(translated)

        return {
            "success": True,
            "translated_text": translated,
            "strategy": strategy,
            "strategy_name": strategy_name,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "domain": actual_domain or "",
            "domain_name": {"academic":"学术论文","business":"商务","technical":"技术文档","daily":"日常","law":"法律","medicine":"医学","cs":"计算机科学"}.get(actual_domain, "无") if actual_domain else "无",
            "source_lang_name": LANGUAGE_NAMES.get(source_lang, source_lang),
            "target_lang_name": LANGUAGE_NAMES.get(target_lang, target_lang),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "translated_text": "",
            "strategy": strategy,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "error": str(e),
        }

# 批量翻译多条文本
def batch_translate(texts: List[str], source_lang: str = "zh", target_lang: str = "en",
                    strategy: str = "semantic", model: str = "",) -> List[Dict[str, Any]]:
    results = []
    for i, text in enumerate(texts):
        result = translate(text, source_lang, target_lang, strategy, model)
        result["index"] = i
        results.append(result)
    return results

# 对同一文本使用多种翻译策略，用于对比实验
def translate_multiple_strategies(text: str, source_lang: str = "zh", target_lang: str = "en",
                                  strategies: Optional[List[str]] = None, model: str = "",) -> Dict[str, Any]:
    if strategies is None:
        strategies = list(TRANSLATION_STRATEGIES.keys())

    output = {}
    for s in strategies:
        result = translate(text, source_lang, target_lang, s, model)
        output[s] = result

    return {
        "original_text": text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "results": output,
    }

# 同一文本使用不同模型引擎对比翻译
def translate_with_multiple_engines(text: str, source_lang: str = "zh", target_lang: str = "en",
                                    engine_models: Optional[List[str]] = None, strategy: str = "semantic",) -> Dict[str, Any]:
    translations = {}
    for model_name in (engine_models or []):
        result = translate(text, source_lang, target_lang, strategy, model_name)
        translations[model_name] = result

    return {
        "original_text": text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "strategy": strategy,
        "translations": translations,
    }
