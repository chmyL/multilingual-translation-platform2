# 领域术语词典
# 为法律、医学、计算机科学三个领域提供中英文术语对照表，
# 在翻译prompt中注入相关术语以提升领域翻译专业性。

# 法律领域术语（35条）
LAW_GLOSSARY = {
    "原告": "plaintiff",
    "被告": "defendant",
    "合同": "contract",
    "协议": "agreement",
    "侵权": "tort",
    "管辖": "jurisdiction",
    "仲裁": "arbitration",
    "知识产权": "intellectual property",
    "违约责任": "breach of contract",
    "不可抗力": "force majeure",
    "保密协议": "non-disclosure agreement (NDA)",
    "诉讼": "litigation",
    "判决": "judgment",
    "裁决": "ruling / award",
    "调解": "mediation",
    "法条": "statutory provision",
    "条款": "clause / provision",
    "版权": "copyright",
    "专利权": "patent",
    "商标": "trademark",
    "法人": "legal entity",
    "自然人": "natural person",
    "担保": "guarantee / security",
    "抵押": "mortgage",
    "破产": "bankruptcy",
    "清算": "liquidation",
    "善意第三方": "bona fide third party",
    "格式条款": "standard form clause",
    "损害赔偿": "damages",
    "禁令": "injunction",
    "证人": "witness",
    "证据": "evidence",
    "举证责任": "burden of proof",
    "豁免": "exemption / immunity",
    "终止": "termination",
}

# 医学领域术语（40条）
MEDICINE_GLOSSARY = {
    "诊断": "diagnosis",
    "预后": "prognosis",
    "病理": "pathology",
    "临床试验": "clinical trial",
    "安慰剂": "placebo",
    "双盲": "double-blind",
    "随机对照试验": "randomized controlled trial (RCT)",
    "流行病学": "epidemiology",
    "并发症": "complication",
    "适应症": "indication",
    "禁忌症": "contraindication",
    "副作用": "adverse effect / side effect",
    "剂量": "dosage",
    "给药": "administration",
    "静脉注射": "intravenous injection",
    "口服": "oral administration",
    "预后标志物": "prognostic marker",
    "生物标志物": "biomarker",
    "影像学": "imaging",
    "磁共振成像": "magnetic resonance imaging (MRI)",
    "计算机断层扫描": "computed tomography (CT)",
    "病理切片": "histopathology slide",
    "良性": "benign",
    "恶性": "malignant",
    "转移": "metastasis",
    "切除": "resection",
    "化疗": "chemotherapy",
    "放疗": "radiotherapy",
    "免疫治疗": "immunotherapy",
    "靶向治疗": "targeted therapy",
    "抗生素": "antibiotic",
    "疫苗": "vaccine",
    "发病率": "incidence rate",
    "死亡率": "mortality rate",
    "生存率": "survival rate",
    "糖尿病": "diabetes mellitus",
    "高血压": "hypertension",
    "心衰": "heart failure",
    "炎症": "inflammation",
    "自身免疫": "autoimmunity",
}

# 计算机科学领域术语（40条）
CS_GLOSSARY = {
    "算法": "algorithm",
    "数据结构": "data structure",
    "编译器": "compiler",
    "操作系统": "operating system",
    "并发": "concurrency",
    "并行计算": "parallel computing",
    "虚拟化": "virtualization",
    "微服务": "microservice",
    "容器化": "containerization",
    "负载均衡": "load balancing",
    "容错": "fault tolerance",
    "高可用": "high availability",
    "消息队列": "message queue",
    "缓存": "cache",
    "数据库": "database",
    "索引": "index",
    "分布式系统": "distributed system",
    "一致性": "consistency",
    "共识算法": "consensus algorithm",
    "哈希表": "hash table",
    "二叉树": "binary tree",
    "时间复杂度": "time complexity",
    "空间复杂度": "space complexity",
    "递归": "recursion",
    "迭代": "iteration",
    "排序": "sorting",
    "搜索": "search",
    "动态规划": "dynamic programming",
    "贪心算法": "greedy algorithm",
    "回溯": "backtracking",
    "设计模式": "design pattern",
    "工厂模式": "factory pattern",
    "观察者模式": "observer pattern",
    "API端点": "API endpoint",
    "RESTful架构": "RESTful architecture",
    "部署": "deployment",
    "持续集成": "continuous integration (CI)",
    "持续交付": "continuous delivery (CD)",
    "容器编排": "container orchestration",
    "服务网格": "service mesh",
}

# 领域→术语词典映射
DOMAIN_GLOSSARIES = {
    "law": LAW_GLOSSARY,
    "medicine": MEDICINE_GLOSSARY,
    "cs": CS_GLOSSARY,
}

# 扫描输入文本，匹配领域术语词典中的术语。
def get_relevant_terms(text: str, domain: str, max_terms: int = 15) -> dict:
    glossary = DOMAIN_GLOSSARIES.get(domain, {})
    matched = {}
    for cn, en in glossary.items():
        if cn in text:
            matched[cn] = en
            if len(matched) >= max_terms:
                break
    return matched

# 将匹配术语以自然语言融入翻译提示，不做字典列表。
def format_glossary_prompt(terms: dict) -> str:
    if not terms:
        return ""
    examples = []
    for cn, en in terms.items():
        first_en = en.split("/")[0].strip()
        examples.append(f"'{cn}' → '{first_en}'")
    return (
        "\n翻译时请注意：上文出现了这些专业领域术语，"
        + "请使用对应的标准英文表达进行翻译（例如：" + "、".join(examples[:6]) + "）。"
        + "保持译文自然流畅，必要时可根据上下文调整措辞。"
    )
