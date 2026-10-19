# 从FLORES-200和NewsCommentary原始数据集筛选法律/医学/计算机语料
# 构建消融实验语料库（每领域50条，共150条）
import json
import random

random.seed(42)

# 领域关键词（≥2个匹配才收录）
DOMAIN_KW = {
    "law": {
        "合同", "协议", "侵权", "仲裁", "诉讼", "判决", "法院", "法官",
        "原告", "被告", "知识产权", "违约金", "版权", "专利",
        "法人", "抵押", "担保", "破产", "清算", "赔偿",
        "证据", "证人", "管辖", "法条", "条款", "调解", "免责", "终止",
        "不可抗力", "违约", "法律", "司法", "律师", "起诉", "裁决",
    },
    "medicine": {
        "临床", "诊断", "治疗", "患者", "病理", "预后", "药物", "剂量",
        "手术", "双盲", "安慰剂", "副作用", "适应症", "并发症",
        "影像", "检测", "筛查", "发病率", "死亡率", "切除", "恶性", "转移",
        "化疗", "免疫", "靶向", "炎症", "糖尿病", "高血压", "疫苗",
        "病毒", "感染", "抗体", "症状", "疗法", "康复", "医生", "护士",
    },
    "cs": {
        "算法", "编程", "软件", "硬件", "编译", "调试", "部署", "架构",
        "并发", "容器", "虚拟化", "数据库", "缓存", "微服务",
        "API", "分布式", "服务器", "协议", "网络", "安全", "加密",
        "人工智能", "机器学习", "云计算", "开源", "代码", "数据",
        "接口", "指令", "芯片", "处理器", "内存", "存储",
    },
}

OUTPUT = "/mnt/f/PythonProject3/data/corpus/backup/ablation_corpus.json"
MAX_PER_DOMAIN = 50

entries = {"law": [], "medicine": [], "cs": []}

# 返回首个匹配≥2关键词的领域，否则返回
def match_domain(text: str) -> str:
    for domain, kws in DOMAIN_KW.items():
        score = sum(1 for kw in kws if kw in text)
        if score >= 2:
            return domain
    return ""

# 过滤太短或纯标点的行
def is_valid(text: str) -> bool:
    if len(text) < 20:
        return False
    # 纯英文/数字行跳过（中文占比太低没意义）
    cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
    return cn_chars >= 3

# 1. NewsCommentary(en-zh TSV)
NC = "/mnt/d/news-commentary-v16.en-zh.tsv"
print(f"扫描 NewsCommentary...")

try:
    with open(NC, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            en, zh = parts[0].strip(), parts[1].strip()
            if not is_valid(zh):
                continue

            d = match_domain(zh)
            if d and len(entries[d]) < MAX_PER_DOMAIN:
                entries[d].append({
                    "zh": zh,
                    "en": en,
                    "domain": d,
                    "source": "NewsCommentary",
                })

            if all(len(v) >= MAX_PER_DOMAIN for v in entries.values()):
                break
except FileNotFoundError:
    print(f"  跳过: {NC} 不存在")


# 2. FLORES-200 devtest(zh/en平行)
FLORES_ZH = "/mnt/d/flores200_dataset/devtest/zho_Hans.devtest"
FLORES_EN = "/mnt/d/flores200_dataset/devtest/eng_Latn.devtest"

print(f"扫描 FLORES-200 devtest...")

try:
    with open(FLORES_ZH, encoding="utf-8") as fz, open(FLORES_EN, encoding="utf-8") as fe:
        for zh, en in zip(fz, fe):
            zh, en = zh.strip(), en.strip()
            if not is_valid(zh):
                continue

            d = match_domain(zh)
            if d and len(entries[d]) < MAX_PER_DOMAIN:
                entries[d].append({
                    "zh": zh,
                    "en": en,
                    "domain": d,
                    "source": "FLORES-200",
                })
except FileNotFoundError:
    print(f"  跳过: FLORES-200 不存在")


# 3. 如果某领域不足50条，从已匹配的大池补
for d in ("law", "medicine", "cs"):
    if len(entries[d]) < MAX_PER_DOMAIN:
        shortfall = MAX_PER_DOMAIN - len(entries[d])
        print(f"\n{d} 领域不足 {shortfall} 条，从 NewsCommentary 补充...")
        try:
            with open(NC, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    en, zh = parts[0].strip(), parts[1].strip()
                    if not is_valid(zh):
                        continue
                    if match_domain(zh) == d and len(entries[d]) < MAX_PER_DOMAIN:
                        # 去重
                        if not any(e["zh"] == zh for e in entries[d]):
                            entries[d].append({
                                "zh": zh, "en": en, "domain": d,
                                "source": "NewsCommentary",
                            })
                    if len(entries[d]) >= MAX_PER_DOMAIN:
                        break
        except FileNotFoundError:
            pass

# 保存
for d in ("law", "medicine", "cs"):
    random.shuffle(entries[d])

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

print(f"\n=== 生成结果 ===")
total = 0
for d in ("law", "medicine", "cs"):
    n = len(entries[d])
    total += n
    src_counts = {}
    for e in entries[d]:
        s = e.get("source", "?")
        src_counts[s] = src_counts.get(s, 0) + 1
    print(f"  {d}: {n} 条 (来源: {src_counts})")

print(f"  总计: {total} 条")
print(f"  保存到: {OUTPUT}")
