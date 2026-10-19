# 策略评估数据集构建脚本 (strategy_corpus.json) — 维基稳定版
import json, os, random
from tqdm import tqdm

random.seed(42)
OUT = "/mnt/f/PythonProject/data/corpus/backup"
os.makedirs(OUT, exist_ok=True)

# 1. 直译 — NewsCommentary v16
NC_PATH = "/mnt/d/news-commentary-v16.en-zh.tsv"
print("[1/3] 加载 NewsCommentary v16（直译策略 / 商务领域）...")
nc_data = []
with open(NC_PATH, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
    f.seek(0)
    for line in tqdm(f, total=total_lines, desc="筛选句对"):
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            zh, en = parts[1], parts[0]
            if 15 <= len(zh) <= 200 and 15 <= len(en) <= 300:
                nc_data.append({"zh": zh, "en": en, "domain": "business", "source": "NewsCommentary v16"})
print(f"  有效句对: {len(nc_data)}")
business = random.sample(nc_data, min(50, len(nc_data)))

# 2. 意译 — FLORES-200
FLORES_BASE = "/mnt/d/flores200_dataset/devtest"
print(f"\n[2/3] 加载 FLORES-200（意译策略 / 日常领域）...")
zh = open(f"{FLORES_BASE}/zho_Hans.devtest", "r", encoding="utf-8").read().strip().split("\n")
en = open(f"{FLORES_BASE}/eng_Latn.devtest", "r", encoding="utf-8").read().strip().split("\n")
print(f"  {len(zh)} 条四语对齐")

DAILY_KW = ["今天", "昨天", "明天", "早上", "晚上", "吃饭", "喝水", "睡觉", "走路", "朋友",
            "家人", "妈妈", "爸爸", "孩子", "学校", "商店", "衣服", "天气", "雨", "电影",
            "音乐", "书", "狗", "猫", "花", "树", "车", "房子", "门", "开心", "难过", "玩",
            "做饭", "洗澡", "打电话", "上班", "下班", "周末", "假期", "旅行", "医院"]
daily_candidates = []
for i, text in enumerate(tqdm(zh, desc="筛选日常句")):
    if any(kw in text for kw in DAILY_KW) and 10 <= len(text) <= 200:
        daily_candidates.append({
            "zh": text, "en": en[i],
            "domain": "daily", "source": "FLORES-200",
        })
daily = random.sample(daily_candidates, min(50, len(daily_candidates)))
print(f"  日常候选: {len(daily_candidates)}, 选取 {len(daily)}")

# 3. 学术润色 — FLORES-200 学术关键词筛选
print(f"\n[3/3] 从 FLORES-200 筛选学术句对（学术润色策略 / 学术领域）...")

AC_KW = ["研究", "实验", "模型", "算法", "训练", "性能", "分析", "方法", "证明", "假设",
         "理论", "科学", "系统", "计算", "统计", "数据", "学习", "结构", "函数", "定义",
         "定理", "公式", "变量", "参数", "结果", "结论", "表明", "显著", "准确", "特征"]

academic_candidates = []
for i, text in enumerate(tqdm(zh, desc="筛选学术句")):
    if any(kw in text for kw in AC_KW) and 15 <= len(text) <= 200:
        academic_candidates.append({"zh": text, "en": en[i]})

academic = [{"zh": item["zh"], "en": item["en"],
             "domain": "academic", "source": "FLORES-200 (学术关键词筛选)"}
            for item in academic_candidates[:50]]
print(f"  从 FLORES 中筛选到 {len(academic)} 条学术句对")

# 保存
print(f"\n汇总：")
print(f"  商务 (直译): {len(business)} 条")
print(f"  日常 (意译): {len(daily)} 条")
print(f"  学术 (润色): {len(academic)} 条")

corpus = {
    "academic": academic,
    "business": business,
    "daily": daily,
    "meta": {
        "name": "策略评估真实数据集（维基稳定版）",
        "sources": {
            "literal_strategy": "NewsCommentary v16 (WMT, CC0) — 商务×50",
            "semantic_strategy": "FLORES-200 (Meta AI, CC-BY-SA 4.0) — 日常×50",
            "academic_strategy": f"{academic[0]['source'] if academic else 'Fallback'} — 学术×50",
        },
        "total": len(academic) + len(business) + len(daily),
        "domains": ["academic", "business", "daily"],
        "usage": "3 domains × 1 strategy = 150 entries.",
        "note": "Academic part tries WikiMatrix first, falls back to FLORES if network fails.",
        "created": "2026-06-24",
    }
}
with open(f"{OUT}/strategy_corpus.json", "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)
print(f"\nDone: {OUT}/strategy_corpus.json ({len(academic) + len(business) + len(daily)} entries)")