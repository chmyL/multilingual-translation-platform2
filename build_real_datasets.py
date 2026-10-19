# 真实公开数据集构建脚本
import json, os, re, sys, random, gzip, io, urllib.request

random.seed(42)
OUT = "/mnt/f/PythonProject/data/corpus/backup"
os.makedirs(OUT, exist_ok=True)

# 1. 加载FLORES-200(学术+技术)
FLORES_BASE = "/mnt/d/flores200_dataset/devtest"
flores_zh = open(f"{FLORES_BASE}/zho_Hans.devtest","r",encoding="utf-8").read().strip().split("\n")
flores_en = open(f"{FLORES_BASE}/eng_Latn.devtest","r",encoding="utf-8").read().strip().split("\n")
flores_ja = open(f"{FLORES_BASE}/jpn_Jpan.devtest","r",encoding="utf-8").read().strip().split("\n")
flores_ko = open(f"{FLORES_BASE}/kor_Hang.devtest","r",encoding="utf-8").read().strip().split("\n")
print(f"[1/4] FLORES-200: {len(flores_zh)} 条")

# 学术关键词筛选
AC_KW = ["研究","实验","模型","算法","训练","性能","分析","方法","证明","假设",
         "理论","科学","论文","学者","教授","大学","系统","技术","计算","统计",
         "测量","观测","发现","表明","显著","准确","数据","学习","结构","函数"]
TECH_KW = ["服务器","代码","程序","网络","数据","接口","配置","部署","软件",
           "硬件","协议","内存","线程","进程","缓存","端口","版本","更新",
           "错误","日志","安全","加密","API","URL","HTTP","TCP","IP"]

# 从FLORES中按关键词选N条
def pick_flores(keywords, n):
    candidates = []
    for i in range(len(flores_zh)):
        txt = flores_zh[i]
        if len(txt) < 15 or len(txt) > 200: continue
        if any(kw in txt for kw in keywords):
            candidates.append({
                "zh": txt, "en": flores_en[i], "ja": flores_ja[i], "ko": flores_ko[i],
                "source": "FLORES-200",
            })
    return random.sample(candidates, min(n, len(candidates)))

# 2. 加载NewsCommentary v16(商务+直译)
NC_PATH = "/mnt/d/news-commentary-v16.en-zh.tsv"
print(f"[2/4] 加载 NewsCommentary v16 ...")
nc_data = []
with open(NC_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            en_txt, zh_txt = parts[0], parts[1]
            if 15 <= len(zh_txt) <= 250 and 15 <= len(en_txt) <= 250:
                nc_data.append({"zh": zh_txt, "en": en_txt})
print(f"  NewsCommentary: {len(nc_data)} 条有效中英句对")

# 商务/正式关键词筛选
BIZ_KW = ["经济","市场","公司","企业","投资","贸易","金融","银行","政府",
          "政策","协议","合同","谈判","预算","收入","利润","价格","股票",
          "关税","汇率","制裁","合作","发展","增长","部门","委员会","总统"]

def pick_nc(keywords, n):
    candidates = [d for d in nc_data if any(kw in d["zh"] for kw in keywords)]
    return random.sample(candidates, min(n, len(candidates))) if candidates else []

# 3. 加载Tatoeba(日常口语+意译)
TAT_PATH = "/mnt/d/有中文普通话-英语对应句 - 2026-06-24.tsv"
print(f"[3/4] 加载 Tatoeba ...")
tat_data = {}  # zh_text->en_text(取第一条)
with open(TAT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().replace("﻿","")
        if not line: continue
        parts = line.split("\t")
        if len(parts) >= 4:
            zh_txt = parts[1].strip()
            en_txt = parts[3].strip()
            if 8 <= len(zh_txt) <= 200 and 5 <= len(en_txt) <= 300 and zh_txt not in tat_data:
                tat_data[zh_txt] = en_txt
print(f"  Tatoeba: {len(tat_data)} 条中英句对")

def pick_tatoeba(n):
    items = list(tat_data.items())
    random.shuffle(items)
    return [{"zh": z, "en": e, "source": "Tatoeba"} for z, e in items[:n]]

# 4. 加载OpenSubtitles(补充口语)
print(f"[4/4] 加载 OpenSubtitles ...")
opensub_data = []
try:
    OPUS_URL = "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-zh.txt.zip"
    print(f"  尝试在线加载...")
    req = urllib.request.urlopen(OPUS_URL, timeout=30)
    import zipfile
    with zipfile.ZipFile(io.BytesIO(req.read())) as zf:
        for name in zf.namelist():
            if name.endswith(".en") or name.endswith(".zh"):
                pass
    print(f"  OpenSubtitles: ZIP 加载成功")
except Exception as e:
    print(f"  OpenSubtitles 在线加载失败: {e}，跳过。不影响主要数据集。")

# 5. 构建策略评估数据集(200条)
print("\n=== 构建 strategy_corpus.json ===")

strategy_corpus = {
    "academic": [],    # FLORES→50
    "business": [],    # NewsCommentary→50
    "daily": [],       # Tatoeba→50
    "technical": [],   # FLORES→50
}

# 学术(FLORES)
strategy_corpus["academic"] = pick_flores(AC_KW, 50)
# 技术(FLORES)
strategy_corpus["technical"] = pick_flores(TECH_KW, 50)
# 商务(NewsCommentary)
biz_entries = pick_nc(BIZ_KW, 50)
for d in biz_entries: d["domain"] = "business"
strategy_corpus["business"] = biz_entries
# 日常(Tatoeba)
daily_entries = pick_tatoeba(50)
for d in daily_entries: d["domain"] = "daily"
strategy_corpus["daily"] = daily_entries

# 补字段
for domain in strategy_corpus:
    for d in strategy_corpus[domain]:
        d["domain"] = domain
        if "ja" not in d: d["ja"] = ""
        if "ko" not in d: d["ko"] = ""

total = sum(len(v) for v in strategy_corpus.values())
strategy_corpus["meta"] = {
    "name": "策略评估真实数据集",
    "sources": {
        "academic": "FLORES-200 (Meta AI, CC-BY-SA 4.0)",
        "business": "NewsCommentary v16 (WMT, CC0)",
        "daily": "Tatoeba (CC-BY 2.0 FR)",
        "technical": "FLORES-200 (Meta AI, CC-BY-SA 4.0)",
    },
    "total": total,
    "usage": "实验二：不同翻译策略对比评估。四个领域各50条，用于验证模型在不同策略(直译/意译/学术润色)下的BLEU/METEOR表现。",
    "created": "2026-06-24",
}
with open(f"{OUT}/strategy_corpus.json", "w", encoding="utf-8") as f:
    json.dump(strategy_corpus, f, ensure_ascii=False, indent=2)
print(f"  策略数据集: {total} 条 ({', '.join(f'{k}={len(v)}' for k,v in strategy_corpus.items() if k!='meta')})")

# 6. 构建口语↔书面语转换数据集 (50对)
print("\n=== 构建 conversion_corpus.json ===")

# 从 Tatoeba取口语短句，写书面语版本
spoken_data = random.sample(list(tat_data.items()), min(100, len(tat_data)))
conversion_pairs = []
for zh_spoken, en_spoken in spoken_data[:50]:
    # 筛选明确口语句式
    is_spoken = any(kw in zh_spoken for kw in ["吧","呢","吗","啊","哦","啦","呀","嘛","喂","嗯"])
    if not is_spoken: continue
    # 写书面语版本（简单规则：去语气词、补正式词）
    zh_written = zh_spoken.replace("吧","").replace("呢","").replace("啊","").replace("哦","").replace("啦","").replace("呀","").replace("嘛","").strip()
    if zh_written == zh_spoken or len(zh_written) < 5: continue
    conversion_pairs.append({
        "spoken_zh": zh_spoken,
        "written_zh": zh_written,
        "spoken_en": en_spoken,
        "source": "Tatoeba (口语部分)",
    })
    if len(conversion_pairs) >= 50: break

print(f"  口语↔书面语: {len(conversion_pairs)} 对")

if len(conversion_pairs) < 50:
    # 补足：手写
    extra = [
        ["我觉得吧，这个结果还行","初步分析表明，该结果符合预期"],
        ["就是说，数据好像有问题","数据显示可能存在偏差"],
        ["那个，我意思是这个事情比较复杂","应当指出，此事涉及多重因素"],
        ["哎呀，其实没必要这么麻烦","实际上可简化该流程"],
        ["嗯，就是说这个方法可以改进","该方法存在优化空间"],
        ["我意思是，我们得再看看","建议进行进一步审查"],
        ["反正我觉得不太对","综合来看，结果存在异常"],
        ["就是说，现在的情况不太好","当前状况未达理想标准"],
        ["那个，我们接下来要讨论的是","以下内容将就相关议题展开讨论"],
        ["其实吧，没你想的那么复杂","需要澄清的是，事态并未如预想般复杂"],
    ]
    for sp, wr in extra:
        if len(conversion_pairs) >= 50: break
        conversion_pairs.append({"spoken_zh":sp, "written_zh":wr, "spoken_en":"", "source":"hand-written"})

conv_data = {
    "pairs": conversion_pairs,
    "meta": {
        "name": "口语↔书面语转换数据集",
        "total": len(conversion_pairs),
        "source": "Tatoeba (CC-BY 2.0 FR) + hand-written supplement",
        "usage": "Tab 3 口语/书面语转换功能的质量评估（用于验证转换准确性）",
        "created": "2026-06-24",
    }
}
with open(f"{OUT}/conversion_corpus.json", "w", encoding="utf-8") as f:
    json.dump(conv_data, f, ensure_ascii=False, indent=2)
print(f"  转换数据集: {len(conversion_pairs)} 对")

print(f"\n=== 完成 ===")
print(f"  {OUT}/strategy_corpus.json  ({os.path.getsize(f'{OUT}/strategy_corpus.json')/1024:.0f} KB)")
print(f"  {OUT}/conversion_corpus.json ({os.path.getsize(f'{OUT}/conversion_corpus.json')/1024:.0f} KB)")
print(f"\n  构建脚本: /mnt/f/PythonProject/build_real_datasets.py")
