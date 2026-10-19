# 口语↔书面语转换数据集构建脚本 (conversion_corpus.json)
import json, os, random

random.seed(42)
OUT = "/mnt/f/PythonProject/data/corpus/backup"
os.makedirs(OUT, exist_ok=True)

TAT_PATH = "/mnt/d/有中文普通话-英语对应句 - 2026-06-24.tsv"

# 1. 加载 Tatoeba 口语句对
print("[1/2] 加载 Tatoeba 中英口语句对...")
tat_zh = {}
with open(TAT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().replace("﻿","")
        if not line: continue
        parts = line.split("\t")
        if len(parts) >= 4:
            zh = parts[1].strip()
            en = parts[3].strip()
            if 8 <= len(zh) <= 100 and 5 <= len(en) <= 200 and zh not in tat_zh:
                tat_zh[zh] = en
print(f"  Tatoeba: {len(tat_zh)} 条")

# 口语关键词（有语气词/口语标记的句子）
SPOKEN_KW = ["吧","呢","吗","啊","哦","啦","呀","嘛","喂","嗯","哎","嘿","哈",
             "我觉得","就是说","那个","这个","反正","其实","真的","好","太",
             "有点","挺","很","特别","超级","想","要","可以","能","会"]
spoken_candidates = []
for zh, en in tat_zh.items():
    if any(kw in zh for kw in SPOKEN_KW):
        spoken_candidates.append((zh, en))
random.shuffle(spoken_candidates)
print(f"  口语句子候选: {len(spoken_candidates)}")

# 2. 构建口语↔书面语对
print("\n[2/2] 构建转换对...")

# 手写口语→书面语 30 对（高质量人工参考）
HAND_S2W = [
    ("我觉得吧，这个结果还行。", "初步分析表明，该实验结果符合预期。", "I think the results are okay."),
    ("就是说，数据好像有点问题。", "数据可能存在统计偏差。", "The data seems to have some issues."),
    ("那个，我意思是这个事情比较复杂。", "应当指出，此事涉及多方面因素。", "This matter is rather complex."),
    ("哎呀，其实没必要这么麻烦。", "实际上，该流程可以进一步简化。", "There's actually no need to make it this complicated."),
    ("嗯，就是说这个方法可以改进。", "该方法存在优化空间，建议进一步研究。", "This method could be improved."),
    ("我意思是，我们得再看看。", "建议进行进一步的验证与审查。", "We need to take another look."),
    ("反正我觉得不太对。", "综合评估后，结果存在异常。", "I just don't think it's right."),
    ("就是说，现在的情况不太好。", "当前状况未达到预定的理想标准。", "The current situation is not good."),
    ("那个，我们接下来要讨论的是。", "以下内容将就相关议题展开深入讨论。", "What we're going to discuss next is."),
    ("其实吧，没你想的那么复杂。", "需要澄清的是，事态并未如预期般复杂。", "It's not as complicated as you think."),
    ("我今天真的超级累，不想做饭了。", "本人今日极度疲惫，无法进行烹饪活动。", "I'm super tired today and don't want to cook."),
    ("你帮我看下这个对不对？", "请协助核实以下内容的准确性。", "Can you check if this is correct?"),
    ("哎呀我去，这太厉害了吧！", "此成果令人印象深刻，值得高度评价。", "Wow, that's amazing!"),
    ("就是说，我们得想办法解决这个问题。", "当前需制定有效的解决方案以应对该问题。", "We need to find a way to fix this."),
    ("嗯，我觉得大概可以试试。", "初步判断，该方案具备尝试的可行性。", "I think we can give it a try."),
    ("那个谁，你过来一下。", "请有关人员前来商议。", "Hey you, come over here."),
    ("啊？真的假的？我怎么不知道。", "本人对此事表示质疑，此前并未获悉相关信息。", "Really? I didn't know that."),
    ("好吧，那就这样吧。", "既已如此，便按此方案执行。", "Fine, let's go with that."),
    ("我觉得吧，这事没那么难。", "根据初步分析，该任务的难度被高估了。", "I think it's not that hard."),
    ("哎呀算了算了，不说了。", "鉴于讨论无法继续推进，建议暂且搁置。", "Forget it, I'm done talking."),
    ("嗯，就是说我大概明白了。", "综上所述，本人已基本理解相关内容。", "I think I get it now."),
    ("那个，你知道我想说什么对吧？", "相信您已理解我方的意图与诉求。", "You know what I'm trying to say, right?"),
    ("哇塞，这个真的可以啊！", "该成果表现出超出预期的优异性能。", "Wow, this really works!"),
    ("就是说，现在的方案还不够好。", "当前方案仍有改进空间，建议进一步优化。", "The current plan isn't good enough yet."),
    ("其实吧，我就是随口问问。", "需要澄清的是，此问题仅为非正式咨询。", "Actually, I was just asking casually."),
    ("嗯嗯，好的好的，我知道了。", "确认收到，已了解相关情况。", "Okay okay, I got it."),
    ("哎呀坏了，我把这事给忘了。", "本人因疏忽未能及时处理此事，深表歉意。", "Oh no, I forgot about it."),
    ("那个，你先等一下。", "请稍候片刻。", "Hang on a moment."),
    ("算了，不搞了，太麻烦了。", "鉴于过程过于繁琐，决定终止当前操作。", "Forget it, it's too much trouble."),
    ("就是说，你觉得怎么样？", "请问您对此有何意见或建议？", "So, what do you think?"),
]

conversion_pairs = []
for sp, wr, en in HAND_S2W:
    conversion_pairs.append({
        "spoken_zh": sp, "written_zh": wr, "spoken_en": en,
        "direction": "spoken_to_written", "source": "hand-written",
    })

# 书面语→口语 20 对：从 Tatoeba 取正式句，添语气词生成伪口语
FORMAL_KW = ["根据", "分析", "表明", "研究", "建议", "采取", "实施", "评估", "当前",
             "进行", "提供", "予以", "探讨", "讨论", "结论", "结果", "实验", "机制",
             "具有", "相关", "显著", "方案", "措施", "项目", "系统", "规定", "申请",
             "产品", "市场", "总结", "趋势", "会议", "决定", "筹备", "合同", "应用",
             "已经", "必须", "需要", "可能", "影响", "发展", "问题", "方法", "数据"]

formal_candidates = [(zh, en) for zh, en in tat_zh.items()
                     if any(kw in zh for kw in FORMAL_KW) and 12 <= len(zh) <= 100]
random.shuffle(formal_candidates)
w2s_count = 0
for zh, en in formal_candidates:
    if w2s_count >= 20:
        break
    spoken = zh.replace("根据", "按").replace("分析", "研究").replace("表明", "说明")
    spoken = spoken.replace("建议", "觉得").replace("采取", "用").replace("实施", "做")
    spoken = spoken.replace("评估", "看看").replace("提供", "给").replace("予以", "给")
    spoken = spoken.replace("统计显著性", "显著性差异")
    if "。" not in spoken:
        spoken += "。"
    endings = ["吧。", "呢。", "啊。", "哈。", "咯。"]
    spoken = spoken.rstrip("。") + random.choice(endings)
    conversion_pairs.append({
        "spoken_zh": spoken, "written_zh": zh, "spoken_en": en,
        "direction": "written_to_spoken", "source": "Tatoeba",
    })
    w2s_count += 1

print(f"  口语→书面语: {len(HAND_S2W)} 对 (手写高质量参考)")
print(f"  书面语→口语: {w2s_count} 对 (Tatoeba)")
print(f"  总计: {len(conversion_pairs)} 对")

# 保存
data = {
    "pairs": conversion_pairs,
    "meta": {
        "name": "口语↔书面语转换数据集",
        "directions": {
            "spoken_to_written": 30,
            "written_to_spoken": 20,
        },
        "total": len(conversion_pairs),
        "source": "Tatoeba (CC-BY 2.0 FR) — 口语→书面语30对手写 + 书面语→口语20对Tatoeba",
        "usage": "Tab 3 口语/书面语转换功能的质量评估",
        "created": "2026-06-24",
    }
}
with open(f"{OUT}/conversion_corpus.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"\nDone: {OUT}/conversion_corpus.json")
