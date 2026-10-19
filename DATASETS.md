# 数据集清单与下载说明

## 一、自建平行语料库（默认语料库）

**位置**：`data/corpus/*.json`

**规模**：546 条，7 领域 × 中英日韩四语

**构建方式**：手动编写学术/商务/技术/日常四个领域的多语言平行语料，并通过领域关键词匹配从 NewsCommentary 扩展法律/医学/计算机科学三个领域。每条语料包含中文原文、英文译文，部分条目含日文/韩文译文及直译/意译/学术润色三种参考译文。

**格式**：JSON，字段 `zh` / `en` / `ja` / `ko` / `domain` / `en_semantic` / `en_literal` / `en_academic`


## 二、外部公开数据集

### 2.1 FLORES-200（Wikipedia 语体多语言平行语料）

- **发布方**：Meta AI
- **下载地址**：https://github.com/facebookresearch/flores
- **使用文件**：`flores200_dataset/devtest/zho_Hans.devtest` + `eng_Latn.devtest`
- **用途**：从中抽取学术/商务/技术/日常四个领域各 50 条，构建真实语料库

### 2.2 NewsCommentary v16（WMT 新闻评论平行语料）

- **发布方**：WMT（Workshop on Machine Translation）
- **下载地址**：https://data.statmt.org/news-commentary/v16/
- **使用文件**：`news-commentary-v16.en-zh.tsv`（约 32 万对中英平行句）
- **用途**：构建策略语料库（150 条，按领域分层抽样）及消融实验语料库（150 条，法律/医学/计算机关键词筛选）

### 2.3 Tatoeba（多语言例句库）

- **发布方**：Tatoeba 社区
- **下载地址**：https://tatoeba.org/en/downloads
- **使用文件**：中英/中日/中韩平行句对
- **用途**：补充日常对话领域语料至真实语料库


## 三、自建专项数据集

### 3.1 口语书面语转换语料库

**位置**：`data/corpus/backup/conversion_corpus.json`，50 对

**构建方式**：人工编写 50 组口语→书面语对照句，涵盖日常/学术/商务三种场景，每组含中文口语、中文书面语、英文口语译文。

### 3.2 领域术语词典

**位置**：`domain_glossary.py`

**规模**：法律 35 条 + 医学 40 条 + 计算机 40 条 = 115 条中英术语对照。术语来源为领域专业教材和行业标准词汇表。


## 四、大作业 vs 实训数据集区别

| 数据集 | 大作业 | 实训 |
|--------|:---:|:---:|
| 默认语料库（4 领域） | ✅ | ✅ |
| 默认语料库（7 领域） | ❌ | ✅ |
| 真实语料库（FLORES+Tatoeba） | ✅ | ✅ |
| 策略语料库（NewsCommentary） | ✅ | ✅ |
| 消融实验语料库 | ❌ | ✅ |
| 口语书面语转换语料库 | ✅ | ✅ |
| 领域术语词典（115 条） | ❌ | ✅ |
