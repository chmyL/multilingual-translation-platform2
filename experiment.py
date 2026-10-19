# 对比实验模块
import json
import os
import time
import threading
import glob
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import (MODELS, TRANSLATION_STRATEGIES, RESULTS_DIR, CORPUS_DIR, EXPERIMENT_SAMPLE_SIZE,)
from evaluator import compute_bleu, compute_meteor_simple

# 设置中文显示
matplotlib.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 占位符关键词
_PLACEHOLDER_KEYWORDS = [
    "corpus entry", "test data entry", "test entry",
    "expansion test", "扩充测试", "语料第",
    "business corpus expansion", "daily conversation corpus test",
    "これはビジネスコーパス", "일상 대화 말뭉치",
    "これは日常会話コーパス", "이것은 비즈니스 말뭉치",
]

# 实验进度追踪
_experiment_progress = {"running": False, "current": 0, "total": 0, "message": "", "type": ""}


def get_experiment_progress() -> dict:
    return dict(_experiment_progress)

# 翻译调用，失败时返回错误记录不中断实验。
# source_domain:传入空字符串表示不指定领域（默认），传入真实领域名用于消融等场景。
def _safe_translate(translate_fn, source_text, source_lang, target_lang, strategy, source_domain=""):
    try:
        t_start = time.time()
        # 注意这里传入了source_domain
        result = translate_fn(source_text, source_lang=source_lang,
                              target_lang=target_lang, strategy=strategy,
                              source_domain=source_domain, model="")
        elapsed = time.time() - t_start
        result["_response_time"] = elapsed
        return result
    except Exception as e:
        # 如果发生异常，不仅在结果里记录，还要在终端打印出具体原因
        print(f"[_safe_translate] 翻译失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "translated_text": "", "error": str(e), "_response_time": 0}

# 数据集注册与管理
DATASETS = {
    "default": {
        "name": "默认语料库",
        "description": "自建语料库，含学术/商务/技术/日常/法律/医学/计算机七领域共546条，支持中英日韩四语",
        "type": "directory",
        "path": CORPUS_DIR,
        "languages": ["zh", "en", "ja", "ko"],
        "has_strategy_refs": True,
    },
    "real": {
        "name": "真实语料库 (FLORES-200)",
        "description": "FLORES-200 + Tatoeba 真实数据，四领域各50条，支持中英日韩",
        "type": "dict_file",
        "path": os.path.join(CORPUS_DIR, "backup", "real_corpus.json"),
        "languages": ["zh", "en", "ja", "ko"],
        "has_strategy_refs": False,
    },
    "strategy": {
        "name": "策略语料库",
        "description": "NewsCommentary + FLORES-200 + Tatoeba 混合数据，三领域各50条",
        "type": "dict_file",
        "path": os.path.join(CORPUS_DIR, "backup", "strategy_corpus.json"),
        "languages": ["zh", "en"],
        "has_strategy_refs": False,
    },
    "ablation": {
        "name": "消融实验语料库",
        "description": "法律+医学+计算机三专业领域，各50条，含211个专业术语对，用于消融对比",
        "type": "dict_file",
        "path": os.path.join(CORPUS_DIR, "backup", "ablation_corpus.json"),
        "languages": ["zh", "en"],
        "has_strategy_refs": False,
        "domains": ["law", "medicine", "cs"],
    },
    "conversion": {
        "name": "口语书面语转换语料库",
        "description": "50对口语↔书面语对照语料，含中文口语、中文书面语、英文口语，用于口语/书面语转换评估",
        "type": "dict_file",
        "path": os.path.join(CORPUS_DIR, "backup", "conversion_corpus.json"),
        "languages": ["zh", "en"],
        "has_strategy_refs": False,
        "key": "pairs",
    },
}

# 从目录加载语料（默认语料库）
def _get_flat_corpus(corpus_dir: str) -> list:
    items = []
    for fp in sorted(glob.glob(f"{corpus_dir}/*.json")):
        if "full_corpus" in fp or "corpus_stats" in fp:
            continue
        try:
            data = json.load(open(fp, encoding="utf-8"))
            if isinstance(data, list):
                items.extend(data)
        except Exception:
            pass
    return items

# 根据数据集键名加载语料
def _load_dataset_corpus(dataset_key: str = "default") -> list:
    if dataset_key not in DATASETS:
        raise ValueError(f"未知数据集: {dataset_key}。可用: {list(DATASETS.keys())}")

    ds = DATASETS[dataset_key]
    if ds["type"] == "directory":
        return _get_flat_corpus(ds["path"])
    elif ds["type"] == "dict_file":
        path = ds["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据集文件不存在: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # 口语书面语转换语料库：特殊结构 {"pairs": [...]}
        key = ds.get("key", "")
        if key and isinstance(data, dict) and key in data:
            items = []
            for i, item in enumerate(data[key]):
                item["domain"] = item.get("source", "conversion")
                items.append(item)
            return items
        # 常规dict_file：{domain: [items]}
        items = []
        for domain, domain_items in data.items():
            if domain == "meta" or domain == "pairs":
                continue
            if not isinstance(domain_items, list):
                continue
            for item in domain_items:
                item["domain"] = domain
                items.append(item)
        return items
    return []

# 获取数据集统计信息
def get_dataset_stats(dataset_key: str) -> dict:
    try:
        corpus = _load_dataset_corpus(dataset_key)
    except Exception as e:
        return {"error": str(e)}
    ds = DATASETS.get(dataset_key, {})
    domains = {}
    langs = set()
    for item in corpus:
        d = item.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
        for k in item:
            if len(k) == 2 and not k.startswith("en_"):
                langs.add(k)
    return {
        "name": ds.get("name", dataset_key),
        "description": ds.get("description", ""),
        "total": len(corpus),
        "domains": domains,
        "languages": sorted(langs),
        "has_strategy_refs": ds.get("has_strategy_refs", False),
    }

# 列出所有可用数据集
def list_datasets() -> List[dict]:
    result = []
    for key, ds in DATASETS.items():
        stats = get_dataset_stats(key)
        result.append({
            "key": key,
            "name": ds["name"],
            "description": ds["description"],
            "total": stats.get("total", 0),
            "domains": stats.get("domains", {}),
            "languages": ds.get("languages", []),
            "has_strategy_refs": ds.get("has_strategy_refs", False),
        })
    return result

# 获取有效语料（过滤占位符数据）
def _get_valid_corpus(dataset_key: str = "default") -> list:
    corpus = _load_dataset_corpus(dataset_key)
    valid = []
    for item in corpus:
        en = item.get("en", "")
        ja = item.get("ja", "")
        is_placeholder = any(kw.lower() in f"{en}{ja}".lower() for kw in _PLACEHOLDER_KEYWORDS)
        if not is_placeholder:
            valid.append(item)
    return valid

# 均匀分层抽样：每个领域等比例抽取样本，确保各领域评估公平。
# 如果某领域数据不足，从其他领域补充以保证总数。
def _stratified_sample(corpus: list, sample_size: int, seed: int = 42) -> list:
    import random
    random.seed(seed)

    # 按领域分组
    by_domain = {}
    for item in corpus:
        d = item.get("domain", "unknown")
        if d not in by_domain:
            by_domain[d] = []
        by_domain[d].append(item)

    domains = sorted(by_domain.keys())
    n_domains = len(domains)

    if n_domains == 0:
        return []

    total_available = len(corpus)
    actual_size = min(sample_size, total_available)

    # 每个领域至少1条，剩余均匀分配
    base_per_domain = max(1, actual_size // n_domains)
    remainder = actual_size - base_per_domain * n_domains

    # 统计每个领域实际能抽多少
    domain_quota = {}
    shortfall = 0
    for i, domain in enumerate(domains):
        target = base_per_domain + (1 if i < remainder else 0)
        pool = by_domain[domain]
        take = min(target, len(pool))
        domain_quota[domain] = take
        if take < target:
            shortfall += target - take

    # 不足的从其他领域补上
    if shortfall > 0:
        # 找出哪些领域还有剩余
        extra_pools = {}
        for domain in domains:
            remaining_in_domain = len(by_domain[domain]) - domain_quota[domain]
            if remaining_in_domain > 0:
                extra_pools[domain] = remaining_in_domain

        # 从有余量的领域多抽
        for _ in range(shortfall):
            if not extra_pools:
                break
            # 随机选一个有余量的领域
            donor = random.choice(list(extra_pools.keys()))
            domain_quota[donor] += 1
            extra_pools[donor] -= 1
            if extra_pools[donor] <= 0:
                del extra_pools[donor]

    sampled = []
    for domain, n in domain_quota.items():
        pool = by_domain[domain]
        sampled.extend(random.sample(pool, n))

    # 打乱顺序
    random.shuffle(sampled)
    return sampled

# 实验数据存储
class ExperimentResults:
    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save(self, name: str, data: Any):
        filepath = os.path.join(self.results_dir, f"{name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, name: str) -> Optional[Any]:
        filepath = os.path.join(self.results_dir, f"{name}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def list_experiments(self) -> List[str]:
        if not os.path.exists(self.results_dir):
            return []
        return sorted([
            f.replace(".json", "")
            for f in os.listdir(self.results_dir)
            if f.endswith(".json") and f.startswith("experiment")
        ])

    # 列出指定数据集的所有实验
    def list_by_dataset(self, dataset_key: str) -> List[str]:
        suffix = f"_{dataset_key}"
        exps = self.list_experiments()
        return [e for e in exps if e.endswith(suffix)]

# 生成带数据集后缀的结果文件名
def _results_name(base: str, dataset_key: str = "default") -> str:
    if dataset_key == "default":
        return base  # 保持向后兼容
    return f"{base}_{dataset_key}"

# 实验一：模型对比
def run_model_comparison(translate_fn: Callable, models: Optional[List[str]] = None, sample_size: int = EXPERIMENT_SAMPLE_SIZE,
                         source_lang: str = "zh", target_lang: str = "en", dataset: str = "default",
                         stratified: bool = True, domain_filter: Optional[List[str]] = None,  # 限定领域，如["law","medicine","cs"]
                         pass_domain: bool = False,  # 消融实验用：传递语料真实领域标签
                         progress_callback: Optional[Callable] = None,) -> Dict[str, Any]:
    global _experiment_progress
    if models is None:
        models = list(MODELS.keys())

    corpus = _get_valid_corpus(dataset)
    if domain_filter:
        corpus = [item for item in corpus if item.get("domain", "") in domain_filter]
    if stratified:
        test_data = _stratified_sample(corpus, sample_size)
    else:
        import random
        random.seed(42)
        test_data = random.sample(corpus, min(sample_size, len(corpus)))

    actual_size = len(test_data)
    ds_info = DATASETS.get(dataset, {})
    has_strategy_refs = ds_info.get("has_strategy_refs", False)

    total = actual_size
    results = {
        "experiment": "model_comparison",
        "dataset": dataset,
        "dataset_name": ds_info.get("name", dataset),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": models,
            "sample_size": actual_size,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "strategy": "semantic",
            "stratified": stratified,
        },
        "per_sample": [],
        "summary": {},
    }

    per_model_records = {m: [] for m in models}
    _experiment_progress = {
        "running": True, "current": 0, "total": total * len(models),
        "message": f"实验一 [{ds_info.get('name', dataset)}]: 开始...", "type": "exp1",
    }

    for i, item in enumerate(test_data):
        # 检查停止信号
        if not _experiment_progress["running"]:
            _experiment_progress["message"] = "已被用户停止"
            break

        source_text = item.get(source_lang, item.get("zh", ""))
        item_domain = item.get("domain", "general")
        if has_strategy_refs:
            reference_text = item.get("en_semantic", item.get(target_lang, item.get("en", "")))
        else:
            reference_text = item.get(target_lang, item.get("en", ""))
        sample_result = {
            "id": i, "source": source_text[:200], "reference": reference_text[:200],
            "domain": item_domain, "models": {},
        }

        for mi, model_key in enumerate(models):
            _experiment_progress["current"] = i * len(models) + mi + 1
            _experiment_progress["message"] = (
                f"实验一: 第{i+1}/{total}条 · {MODELS[model_key]['name']}"
            )
            model_name = MODELS[model_key]["name"]

            trans_result = _safe_translate(
                translate_fn, source_text, source_lang=source_lang,
                target_lang=target_lang, strategy="semantic",
                source_domain=item_domain if pass_domain else "",
            )
            elapsed = trans_result.get("_response_time", 0)

            translated_text = trans_result.get("translated_text", "")
            bleu_score, meteor_score_val = 0.0, 0.0
            if trans_result.get("success") and translated_text and reference_text:
                bleu_score = compute_bleu(reference_text, translated_text, lang=target_lang).get("bleu", 0)
                meteor_score_val = compute_meteor_simple(reference_text, translated_text, lang=target_lang).get("meteor", 0)

            sample_result["models"][model_key] = {
                "model": model_key, "model_name": model_name,
                "translated_text": translated_text,
                "success": trans_result.get("success", False),
                "response_time": round(elapsed, 3),
                "bleu": bleu_score, "meteor": meteor_score_val,
                "error": trans_result.get("error"),
                "domain": item_domain,
            }
            per_model_records[model_key].append(sample_result["models"][model_key])

        results["per_sample"].append(sample_result)

    # 汇总
    domains_seen = set()
    for item in test_data:
        d = item.get("domain", "general")
        if d:
            domains_seen.add(d)
    domain_list = sorted(domains_seen)

    summary = {"models": {}}
    for key in models:
        records = per_model_records[key]
        success_records = [r for r in records if r["success"]]
        times = [r["response_time"] for r in success_records]
        bleu_s = [r["bleu"] for r in success_records if r.get("bleu", 0) > 0]
        meteor_s = [r["meteor"] for r in success_records if r.get("meteor", 0) > 0]

        # 各领域独立统计
        domain_stats = {}
        for domain in domain_list:
            domain_records = [
                r for r in records
                if r.get("domain", "") == domain and r["success"]
            ]
            d_bleu = [r["bleu"] for r in domain_records if r.get("bleu", 0) > 0]
            d_meteor = [r["meteor"] for r in domain_records if r.get("meteor", 0) > 0]
            d_times = [r["response_time"] for r in domain_records]
            domain_stats[domain] = {
                "count": len(domain_records),
                "sample_total": sum(
                    1 for item in test_data if item.get("domain") == domain
                ),
                "avg_bleu": round(sum(d_bleu) / len(d_bleu), 2) if d_bleu else 0,
                "avg_meteor": round(sum(d_meteor) / len(d_meteor), 2) if d_meteor else 0,
                "avg_response_time": round(sum(d_times) / len(d_times), 3) if d_times else 0,
            }

        summary["models"][key] = {
            "model_name": MODELS[key]["name"],
            "total": len(records),
            "success_count": len(success_records),
            "success_rate": round(len(success_records) / len(records) * 100, 1) if records else 0,
            "avg_response_time": round(sum(times) / len(times), 3) if times else 0,
            "min_response_time": round(min(times), 3) if times else 0,
            "max_response_time": round(max(times), 3) if times else 0,
            "avg_bleu": round(sum(bleu_s) / len(bleu_s), 2) if bleu_s else 0,
            "avg_meteor": round(sum(meteor_s) / len(meteor_s), 2) if meteor_s else 0,
            "domains": domain_stats,
        }

    _experiment_progress["running"] = False
    results["summary"] = summary
    return results

# 实验二：策略对比
def run_strategy_comparison(translate_fn: Callable, model: str = "Hy-MT2-1.8B", strategies: Optional[List[str]] = None,
                            sample_size: int = EXPERIMENT_SAMPLE_SIZE, source_lang: str = "zh", target_lang: str = "en",
                            dataset: str = "default", stratified: bool = True,
                            strategy_domain_map: Optional[Dict[str, str]] = None,
                            progress_callback: Optional[Callable] = None,
                            ) -> Dict[str, Any]:
    global _experiment_progress
    if strategies is None:
        strategies = list(TRANSLATION_STRATEGIES.keys())

    corpus = _get_valid_corpus(dataset)
    if stratified:
        test_data = _stratified_sample(corpus, sample_size)
    else:
        import random
        random.seed(42)
        test_data = random.sample(corpus, min(sample_size, len(corpus)))

    actual_size = len(test_data)
    ds_info = DATASETS.get(dataset, {})
    has_strategy_refs = ds_info.get("has_strategy_refs", False)

    total = actual_size
    results = {
        "experiment": "strategy_comparison",
        "dataset": dataset,
        "dataset_name": ds_info.get("name", dataset),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": model,
            "model_name": MODELS[model]["name"],
            "strategies": strategies,
            "sample_size": actual_size,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "stratified": stratified,
        },
        "per_sample": [],
        "summary": {},
    }

    # 预创建per_sample骨架
    for i, item in enumerate(test_data):
        source_text = item.get(source_lang, item.get("zh", ""))
        reference_text = item.get("en", "")
        results["per_sample"].append({
            "id": i,
            "source": source_text[:200],
            "reference": reference_text[:200],
            "domain": item.get("domain", "unknown"),
            "strategies": {},
        })

    per_strategy_records = {s: [] for s in strategies}

    if strategy_domain_map:
        # 领域-策略配对模式（策略语料库专用）
        # 每个领域只测其对应的最优策略
        total_steps = total  # 每条只测1个策略
        _experiment_progress = {
            "running": True, "current": 0, "total": total_steps,
            "message": f"实验二 [{ds_info.get('name', dataset)}] 策略配对模式: 开始...", "type": "exp2",
        }
        for i, item in enumerate(test_data):
            # 检查停止信号
            if not _experiment_progress["running"]:
                _experiment_progress["message"] = "已被用户停止"
                break

            item_domain = item.get("domain", "")
            strategy = strategy_domain_map.get(item_domain)
            if not strategy:
                continue  # 跳过无映射的领域
            sname = TRANSLATION_STRATEGIES[strategy]["name"]
            source_text = item.get(source_lang, item.get("zh", ""))
            reference_text = item.get("en", "")

            _experiment_progress["current"] = i + 1
            _experiment_progress["message"] = (
                f"实验二: [{sname}] 第{i+1}/{total}条 ({item_domain})"
            )

            trans_result = _safe_translate(
                translate_fn, source_text, source_lang=source_lang,
                target_lang=target_lang, strategy=strategy,
            )
            elapsed = trans_result.get("_response_time", 0)

            translated_text = trans_result.get("translated_text", "")
            bleu_score, meteor_score_val = 0.0, 0.0
            if trans_result.get("success") and translated_text and reference_text:
                bleu_score = compute_bleu(reference_text, translated_text, lang=target_lang).get("bleu", 0)
                meteor_score_val = compute_meteor_simple(reference_text, translated_text, lang=target_lang).get("meteor", 0)

            record = {
                "strategy": strategy, "strategy_name": sname,
                "model": model, "model_name": MODELS[model]["name"],
                "translated_text": translated_text,
                "success": trans_result.get("success", False),
                "response_time": round(elapsed, 3),
                "bleu": bleu_score, "meteor": meteor_score_val,
                "error": trans_result.get("error"),
                "domain": item_domain,
            }
            results["per_sample"][i]["strategies"][strategy] = record
            per_strategy_records[strategy].append(record)


    else:
        # 原模式：所有策略跑所有条目
        total_steps = total * len(strategies)
        _experiment_progress = {
            "running": True, "current": 0, "total": total_steps,
            "message": f"实验二 [{ds_info.get('name', dataset)}]: 开始...", "type": "exp2",
        }

        step = 0
        stopped = False
        for strategy in strategies:
            if stopped:
                break
            sname = TRANSLATION_STRATEGIES[strategy]["name"]
            for i, item in enumerate(test_data):
                # 检查停止信号
                if not _experiment_progress["running"]:
                    _experiment_progress["message"] = "已被用户停止"
                    stopped = True
                    break
                step += 1
                source_text = item.get(source_lang, item.get("zh", ""))
                if has_strategy_refs:
                    style_key = f"en_{strategy}"
                    reference_text = item.get(style_key, item.get("en", ""))
                else:
                    reference_text = item.get("en", "")
                _experiment_progress["current"] = step
                _experiment_progress["message"] = (
                    f"实验二: [{sname}] 第{i + 1}/{total}条"
                )
                trans_result = _safe_translate(
                    translate_fn, source_text, source_lang=source_lang,
                    target_lang=target_lang, strategy=strategy,
                )
                elapsed = trans_result.get("_response_time", 0)

                translated_text = trans_result.get("translated_text", "")
                bleu_score, meteor_score_val = 0.0, 0.0
                if trans_result.get("success") and translated_text and reference_text:
                    bleu_score = compute_bleu(reference_text, translated_text, lang=target_lang).get("bleu", 0)
                    meteor_score_val = compute_meteor_simple(reference_text, translated_text, lang=target_lang).get("meteor", 0)

                record = {
                    "strategy": strategy, "strategy_name": sname,
                    "model": model, "model_name": MODELS[model]["name"],
                    "translated_text": translated_text,
                    "success": trans_result.get("success", False),
                    "response_time": round(elapsed, 3),
                    "bleu": bleu_score, "meteor": meteor_score_val,
                    "error": trans_result.get("error"),
                    "domain": item.get("domain", "unknown"),
                }
                results["per_sample"][i]["strategies"][strategy] = record
                per_strategy_records[strategy].append(record)

    # 汇总
    domains_seen = set()
    for item in test_data:
        d = item.get("domain", "unknown")
        if d:
            domains_seen.add(d)
    domain_list = sorted(domains_seen)

    summary = {"strategies": {}}
    for strategy in strategies:
        records = per_strategy_records[strategy]
        success_records = [r for r in records if r["success"]]
        times = [r["response_time"] for r in success_records]
        bleu_s = [r["bleu"] for r in success_records if r.get("bleu", 0) > 0]
        meteor_s = [r["meteor"] for r in success_records if r.get("meteor", 0) > 0]

        # 各领域独立统计
        domain_stats = {}
        for domain in domain_list:
            domain_records = [
                r for r in records
                if r.get("domain", "") == domain and r["success"]
            ]
            d_bleu = [r["bleu"] for r in domain_records if r.get("bleu", 0) > 0]
            d_meteor = [r["meteor"] for r in domain_records if r.get("meteor", 0) > 0]
            d_times = [r["response_time"] for r in domain_records]
            domain_stats[domain] = {
                "count": len(domain_records),
                "sample_total": sum(
                    1 for item in test_data if item.get("domain") == domain
                ),
                "avg_bleu": round(sum(d_bleu) / len(d_bleu), 2) if d_bleu else 0,
                "avg_meteor": round(sum(d_meteor) / len(d_meteor), 2) if d_meteor else 0,
                "avg_response_time": round(sum(d_times) / len(d_times), 3) if d_times else 0,
            }

        summary["strategies"][strategy] = {
            "strategy_name": TRANSLATION_STRATEGIES[strategy]["name"],
            "total": len(records),
            "success_count": len(success_records),
            "success_rate": round(len(success_records) / len(records) * 100, 1) if records else 0,
            "avg_response_time": round(sum(times) / len(times), 3) if times else 0,
            "avg_bleu": round(sum(bleu_s) / len(bleu_s), 2) if bleu_s else 0,
            "avg_meteor": round(sum(meteor_s) / len(meteor_s), 2) if meteor_s else 0,
            "domains": domain_stats,
        }

    _experiment_progress["running"] = False
    results["summary"] = summary
    return results


# 可视化模块（matplotlib PNG）
def plot_model_comparison(data: Dict[str, Any], save_path: Optional[str] = None) -> str:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    summary = data.get("summary", {}).get("models", {})
    if not summary:
        return ""
    model_names = [info["model_name"][:12] for info in summary.values()]
    avg_times = [info["avg_response_time"] for info in summary.values()]
    success_rates = [info["success_rate"] for info in summary.values()]
    avg_bleu = [info.get("avg_bleu", 0) for info in summary.values()]
    avg_meteor = [info.get("avg_meteor", 0) for info in summary.values()]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Experiment 1: Model Performance Comparison", fontsize=16, fontweight='bold')
    colors = ["#5470C6", "#91CC75", "#FAC858"]
    x = np.arange(len(model_names))

    axes[0, 0].bar(x, avg_bleu, color=colors[:len(model_names)])
    axes[0, 0].set_title("Avg BLEU Score"); axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(model_names)
    for i_, v in enumerate(avg_bleu): axes[0, 0].text(i_, v+1, str(v), ha='center', fontweight='bold')

    axes[0, 1].bar(x, avg_meteor, color=colors[:len(model_names)])
    axes[0, 1].set_title("Avg METEOR Score"); axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(model_names)
    for i_, v in enumerate(avg_meteor): axes[0, 1].text(i_, v+0.02, str(v), ha='center', fontweight='bold')

    axes[1, 0].bar(x, success_rates, color=colors[:len(model_names)])
    axes[1, 0].set_title("Success Rate (%)"); axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(model_names)
    axes[1, 0].set_ylim(0, 105)
    for i_, v in enumerate(success_rates): axes[1, 0].text(i_, v+1, f"{v}%", ha='center', fontweight='bold')

    axes[1, 1].bar(x, avg_times, color=colors[:len(model_names)])
    axes[1, 1].set_title("Avg Response Time (s)"); axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(model_names)
    for i_, v in enumerate(avg_times): axes[1, 1].text(i_, v+0.05, f"{v:.2f}s", ha='center', fontweight='bold')

    plt.tight_layout()
    if not save_path:
        save_path = os.path.join(RESULTS_DIR, "experiment1_model_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_strategy_comparison(data: Dict[str, Any], save_path: Optional[str] = None) -> str:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    summary = data.get("summary", {}).get("strategies", {})
    if not summary:
        return ""
    s_names = [info["strategy_name"] for info in summary.values()]
    avg_times = [info["avg_response_time"] for info in summary.values()]
    success_rates = [info["success_rate"] for info in summary.values()]
    avg_bleu = [info.get("avg_bleu", 0) for info in summary.values()]
    avg_meteor = [info.get("avg_meteor", 0) for info in summary.values()]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Experiment 2: Prompt Strategy Comparison", fontsize=16, fontweight='bold')
    colors = ["#EE6666", "#5470C6", "#3BA272"]
    x = np.arange(len(s_names))

    axes[0, 0].bar(x, avg_bleu, color=colors[:len(s_names)])
    axes[0, 0].set_title("Avg BLEU Score"); axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(s_names)
    for i_, v in enumerate(avg_bleu): axes[0, 0].text(i_, v+1, str(v), ha='center', fontweight='bold')

    axes[0, 1].bar(x, avg_meteor, color=colors[:len(s_names)])
    axes[0, 1].set_title("Avg METEOR Score"); axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(s_names)
    for i_, v in enumerate(avg_meteor): axes[0, 1].text(i_, v+0.02, str(v), ha='center', fontweight='bold')

    axes[1, 0].bar(x, success_rates, color=colors[:len(s_names)])
    axes[1, 0].set_title("Success Rate (%)"); axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(s_names)
    axes[1, 0].set_ylim(0, 105)
    for i_, v in enumerate(success_rates): axes[1, 0].text(i_, v+1, f"{v}%", ha='center', fontweight='bold')

    axes[1, 1].bar(x, avg_times, color=colors[:len(s_names)])
    axes[1, 1].set_title("Avg Response Time (s)"); axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(s_names)
    for i_, v in enumerate(avg_times): axes[1, 1].text(i_, v+0.05, f"{v:.2f}s", ha='center', fontweight='bold')

    plt.tight_layout()
    if not save_path:
        save_path = os.path.join(RESULTS_DIR, "experiment2_strategy_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_response_time_distribution(data: Dict[str, Any], experiment_type: str = "model", save_path: Optional[str] = None) -> str:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    key = "models" if experiment_type == "model" else "strategies"
    per_sample = data.get("per_sample", [])
    groups = {}
    for sample in per_sample:
        for name, d in sample.get(key, {}).items():
            if d.get("success"):
                label = d.get("model_name", d.get("strategy_name", name))[:15]
                if label not in groups:
                    groups[label] = []
                groups[label].append(d["response_time"])

    if not groups:
        return ""
    fig, ax = plt.subplots(figsize=(10, 6))
    data_list = [groups[k] for k in groups.keys()]
    bp = ax.boxplot(data_list, tick_labels=list(groups.keys()), patch_artist=True)
    colors = ["#5470C6", "#91CC75", "#FAC858", "#EE6666"]
    for patch, color in zip(bp['boxes'], colors[:len(data_list)]):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    title = f"Response Time Distribution ({'Model Comparison' if experiment_type == 'model' else 'Strategy Comparison'})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Response Time (s)"); ax.set_xlabel("Engine")
    plt.tight_layout()

    prefix = "experiment1" if experiment_type == "model" else "experiment2"
    if not save_path:
        save_path = os.path.join(RESULTS_DIR, f"{prefix}_response_distribution.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_radar_comparison(*args, **kwargs) -> str:
    return ""


def generate_experiment_report(e1_data=None, e2_data=None, save_path: str = None) -> str:
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "experiment_report.md")
    lines = [
        "# 翻译对比实验报告",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if e1_data:
        config = e1_data.get("config", {})
        ds = e1_data.get("dataset_name", e1_data.get("dataset", "未知"))
        lines += [
            "## 实验一：不同翻译模型性能对比",
            f"- **数据集**: {ds}",
            f"- **模型**: {', '.join(config.get('models', []))}",
            f"- **样本数**: {config.get('sample_size', 0)}",
            f"- **语言对**: {config.get('source_lang', '')} → {config.get('target_lang', '')}",
            f"- **抽样方式**: {'均匀分层' if config.get('stratified') else '随机'}",
            "",
            "### 模型性能汇总",
            "",
            "| 模型 | 成功率 | 平均BLEU | 平均METEOR | 平均响应(s) |",
            "|------|--------|----------|------------|-------------|",
        ]
        for key, info in e1_data.get("summary", {}).get("models", {}).items():
            lines.append(
                f"| {info['model_name'][:20]} | {info['success_rate']}% | "
                f"{info['avg_bleu']} | {info['avg_meteor']} | {info['avg_response_time']} |"
            )

        # 各领域细分
        domain_lines = []
        for key, info in e1_data.get("summary", {}).get("models", {}).items():
            for domain, ds_stats in info.get("domains", {}).items():
                domain_lines.append(
                    f"| {info['model_name'][:20]} | {domain} | {ds_stats.get('count', 0)} | "
                    f"{ds_stats.get('avg_bleu', 0)} | {ds_stats.get('avg_meteor', 0)} |"
                )
        if domain_lines:
            lines += [
                "", "### 各领域详细评估", "",
                "| 模型 | 领域 | 样本数 | BLEU | METEOR |",
                "|------|------|--------|------|--------|",
            ] + domain_lines
        lines.append("")

    if e2_data:
        config = e2_data.get("config", {})
        ds = e2_data.get("dataset_name", e2_data.get("dataset", "未知"))
        lines += [
            "## 实验二：不同提示策略效果对比",
            f"- **数据集**: {ds}",
            f"- **模型**: {config.get('model_name', '')}",
            f"- **样本数**: {config.get('sample_size', 0)}",
            f"- **抽样方式**: {'均匀分层' if config.get('stratified') else '随机'}",
            "",
            "### 策略性能汇总",
            "",
            "| 策略 | 成功率 | 平均BLEU | 平均METEOR | 平均响应(s) |",
            "|------|--------|----------|------------|-------------|",
        ]
        for key, info in e2_data.get("summary", {}).get("strategies", {}).items():
            lines.append(
                f"| {info['strategy_name']} | {info['success_rate']}% | "
                f"{info['avg_bleu']} | {info['avg_meteor']} | {info['avg_response_time']} |"
            )

        # 各领域细分
        domain_lines = []
        for key, info in e2_data.get("summary", {}).get("strategies", {}).items():
            for domain, ds_stats in info.get("domains", {}).items():
                domain_lines.append(
                    f"| {info['strategy_name']} | {domain} | {ds_stats.get('count', 0)} | "
                    f"{ds_stats.get('avg_bleu', 0)} | {ds_stats.get('avg_meteor', 0)} |"
                )
        if domain_lines:
            lines += [
                "", "### 各领域详细评估", "",
                "| 策略 | 领域 | 样本数 | BLEU | METEOR |",
                "|------|------|--------|------|--------|",
            ] + domain_lines
        lines.append("")

    lines += ["---", "> 注：本报告由实验模块自动生成"]
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return save_path

# 绘制各领域BLEU/BETEOR分组柱状图
def plot_domain_breakdown(data: Dict[str, Any], experiment_type: str = "model", save_path: Optional[str] = None) -> str:
    key = "models" if experiment_type == "model" else "strategies"
    summary = data.get("summary", {}).get(key, {})
    if not summary:
        return ""

    # 收集所有领域
    all_domains = set()
    for info in summary.values():
        for d in info.get("domains", {}):
            all_domains.add(d)
    domain_list = sorted(all_domains)
    if not domain_list:
        return ""

    groups = list(summary.keys())
    group_labels = [info.get("model_name", info.get("strategy_name", k))[:10] for k, info in summary.items()]
    n_groups = len(groups)
    n_domains = len(domain_list)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n_domains * 3), 6))
    fig.suptitle("Per-Domain Evaluation Breakdown", fontsize=14, fontweight='bold')

    x = np.arange(n_domains)
    width = 0.8 / n_groups
    colors = ["#5470C6", "#91CC75", "#FAC858"]

    for plot_idx, (metric_name, metric_key) in enumerate([("BLEU", "avg_bleu"), ("METEOR", "avg_meteor")]):
        ax = axes[plot_idx]
        for gi, gk in enumerate(groups):
            vals = [summary[gk].get("domains", {}).get(d, {}).get(metric_key, 0) for d in domain_list]
            offset = (gi - n_groups / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=group_labels[gi], color=colors[gi % len(colors)])

        ax.set_title(f"Avg {metric_name} by Domain")
        ax.set_xticks(x)
        ax.set_xticklabels(domain_list, rotation=30, ha='right')
        if plot_idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    prefix = "experiment1" if experiment_type == "model" else "experiment2"
    if not save_path:
        save_path = os.path.join(RESULTS_DIR, f"{prefix}_domain_breakdown.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# 多语言对性能对比（实训要求：展示不同引擎在各语言对上的性能）
# 在所有语言对上跑模型对比实验，汇总生成跨语言对比数据。
def run_multi_lang_experiment(translate_fn: Callable, models: Optional[List[str]] = None,
                              sample_size: int = EXPERIMENT_SAMPLE_SIZE, lang_pairs: Optional[List[tuple]] = None,
                              dataset: str = "default", stratified: bool = True,) -> Dict[str, Any]:
    if models is None:
        models = list(MODELS.keys())
    if lang_pairs is None:
        lang_pairs = [("zh", "en"), ("en", "zh"), ("zh", "ja"), ("zh", "ko")]

    global _experiment_progress
    all_results = {}
    total_pairs = len(lang_pairs)
    ds_info = DATASETS.get(dataset, {})

    for pi, (src, tgt) in enumerate(lang_pairs):
        # 检查停止信号
        if not _experiment_progress["running"]:
            _experiment_progress["message"] = "已被用户停止"
            break

        pair_key = f"{src}-{tgt}"
        _experiment_progress = {
            "running": True, "current": pi, "total": total_pairs,
            "message": f"多语言对对比: [{pi + 1}/{total_pairs}] {src}→{tgt} 运行中...",
            "type": "multilang",
        }

        pair_result = run_model_comparison(
            translate_fn=translate_fn, models=models,
            sample_size=sample_size, source_lang=src, target_lang=tgt,
            dataset=dataset, stratified=stratified,
        )
        for mk in list(pair_result.get("summary", {}).get("models", {}).keys()):
            pair_result["summary"]["models"][f"{mk}__{pair_key}"] = \
                pair_result["summary"]["models"].pop(mk)
        all_results[pair_key] = pair_result

    _experiment_progress["running"] = False
    _experiment_progress["message"] = "多语言对对比实验完成"

    exp_results = ExperimentResults()
    save_name = _results_name("experiment_langpairs", dataset)

    # 合并已有数据（不同模型在不同语言对上的结果），不覆盖
    existing = exp_results.load(save_name)
    existing_per_pair = existing.get("per_pair", {}) if existing else {}
    for pair_key, pair_data in all_results.items():
        if pair_key not in existing_per_pair:
            existing_per_pair[pair_key] = pair_data
        else:
            # 合并per_sample：按id去重，追加新模型
            existing_samples = {s["id"]: s for s in existing_per_pair[pair_key].get("per_sample", [])}
            for s in pair_data.get("per_sample", []):
                sid = s["id"]
                if sid in existing_samples:
                    existing_samples[sid]["models"].update(s.get("models", {}))
                else:
                    existing_samples[sid] = s
            existing_per_pair[pair_key]["per_sample"] = list(existing_samples.values())
            # 合并 summary
            existing_per_pair[pair_key]["summary"]["models"].update(
                pair_data.get("summary", {}).get("models", {})
            )
    # 合并config.models
    merged_models = list(set(
        (existing.get("config", {}).get("models", []) if existing else []) +
        models
    ))

    save_data = {
        "experiment": "multi_lang_comparison",
        "dataset": dataset,
        "dataset_name": ds_info.get("name", dataset),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": merged_models,
            "sample_size": sample_size,
            "lang_pairs": [f"{s}-{t}" for s, t in lang_pairs],
            "stratified": stratified,
        },
        "per_pair": existing_per_pair,
    }
    exp_results.save(save_name, save_data)
    return all_results

# 绘制模型×语言对BLEU/METEOR分组柱状图。
def plot_lang_pair_comparison(data: Dict[str, Any], save_path: Optional[str] = None) -> str:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    per_pair = data.get("per_pair", data)
    if not per_pair:
        return ""

    all_models = set()
    all_pairs = sorted(per_pair.keys())
    for pk in all_pairs:
        summary = per_pair[pk].get("summary", {}).get("models", {})
        for mk, info in summary.items():
            all_models.add(info.get("model_name", mk)[:12])

    model_list = sorted(all_models)
    pair_list = all_pairs

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(pair_list) * 3), 7))
    fig.suptitle("Multi-Language Pair Performance Comparison", fontsize=14, fontweight='bold')

    colors = ["#5470C6", "#91CC75", "#FAC858"]
    x = np.arange(len(pair_list))
    width = 0.8 / len(model_list)

    for plot_idx, (metric, key) in enumerate([("BLEU", "avg_bleu"), ("METEOR", "avg_meteor")]):
        ax = axes[plot_idx]
        for mi, mname in enumerate(model_list):
            vals = []
            for pk in pair_list:
                summary = per_pair[pk].get("summary", {}).get("models", {})
                found = 0
                for mk, info in summary.items():
                    if info.get("model_name", mk)[:12] == mname:
                        found = info.get(key, 0)
                        break
                vals.append(found)
            offset = (mi - len(model_list) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=mname, color=colors[mi % len(colors)])

        ax.set_title(f"Avg {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(pair_list)
        if plot_idx == 0:
            ax.legend(fontsize=9)

    plt.tight_layout()
    if not save_path:
        save_path = os.path.join(RESULTS_DIR, "experiment_langpair_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path

# 消融实验与多语言报告
# 绘制消融实验ONvsOFF对比柱状图
def plot_ablation_comparison(data: Dict[str, Any], save_path: Optional[str] = None) -> str:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    on_data = data.get("custom_on") or data.get("all_on", {})
    off_data = data.get("all_off", {})
    if not on_data.get("summary") or not off_data.get("summary"):
        return ""

    on_model = list(on_data["summary"].get("models", {}).values())
    off_model = list(off_data["summary"].get("models", {}).values())
    if not on_model or not off_model:
        return ""
    on_info, off_info = on_model[0], off_model[0]

    metrics = [
        ("BLEU", on_info.get("avg_bleu", 0), off_info.get("avg_bleu", 0)),
        ("METEOR", on_info.get("avg_meteor", 0), off_info.get("avg_meteor", 0)),
        ("Success %", on_info.get("success_rate", 0), off_info.get("success_rate", 0)),
        ("Time(s)", on_info.get("avg_response_time", 0), off_info.get("avg_response_time", 0)),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(metrics))
    width = 0.35
    bars_on = ax.bar(x - width/2, [m[1] for m in metrics], width, label="Domain ON", color="#5470C6")
    bars_off = ax.bar(x + width/2, [m[2] for m in metrics], width, label="Domain OFF", color="#EE6666")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title("Ablation: Domain-Aware ON vs OFF", fontsize=14, fontweight='bold')
    ax.set_ylabel("Score")
    for bar in bars_on:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha='center', fontsize=10, fontweight='bold')
    for bar in bars_off:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    if not save_path:
        save_path = os.path.join(RESULTS_DIR, "experiment_ablation_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path

# 生成消融实验Markdown报告
def generate_ablation_report(data: Dict[str, Any], save_path: Optional[str] = None) -> str:
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "experiment_ablation_report.md")
    lines = [
        "# 消融实验报告",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    config = data.get("config", {})
    lines += [
        f"- **模型**: {config.get('model', '未知')}",
        f"- **样本数**: {config.get('sample_size', 0)}",
        f"- **语言对**: {config.get('lang_pair', '')}",
        f"- **数据集**: {config.get('dataset', 'default')}",
        "",
        "## 领域感知 ON vs OFF 对比",
        "",
        "| 条件 | 成功率 | BLEU | METEOR | 响应(s) |",
        "|------|--------|------|--------|---------|",
    ]

    def _model_info(d, key):
        summary = d.get("summary", {}).get("models", {})
        if not summary:
            return {}
        return list(summary.values())[0] if summary else {}

    on_data = data.get("custom_on") or data.get("all_on", {})
    off_data = data.get("all_off", {})
    on_info = _model_info(on_data, "on")
    off_info = _model_info(off_data, "off")

    lines.append(
        f"| 🟢 领域感知 ON | {on_info.get('success_rate', 0)}% | "
        f"{on_info.get('avg_bleu', 0)} | {on_info.get('avg_meteor', 0)} | "
        f"{on_info.get('avg_response_time', 0)} |"
    )
    lines.append(
        f"| 🔴 领域感知 OFF | {off_info.get('success_rate', 0)}% | "
        f"{off_info.get('avg_bleu', 0)} | {off_info.get('avg_meteor', 0)} | "
        f"{off_info.get('avg_response_time', 0)} |"
    )
    delta = on_info.get("avg_bleu", 0) - off_info.get("avg_bleu", 0)
    lines.append("")
    lines.append(f"**BLEU 增益**: {'+' if delta >= 0 else ''}{delta}")
    lines.append("")
    lines.append("---")
    lines.append("> 注：本报告由消融实验模块自动生成")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return save_path

# 生成多语言对对比实验Markdown报告
def generate_multilang_report(data: Dict[str, Any], save_path: Optional[str] = None) -> str:
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "experiment_multilang_report.md")
    lines = [
        "# 多语言对对比实验报告",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    config = data.get("config", {})
    lines += [
        f"- **模型**: {', '.join(config.get('models', []))}",
        f"- **样本数**: {config.get('sample_size', 0)}",
        f"- **语言对**: {', '.join(config.get('lang_pairs', []))}",
        f"- **数据集**: {data.get('dataset', 'default')}",
        "",
        "## 各语言对性能指标",
        "",
        "| 语言对 | BLEU | METEOR | 成功率 | 响应(s) |",
        "|--------|------|--------|--------|---------|",
    ]

    per_pair = data.get("per_pair", {})
    for pair_key in sorted(per_pair.keys()):
        pair_data = per_pair[pair_key]
        summary = pair_data.get("summary", {}).get("models", {})
        if not summary:
            continue
        info = list(summary.values())[0]
        lines.append(
            f"| {pair_key} | {info.get('avg_bleu', 0)} | {info.get('avg_meteor', 0)} | "
            f"{info.get('success_rate', 0)}% | {info.get('avg_response_time', 0)} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("> 注：本报告由多语言对对比实验模块自动生成")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return save_path
