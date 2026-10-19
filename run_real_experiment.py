#!/usr/bin/env python3
"""
真实数据集实验脚本（已重构，使用 experiment.py 公共模块）

实验一：模型对比 — real_corpus.json（FLORES-200 + Tatoeba 真实数据）
实验二：策略对比 — strategy_corpus.json（NewsCommentary + FLORES-200 + Tatoeba）

支持：
- 均匀分层抽样（各领域等比例抽取）
- 各场景独立评估
- 结果统一存储，前端可直接查看
"""

import sys
sys.path.insert(0, "/mnt/f/PythonProject3")

import time
from config import MODELS, TRANSLATION_STRATEGIES, EXPERIMENT_SAMPLE_SIZE
from vllm_manager import switch_model, get_current_model, stop_vllm
from experiment import (
    run_model_comparison,
    run_strategy_comparison,
    ExperimentResults,
    _results_name,
    DATASETS,
)
from translator import translate


def run_real_exp1(sample_size=50, source_lang="zh", target_lang="en"):
    """
    实验一：三模型在真实语料上对比（依次切换模型，每个模型独立翻译所有样本）
    """
    dataset_key = "real"
    ds_name = DATASETS[dataset_key]["name"]
    print("=" * 60)
    print(f"  实验一：真实数据集 — 模型对比 [{ds_name}]")
    print("=" * 60)

    models = list(MODELS.keys())
    print(f"Models: {[MODELS[m]['name'] for m in models]}")
    print(f"Sample size: {sample_size} (stratified)")

    stop_vllm()
    time.sleep(2)

    # 使用 ExperimentResults 统一存储
    exp_store = ExperimentResults()
    save_name = _results_name("experiment1", dataset_key)

    all_results = None

    for model_key in models:
        model_name = MODELS[model_key]["name"]
        print(f"\n{'─' * 50}")
        print(f"  切换模型: {model_name}")
        print(f"{'─' * 50}")

        switch_result = switch_model(model_key)
        if not switch_result.get("success"):
            print(f"  ❌ 切换失败: {switch_result.get('message', '')}")
            continue

        # 调用公共实验函数（单模型模式：只传当前模型）
        result = run_model_comparison(
            translate_fn=translate,
            models=[model_key],
            sample_size=sample_size,
            source_lang=source_lang,
            target_lang=target_lang,
            dataset=dataset_key,
            stratified=True,
        )

        # 合并到已有结果
        if all_results is None:
            all_results = result
        else:
            if all_results.get("per_sample") and result.get("per_sample"):
                for i, sample in enumerate(result["per_sample"]):
                    if i < len(all_results["per_sample"]):
                        all_results["per_sample"][i]["models"].update(sample["models"])
            for mk, mv in result.get("summary", {}).get("models", {}).items():
                all_results["summary"]["models"][mk] = mv

        print(f"  [完成] {model_name} 批次结束")
        time.sleep(3)

    if all_results:
        # 使用所有模型的列表更新 config
        all_results["config"]["models"] = models
        exp_store.save(save_name, all_results)
        print(f"\n{'=' * 60}")
        print(f"  实验一 结果已保存: {save_name}.json")
        print(f"{'=' * 60}")
        summary = all_results.get("summary", {}).get("models", {})
        for k, v in summary.items():
            print(f"  {v['model_name'][:30]}: BLEU={v['avg_bleu']} METEOR={v['avg_meteor']} "
                  f"OK={v['success_rate']}% T={v['avg_response_time']}s")
            # 各领域细分
            for d, ds in v.get("domains", {}).items():
                print(f"    [{d}] BLEU={ds.get('avg_bleu', 0)} METEOR={ds.get('avg_meteor', 0)} "
                      f"n={ds.get('count', 0)}/{ds.get('sample_total', 0)}")

    return all_results


def run_real_exp2(sample_size=50, source_lang="zh", target_lang="en", model=None):
    """
    实验二：三策略在策略语料上对比
    """
    dataset_key = "strategy"
    ds_name = DATASETS[dataset_key]["name"]
    print("=" * 60)
    print(f"  实验二：真实数据集 — 策略对比 [{ds_name}]")
    print("=" * 60)

    if not model:
        model = get_current_model()
    if not model or model not in MODELS:
        print(f"ERROR: No valid model. model={model}")
        return None

    model_name = MODELS[model]["name"]
    print(f"Model: {model_name}")
    print(f"Sample size: {sample_size} (stratified)")

    result = run_strategy_comparison(
        translate_fn=translate,
        model=model,
        sample_size=sample_size,
        source_lang=source_lang,
        target_lang=target_lang,
        dataset=dataset_key,
        stratified=True,
    )

    exp_store = ExperimentResults()
    save_name = _results_name("experiment2", dataset_key)
    exp_store.save(save_name, result)

    print(f"\n{'=' * 60}")
    print(f"  实验二 结果已保存: {save_name}.json")
    print(f"{'=' * 60}")
    summary = result.get("summary", {}).get("strategies", {})
    for k, v in summary.items():
        print(f"  {v['strategy_name']}: BLEU={v['avg_bleu']} METEOR={v['avg_meteor']} "
              f"OK={v['success_rate']}% T={v['avg_response_time']}s")
        for d, ds in v.get("domains", {}).items():
            print(f"    [{d}] BLEU={ds.get('avg_bleu', 0)} METEOR={ds.get('avg_meteor', 0)} "
                  f"n={ds.get('count', 0)}/{ds.get('sample_total', 0)}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run experiments on real datasets")
    parser.add_argument("exp", choices=["1", "2", "both"], default="both", nargs="?")
    parser.add_argument("-n", "--samples", type=int, default=50)
    parser.add_argument("-m", "--model", type=str, default=None, help="Model for exp2")
    args = parser.parse_args()

    if args.exp in ("1", "both"):
        run_real_exp1(sample_size=args.samples)
    if args.exp in ("2", "both"):
        run_real_exp2(sample_size=args.samples, model=args.model)
