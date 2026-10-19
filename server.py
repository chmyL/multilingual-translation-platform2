# FastAPI 后端服务
# 将 frontend/app.html 作为前端界面
# 提供 REST API 对接翻译/润色/转换/评估/实验/导出/模型管理
# 底层调用 vLLM 本地模型
import random
import sys
import os
import re
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

sys.path.insert(0, "/mnt/f/PythonProject3")

import time
import threading
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from config import (
    MODELS, LANGUAGE_PAIRS, TRANSLATION_STRATEGIES, POLISH_LEVELS,
    RESULTS_DIR, OUTPUT_DIR, CORPUS_DIR, EXPERIMENT_SAMPLE_SIZE,
)
from vllm_manager import start_vllm, stop_vllm, switch_model, get_status, get_current_model
from translator import translate, batch_translate
from polisher import polish
from converter import spoken_to_written, written_to_spoken
from evaluator import evaluate, ManualScoring, compute_bleu, compute_meteor_simple
from domain_aware import get_ablation_state, set_ablation
from sentence_splitter import translate_long_text, extract_special_tokens, restore_special_tokens
from style_analyzer import compare as compare_styles
from exporter import export_json, export_txt, export_markdown
from experiment import (
    run_model_comparison, run_strategy_comparison,
    run_multi_lang_experiment, plot_lang_pair_comparison,
    plot_model_comparison, plot_strategy_comparison,
    plot_response_time_distribution, plot_domain_breakdown,
    plot_ablation_comparison,
    generate_experiment_report, generate_ablation_report, generate_multilang_report,
    ExperimentResults,
    DATASETS, list_datasets, get_dataset_stats, _load_dataset_corpus,
    _results_name,
)
def _load_corpus(path):
    import glob
    items=[]
    for fp in sorted(glob.glob(f"{path}/*.json")):
        if "full_corpus" in fp or "corpus_stats" in fp: continue
        try:
            data=json.load(open(fp,encoding="utf-8"))
            if isinstance(data,list): items.extend(data)
        except: pass
    return items

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="多语言翻译与润色工具", version="1.0.0")

# 人工评分存储
manual_scoring = ManualScoring(os.path.join(RESULTS_DIR, "manual_scores.json"))
# 实验数据存储
exp_store = ExperimentResults(RESULTS_DIR)

# Pydantic 请求模型
class ModelSwitchRequest(BaseModel):
    model: str

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "zh"
    target_lang: str = "en"
    strategy: str = "semantic"
    domain: str = ""
    temperature: float = 0.3
    max_tokens: int = 128

class PolishRequest(BaseModel):
    text: str
    level: str = "moderate"
    target_lang: str = "zh"

class ConvertRequest(BaseModel):
    text: str
    direction: str = "spoken_to_written"  # or "written_to_spoken"
    lang: str = "zh"

class EvaluateRequest(BaseModel):
    reference: str
    candidate: str
    lang: str = "en"

class ManualScoreRequest(BaseModel):
    source_text: str
    translated_text: str
    reference_text: str = ""
    accuracy: int = 3
    fluency: int = 3
    adequacy: int = 3
    engine: str = ""
    notes: str = ""
    # 实训新增：错误类别、建议译文
    error_categories: Optional[List[str]] = []
    suggested_translation: str = ""

class ExperimentRequest(BaseModel):
    sample_size: int = EXPERIMENT_SAMPLE_SIZE
    lang_pair: str = "zh-en"
    strategies: Optional[List[str]] = None
    dataset: str = "default"
    ablation_flags: Optional[Dict[str, bool]] = None  # 消融实验：逐个开关
    ablation_type: str = "abl1"  # abl1=场景识别, abl2=术语注入

class ExportRequest(BaseModel):
    items: List[str]
    format: str = "json"

# 历史记录持久化（JSON 文件）
import time
import threading
_history_lock = threading.Lock()

def _load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def _save_json(path, data):
    with _history_lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# 追加一条历史记录到文件，保留最近 max_items 条
def _append_history(filepath, entry, max_items=100):
    data = _load_json(filepath, [])
    data.append(entry)
    if len(data) > max_items:
        data = data[-max_items:]
    _save_json(filepath, data)
    return data

_HISTORY_FILES = {
    "translation": os.path.join(RESULTS_DIR, "translation_history.json"),
    "polish": os.path.join(RESULTS_DIR, "polish_history.json"),
    "conversion": os.path.join(RESULTS_DIR, "conversion_history.json"),
    "evaluation": os.path.join(RESULTS_DIR, "evaluation_results.json"),
    "batch": os.path.join(RESULTS_DIR, "batch_results.json"),
}

# 获取历史记录
@app.get("/api/history/{history_type}")
async def get_history(history_type: str):
    if history_type not in _HISTORY_FILES:
        return {"success": False, "error": "未知的历史类型"}
    data = _load_json(_HISTORY_FILES[history_type], [])
    return {"success": True, "data": data}

# 保存一条历史记录
@app.post("/api/history/{history_type}")
async def save_history(history_type: str, entry: Dict[str, Any]):
    if history_type not in _HISTORY_FILES:
        return {"success": False, "error": "未知的历史类型"}
    data = _append_history(_HISTORY_FILES[history_type], entry)
    return {"success": True, "count": len(data)}

# 前端页面
@app.get("/")
async def root():
    return FileResponse("/mnt/f/PythonProject3/frontend/app.html",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

# 模型管理 API
# 获取 vLLM 服务状态
@app.get("/api/status")
async def api_status():
    status = get_status()
    return {
        "running": status["running"],
        "current_model": status["current_model"],
        "current_model_name": status["current_model_name"],
        "backend": status.get("backend", "未知"),
        "models": list(MODELS.keys()),
        "model_details": {k: {"name": v["name"], "description": v["description"]} for k, v in MODELS.items()},
    }

_model_switch_lock = threading.Lock()
_model_switch_status = {"switching": False, "message": ""}

# 启动或切换到指定模型（后台线程执行，立即返回）
@app.post("/api/model/start")
async def api_start_model(req: ModelSwitchRequest):
    if req.model not in MODELS:
        return {"success": False, "message": f"未知模型: {req.model}"}

    if _model_switch_lock.locked():
        return {"success": False, "message": "模型正在切换中，请等待..."}

    def _switch_background():
        with _model_switch_lock:
            _model_switch_status["switching"] = True
            _model_switch_status["message"] = "正在切换模型..."
            try:
                result = switch_model(req.model)
                _model_switch_status["message"] = result.get("message", "")
            except Exception as e:
                _model_switch_status["message"] = f"切换失败: {e}"
            finally:
                _model_switch_status["switching"] = False

    threading.Thread(target=_switch_background, daemon=True).start()
    return {
        "success": True,
        "message": f"正在后台启动 {MODELS[req.model]['name']}，请等待约60-90秒后刷新状态",
        "model": req.model,
        "background": True,
    }

@app.get("/api/model/switch-status")
async def api_switch_status():
    return _model_switch_status

# 停止 vLLM 服务
@app.post("/api/model/stop")
async def api_stop_model():
    result = stop_vllm()
    return {"success": result, "message": "已停止" if result else "停止失败"}

# 翻译 API
# 单条翻译
@app.post("/api/translate")
async def api_translate(req: TranslateRequest):
    if not get_status()["running"]:
        return {"success": False, "error": "vLLM 服务未启动，请先在侧边栏启动模型"}
    result = translate(
        req.text, req.source_lang, req.target_lang,
        req.strategy, domain=req.domain,
        temperature=req.temperature, max_tokens=req.max_tokens,
    )
    return result

# 批量翻译（同语言对多文本）
@app.post("/api/translate/batch")
async def api_translate_batch(req: TranslateRequest):
    if not get_status()["running"]:
        return {"success": False, "error": "vLLM 服务未启动"}
    texts = req.text.split("\n---\n")  # 用 --- 分隔多段
    results = batch_translate(
        texts, req.source_lang, req.target_lang, req.strategy,
    )
    return {"results": results}

# 润色 API
# 学术文本润色
@app.post("/api/polish")
async def api_polish(req: PolishRequest):
    if not get_status()["running"]:
        return {"success": False, "error": "vLLM 服务未启动"}
    result = polish(req.text, req.level, req.target_lang)
    return result

# 润色对比：用三种级别润色同一段文本，返回结果+风格分析
@app.post("/api/polish/compare")
async def api_polish_compare(req: PolishRequest):
    if not get_status()["running"]:
        return {"success": False, "error": "vLLM 服务未启动"}

    from style_analyzer import analyze as analyze_style
    results = {}
    for level in ["light", "moderate", "heavy"]:
        r = polish(req.text, level, req.target_lang)
        polished_text = r.get("polished_text", "") if r.get("success") else ""
        style_info = None
        if polished_text:
            style_info = analyze_style(polished_text)
        results[level] = {
            "polished_text": polished_text,
            "success": r.get("success", False),
            "style": style_info,
        }

    return {
        "success": True,
        "original": req.text,
        "results": results,
    }

# 口语/书面语转换 API
# 口语/书面语转换
@app.post("/api/convert")
async def api_convert(req: ConvertRequest):
    if not get_status()["running"]:
        return {"success": False, "error": "vLLM 服务未启动"}
    if req.direction == "spoken_to_written":
        result = spoken_to_written(req.text, req.lang)
    else:
        result = written_to_spoken(req.text, req.lang)
    return result

# 质量评估 API
# 自动评估（BLEU/METEOR）
@app.post("/api/evaluate")
async def api_evaluate(req: EvaluateRequest):
    result = evaluate(req.reference, req.candidate, req.lang)
    return result

# 批量评估进度 + 结果
_eval_batch_progress = {"running": False, "current": 0, "total": 0}
_eval_batch_results = []

@app.get("/api/evaluate/batch/progress")
async def api_eval_batch_progress():
    return {"progress": dict(_eval_batch_progress), "results": _eval_batch_results}

class BatchEvalRequest(BaseModel):
    sample_size: int = 10
    lang_pair: str = "zh-en"
    domain: str = ""
    strategy: str = "semantic"

# 从语料抽 N 条，模型逐个翻译，与语料自带金标准对比，汇总 BLEU/METEOR
@app.post("/api/evaluate/batch")
async def api_evaluate_batch(req: BatchEvalRequest):
    global _eval_batch_progress, _eval_batch_results
    if not get_status()["running"]:
        return {"success": False, "error": "模型服务未启动"}

    corpus = _load_corpus(CORPUS_DIR)
    if req.domain:
        corpus = [item for item in corpus if item.get("domain") == req.domain]
    if len(corpus) > req.sample_size:
        random.seed(42)
        corpus = random.sample(corpus, req.sample_size)

    source_lang, target_lang = req.lang_pair.split("-")
    _eval_batch_results = []
    _eval_batch_progress = {"running": True, "current": 0, "total": len(corpus)}

    def _run():
        global _eval_batch_progress, _eval_batch_results
        results = []
        total = len(corpus)
        for i, item in enumerate(corpus):
            _eval_batch_progress["current"] = i + 1
            src = item.get(source_lang, item.get("zh", ""))
            style_key = f"en_{req.strategy}" if req.strategy in ("literal","semantic","academic") else "en"
            ref = item.get(style_key, item.get("en", ""))
            trans_result = translate(src, source_lang=source_lang, target_lang=target_lang, strategy=req.strategy)
            bleu_score = 0.0
            meteor_score = 0.0
            if trans_result["success"] and trans_result.get("translated_text") and ref:
                bleu_score = compute_bleu(ref, trans_result["translated_text"]).get("bleu", 0)
                meteor_score = compute_meteor_simple(ref, trans_result["translated_text"], lang=target_lang).get("meteor", 0)
            results.append({
                "id": i + 1,
                "source": src[:120],
                "translated": trans_result.get("translated_text", "")[:120],
                "reference": ref[:120],
                "bleu": bleu_score,
                "meteor": meteor_score,
                "success": trans_result["success"],
            })
        _eval_batch_results = results
        _eval_batch_progress = {"running": False, "current": total, "total": total}

    threading.Thread(target=_run, daemon=True).start()
    return {"success": True, "message": "批量评估已启动", "started": True}

# 提交人工打分（含错误类别和建议译文）
@app.post("/api/score")
async def api_submit_score(req: ManualScoreRequest):
    record = manual_scoring.add_score(
        req.source_text, req.translated_text, req.reference_text,
        req.accuracy, req.fluency, req.adequacy,
        engine=req.engine, notes=req.notes,
        error_categories=req.error_categories or [],
        suggested_translation=req.suggested_translation or "",
    )
    manual_scoring.save(os.path.join(RESULTS_DIR, "manual_scores.json"))
    return {"success": True, "record": record}

# 获取所有人工评分
@app.get("/api/scores")
async def api_get_scores():
    scores = manual_scoring.get_all_scores()
    summary = manual_scoring.get_summary()
    return {"scores": scores, "summary": summary}

# 获取反馈分析与提示优化建议
@app.get("/api/feedback/insights")
async def api_feedback_insights():
    summary = manual_scoring.get_feedback_summary()
    return {"success": True, "feedback": summary}

# 批量处理进度 + 结果
_batch_progress = {"running": False, "current": 0, "total": 0}
_batch_results = []

@app.get("/api/batch/progress")
async def api_batch_progress():
    return {"progress": dict(_batch_progress), "results": _batch_results}

# 批量文件处理 API
# 上传文件并批量翻译（后台线程，前端轮询进度）
@app.post("/api/batch/upload")
async def api_batch_upload(file: UploadFile = File(...), source_lang: str = Form("zh"), target_lang: str = Form("en"),
                           strategy: str = Form("semantic"), polish_level: str = Form("none"), context_window: int = Form(0),):
    global _batch_progress, _batch_results
    if not get_status()["running"]:
        return {"success": False, "error": "vLLM 服务未启动"}

    try:
        content = await file.read()
        file_text = ""
        if file.filename.endswith(".txt"):
            file_text = content.decode("utf-8")
        elif file.filename.endswith(".pdf"):
            import pypdf, io
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                file_text += page.extract_text() or ""
        elif file.filename.endswith(".docx"):
            import docx, io
            doc = docx.Document(io.BytesIO(content))
            file_text = "\n".join([p.text for p in doc.paragraphs])
        else:
            return {"success": False, "error": "不支持的文件格式"}

        paragraphs = [p.strip() for p in file_text.split("\n") if p.strip() and len(p.strip()) > 10]
        if not paragraphs:
            return {"success": False, "error": "文件中未检测到有效文本段落"}

        max_paragraphs = 50
        if len(paragraphs) > max_paragraphs:
            paragraphs = paragraphs[:max_paragraphs]

        _batch_results = []
        _batch_progress = {"running": True, "current": 0, "total": len(paragraphs)}

        def _run_batch():
            global _batch_progress, _batch_results
            from sentence_splitter import _split_sentences, extract_special_tokens, restore_special_tokens
            results = []
            total = len(paragraphs)
            ctx_window = context_window if context_window else 0
            prev_translations = []
            for i, para in enumerate(paragraphs):
                _batch_progress["current"] = i + 1
                t_start = time.time()

                # 短段落直接翻译，长段落用分句翻译
                if len(para) > 200:
                    clean, placeholders = extract_special_tokens(para)
                    sentences = _split_sentences(clean)
                    parts = []
                    ctx_buf = []
                    for s in sentences:
                        if "▣" in s:  # 含占位符的行直接保留
                            parts.append(s)
                            continue
                        use_ctx = ctx_window > 0 and len(ctx_buf) > 0
                        prompt = f"(Previous: {' '.join(ctx_buf[-ctx_window:])})  {s}" if use_ctx else s
                        r = translate(prompt, source_lang, target_lang, strategy)
                        t = r.get("translated_text", "") if r.get("success") else s
                        if not t or len(t) < 2: t = s
                        # 去上下文回声
                        if use_ctx and t and "(" in t[:5] and "Previous:" in t[:40]:
                            idx = t.find(") ")
                            t = t[idx+2:].strip() if idx > 0 else t
                        parts.append(t)
                        if t and t != s: ctx_buf.append(re.sub(r'[一-鿿]', '', t).strip() or t)
                    translated = restore_special_tokens(" ".join(parts), placeholders) if placeholders else " ".join(parts)
                else:
                    if ctx_window > 0 and prev_translations:
                        to_translate = f"(Previous: {' / '.join(prev_translations[-ctx_window:])})  {para}"
                    else:
                        to_translate = para
                    trans_result = translate(to_translate, source_lang, target_lang, strategy)
                    translated = trans_result.get("translated_text", "")
                    if ctx_window > 0 and translated:
                        for bad in ("(Previous:", "Previous:", "Now translate:", "术语映射", "术语对应", "使用", "【.."):
                            if bad in translated:
                                idx2 = translated.rfind(bad)
                                after = translated[idx2+len(bad):].strip().lstrip('：:」」「『"\'')
                                if len(after) > 3: translated = after
                                break
                        if len(translated) > len(para) * 4:
                            lines = [l.strip() for l in translated.split("\\n") if l.strip() and not any(b in l for b in ("Previous:", "Now translate:", "术语", "使用", "【"))]
                            translated = lines[-1] if lines else translated

                elapsed = round(time.time() - t_start, 3)
                entry = {
                    "index": i + 1, "original": para,
                    "translated": translated,
                    "polished": "", "success": trans_result["success"], "elapsed_s": elapsed,
                    "has_context": ctx_window > 0 and len(prev_translations) > 0,
                }
                if polish_level != "none" and trans_result["success"]:
                    polish_result = polish(trans_result["translated_text"], level=polish_level, target_lang=target_lang)
                    if polish_result["success"]:
                        entry["polished"] = polish_result["polished_text"]
                results.append(entry)
                if trans_result["success"]:
                    prev_translations.append(translated)
            _batch_results = results
            _save_json(_HISTORY_FILES.get("batch", os.path.join(RESULTS_DIR, "batch_results.json")), results)
            # 汇总统计
            valid = [r for r in results if r.get("success")]
            times = [r.get("elapsed_s", 0) for r in valid]
            total_time = sum(times)
            avg_time = round(total_time / len(times), 3) if times else 0
            _batch_progress = {
                "running": False, "current": total, "total": total,
                "stats": {
                    "total_segments": total,
                    "success_count": len(valid),
                    "total_time_s": round(total_time, 1),
                    "avg_time_s": avg_time,
                    "min_time_s": round(min(times), 3) if times else 0,
                    "max_time_s": round(max(times), 3) if times else 0,
                }
            }

        threading.Thread(target=_run_batch, daemon=True).start()
        return {"success": True, "message": "批量处理已启动", "started": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

class LongTextRequest(BaseModel):
    text: str
    source_lang: str = "zh"
    target_lang: str = "en"
    strategy: str = "semantic"
    domain: str = ""
    preserve_format: bool = True  # 是否保护特殊格式（代码/URL/Markdown等）
    context_window: int = 2  # 上下文句子数

_longtext_progress = {"running": False, "current": 0, "total": 0, "message": ""}
_longtext_results = []

@app.get("/api/translate/longtext/progress")
async def api_longtext_progress():
    return {"progress": dict(_longtext_progress), "results": _longtext_results}

# 长文本分句翻译：分句→上下文关联→拼接，保护特殊格式
@app.post("/api/translate/longtext")
async def api_translate_longtext(req: LongTextRequest):
    global _longtext_progress, _longtext_results
    if not get_status()["running"]:
        return {"success": False, "error": "vLLM 服务未启动"}

    _longtext_results = []
    _longtext_progress = {"running": True, "current": 0, "total": 0, "message": "分句中..."}

    def _translate_wrapper(text, source_lang, target_lang, strategy, domain="", **kw):
        return translate(text, source_lang=source_lang, target_lang=target_lang,
                         strategy=strategy, domain=domain)

    def _run():
        global _longtext_progress, _longtext_results
        try:
            merged, details = translate_long_text(
                translate_fn=_translate_wrapper,
                text=req.text,
                source_lang=req.source_lang,
                target_lang=req.target_lang,
                strategy=req.strategy,
                domain=req.domain or "",
                context_window=req.context_window if req.preserve_format else 0,
            )
            _longtext_results = details
            _longtext_progress = {
                "running": False, "current": len(details), "total": len(details),
                "message": f"完成！{len(details)} 句",
                "merged": merged,
            }
        except Exception as e:
            _longtext_progress = {"running": False, "current": 0, "total": 0, "message": f"错误: {e}"}

    threading.Thread(target=_run, daemon=True).start()
    return {"success": True, "message": "长文本分句翻译已启动", "started": True}

# 语料库/数据集管理 API
# 列出所有可用数据集
@app.get("/api/datasets")
async def api_list_datasets():
    try:
        datasets = list_datasets()
        return {"success": True, "datasets": datasets}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 获取指定数据集的详细信息
@app.get("/api/datasets/{dataset_key}")
async def api_dataset_detail(dataset_key: str):
    try:
        stats = get_dataset_stats(dataset_key)
        corpus = _load_dataset_corpus(dataset_key)
        # 返回前5条作为预览
        preview = []
        for i, item in enumerate(corpus[:5]):
            preview.append({
                "id": i,
                "zh": item.get("zh", "")[:120],
                "en": item.get("en", "")[:120],
                "domain": item.get("domain", "unknown"),
            })
        return {
            "success": True,
            "stats": stats,
            "preview": preview,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# 对比实验 API
_exp_thread = None
_exp_state = {
    "running": False,
    "type": "",
    "dataset": "",
    "started_at": None,
}

# 实验一：模型对比（支持多数据集）
@app.post("/api/experiment/model")
async def api_experiment_model(req: ExperimentRequest):
    global _exp_thread
    if not get_status()["running"]:
        return {"success": False, "error": "请先启动模型服务"}

    # 并发保护
    if _exp_state["running"]:
        return {"success": False,
                "error": f"已有实验进行中（{_exp_state['type']} - {_exp_state['dataset']}），请等待完成或先点「停止实验」"}

    source_lang, target_lang = req.lang_pair.split("-")
    current_model = get_current_model()
    dataset_key = req.dataset if hasattr(req, 'dataset') and req.dataset else "default"

    def _run():
        from experiment import _experiment_progress
        try:
            _exp_state["running"] = True
            _exp_state["type"] = "model"
            _exp_state["dataset"] = dataset_key
            _experiment_progress["running"] = True

            result = run_model_comparison(
                translate_fn=translate, models=[current_model],
                sample_size=req.sample_size,
                source_lang=source_lang, target_lang=target_lang,
                dataset=dataset_key,
            )

            save_name = _results_name("experiment1", dataset_key)
            existing = exp_store.load(save_name)
            if existing and existing.get("per_sample"):
                base_id = max([s.get("id", 0) for s in existing["per_sample"]] or [-1]) + 1
                for s in result["per_sample"]:
                    s["id"] = base_id + s["id"]
                existing["per_sample"].extend(result["per_sample"])
                for mk, mv in result["summary"]["models"].items():
                    existing["summary"]["models"][mk] = mv
                result = existing
            exp_store.save(save_name, result)
            print(f"[experiment1] 运行完成，共 {len(result['per_sample'])} 条数据")
        except Exception as e:
            import traceback
            print(f"[experiment1] 运行崩溃: {e}")
            traceback.print_exc()
            _experiment_progress["running"] = False
            _experiment_progress["message"] = f"实验出错: {e}"
        finally:
            _exp_state["running"] = False
            _exp_state["type"] = ""
            _exp_state["dataset"] = ""
            _experiment_progress["running"] = False

    _exp_thread = threading.Thread(target=_run, daemon=True)
    _exp_thread.start()
    return {"success": True, "message": f"实验一已启动", "started": True}

# 实验二：提示策略对比（支持多数据集）
@app.post("/api/experiment/strategy")
async def api_experiment_strategy(req: ExperimentRequest):
    global _exp_thread
    if not get_status()["running"]:
        return {"success": False, "error": "请先启动模型服务"}

    if _exp_state["running"]:
        return {"success": False, "error": f"已有实验进行中（{_exp_state['type']} - {_exp_state['dataset']}），请等待完成或先点「停止实验」",}

    source_lang, target_lang = req.lang_pair.split("-")
    current_model = get_current_model()
    dataset_key = req.dataset if hasattr(req, 'dataset') and req.dataset else "default"

    model_name = MODELS[current_model]["name"]

    def _run():
        from experiment import _experiment_progress
        try:
            _exp_state["running"] = True
            _exp_state["type"] = "strategy"
            _exp_state["dataset"] = dataset_key
            _experiment_progress["running"] = True

            sd_map = {"academic": "academic", "business": "literal",
                      "daily": "semantic"} if dataset_key == "strategy" else None
            result = run_strategy_comparison(
                translate_fn=translate, model=current_model,
                sample_size=req.sample_size,
                source_lang=source_lang, target_lang=target_lang,
                strategies=req.strategies,
                dataset=dataset_key,
                strategy_domain_map=sd_map,
            )
            raw_strategies = result.get("summary", {}).get("strategies", {})
            new_strategies = {}
            for sk, sv in raw_strategies.items():
                key = f"{model_name[:12]} - {sv['strategy_name']}"
                new_strategies[key] = sv
            result["summary"]["strategies"] = new_strategies

            save_name = _results_name("experiment2", dataset_key)
            existing = exp_store.load(save_name)
            if existing and existing.get("per_sample"):
                base_id = max([s.get("id", 0) for s in existing["per_sample"]] or [-1]) + 1
                for s in result["per_sample"]:
                    s["id"] = base_id + s["id"]
                existing["per_sample"].extend(result["per_sample"])
                for k, v in new_strategies.items():
                    existing["summary"]["strategies"][k] = v
                result = existing
            exp_store.save(save_name, result)
        except Exception as e:
            import traceback
            print(f"[experiment2] 运行崩溃: {e}")
            traceback.print_exc()
            _experiment_progress["running"] = False
            _experiment_progress["message"] = f"实验出错: {e}"
        finally:
            _exp_state["running"] = False
            _exp_state["type"] = ""
            _exp_state["dataset"] = ""
            _experiment_progress["running"] = False

    _exp_thread = threading.Thread(target=_run, daemon=True)
    _exp_thread.start()
    return {"success": True, "message": f"实验二已启动（数据集: {DATASETS.get(dataset_key, {}).get('name', dataset_key)}）", "started": True}

# 获取实验可视化图表
@app.get("/api/experiment/charts/{exp_type}")
async def api_experiment_charts(exp_type: str, dataset: str = "default"):
    if exp_type not in ("model", "strategy"):
        return {"success": False, "error": "无效的实验类型"}

    base_name = "experiment1" if exp_type == "model" else "experiment2"
    data_key = _results_name(base_name, dataset)
    data = exp_store.load(data_key)

    if not data:
        # fallback to default
        data = exp_store.load(base_name)

    if not data:
        return {"success": False, "error": "无实验数据"}

    try:
        # 图表文件加数据集后缀，防止不同数据集互相覆盖
        ds_suffix = f"_{dataset}" if dataset != "default" else ""
        if exp_type == "model":
            plot_model_comparison(data, os.path.join(RESULTS_DIR, f"experiment1_model_comparison{ds_suffix}.png"))
            plot_response_time_distribution(data, "model", os.path.join(RESULTS_DIR, f"experiment1_response_distribution{ds_suffix}.png"))
            plot_domain_breakdown(data, "model", os.path.join(RESULTS_DIR, f"experiment1_domain_breakdown{ds_suffix}.png"))
        else:
            plot_strategy_comparison(data, os.path.join(RESULTS_DIR, f"experiment2_strategy_comparison{ds_suffix}.png"))
            plot_response_time_distribution(data, "strategy", os.path.join(RESULTS_DIR, f"experiment2_response_distribution{ds_suffix}.png"))
            plot_domain_breakdown(data, "strategy", os.path.join(RESULTS_DIR, f"experiment2_domain_breakdown{ds_suffix}.png"))

        # 返回生成的全部 PNG 图表路径（只返回当前数据集的）
        import glob
        prefix = "experiment1" if exp_type == "model" else "experiment2"
        pattern = f"{prefix}_*{ds_suffix}.png"
        charts = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
        return {
            "success": True,
            "chart_urls": [
                f"/api/experiment/chart-image/{os.path.basename(c)}"
                for c in charts
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# 返回图表图片文件（保留兼容）
@app.get("/api/experiment/chart-image/{filename}")
async def api_chart_image(filename: str):
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    return JSONResponse({"error": "文件不存在"}, status_code=404)

# 返回图表 HTML 文件
@app.get("/api/experiment/chart-html/{filename}")
async def api_chart_html(filename: str):
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="text/html")
    return JSONResponse({"error": "文件不存在"}, status_code=404)

# 获取实验报告（Markdown）
@app.get("/api/experiment/report")
async def api_experiment_report(dataset: str = "default"):
    e1_data = exp_store.load(_results_name("experiment1", dataset)) or exp_store.load("experiment1")
    e2_data = exp_store.load(_results_name("experiment2", dataset)) or exp_store.load("experiment2")

    if not e1_data and not e2_data:
        return {"success": False, "error": "无实验数据"}

    try:
        ds_suffix = f"_{dataset}" if dataset != "default" else ""
        report_path = generate_experiment_report(e1_data, e2_data,
                         save_path=os.path.join(RESULTS_DIR, f"experiment_report{ds_suffix}.md"))
        return {"success": True, "report_path": report_path}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 获取实验数据（支持数据集筛选，四个实验全返回）
@app.get("/api/experiment/data")
async def api_experiment_data(dataset: str = "default"):
    if dataset and dataset != "default":
        e1 = exp_store.load(_results_name("experiment1", dataset))
        e2 = exp_store.load(_results_name("experiment2", dataset))
        e_lang = exp_store.load(_results_name("experiment_langpairs", dataset))
        e_lang = e_lang or exp_store.load("experiment_langpairs")
        e_abl1 = exp_store.load(_results_name("experiment_abl1", dataset))
        e_abl1 = e_abl1 or exp_store.load("experiment_abl1")
        e_abl2 = exp_store.load(_results_name("experiment_abl2", dataset))
        e_abl2 = e_abl2 or exp_store.load("experiment_abl2")
        return {
            "success": True,
            "experiment1": e1,
            "experiment2": e2,
            "experiment_langpairs": e_lang,
            "experiment_abl1": e_abl1,
            "experiment_abl2": e_abl2,
            "dataset": dataset,
        }
    # 默认：返回全部实验数据
    e1 = exp_store.load("experiment1")
    e2 = exp_store.load("experiment2")
    e_lang = exp_store.load("experiment_langpairs")
    e_abl1 = exp_store.load("experiment_abl1")
    e_abl2 = exp_store.load("experiment_abl2")
    return {
        "success": True,
        "experiment1": e1,
        "experiment2": e2,
        "experiment_langpairs": e_lang,
        "experiment_abl1": e_abl1,
        "experiment_abl2": e_abl2,
        "dataset": "default",
    }

# 列出已有实验数据的数据集
@app.get("/api/experiment/datasets")
async def api_experiment_dataset_list():
    all_exps = exp_store.list_experiments()
    datasets_seen = set()
    for exp_name in all_exps:
        data = exp_store.load(exp_name)
        if data:
            ds = data.get("dataset", "default")
            datasets_seen.add(ds)
    # 构建数据集列表（含统计）
    result = []
    for key in sorted(datasets_seen):
        ds_info = DATASETS.get(key, {"name": key, "description": ""})
        exps_for_ds = []
        for exp_name in all_exps:
            if exp_name.endswith(f"_{key}") or (key == "default" and "_" not in exp_name.replace("experiment1", "").replace("experiment2", "")):
                exps_for_ds.append(exp_name)
        result.append({
            "key": key,
            "name": ds_info.get("name", key),
            "description": ds_info.get("description", ""),
            "experiments": exps_for_ds,
        })
    return {"success": True, "datasets": result}

# 获取当前实验进度
@app.get("/api/experiment/progress")
async def api_experiment_progress():
    from experiment import get_experiment_progress
    return get_experiment_progress()

# 停止正在运行的实验
@app.post("/api/experiment/stop")
async def api_experiment_stop():
    from experiment import _experiment_progress
    if not _experiment_progress["running"]:
        return {"success": False, "message": "当前没有正在运行的实验"}
    _experiment_progress["running"] = False
    _experiment_progress["message"] = "已被用户停止"
    return {"success": True, "message": "停止信号已发送，等待当前样本完成"}

# 多语言对对比实验：在所有语言对上跑模型对比
@app.post("/api/experiment/langpairs")
async def api_experiment_langpairs(req: ExperimentRequest):
    global _exp_thread
    if not get_status()["running"]:
        return {"success": False, "error": "请先启动模型服务"}

    # 并发锁
    if _exp_state["running"]:
        return {"success": False,
                "error": f"已有实验进行中（{_exp_state['type']} - {_exp_state['dataset']}），请等待完成或先点「停止实验」"}

    current_model = get_current_model()
    dataset_key = req.dataset if hasattr(req, 'dataset') and req.dataset else "default"

    def _run():
        from experiment import _experiment_progress
        try:
            _exp_state["running"] = True
            _exp_state["type"] = "langpairs"
            _exp_state["dataset"] = dataset_key
            _experiment_progress["running"] = True

            run_multi_lang_experiment(
                translate_fn=translate, models=[current_model],
                sample_size=req.sample_size,
                dataset=dataset_key,
            )
        except Exception as e:
            import traceback
            print(f"[experiment3] 运行崩溃: {e}")
            traceback.print_exc()
            _experiment_progress["running"] = False
            _experiment_progress["message"] = f"实验出错: {e}"
        finally:
            _exp_state["running"] = False
            _exp_state["type"] = ""
            _exp_state["dataset"] = ""
            _experiment_progress["running"] = False

    _exp_thread = threading.Thread(target=_run, daemon=True)
    _exp_thread.start()
    return {"success": True, "message": "多语言对对比实验已启动", "started": True}

# 获取多语言对实验原始数据
@app.get("/api/experiment/langpairs/data")
async def api_langpairs_data(dataset: str = "default"):
    data = exp_store.load(_results_name("experiment_langpairs", dataset))
    if not data:
        data = exp_store.load("experiment_langpairs")
    if not data:
        return {"success": False, "error": "无多语言对实验数据"}
    return {"success": True, "data": data}

# 获取多语言对对比图表
@app.get("/api/experiment/langcharts/{dataset}")
async def api_experiment_langcharts(dataset: str = "default"):
    data = exp_store.load(_results_name("experiment_langpairs", dataset))
    if not data:
        data = exp_store.load("experiment_langpairs")
    if not data:
        return {"success": False, "error": "无多语言对实验数据，请先运行多语言对对比实验"}
    try:
        ds_suffix = f"_{dataset}" if dataset != "default" else ""
        path = os.path.join(RESULTS_DIR, f"experiment_langpair_comparison{ds_suffix}.png")
        plot_lang_pair_comparison(data, path)
        return {"success": True, "chart_urls": [f"/api/experiment/chart-image/experiment_langpair_comparison{ds_suffix}.png"]}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 消融实验 API
# 获取消融实验开关状态
@app.get("/api/ablation")
async def api_get_ablation():
    return {"success": True, "ablation": get_ablation_state()}

# 设置消融实验开关
@app.post("/api/ablation")
async def api_set_ablation(data: Dict[str, Any]):
    for feature, enabled in data.items():
        if feature in ("domain_detection", "domain_instruction", "glossary_injection"):
            set_ablation(feature, bool(enabled))
    return {"success": True, "ablation": get_ablation_state()}

# 获取消融实验数据
@app.get("/api/experiment/ablation/data")
async def api_ablation_data(dataset: str = "default", type: str = "abl1"):
    save_key = f"experiment_{type}"
    data = exp_store.load(_results_name(save_key, dataset))
    if not data:
        return {"success": False, "error": "无消融实验数据"}
    return {"success": True, "data": data}

# 消融实验：对比开启/关闭领域感知功能的翻译质量
@app.post("/api/experiment/ablation")
async def api_experiment_ablation(req: ExperimentRequest):
    global _exp_thread
    if not get_status()["running"]:
        return {"success": False, "error": "请先启动模型服务"}

    # 并发保护
    if _exp_state["running"]:
        return {"success": False,
                "error": f"已有实验进行中（{_exp_state['type']} - {_exp_state['dataset']}），请等待完成或先点「停止实验」"}

    # 【新增】防止模型为空时崩溃
    current_model = get_current_model()
    if not current_model:
        return {"success": False, "error": "未获取到当前运行模型，请稍后重试或重新启动模型"}

    source_lang, target_lang = req.lang_pair.split("-")
    dataset_key = req.dataset if hasattr(req, 'dataset') and req.dataset else "default"

    def _run():
        from experiment import _experiment_progress
        ab_type = req.ablation_type if hasattr(req, 'ablation_type') and req.ablation_type else "abl1"

        try:
            _exp_state["running"] = True
            _exp_state["type"] = f"ablation_{ab_type}"
            _exp_state["dataset"] = dataset_key

            if ab_type == "abl1":
                # 消融一：场景识别+领域提示 ON vs OFF（全七领域）
                set_ablation("glossary_injection", True)

                def _abl1_on(text, source_lang, target_lang, strategy, model="", **kw):
                    from domain_aware import detect_domain as domain_detect
                    domain = domain_detect(text)
                    return translate(text, source_lang=source_lang, target_lang=target_lang,
                                     strategy=strategy, domain=domain, model=model)

                _experiment_progress["message"] = "消融一 [1/2]: 场景识别+领域提示 ON..."
                result_on = run_model_comparison(
                    translate_fn=_abl1_on, models=[current_model],
                    sample_size=req.sample_size, source_lang=source_lang,
                    target_lang=target_lang, dataset="default",
                )
                result_on["ablation"] = "domain_on"

                _detect_correct = 0
                from domain_aware import detect_domain as _dd
                for _s in result_on.get("per_sample", []):
                    _src = _s.get("source", "")
                    _true = _s.get("domain", "")
                    if _src and _true and _dd(_src) == _true:
                        _detect_correct += 1
                _total = len(result_on.get("per_sample", []))
                result_on["detection_accuracy"] = round(_detect_correct / _total * 100, 1) if _total else 0
                result_on["detection_correct"] = _detect_correct
                result_on["detection_total"] = _total

                set_ablation("glossary_injection", False)

                def _abl1_off(text, source_lang, target_lang, strategy, model="", **kw):
                    return translate(text, source_lang=source_lang, target_lang=target_lang,
                                     strategy=strategy, domain="", model=model)

                _experiment_progress["message"] = "消融一 [2/2]: 场景识别+领域提示 OFF..."
                result_off = run_model_comparison(
                    translate_fn=_abl1_off, models=[current_model],
                    sample_size=req.sample_size, source_lang=source_lang,
                    target_lang=target_lang, dataset="default",
                )
                result_off["ablation"] = "domain_off"

            else:
                # 消融二：术语注入 ON vs OFF
                set_ablation("glossary_injection", True)

                def _abl2_on(text, source_lang, target_lang, strategy, model="", **kw):
                    from domain_aware import detect_domain as domain_detect
                    domain = domain_detect(text)
                    return translate(text, source_lang=source_lang, target_lang=target_lang,
                                     strategy=strategy, domain=domain, model=model)

                _experiment_progress["message"] = "消融二 [1/2]: 术语注入 ON..."
                result_on = run_model_comparison(
                    translate_fn=_abl2_on, models=[current_model],
                    sample_size=req.sample_size, source_lang=source_lang,
                    target_lang=target_lang, dataset="ablation",
                )
                result_on["ablation"] = "glossary_on"

                set_ablation("glossary_injection", False)

                def _abl2_off(text, source_lang, target_lang, strategy, model="", **kw):
                    from domain_aware import detect_domain as domain_detect
                    domain = domain_detect(text)
                    return translate(text, source_lang=source_lang, target_lang=target_lang,
                                     strategy=strategy, domain=domain, model=model)

                _experiment_progress["message"] = "消融二 [2/2]: 术语注入 OFF..."
                result_off = run_model_comparison(
                    translate_fn=_abl2_off, models=[current_model],
                    sample_size=req.sample_size, source_lang=source_lang,
                    target_lang=target_lang, dataset="ablation",
                )
                result_off["ablation"] = "glossary_off"

            # 保存
            save_key = f"experiment_{ab_type}"
            ablation_data = {"experiment": ab_type, "custom_on": result_on, "all_off": result_off,
                             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "config": {"model": current_model, "sample_size": req.sample_size,
                                        "lang_pair": req.lang_pair, "dataset": dataset_key}}
            exp_store.save(_results_name(save_key, dataset_key), ablation_data)

        except Exception as e:
            import traceback
            print(f"[ablation] 运行崩溃: {e}")
            traceback.print_exc()
            _experiment_progress["running"] = False
            _experiment_progress["message"] = f"实验出错: {e}"
        finally:
            _exp_state["running"] = False
            _exp_state["type"] = ""
            _exp_state["dataset"] = ""
            _experiment_progress["running"] = False
            set_ablation("domain_detection", True)
            set_ablation("domain_instruction", True)
            set_ablation("glossary_injection", True)

    _exp_thread = threading.Thread(target=_run, daemon=True)
    _exp_thread.start()
    return {"success": True, "message": "消融实验已启动（两轮：领域感知 ON vs OFF）", "started": True, "progress": True}

# 获取消融实验图表
@app.get("/api/experiment/ablation/charts")
async def api_ablation_charts(dataset: str = "default", type: str = "abl1"):
    save_key = f"experiment_{type}"
    data = exp_store.load(_results_name(save_key, dataset))
    if not data:
        return {"success": False, "error": "无消融实验数据"}
    try:
        ds_suffix = f"_{dataset}" if dataset != "default" else ""
        path = plot_ablation_comparison(data, os.path.join(RESULTS_DIR, f"{save_key}_comparison{ds_suffix}.png"))
        if not path:
            return {"success": False, "error": "生成图表失败"}
        return {"success": True, "chart_urls": [f"/api/experiment/chart-image/{save_key}_comparison{ds_suffix}.png"]}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 获取消融实验报告
@app.get("/api/experiment/ablation/report")
async def api_ablation_report(dataset: str = "default", type: str = "abl1"):
    save_key = f"experiment_{type}"
    data = exp_store.load(_results_name(save_key, dataset))
    if not data:
        return {"success": False, "error": "无消融实验数据"}
    try:
        ds_suffix = f"_{dataset}" if dataset != "default" else ""
        path = generate_ablation_report(data, os.path.join(RESULTS_DIR, f"{save_key}_report{ds_suffix}.md"))
        return {"success": True, "report_path": path}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 获取多语言对实验报告
@app.get("/api/experiment/langpairs/report")
async def api_langpairs_report(dataset: str = "default"):
    data = exp_store.load(_results_name("experiment_langpairs", dataset))
    if not data:
        return {"success": False, "error": "无多语言对实验数据"}
    try:
        ds_suffix = f"_{dataset}" if dataset != "default" else ""
        path = generate_multilang_report(data, os.path.join(RESULTS_DIR, f"experiment_multilang_report{ds_suffix}.md"))
        return {"success": True, "report_path": path}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 导出 API
# 导出数据为指定格式
@app.post("/api/export")
async def api_export(req: ExportRequest):
    if not req.items:
        return {"success": False, "error": "未选择导出内容"}

    # 收集数据
    export_data = {}
    for item_key in req.items:
        if item_key == "experiment1_data":
            data = exp_store.load("experiment1")
            if data:
                export_data[item_key] = data
        elif item_key == "experiment2_data":
            data = exp_store.load("experiment2")
            if data:
                export_data[item_key] = data
        elif item_key == "manual_scores":
            scores = manual_scoring.get_all_scores()
            if scores:
                export_data[item_key] = scores
        # 其他数据类型通过 session 无法持久化，这里只处理可持久化的

    if not export_data:
        return {"success": False, "error": "所选内容无可用数据"}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exported_files = []

    for key, data in export_data.items():
        if req.format == "json":
            filepath = export_json(data, OUTPUT_DIR, f"{key}_{timestamp}")
        elif req.format == "markdown":
            filepath = export_markdown(data, OUTPUT_DIR, f"{key}_{timestamp}")
        else:
            filepath = export_txt(data, OUTPUT_DIR, f"{key}_{timestamp}")
        exported_files.append(filepath)

    return {
        "success": True,
        "files": exported_files,
        "output_dir": OUTPUT_DIR,
    }

# 下载导出的文件
@app.get("/api/export/download/{filename}")
async def api_download(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, filename=filename)
    return JSONResponse({"error": "文件不存在"}, status_code=404)

# 语料库查询 API
# 获取语料库统计
@app.get("/api/corpus/stats")
async def api_corpus_stats():
    corpus = _load_corpus(CORPUS_DIR)
    domains = {}
    for item in corpus:
        d = item.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
    return {"total": len(corpus), "domains": domains}

# 启动
if __name__ == "__main__":
    print("=" * 60)
    print("  多语言翻译与润色工具 — FastAPI 后端")
    print("=" * 60)
    print(f"  前端: http://localhost:8501")
    print(f"  API 文档: http://localhost:8501/docs")
    print(f"  默认语料库: {len(_load_corpus(CORPUS_DIR))} 条")
    try:
        from experiment import list_datasets
        for ds in list_datasets():
            print(f"    {ds['name']}: {ds['total']} 条")
    except Exception:
        pass
    print("=" * 60)
    # 预热场景分类器
    def _warmup():
        try:
            from domain_classifier import classify_domain
            classify_domain("预热测试")
            print("[server] 场景分类器预热完成")
        except Exception as e:
            print(f"[server] 分类器预热失败: {e}")
    threading.Thread(target=_warmup, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8501)
