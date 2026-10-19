"""
多语言翻译与润色工具 — Streamlit 前端
支持中英日韩互译、学术润色、口语/书面语转换、质量评估、批量处理、对比实验、结果导出
"""

import sys
import os
import time
import json
from datetime import datetime

# 将项目根目录添加到 Python 路径
sys.path.insert(0, "/mnt/f/PythonProject")

import streamlit as st
import pandas as pd

from config import (
    MODELS,
    LANGUAGE_PAIRS,
    LANGUAGE_NAMES,
    TRANSLATION_STRATEGIES,
    POLISH_LEVELS,
    RESULTS_DIR,
    OUTPUT_DIR,
    CORPUS_DIR,
    EXPERIMENT_SAMPLE_SIZE,
)
from vllm_manager import (
    start_vllm,
    stop_vllm,
    switch_model,
    check_health,
    get_status,
    get_current_model,
)
from translator import translate, batch_translate
from polisher import polish
from converter import spoken_to_written, written_to_spoken
from evaluator import evaluate, ManualScoring
from exporter import export_json, export_txt, export_markdown
from experiment import (
    run_model_comparison,
    run_strategy_comparison,
    plot_model_comparison,
    plot_strategy_comparison,
    plot_response_time_distribution,
    generate_experiment_report,
    ExperimentResults,
)

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="多语言翻译与润色工具",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 初始化 session state
# ============================================================
if "vllm_running" not in st.session_state:
    st.session_state.vllm_running = False
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "translation_history" not in st.session_state:
    st.session_state.translation_history = []
if "polish_history" not in st.session_state:
    st.session_state.polish_history = []
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = []
if "manual_scores" not in st.session_state:
    st.session_state.manual_scores = ManualScoring()
if "experiment1_data" not in st.session_state:
    st.session_state.experiment1_data = None
if "experiment2_data" not in st.session_state:
    st.session_state.experiment2_data = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 侧边栏 — 模型管理与状态
# ============================================================
st.sidebar.title("🌐 系统控制")

# vLLM 服务状态
st.sidebar.subheader("vLLM 服务状态")
try:
    is_alive = check_health()
    st.session_state.vllm_running = is_alive
    current_model = get_current_model()
    st.session_state.current_model = current_model

    if is_alive and current_model:
        model_name = MODELS.get(current_model, {}).get("name", current_model)
        st.sidebar.success(f"🟢 运行中: {model_name}")
    elif is_alive:
        st.sidebar.success("🟢 vLLM 服务运行中")
    else:
        st.sidebar.warning("🔴 vLLM 服务未启动")
except Exception:
    st.session_state.vllm_running = False
    st.sidebar.error("🔴 无法连接 vLLM 服务")

# 模型切换
st.sidebar.subheader("🔄 切换模型")
model_options = list(MODELS.keys())
default_idx = list(MODELS.keys()).index(current_model) if current_model in MODELS else 0
selected_model = st.sidebar.selectbox(
    "选择要运行的模型",
    options=model_options,
    index=default_idx,
    format_func=lambda x: MODELS[x]["name"],
)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶️ 启动/切换", use_container_width=True):
        with st.spinner(f"正在启动 {MODELS[selected_model]['name']} ..."):
            try:
                result = switch_model(selected_model)
                if result["success"]:
                    st.session_state.vllm_running = True
                    st.session_state.current_model = selected_model
                    st.success(result["message"])
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(result["message"])
            except Exception as e:
                st.error(f"启动失败: {str(e)}")

with col2:
    if st.button("⏹️ 停止", use_container_width=True):
        with st.spinner("正在停止 vLLM 服务..."):
            try:
                stop_vllm()
                st.session_state.vllm_running = False
                st.session_state.current_model = None
                st.success("已停止 vLLM 服务")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"停止失败: {str(e)}")

# 模型信息
if selected_model:
    st.sidebar.divider()
    st.sidebar.caption(f"**模型描述**: {MODELS[selected_model]['description']}")
    st.sidebar.caption(f"**架构**: {MODELS[selected_model]['architecture']}")

# 刷新状态
if st.sidebar.button("🔄 刷新状态"):
    st.rerun()

# ============================================================
# 标题
# ============================================================
st.title("🌐 多语言翻译与润色工具")
st.caption("支持中英日韩互译 | 学术润色 | 口语/书面语转换 | 质量评估 | 批量处理 | 对比实验")

# vLLM 未启动时的警告
if not st.session_state.vllm_running:
    st.warning("⚠️ vLLM 服务未启动。请在左侧边栏启动模型后再使用翻译功能。")

# ============================================================
# 主界面 Tab
# ============================================================
tabs = st.tabs([
    "🔄 翻译",
    "✨ 润色",
    "💬 口语/书面语转换",
    "📊 评估",
    "📁 批量处理",
    "🔬 对比实验",
    "💾 导出",
])

# ============================================================
# Tab 1: 翻译
# ============================================================
with tabs[0]:
    st.subheader("🔄 多语言翻译")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        input_text = st.text_area(
            "输入文本",
            placeholder="请输入要翻译的文本...",
            height=150,
            key="trans_input",
        )

    with col_right:
        # 语言对选择
        lang_pair = st.selectbox(
            "语言对",
            options=list(LANGUAGE_PAIRS.keys()),
            format_func=lambda x: LANGUAGE_PAIRS[x]["label"],
        )

        # 翻译策略
        strategy = st.selectbox(
            "翻译策略",
            options=list(TRANSLATION_STRATEGIES.keys()),
            format_func=lambda x: f"{TRANSLATION_STRATEGIES[x]['name']} — {TRANSLATION_STRATEGIES[x]['description']}",
        )

        # 高级选项
        with st.expander("高级选项"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
            max_tokens = st.number_input("最大 Token 数", 64, 1024, 512, 64)

    if st.button("🔍 翻译", type="primary", use_container_width=True):
        if not st.session_state.vllm_running:
            st.error("请先在侧边栏启动 vLLM 服务")
        elif not input_text.strip():
            st.warning("请输入要翻译的文本")
        else:
            pair = LANGUAGE_PAIRS[lang_pair]
            source_lang = lang_pair.split("-")[0]
            target_lang = lang_pair.split("-")[1]

            with st.spinner(f"正在翻译 ({pair['label']}) ..."):
                result = translate(
                    input_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    strategy=strategy,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            if result["success"]:
                st.success(f"翻译完成 — {result['strategy_name']}")
                st.markdown("### 📝 翻译结果")
                st.info(result["translated_text"])

                # 复制按钮
                st.code(result["translated_text"], language="")

                # 保存到历史
                st.session_state.translation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "original": input_text,
                    "translated": result["translated_text"],
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "strategy": strategy,
                    "strategy_name": result["strategy_name"],
                })
            else:
                st.error(f"翻译失败: {result.get('error', '未知错误')}")

    # 历史记录
    if st.session_state.translation_history:
        with st.expander(f"📋 翻译历史 ({len(st.session_state.translation_history)})"):
            for i, h in enumerate(reversed(st.session_state.translation_history[-10:])):
                st.caption(f"#{len(st.session_state.translation_history) - i} | {h['strategy_name']} | {h.get('source_lang', '')}→{h.get('target_lang', '')}")
                st.text_area("原文", h["original"], height=60, key=f"hist_orig_{i}", disabled=True)
                st.text_area("译文", h["translated"], height=60, key=f"hist_trans_{i}", disabled=True)
                st.divider()

# ============================================================
# Tab 2: 润色
# ============================================================
with tabs[1]:
    st.subheader("✨ 学术文本润色")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        polish_input = st.text_area(
            "输入文本",
            placeholder="请输入待润色的文本（支持中英日韩）...",
            height=150,
            key="polish_input",
        )

    with col_right:
        polish_level = st.selectbox(
            "润色级别",
            options=list(POLISH_LEVELS.keys()),
            format_func=lambda x: f"{POLISH_LEVELS[x]['name']} — {POLISH_LEVELS[x]['description']}",
        )
        polish_lang = st.selectbox(
            "文本语言",
            options=list(LANGUAGE_NAMES.keys()),
            format_func=lambda x: LANGUAGE_NAMES[x],
        )

    if st.button("✨ 润色", type="primary", use_container_width=True):
        if not st.session_state.vllm_running:
            st.error("请先在侧边栏启动 vLLM 服务")
        elif not polish_input.strip():
            st.warning("请输入要润色的文本")
        else:
            with st.spinner(f"正在润色 ({POLISH_LEVELS[polish_level]['name']}) ..."):
                result = polish(polish_input, level=polish_level, target_lang=polish_lang)

            if result["success"]:
                st.success(f"润色完成 — {result['level_name']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 📄 原文")
                    st.info(result["original_text"])
                with col2:
                    st.markdown("#### ✨ 润色后")
                    st.success(result["polished_text"])

                st.session_state.polish_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "original": result["original_text"],
                    "polished": result["polished_text"],
                    "level": polish_level,
                    "level_name": result["level_name"],
                })
            else:
                st.error(f"润色失败: {result.get('error', '未知错误')}")

    if st.session_state.polish_history:
        with st.expander(f"📋 润色历史 ({len(st.session_state.polish_history)})"):
            for i, h in enumerate(reversed(st.session_state.polish_history[-10:])):
                st.caption(f"#{len(st.session_state.polish_history) - i} | {h['level_name']}")
                c1, c2 = st.columns(2)
                with c1:
                    st.text_area("原文", h["original"], height=80, key=f"pol_orig_{i}", disabled=True)
                with c2:
                    st.text_area("润色", h["polished"], height=80, key=f"pol_pol_{i}", disabled=True)
                st.divider()

# ============================================================
# Tab 3: 口语/书面语转换
# ============================================================
with tabs[2]:
    st.subheader("💬 口语 ↔ 书面语转换")

    conv_direction = st.radio(
        "转换方向",
        options=["spoken_to_written", "written_to_spoken"],
        format_func=lambda x: "口语 → 书面语" if x == "spoken_to_written" else "书面语 → 口语",
        horizontal=True,
    )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        conv_input = st.text_area(
            "输入文本",
            placeholder="请输入要转换的文本...",
            height=150,
            key="conv_input",
        )

    with col_right:
        conv_lang = st.selectbox(
            "语言",
            options=list(LANGUAGE_NAMES.keys()),
            format_func=lambda x: LANGUAGE_NAMES[x],
            key="conv_lang",
        )

    if st.button("💬 转换", type="primary", use_container_width=True):
        if not st.session_state.vllm_running:
            st.error("请先在侧边栏启动 vLLM 服务")
        elif not conv_input.strip():
            st.warning("请输入要转换的文本")
        else:
            with st.spinner("正在转换..."):
                if conv_direction == "spoken_to_written":
                    result = spoken_to_written(conv_input, lang=conv_lang)
                else:
                    result = written_to_spoken(conv_input, lang=conv_lang)

            if result["success"]:
                st.success(f"转换完成 — {result['direction_name']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 📄 原文")
                    st.info(result["original_text"])
                with col2:
                    st.markdown("#### 💬 转换后")
                    st.success(result["converted_text"])
            else:
                st.error(f"转换失败: {result.get('error', '未知错误')}")

# ============================================================
# Tab 4: 质量评估
# ============================================================
with tabs[3]:
    st.subheader("📊 翻译质量评估")

    eval_mode = st.radio(
        "评估模式",
        options=["auto", "manual"],
        format_func=lambda x: "🤖 自动评估 (BLEU/METEOR)" if x == "auto" else "✍️ 人工打分",
        horizontal=True,
    )

    if eval_mode == "auto":
        col1, col2 = st.columns(2)
        with col1:
            reference_text = st.text_area(
                "参考译文 (黄金标准)",
                placeholder="请输入高质量的参考译文...",
                height=120,
                key="ref_text",
            )
        with col2:
            candidate_text = st.text_area(
                "候选译文 (待评估)",
                placeholder="请输入待评估的翻译结果...",
                height=120,
                key="cand_text",
            )

        eval_lang = st.selectbox("语言", ["en", "zh", "ja", "ko"],
                                 format_func=lambda x: LANGUAGE_NAMES.get(x, x),
                                 key="eval_lang")

        if st.button("📊 评估", type="primary", use_container_width=True):
            if not reference_text.strip() or not candidate_text.strip():
                st.warning("请同时输入参考译文和候选译文")
            else:
                with st.spinner("正在计算评分..."):
                    result = evaluate(reference_text, candidate_text, lang=eval_lang)

                st.markdown("### 📈 评估结果")
                metrics = result["metrics"]

                col_b, col_m = st.columns(2)
                with col_b:
                    bleu = metrics.get("BLEU", {})
                    st.metric("BLEU", f"{bleu.get('bleu', 'N/A')}")
                with col_m:
                    meteor = metrics.get("METEOR", {})
                    st.metric("METEOR", f"{meteor.get('meteor', 'N/A')}")

                st.session_state.evaluation_results.append(result)

    else:  # 人工打分
        st.markdown("### ✍️ 人工评分")
        man_src = st.text_area("源文本", placeholder="请输入源文本...", key="man_src")
        man_trans = st.text_area("翻译结果", placeholder="请输入翻译结果...", key="man_trans")
        man_ref = st.text_area("参考译文", placeholder="请输入参考译文...", key="man_ref")
        man_engine = st.text_input("翻译引擎", placeholder="如 HY-MT1.5 / Qwen2.5 ...", key="man_engine")

        st.markdown("#### 评分 (1-5分)")
        c1, c2, c3 = st.columns(3)
        with c1:
            acc_score = st.slider("准确度", 1, 5, 3)
        with c2:
            flu_score = st.slider("流畅度", 1, 5, 3)
        with c3:
            ade_score = st.slider("充分性", 1, 5, 3)

        man_notes = st.text_area("备注", placeholder="可选备注...", key="man_notes")

        if st.button("💾 提交评分", type="primary", use_container_width=True):
            if not man_src.strip() or not man_trans.strip():
                st.warning("请至少填写源文本和翻译结果")
            else:
                record = st.session_state.manual_scores.add_score(
                    man_src, man_trans, man_ref,
                    acc_score, flu_score, ade_score,
                    engine=man_engine, notes=man_notes,
                )
                st.success(f"评分已保存 (ID: {record['id']})")
                # 保存到文件
                st.session_state.manual_scores.save(
                    os.path.join(RESULTS_DIR, "manual_scores.json")
                )

        # 显示已有评分
        all_scores = st.session_state.manual_scores.get_all_scores()
        if all_scores:
            with st.expander(f"📋 已有评分 ({len(all_scores)} 条)"):
                summary = st.session_state.manual_scores.get_summary()
                st.write(f"**平均总分**: {summary['total_avg']}/15 | "
                         f"准确度: {summary['avg_scores'].get('accuracy', 0)} | "
                         f"流畅度: {summary['avg_scores'].get('fluency', 0)} | "
                         f"充分性: {summary['avg_scores'].get('adequacy', 0)}")

                df = pd.DataFrame([
                    {"ID": s["id"], "准确度": s["scores"]["accuracy"],
                     "流畅度": s["scores"]["fluency"], "充分性": s["scores"]["adequacy"],
                     "总分": s["total_score"], "引擎": s["engine"], "时间": s["timestamp"][:19]}
                    for s in all_scores
                ])
                st.dataframe(df, use_container_width=True)

# ============================================================
# Tab 5: 批量处理
# ============================================================
with tabs[4]:
    st.subheader("📁 批量翻译处理")

    uploaded_file = st.file_uploader(
        "上传文档",
        type=["txt", "pdf", "docx"],
        help="支持 TXT、PDF、Word 文档格式",
    )

    batch_col1, batch_col2, batch_col3 = st.columns(3)
    with batch_col1:
        batch_lang_pair = st.selectbox(
            "语言对",
            options=list(LANGUAGE_PAIRS.keys()),
            format_func=lambda x: LANGUAGE_PAIRS[x]["label"],
            key="batch_lang",
        )
    with batch_col2:
        batch_strategy = st.selectbox(
            "翻译策略",
            options=list(TRANSLATION_STRATEGIES.keys()),
            format_func=lambda x: TRANSLATION_STRATEGIES[x]["name"],
            key="batch_strategy",
        )
    with batch_col3:
        batch_polish = st.selectbox(
            "后续润色",
            options=["none"] + list(POLISH_LEVELS.keys()),
            format_func=lambda x: "不润色" if x == "none" else POLISH_LEVELS[x]["name"],
        )

    if uploaded_file is not None:
        st.info(f"已上传: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        # 读取文件内容
        file_text = ""
        try:
            if uploaded_file.name.endswith(".txt"):
                file_text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith(".pdf"):
                import pypdf
                import io
                pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
                for page in pdf_reader.pages:
                    file_text += page.extract_text() or ""
            elif uploaded_file.name.endswith(".docx"):
                import docx
                doc = docx.Document(io.BytesIO(uploaded_file.read()))
                file_text = "\n".join([p.text for p in doc.paragraphs])

            if file_text.strip():
                # 按段落/句子分割
                paragraphs = [p.strip() for p in file_text.split("\n") if p.strip()]
                st.write(f"检测到 **{len(paragraphs)}** 个段落")

                preview_limit = min(3, len(paragraphs))
                with st.expander("📄 内容预览"):
                    for i in range(preview_limit):
                        st.caption(f"段落 {i+1}: {paragraphs[i][:200]}...")

                if st.button("🚀 开始批量翻译", type="primary", use_container_width=True):
                    if not st.session_state.vllm_running:
                        st.error("请先在侧边栏启动 vLLM 服务")
                    else:
                        pair = LANGUAGE_PAIRS[batch_lang_pair]
                        src_lang = batch_lang_pair.split("-")[0]
                        tgt_lang = batch_lang_pair.split("-")[1]

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results_container = st.container()

                        batch_results = []
                        total = len(paragraphs)

                        for i, para in enumerate(paragraphs):
                            status_text.text(f"正在翻译: {i+1}/{total}")
                            progress_bar.progress((i + 1) / total)

                            trans_res = translate(para, src_lang, tgt_lang, batch_strategy)
                            entry = {
                                "index": i + 1,
                                "original": para,
                                "translated": trans_res.get("translated_text", ""),
                                "polished": "",
                            }

                            if batch_polish != "none" and trans_res["success"]:
                                polish_res = polish(
                                    trans_res["translated_text"],
                                    level=batch_polish,
                                    target_lang=tgt_lang,
                                )
                                if polish_res["success"]:
                                    entry["polished"] = polish_res["polished_text"]

                            batch_results.append(entry)

                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"批量翻译完成！共处理 {total} 条")

                        st.session_state.batch_results = batch_results

                        # 显示结果表格
                        df = pd.DataFrame([
                            {"#": r["index"],
                             "原文": r["original"][:80] + "...",
                             "译文": r["translated"][:80] + "...",
                             "润色": r["polished"][:80] + "..." if r["polished"] else "-"}
                            for r in batch_results
                        ])
                        st.dataframe(df, use_container_width=True)

                    # 下载按钮
                    if st.session_state.batch_results:
                        st.download_button(
                            "📥 下载结果 (JSON)",
                            data=json.dumps(st.session_state.batch_results, ensure_ascii=False, indent=2),
                            file_name=f"batch_translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                        )

        except Exception as e:
            st.error(f"文件读取失败: {str(e)}")

# ============================================================
# Tab 6: 对比实验
# ============================================================
with tabs[5]:
    st.subheader("🔬 对比实验")

    experiment_type = st.radio(
        "选择实验",
        options=["exp1", "exp2"],
        format_func=lambda x: "实验一：不同翻译模型性能对比" if x == "exp1" else "实验二：不同提示策略效果对比",
        horizontal=True,
    )

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        exp_sample = st.number_input("测试样本数", 5, 100, EXPERIMENT_SAMPLE_SIZE, 5,
                                     help="从语料库中随机抽取的样本数量")
    with exp_col2:
        exp_lang_pair = st.selectbox(
            "翻译方向",
            options=list(LANGUAGE_PAIRS.keys()),
            format_func=lambda x: LANGUAGE_PAIRS[x]["label"],
            key="exp_lang",
        )

    if experiment_type == "exp1":
        st.info("**实验说明**: 使用相同的翻译策略（意译），对比三个不同模型的翻译效果。"
                "需要依次切换模型并运行翻译，记录每个模型的响应时间和成功率。")

        if st.button("🔬 运行实验一", type="primary", use_container_width=True):
            if not st.session_state.vllm_running:
                st.error("请先在侧边栏启动 vLLM 服务")
            else:
                st.warning("⚠️ 实验需要依次切换三个模型。当前运行的是: "
                           f"{MODELS.get(st.session_state.current_model, {}).get('name', '未知')}")

                src_lang = exp_lang_pair.split("-")[0]
                tgt_lang = exp_lang_pair.split("-")[1]

                # 使用当前模型运行对比
                # 简化实验：由于一次只能运行一个模型，将所有模型用'当前模型'标记
                st.info("提示：完整对比需运行三次实验（每次切换不同模型），结果会自动汇总。")

                progress_text = st.empty()

                def progress_cb(msg):
                    progress_text.text(msg)

                result = run_model_comparison(
                    translate_fn=translate,
                    models=[st.session_state.current_model],  # 只测当前模型
                    sample_size=exp_sample,
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    progress_callback=progress_cb,
                )

                progress_text.empty()

                # 保存并显示结果
                exp_store = ExperimentResults()
                existing = exp_store.load("experiment1")
                if existing:
                    # 合并已有数据
                    existing["per_sample"].extend(result["per_sample"])
                    for mk, mv in result["summary"]["models"].items():
                        existing["summary"]["models"][mk] = mv
                    result = existing

                exp_store.save("experiment1", result)
                st.session_state.experiment1_data = result

                st.success(f"实验一完成！当前模型: {MODELS[st.session_state.current_model]['name']}")

                # 显示汇总
                summary = result.get("summary", {}).get("models", {})
                if summary:
                    df = pd.DataFrame([
                        {"模型": info["model_name"][:25],
                         "成功率(%)": info["success_rate"],
                         "平均响应(s)": info["avg_response_time"],
                         "最短(s)": info["min_response_time"],
                         "最长(s)": info["max_response_time"]}
                        for info in summary.values()
                    ])
                    st.dataframe(df, use_container_width=True)

    else:
        st.info("**实验说明**: 使用同一个模型，对比三种翻译策略（直译、意译、学术润色翻译）的效果差异。")

        if st.button("🔬 运行实验二", type="primary", use_container_width=True):
            if not st.session_state.vllm_running:
                st.error("请先在侧边栏启动 vLLM 服务")
            else:
                src_lang = exp_lang_pair.split("-")[0]
                tgt_lang = exp_lang_pair.split("-")[1]

                progress_text = st.empty()

                def progress_cb(msg):
                    progress_text.text(msg)

                result = run_strategy_comparison(
                    translate_fn=translate,
                    model=st.session_state.current_model,
                    sample_size=exp_sample,
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    progress_callback=progress_cb,
                )

                progress_text.empty()

                exp_store = ExperimentResults()
                exp_store.save("experiment2", result)
                st.session_state.experiment2_data = result

                st.success("实验二完成！")

                summary = result.get("summary", {}).get("strategies", {})
                if summary:
                    df = pd.DataFrame([
                        {"策略": info["strategy_name"],
                         "成功率(%)": info["success_rate"],
                         "平均响应(s)": info["avg_response_time"]}
                        for info in summary.values()
                    ])
                    st.dataframe(df, use_container_width=True)

    # 显示已有实验结果
    st.divider()
    st.subheader("📊 实验可视化")

    exp_store = ExperimentResults()
    saved_exps = exp_store.list_experiments()

    if "experiment1" in saved_exps or "experiment2" in saved_exps:
        vis_col1, vis_col2 = st.columns(2)

        with vis_col1:
            if st.button("📊 生成可视化图表", type="primary"):
                e1_data = exp_store.load("experiment1")
                e2_data = exp_store.load("experiment2")

                charts_generated = []
                if e1_data:
                    path = plot_model_comparison(e1_data)
                    if path:
                        charts_generated.append(path)
                    path2 = plot_response_time_distribution(e1_data, "model")
                    if path2:
                        charts_generated.append(path2)

                if e2_data:
                    path = plot_strategy_comparison(e2_data)
                    if path:
                        charts_generated.append(path)
                    path2 = plot_response_time_distribution(e2_data, "strategy")
                    if path2:
                        charts_generated.append(path2)

                if charts_generated:
                    st.success(f"已生成 {len(charts_generated)} 张图表")
                    for p in charts_generated:
                        if p.endswith(".png"):
                            st.image(p, caption=os.path.basename(p))
                else:
                    st.warning("没有足够的实验数据生成图表")

        with vis_col2:
            if st.button("📝 生成实验报告"):
                e1_data = exp_store.load("experiment1")
                e2_data = exp_store.load("experiment2")
                if e1_data or e2_data:
                    report_path = generate_experiment_report(e1_data, e2_data)
                    st.success(f"报告已生成: {report_path}")
                    if report_path and os.path.exists(report_path):
                        with open(report_path, "r", encoding="utf-8") as f:
                            st.markdown(f.read())
                else:
                    st.warning("没有实验数据")
    else:
        st.info("暂无实验结果。请先运行实验。")

# ============================================================
# Tab 7: 导出
# ============================================================
with tabs[6]:
    st.subheader("💾 结果导出")

    # 收集所有可导出的数据
    export_options = []
    if st.session_state.translation_history:
        export_options.append("translation_history")
    if st.session_state.polish_history:
        export_options.append("polish_history")
    if st.session_state.evaluation_results:
        export_options.append("evaluation_results")
    if st.session_state.batch_results:
        export_options.append("batch_results")
    if st.session_state.experiment1_data:
        export_options.append("experiment1_data")
    if st.session_state.experiment2_data:
        export_options.append("experiment2_data")

    if not export_options:
        st.info("暂无数据可导出。请先使用各项功能产生数据。")
    else:
        export_items = st.multiselect(
            "选择要导出的数据",
            options=export_options,
            format_func=lambda x: {
                "translation_history": "翻译历史",
                "polish_history": "润色历史",
                "evaluation_results": "评估结果",
                "batch_results": "批量翻译结果",
                "experiment1_data": "实验一数据",
                "experiment2_data": "实验二数据",
            }.get(x, x),
        )

        export_format = st.selectbox(
            "导出格式",
            options=["json", "txt", "markdown"],
            format_func=lambda x: {"json": "JSON (结构化)", "txt": "TXT (纯文本)", "markdown": "Markdown (格式化)"}.get(x, x),
        )

        if st.button("💾 导出", type="primary", use_container_width=True):
            if not export_items:
                st.warning("请选择要导出的数据")
            else:
                exported_files = []
                for item_key in export_items:
                    data = getattr(st.session_state, item_key, None)
                    if data is None:
                        continue

                    if export_format == "json":
                        filepath = export_json(data, OUTPUT_DIR, f"{item_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    elif export_format == "txt":
                        filepath = export_txt(data, OUTPUT_DIR, f"{item_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    else:
                        filepath = export_markdown(data, OUTPUT_DIR, f"{item_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

                    exported_files.append(filepath)

                st.success(f"已导出 {len(exported_files)} 个文件到 {OUTPUT_DIR}/")
                for fp in exported_files:
                    st.write(f"📄 `{fp}`")
                    # 提供下载
                    if os.path.exists(fp):
                        with open(fp, "r", encoding="utf-8") as f:
                            content = f.read()
                        st.download_button(
                            f"下载 {os.path.basename(fp)}",
                            data=content,
                            file_name=os.path.basename(fp),
                            mime="application/octet-stream",
                        )

# ============================================================
# 底部
# ============================================================
st.divider()
st.caption("🌐 多语言翻译与润色工具 | 基于 vLLM + Streamlit | 中英日韩互译")
