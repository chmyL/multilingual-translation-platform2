# 结果导出模块
# 支持将翻译和润色结果导出为JSON、TXT或Markdown格式。
import json
import os
from datetime import datetime
from typing import Dict, Any, List

# 导出为JSON格式
def export_json(data: Any, output_path: str = "", filename: str = "",) -> str:
    if not filename:
        filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    filepath = os.path.join(output_path, f"{filename}.json") if output_path else f"{filename}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return filepath

# 导出为TXT格式（双语对照纯文本）
def export_txt(data: Any, output_path: str = "", filename: str = "",) -> str:
    if not filename:
        filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    filepath = os.path.join(output_path, f"{filename}.txt") if output_path else f"{filename}.txt"

    lines = []
    lines.append("=" * 60)
    lines.append(f"翻译/润色结果导出")
    lines.append(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    if isinstance(data, list):
        for i, item in enumerate(data):
            lines.append(f"--- 条目 {i + 1} ---")
            _write_item_txt(lines, item)
            lines.append("")
    elif isinstance(data, dict):
        _write_item_txt(lines, data)
    else:
        lines.append(str(data))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath

# 将单个翻译条目写入TXT行列表
def _write_item_txt(lines: List[str], item: Dict[str, Any]):
    if isinstance(item, dict):
        # 翻译结果
        if "original_text" in item:
            lines.append(f"原文: {item.get('original_text', '')}")
        if "source_text" in item:
            lines.append(f"源文本: {item.get('source_text', '')}")

        if "translated_text" in item:
            src = item.get("source_lang_name", "")
            tgt = item.get("target_lang_name", "")
            strategy = item.get("strategy_name", "")
            lines.append(f"翻译 ({src}→{tgt}, {strategy}): {item['translated_text']}")

        if "polished_text" in item:
            level = item.get("level_name", "")
            lines.append(f"润色后 ({level}): {item['polished_text']}")

        if "converted_text" in item:
            direction = item.get("direction_name", "")
            lines.append(f"转换后 ({direction}): {item['converted_text']}")

        # 评估结果
        if "metrics" in item:
            metrics = item["metrics"]
            bleu_info = metrics.get("BLEU", {})
            meteor_info = metrics.get("METEOR", {})
            if bleu_info:
                lines.append(f"BLEU: {bleu_info.get('bleu', 'N/A')}")
            if meteor_info:
                lines.append(f"METEOR: {meteor_info.get('meteor', 'N/A')}")

        # 递归处理嵌套字典
        for key, value in item.items():
            if key in ("original_text", "source_text", "translated_text",
                       "polished_text", "converted_text", "metrics"):
                continue
            if isinstance(value, dict):
                _write_item_txt(lines, value)

# 导出为Markdown格式
def export_markdown(data: Any, output_path: str = "", filename: str = "",) -> str:
    if not filename:
        filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    filepath = os.path.join(output_path, f"{filename}.md") if output_path else f"{filename}.md"

    lines = []
    lines.append("# 翻译/润色结果导出")
    lines.append(f"> 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 翻译列表→表格
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        _build_translation_table(lines, data)
    elif isinstance(data, dict):
        # 单个结果
        lines.append("## 结果详情")
        lines.append("")
        _write_item_markdown(lines, data)
    else:
        lines.append(f"```\n{data}\n```")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath

# 构建翻译结果Markdown表格
def _build_translation_table(lines: List[str], items: List[Dict[str, Any]]):
    # 判断类型
    has_original = any("original_text" in item for item in items)
    has_translated = any("translated_text" in item for item in items)
    has_polished = any("polished_text" in item for item in items)
    has_metrics = any("metrics" in item for item in items)

    headers = ["序号"]
    if has_original:
        headers.append("原文")
    if has_translated:
        headers.append("译文")
    if has_polished:
        headers.append("润色后")
    if has_metrics:
        headers.append("BLEU")
        headers.append("METEOR")

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for i, item in enumerate(items):
        row = [str(i + 1)]
        if has_original:
            row.append(_escape_md(item.get("original_text", "")))
        if has_translated:
            row.append(_escape_md(item.get("translated_text", "")))
        if has_polished:
            row.append(_escape_md(item.get("polished_text", "")))
        if has_metrics:
            metrics = item.get("metrics", {})
            bleu_val = metrics.get("BLEU", {}).get("bleu", "")
            meteor_val = metrics.get("METEOR", {}).get("meteor", "")
            row.append(str(bleu_val))
            row.append(str(meteor_val))
        lines.append("| " + " | ".join(row) + " |")

# 将单个条目写入Markdown
def _write_item_markdown(lines: List[str], item: Dict[str, Any]):
    if isinstance(item, dict):
        for key, value in item.items():
            if key in ("timestamp", "error"):
                continue
            if isinstance(value, dict):
                lines.append(f"### {key}")
                lines.append("")
                _write_item_markdown(lines, value)
            elif isinstance(value, list):
                lines.append(f"### {key}")
                lines.append("")
                for v in value:
                    lines.append(f"- {_escape_md(str(v))}")
            else:
                lines.append(f"**{key}**: {_escape_md(str(value))}")
                lines.append("")

# 转义Markdown特殊字符
def _escape_md(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # 截断长文本用于表格
    if len(text) > 120:
        text = text[:117] + "..."
    return text.replace("|", "\\|").replace("\n", " ")
