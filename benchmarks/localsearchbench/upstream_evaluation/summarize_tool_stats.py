#!/usr/bin/env python3
"""
批量统计 Agent 评测结果的工具调用/对话轮数

特性
- 遍历指定目录下的 *_agent_results_*.json（自动跳过 *_search_details.json）
- 输出 overall / L3 / L4 的计数、总次数、平均次数
- 可排除指定模型（默认排除 gpt-4.1）

用法示例
  python summarize_tool_stats.py \
    --root ./evaluation_results_1211 \
    --output ./evaluation_results_1211/tool_stats_summary.csv
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def iter_result_files(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if "_agent_results_" in lower and lower.endswith(".json") and "search_details" not in lower:
                files.append(os.path.join(dirpath, name))
    return files


def load_results(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metadata", {}), data.get("results", [])


def summarize_results(results: List[Dict[str, Any]], difficulties: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    返回 {"overall": {...}, "L3": {...}, "L4": {...}}（若无样本则为空 dict）
    """
    groups: Dict[str, List[Dict[str, Any]]] = {"overall": results}
    for d in difficulties:
        groups[d] = [r for r in results if r.get("difficulty") == d]

    summary: Dict[str, Dict[str, Any]] = {}
    for key, items in groups.items():
        if not items:
            continue
        tool_counts = [len(r.get("tool_calls", [])) for r in items]
        rounds_counts = [len(r.get("conversation_history", [])) for r in items]
        summary[key] = {
            "count": len(items),
            "total_tool_calls": sum(tool_counts),
            "avg_tool_calls": sum(tool_counts) / len(items) if items else 0,
            "total_rounds": sum(rounds_counts),
            "avg_rounds": sum(rounds_counts) / len(items) if items else 0,
        }
    return summary


def merge_model_city(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    聚合同一模型跨城市的 overall/L3/L4。
    输入行应包含: model, difficulty, count, total_tool_calls, total_rounds
    """
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["model"], row["difficulty"])
        if key not in merged:
            merged[key] = {
                "model": row["model"],
                "difficulty": row["difficulty"],
                "count": 0,
                "total_tool_calls": 0,
                "total_rounds": 0,
                "source_files": [],
            }
        merged[key]["count"] += row["count"]
        merged[key]["total_tool_calls"] += row["total_tool_calls"]
        merged[key]["total_rounds"] += row["total_rounds"]
        merged[key]["source_files"].append(row["file"])

    merged_rows: List[Dict[str, Any]] = []
    for (_, diff), agg in merged.items():
        cnt = agg["count"] or 1
        merged_rows.append(
            {
                "model": agg["model"],
                "difficulty": diff,
                "count": agg["count"],
                "total_tool_calls": agg["total_tool_calls"],
                "avg_tool_calls": round(agg["total_tool_calls"] / cnt, 3),
                "total_rounds": agg["total_rounds"],
                "avg_rounds": round(agg["total_rounds"] / cnt, 3),
                "file": "|".join(agg["source_files"]),
                "city": "ALL",
            }
        )
    return merged_rows


def write_csv(rows: List[Dict[str, Any]], output: str) -> None:
    fieldnames = [
        "model",
        "city",
        "file",
        "difficulty",
        "count",
        "total_tool_calls",
        "avg_tool_calls",
        "total_rounds",
        "avg_rounds",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量统计 Agent 结果的工具调用/对话轮数")
    parser.add_argument("--root", default="./evaluation_results_1211", help="结果根目录")
    parser.add_argument("--output", default="./tool_stats_summary.csv", help="输出 CSV 路径")
    parser.add_argument(
        "--exclude-model",
        action="append",
        dest="exclude_models",
        help="排除的模型名称，可多次提供；默认排除 gpt-4.1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclude = set(m.strip() for m in (args.exclude_models or ["gpt-4.1"]))

    rows: List[Dict[str, Any]] = []
    for path in iter_result_files(args.root):
        metadata, results = load_results(path)
        model = metadata.get("model") or ""

        # 兼容 _nothinking 变体：如果目录名包含 _nothinking，强制使用目录名作为 model
        parts = os.path.normpath(path).split(os.sep)
        if len(parts) >= 2:
            parent_dir = parts[-2]
            if "_nothinking" in parent_dir:
                model = parent_dir

        if model in exclude:
            continue

        # city 兼容目录名或 metadata.dataset 前缀
        # 目录： .../<city>/<model>/file.json
        city = ""
        if len(parts) >= 3:
            city = parts[-3]  # city 层
        dataset = metadata.get("dataset") or ""
        if not city and dataset:
            city = dataset.split("_")[0]

        summary = summarize_results(results, difficulties=["L3", "L4"])
        for diff, stat in summary.items():
            rows.append(
                {
                    "model": model,
                    "city": city,
                    "file": os.path.relpath(path, args.root),
                    "difficulty": diff,
                    "count": stat["count"],
                    "total_tool_calls": stat["total_tool_calls"],
                    "avg_tool_calls": round(stat["avg_tool_calls"], 3),
                    "total_rounds": stat["total_rounds"],
                    "avg_rounds": round(stat["avg_rounds"], 3),
                }
            )

    # 追加跨城市汇总（按模型、difficulty 聚合）
    merged_rows = merge_model_city(rows)
    all_rows = rows + merged_rows

    write_csv(all_rows, args.output)
    print(f"✅ 已生成: {args.output} (共 {len(all_rows)} 行，单城/文件行 {len(rows)}，跨城汇总行 {len(merged_rows)})")


if __name__ == "__main__":
    main()

