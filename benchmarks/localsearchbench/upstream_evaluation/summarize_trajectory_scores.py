#!/usr/bin/env python3
"""
统计每个模型在所有城市的轨迹评测分数（overall、L3、L4）

输出 CSV 格式:
model, split, action_relevance, evidence_sufficiency, causal_coherence, search_efficiency, avg_score, count

使用方法:
  python3 summarize_trajectory_scores.py --input-dir trajectory_evaluations --output trajectory_scores_summary_6.csv
  
  
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Any


def extract_model_from_filename(filename: str) -> str:
    """从文件名提取模型名，如 trajectory_eval_beijing_deepseek-v32-meituan_20251218_180000.json -> deepseek-v32-meituan"""
    # 格式: trajectory_eval_{city}_{model}_{timestamp}.json
    match = re.match(r"trajectory_eval_[^_]+_(.+)_\d{8}_\d{6}\.json", filename)
    if match:
        return match.group(1)
    return ""


def extract_city_from_filename(filename: str) -> str:
    """从文件名提取城市名"""
    match = re.match(r"trajectory_eval_([^_]+)_.+_\d{8}_\d{6}\.json", filename)
    if match:
        return match.group(1)
    return ""


def get_difficulty(eval_item: Dict[str, Any]) -> str:
    """获取难度级别，返回 'L3' 或 'L4'，无法判断则返回 'unknown'"""
    diff = eval_item.get("difficulty")
    if diff:
        return str(diff).upper()
    # 尝试从 question_id 推断（如果有规律的话）
    return "unknown"


def load_eval_file(filepath: str) -> List[Dict[str, Any]]:
    """加载评测结果文件"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("evaluations", [])
    except Exception as e:
        print(f"警告: 无法加载 {filepath}: {e}")
        return []


def compute_avg_scores(evals: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算平均分数"""
    if not evals:
        return {
            "action_relevance": 0.0,
            "evidence_sufficiency": 0.0,
            "causal_coherence": 0.0,
            "search_efficiency": 0.0,
            "avg_score": 0.0,
            "count": 0,
        }

    ar_sum = es_sum = cc_sum = se_sum = 0.0
    valid_count = 0

    for e in evals:
        evaluation = e.get("evaluation", {})
        if "error" in evaluation:
            continue
        scores = evaluation.get("scores", {})
        if not scores:
            continue

        ar = scores.get("action_relevance", 0)
        es = scores.get("evidence_sufficiency", 0)
        cc = scores.get("causal_coherence", 0)
        se = scores.get("search_efficiency", 0)

        ar_sum += ar
        es_sum += es
        cc_sum += cc
        se_sum += se
        valid_count += 1

    if valid_count == 0:
        return {
            "action_relevance": 0.0,
            "evidence_sufficiency": 0.0,
            "causal_coherence": 0.0,
            "search_efficiency": 0.0,
            "avg_score": 0.0,
            "count": 0,
        }

    ar_avg = ar_sum / valid_count
    es_avg = es_sum / valid_count
    cc_avg = cc_sum / valid_count
    se_avg = se_sum / valid_count
    overall_avg = (ar_avg + es_avg + cc_avg + se_avg) / 4

    return {
        "action_relevance": ar_avg,
        "evidence_sufficiency": es_avg,
        "causal_coherence": cc_avg,
        "search_efficiency": se_avg,
        "avg_score": overall_avg,
        "count": valid_count,
    }


def main():
    parser = argparse.ArgumentParser(description="统计轨迹评测分数")
    parser.add_argument("--input-dir", required=True, help="评测结果目录")
    parser.add_argument("--output", default="trajectory_scores_summary.csv", help="输出 CSV 文件")
    args = parser.parse_args()

    # 收集所有评测文件
    # 结构: model -> city -> [evals]
    model_city_evals: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for filename in os.listdir(args.input_dir):
        if not filename.startswith("trajectory_eval_") or not filename.endswith(".json"):
            continue

        model = extract_model_from_filename(filename)
        city = extract_city_from_filename(filename)
        if not model or not city:
            continue

        filepath = os.path.join(args.input_dir, filename)
        evals = load_eval_file(filepath)
        model_city_evals[model][city].extend(evals)

    # 统计每个模型的 overall、L3、L4 分数
    results = []

    for model in sorted(model_city_evals.keys()):
        city_data = model_city_evals[model]

        # 合并所有城市的数据
        all_evals = []
        l3_evals = []
        l4_evals = []

        for city, evals in city_data.items():
            for e in evals:
                all_evals.append(e)
                diff = get_difficulty(e)
                if diff == "L3":
                    l3_evals.append(e)
                elif diff == "L4":
                    l4_evals.append(e)

        # 计算 overall
        overall_scores = compute_avg_scores(all_evals)
        results.append({
            "model": model,
            "split": "overall",
            **overall_scores,
        })

        # 计算 L3
        l3_scores = compute_avg_scores(l3_evals)
        results.append({
            "model": model,
            "split": "L3",
            **l3_scores,
        })

        # 计算 L4
        l4_scores = compute_avg_scores(l4_evals)
        results.append({
            "model": model,
            "split": "L4",
            **l4_scores,
        })

    # 输出 CSV
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("model,split,action_relevance,evidence_sufficiency,causal_coherence,search_efficiency,avg_score,count\n")
        for r in results:
            f.write(f"{r['model']},{r['split']},{r['action_relevance']:.3f},{r['evidence_sufficiency']:.3f},{r['causal_coherence']:.3f},{r['search_efficiency']:.3f},{r['avg_score']:.3f},{r['count']}\n")

    print(f"✅ 统计完成，结果已保存到: {args.output}")
    print(f"   共 {len(model_city_evals)} 个模型")

    # 打印摘要
    print("\n📊 模型分数摘要 (overall):")
    print("-" * 100)
    print(f"{'模型':<45} {'AR':>6} {'ES':>6} {'CC':>6} {'SE':>6} {'Avg':>6} {'Count':>6}")
    print("-" * 100)
    for r in results:
        if r["split"] == "overall" and r["count"] > 0:
            print(f"{r['model']:<45} {r['action_relevance']:>6.2f} {r['evidence_sufficiency']:>6.2f} {r['causal_coherence']:>6.2f} {r['search_efficiency']:>6.2f} {r['avg_score']:>6.2f} {r['count']:>6}")


if __name__ == "__main__":
    main()
