#!/usr/bin/env python3
"""
统计每个模型在所有城市的答案评测分数（overall、L3、L4）

输出 CSV 格式:
model, split, correctness, completeness, fluency, safety, total_score, count

使用方法:
  python summarize_answer_scores.py --input-dir answer_judgements --output answer_scores_summary.csv
  
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Any


def extract_model_from_path(filepath: str) -> str:
    """从文件路径提取模型名，如 answer_judgements/beijing/deepseek-v32-meituan/xxx.json -> deepseek-v32-meituan"""
    # 路径格式: answer_judgements/{city}/{model}/xxx.json
    parts = filepath.split(os.sep)
    if len(parts) >= 3:
        return parts[-2]  # 倒数第二层是模型名
    return ""


def extract_city_from_path(filepath: str) -> str:
    """从文件路径提取城市名"""
    # 路径格式: answer_judgements/{city}/{model}/xxx.json
    parts = filepath.split(os.sep)
    if len(parts) >= 4:
        return parts[-3]  # 倒数第三层是城市名
    return ""


def get_difficulty(result_item: Dict[str, Any]) -> str:
    """获取难度级别，返回 'L3' 或 'L4'，无法判断则返回 'unknown'"""
    diff = result_item.get("difficulty")
    if diff:
        return str(diff).upper()
    return "unknown"


def load_judged_file(filepath: str) -> List[Dict[str, Any]]:
    """加载答案评测结果文件"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 从 results 数组中提取所有结果
        return data.get("results", [])
    except Exception as e:
        print(f"警告: 无法加载 {filepath}: {e}")
        return []


def compute_avg_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算平均分数"""
    if not results:
        return {
            "correctness": 0.0,
            "completeness": 0.0,
            "fluency": 0.0,
            "safety": 0.0,
            "hallucination": 0.0,
            "total_score": 0.0,
            "count": 0,
        }

    corr_sum = comp_sum = flu_sum = safe_sum = hall_sum = total_sum = 0.0
    valid_count = 0

    for r in results:
        judge_scores = r.get("judge_scores")
        if not judge_scores or not isinstance(judge_scores, dict):
            continue
        if "error" in judge_scores:
            continue

        # 提取四个维度的分数
        corr = judge_scores.get("correctness", {}).get("score", 0)
        comp = judge_scores.get("completeness", {}).get("score", 0)
        flu = judge_scores.get("fluency", {}).get("score", 0)
        safe = judge_scores.get("safety", {}).get("score", 0)
        # 幻觉评分可能存在，格式可能为 dict 或数字
        hall_val = judge_scores.get("hallucination", None)
        hall = 0.0
        if isinstance(hall_val, dict):
            hall = hall_val.get("score", 0.0) or 0.0
        else:
            try:
                if hall_val is not None:
                    hall = float(hall_val)
            except Exception:
                hall = 0.0
        total = judge_scores.get("total_score", 0)

        # 验证分数是否有效
        if corr is None or comp is None or flu is None or safe is None:
            continue

        corr_sum += corr
        comp_sum += comp
        flu_sum += flu
        safe_sum += safe
        hall_sum += hall
        total_sum += total
        valid_count += 1

    if valid_count == 0:
        return {
            "correctness": 0.0,
            "completeness": 0.0,
            "fluency": 0.0,
            "safety": 0.0,
            "total_score": 0.0,
            "count": 0,
        }

    corr_avg = corr_sum / valid_count
    comp_avg = comp_sum / valid_count
    flu_avg = flu_sum / valid_count
    safe_avg = safe_sum / valid_count
    hall_avg = hall_sum / valid_count
    total_avg = total_sum / valid_count

    return {
        "correctness": corr_avg,
        "completeness": comp_avg,
        "fluency": flu_avg,
        "safety": safe_avg,
        "hallucination": hall_avg,
        "total_score": total_avg,
        "count": valid_count,
    }


def find_judged_files(input_dir: str) -> List[str]:
    """递归查找所有 _judged.json 文件"""
    judged_files = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith("_judged.json"):
                filepath = os.path.join(root, filename)
                judged_files.append(filepath)
    return judged_files


def main():
    parser = argparse.ArgumentParser(description="统计答案评测分数")
    parser.add_argument("--input-dir", required=True, help="答案评测结果目录")
    parser.add_argument("--output", default="answer_scores_summary.csv", help="输出 CSV 文件")
    args = parser.parse_args()

    # 收集所有评测文件
    # 结构: model -> city -> [results]
    model_city_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    # 递归查找所有 _judged.json 文件
    judged_files = find_judged_files(args.input_dir)
    
    if not judged_files:
        print(f"⚠️  在 {args.input_dir} 中未找到任何 _judged.json 文件")
        return

    print(f"📂 找到 {len(judged_files)} 个评测文件")

    for filepath in judged_files:
        model = extract_model_from_path(filepath)
        city = extract_city_from_path(filepath)
        
        if not model or not city:
            # 如果无法从路径提取，尝试从文件名提取
            filename = os.path.basename(filepath)
            # 尝试从文件名匹配: {city}_{model}_xxx_judged.json 或类似格式
            match = re.search(r"([^_]+)_([^_]+(?:_[^_]+)*)_\d{8}_\d{6}_judged\.json", filename)
            if match:
                city = match.group(1)
                model = match.group(2)
            else:
                print(f"⚠️  无法从路径提取模型和城市: {filepath}")
                continue

        results = load_judged_file(filepath)
        model_city_results[model][city].extend(results)

    # 统计每个模型的 overall、L3、L4 分数
    results = []

    for model in sorted(model_city_results.keys()):
        city_data = model_city_results[model]

        # 合并所有城市的数据
        all_results = []
        l3_results = []
        l4_results = []

        for city, results_list in city_data.items():
            for r in results_list:
                all_results.append(r)
                diff = get_difficulty(r)
                if diff == "L3":
                    l3_results.append(r)
                elif diff == "L4":
                    l4_results.append(r)

        # 计算 overall
        overall_scores = compute_avg_scores(all_results)
        results.append({
            "model": model,
            "split": "overall",
            **overall_scores,
        })

        # 计算 L3
        l3_scores = compute_avg_scores(l3_results)
        results.append({
            "model": model,
            "split": "L3",
            **l3_scores,
        })

        # 计算 L4
        l4_scores = compute_avg_scores(l4_results)
        results.append({
            "model": model,
            "split": "L4",
            **l4_scores,
        })

    # 输出 CSV
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("model,split,correctness,completeness,fluency,safety,hallucination,total_score,count\n")
        for r in results:
            f.write(f"{r['model']},{r['split']},{r['correctness']:.3f},{r['completeness']:.3f},{r['fluency']:.3f},{r['safety']:.3f},{r.get('hallucination', 0.0):.3f},{r['total_score']:.3f},{r['count']}\n")

    print(f"✅ 统计完成，结果已保存到: {args.output}")
    print(f"   共 {len(model_city_results)} 个模型")

    # 打印摘要
    print("\n📊 模型分数摘要 (overall):")
    print("-" * 110)
    print(f"{'模型':<45} {'Corr':>6} {'Comp':>6} {'Flu':>6} {'Safe':>6} {'Hall':>6} {'Total':>6} {'Count':>6}")
    print("-" * 110)
    for r in results:
        if r["split"] == "overall" and r["count"] > 0:
            print(f"{r['model']:<45} {r['correctness']:>6.2f} {r['completeness']:>6.2f} {r['fluency']:>6.2f} {r['safety']:>6.2f} {r.get('hallucination', 0.0):>6.2f} {r['total_score']:>6.2f} {r['count']:>6}")


if __name__ == "__main__":
    main()

