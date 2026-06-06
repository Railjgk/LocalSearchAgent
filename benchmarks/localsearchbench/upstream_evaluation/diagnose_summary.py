#!/usr/bin/env python3
"""
自动提取多个模型的三城市评估结果并计算加权平均
支持的模型: deepseek-v31-meituan, Qwen-Plus-Latest, gpt-4.1, glm-4.5, gemini-2.5-pro, hunyuan-t1-latest,
           LongCat-Large-32K-Chat-0626, Qwen3-235B-A22B-Meituan, Qwen3-32B-Meituan, Qwen3-14B-Meituan
"""

import json
import os
import glob
import argparse
from pathlib import Path
import pandas as pd

def find_result_files(base_dir, model_name, city_name):
    """查找指定模型和城市的结果文件"""
    # 城市名称映射（完整名称到前缀）
    city_prefix_map = {
        'beijing': 'bj',
        'shanghai': 'sh',
        'guangzhou': 'gz'
    }
    
    city_prefix = city_prefix_map.get(city_name.lower(), city_name.lower()[:2])
    
    # 可能的目录路径
    possible_dirs = [
        base_dir,  # 直接在base_dir下（新格式）
        os.path.join(base_dir, city_name, model_name),  # 旧格式1
        os.path.join(base_dir, model_name, city_name),  # 旧格式2
    ]
    
    for city_dir in possible_dirs:
        if not os.path.exists(city_dir):
            continue
        
        # 查找匹配的JSON文件（排除search_details文件）
        # 文件名格式: {city_prefix}_query_1_agent_results_{timestamp}.json
        json_files = []
        for f in os.listdir(city_dir):
            if (f.startswith(f"{city_prefix}_") and 
                f.endswith('.json') and 
                'search_details' not in f and
                'agent_results' in f):
                json_files.append(os.path.join(city_dir, f))
        
        if json_files:
            # 选择最新的文件
            latest_file = max(json_files, key=os.path.getmtime)
            print(f"✅ 找到文件: {latest_file}")
            return latest_file
    
    print(f"❌ 未找到 {city_name} 的结果文件")
    return None

def extract_judge_averages(json_file):
    """从JSON文件中提取judge_averages数据以及工具调用和对话轮次统计
    
    注意：会自动重新计算平均工具调用次数和对话轮次，使用有效样本数量作为分母
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # judge_averages在metadata字段中
        metadata = data.get('metadata', {})
        judge_averages = metadata.get('judge_averages', {})
        
        if not judge_averages:
            print(f"❌ 文件中未找到judge_averages: {json_file}")
            return None
        
        # 提取额外的统计数据
        result = dict(judge_averages)
        
        # 获取总调用次数和轮次
        total_tool_calls = metadata.get('total_tool_calls', 0)
        total_conversation_rounds = metadata.get('total_conversation_rounds', 0)
        
        # 获取有效样本数量（有Judge评分的问题数）
        sample_count = judge_averages.get('count', 0)
        
        # 重新计算平均值，使用有效样本数量作为分母
        if sample_count > 0:
            recalculated_avg_tool_calls = total_tool_calls / sample_count
            recalculated_avg_conversation_rounds = total_conversation_rounds / sample_count
            
            # 检查是否与原始值不同
            old_avg_tool_calls = metadata.get('avg_tool_calls', 0)
            old_avg_conversation_rounds = metadata.get('avg_conversation_rounds', 0)
            
            if abs(recalculated_avg_tool_calls - old_avg_tool_calls) > 0.0001:
                print(f"  ⚠️  修正平均工具调用次数: {old_avg_tool_calls:.4f} → {recalculated_avg_tool_calls:.4f}")
            
            if abs(recalculated_avg_conversation_rounds - old_avg_conversation_rounds) > 0.0001:
                print(f"  ⚠️  修正平均对话轮次: {old_avg_conversation_rounds:.4f} → {recalculated_avg_conversation_rounds:.4f}")
            
            # 使用重新计算的值
            result['avg_tool_calls'] = recalculated_avg_tool_calls
            result['avg_conversation_rounds'] = recalculated_avg_conversation_rounds
        else:
            # 如果没有有效样本，使用原始值
            result['avg_tool_calls'] = metadata.get('avg_tool_calls', 0)
            result['avg_conversation_rounds'] = metadata.get('avg_conversation_rounds', 0)
        
        result['total_tool_calls'] = total_tool_calls
        result['total_conversation_rounds'] = total_conversation_rounds
        
        return result
    except Exception as e:
        print(f"❌ 读取文件失败 {json_file}: {str(e)}")
        return None

def calculate_weighted_averages(cities_data, model_name):
    """计算单个模型的三城市加权平均"""
    print(f"\n{'='*80}")
    print(f"模型: {model_name} - 三城市加权平均指标计算")
    print(f"{'='*80}")
    
    # 显示各城市数据
    print(f"\n📊 各城市数据:")
    total_count = 0
    valid_cities = {}
    
    for city, data in cities_data.items():
        if data and 'count' in data:
            valid_cities[city] = data
            total_count += data['count']
            print(f"\n{city}:")
            print(f"  样本数量: {data.get('count', 0)}")
            print(f"  正确性: {data.get('correctness', 0):.4f}")
            print(f"  完整性: {data.get('completeness', 0):.4f}")
            print(f"  流畅度: {data.get('fluency', 0):.4f}")
            print(f"  安全性: {data.get('safety', 0):.4f}")
            print(f"  幻觉检测: {data.get('hallucination', 0):.4f}")
            print(f"  总分: {data.get('total_score', 0):.4f}")
            print(f"  平均工具调用次数: {data.get('avg_tool_calls', 0):.2f}")
            print(f"  平均对话轮次: {data.get('avg_conversation_rounds', 0):.2f}")
        else:
            print(f"\n{city}: ❌ 数据无效或缺失")
    
    if not valid_cities:
        print("❌ 没有有效的城市数据")
        return None
    
    # 计算加权平均
    metrics = ['correctness', 'completeness', 'fluency', 'safety', 'hallucination', 'total_score', 
               'avg_tool_calls', 'avg_conversation_rounds']
    weighted_averages = {}
    
    print(f"\n📈 加权平均计算 (总样本数: {total_count}):")
    print("-" * 60)
    
    for metric in metrics:
        weighted_sum = 0
        metric_total_count = 0
        
        print(f"\n{metric}:")
        for city, data in valid_cities.items():
            if metric in data:
                count = data['count']
                value = data[metric]
                weight = count / total_count
                contribution = value * weight
                weighted_sum += contribution
                metric_total_count += count
                print(f"  {city}: {value:.4f} × {weight:.4f} = {contribution:.4f}")
        
        if metric_total_count > 0:
            weighted_averages[metric] = weighted_sum
            print(f"  {metric} 加权平均: {weighted_sum:.4f}")
        else:
            weighted_averages[metric] = 0
            print(f"  {metric} 加权平均: 无数据")
    
    # 输出最终结果
    print(f"\n🎯 {model_name} 最终加权平均结果:")
    print("=" * 50)
    print(f"正确性 (correctness): {weighted_averages.get('correctness', 0):.4f}")
    print(f"完整性 (completeness): {weighted_averages.get('completeness', 0):.4f}")
    print(f"流畅度 (fluency): {weighted_averages.get('fluency', 0):.4f}")
    print(f"安全性 (safety): {weighted_averages.get('safety', 0):.4f}")
    print(f"幻觉检测 (hallucination): {weighted_averages.get('hallucination', 0):.4f}")
    print(f"总分 (total_score): {weighted_averages.get('total_score', 0):.4f}")
    print(f"平均工具调用次数: {weighted_averages.get('avg_tool_calls', 0):.2f}")
    print(f"平均对话轮次: {weighted_averages.get('avg_conversation_rounds', 0):.2f}")
    print(f"总样本数: {total_count}")
    
    # 计算百分比形式
    print(f"\n📊 {model_name} 百分比形式:")
    print("=" * 50)
    if weighted_averages.get('correctness', 0) > 0:
        print(f"正确性: {(weighted_averages['correctness']/4.0)*100:.2f}%")
    if weighted_averages.get('completeness', 0) > 0:
        print(f"完整性: {(weighted_averages['completeness']/10.0)*100:.2f}%")
    if weighted_averages.get('fluency', 0) > 0:
        print(f"流畅度: {(weighted_averages['fluency']/10.0)*100:.2f}%")
    if weighted_averages.get('safety', 0) > 0:
        print(f"安全性: {(weighted_averages['safety']/10.0)*100:.2f}%")
    if weighted_averages.get('hallucination', 0) > 0:
        print(f"幻觉检测: {(weighted_averages['hallucination']/10.0)*100:.2f}%")
    if weighted_averages.get('total_score', 0) > 0:
        print(f"总分: {(weighted_averages['total_score']/4.0)*100:.2f}%")
    
    return weighted_averages

def generate_comparison_report(all_models_results):
    """生成所有模型的对比报告"""
    print(f"\n{'='*140}")
    print("🏆 所有模型对比报告")
    print(f"{'='*140}")
    
    # 创建对比表格
    metrics = ['correctness', 'completeness', 'fluency', 'safety', 'hallucination', 'total_score', 
               'avg_tool_calls', 'avg_conversation_rounds']
    metric_names = ['正确性', '完整性', '流畅度', '安全性', '幻觉检测', '总分', 
                    '平均工具调用', '平均对话轮次']
    
    model_list = ['deepseek-v31-meituan', 'Qwen-Plus-Latest', 'gpt-4.1', 'glm-4.5', 'gemini-2.5-pro', 'hunyuan-t1-latest', 
                  'LongCat-Large-32K-Chat-0626', 'Qwen3-235B-A22B-Meituan', 'Qwen3-32B-Meituan', 'Qwen3-14B-Meituan']
    model_display_names = ['deepseek-v31', 'Qwen-Plus', 'gpt-4.1', 'glm-4.5', 'gemini-2.5-pro', 'hunyuan-t1', 
                          'LongCat', 'Qwen3-235B', 'Qwen3-32B', 'Qwen3-14B']
    
    print(f"\n📊 各项指标对比:")
    print("-" * 140)
    header = f"{'指标':<12}"
    for display_name in model_display_names:
        header += f"{display_name:<18}"
    print(header)
    print("-" * 140)
    
    for i, metric in enumerate(metrics):
        row = f"{metric_names[i]:<12}"
        for model in model_list:
            if model in all_models_results and all_models_results[model]:
                value = all_models_results[model].get(metric, 0)
                row += f"{value:<18.4f}"
            else:
                row += f"{'N/A':<18}"
        print(row)
    
    # 百分比对比
    print(f"\n📈 百分比对比:")
    print("-" * 140)
    header = f"{'指标':<12}"
    for display_name in model_display_names:
        header += f"{display_name:<18}"
    print(header)
    print("-" * 140)
    
    scales = [1.0, 10.0, 10.0, 10.0, 10.0, 10.0, None, None]  # 各指标的满分，工具调用和对话轮次无满分
    
    for i, metric in enumerate(metrics):
        row = f"{metric_names[i]:<12}"
        for model in model_list:
            if model in all_models_results and all_models_results[model]:
                value = all_models_results[model].get(metric, 0)
                if scales[i] is not None:  # 只有有满分的指标才计算百分比
                    percentage = (value / scales[i]) * 100
                    row += f"{percentage:<18.2f}%"
                else:
                    row += f"{value:<18.2f}"  # 工具调用和对话轮次直接显示数值
            else:
                row += f"{'N/A':<18}"
        print(row)
    
    # 排名分析
    print(f"\n🏅 各指标排名:")
    print("-" * 80)
    
    for i, metric in enumerate(metrics):
        print(f"\n{metric_names[i]} 排名:")
        model_scores = []
        for j, model in enumerate(model_list):
            if model in all_models_results and all_models_results[model]:
                score = all_models_results[model].get(metric, 0)
                model_scores.append((model_display_names[j], score))
        
        # 按分数排序
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (model_display, score) in enumerate(model_scores, 1):
            if scales[i] is not None:
                percentage = (score / scales[i]) * 100
                print(f"  {rank}. {model_display:<18}: {score:.4f} ({percentage:.2f}%)")
            else:
                print(f"  {rank}. {model_display:<18}: {score:.2f}")

def generate_csv_report(all_models_results, output_csv):
    """生成CSV格式的对比报告"""
    metrics = ['correctness', 'completeness', 'fluency', 'safety', 'hallucination', 'total_score', 
               'avg_tool_calls', 'avg_conversation_rounds']
    metric_names = ['正确性', '完整性', '流畅度', '安全性', '幻觉检测', '总分', 
                    '平均工具调用', '平均对话轮次']
    scales = [1.0, 10.0, 10.0, 10.0, 10.0, 1.0, None, None]  # 各指标的满分，工具调用和对话轮次无满分
    
    model_list = ['deepseek-v31-meituan', 'Qwen-Plus-Latest', 'gpt-4.1', 'glm-4.5', 'gemini-2.5-pro', 'hunyuan-t1-latest',
                  'LongCat-Large-32K-Chat-0626', 'Qwen3-235B-A22B-Meituan', 'Qwen3-32B-Meituan', 'Qwen3-14B-Meituan']
    model_display_names = ['deepseek-v31', 'Qwen-Plus', 'gpt-4.1', 'glm-4.5', 'gemini-2.5-pro', 'hunyuan-t1',
                          'LongCat', 'Qwen3-235B', 'Qwen3-32B', 'Qwen3-14B']
    
    # 创建两个DataFrame：原始分数和百分比
    
    # 1. 原始分数表
    raw_data = []
    for i, metric in enumerate(metrics):
        row = {'指标': metric_names[i], '指标英文': metric}
        for j, model in enumerate(model_list):
            if model in all_models_results and all_models_results[model]:
                value = all_models_results[model].get(metric, 0)
                row[model_display_names[j]] = f"{value:.4f}"
            else:
                row[model_display_names[j]] = "N/A"
        raw_data.append(row)
    
    df_raw = pd.DataFrame(raw_data)
    
    # 2. 百分比表
    percentage_data = []
    for i, metric in enumerate(metrics):
        row = {'指标': metric_names[i], '指标英文': metric, '满分': scales[i] if scales[i] is not None else 'N/A'}
        for j, model in enumerate(model_list):
            if model in all_models_results and all_models_results[model]:
                value = all_models_results[model].get(metric, 0)
                if scales[i] is not None:
                    percentage = (value / scales[i]) * 100
                    row[model_display_names[j]] = f"{percentage:.2f}%"
                else:
                    row[model_display_names[j]] = f"{value:.2f}"  # 工具调用和对话轮次直接显示数值
            else:
                row[model_display_names[j]] = "N/A"
        percentage_data.append(row)
    
    df_percentage = pd.DataFrame(percentage_data)
    
    # 3. 排名表
    ranking_data = []
    for i, metric in enumerate(metrics):
        model_scores = []
        for j, model in enumerate(model_list):
            if model in all_models_results and all_models_results[model]:
                score = all_models_results[model].get(metric, 0)
                model_scores.append((model_display_names[j], score))
        
        # 按分数排序
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        row = {'指标': metric_names[i], '指标英文': metric}
        for rank, (model_name, score) in enumerate(model_scores, 1):
            row[f'第{rank}名'] = model_name
            row[f'第{rank}名分数'] = f"{score:.4f}"
            if scales[i] is not None:
                percentage = (score / scales[i]) * 100
                row[f'第{rank}名百分比'] = f"{percentage:.2f}%"
            else:
                row[f'第{rank}名百分比'] = 'N/A'
        
        ranking_data.append(row)
    
    df_ranking = pd.DataFrame(ranking_data)
    
    # 保存到CSV（使用多个sheet的方式需要Excel，这里分别保存三个CSV）
    base_name = output_csv.replace('.csv', '')
    
    # 保存原始分数
    csv_raw = f"{base_name}_raw_scores.csv"
    df_raw.to_csv(csv_raw, index=False, encoding='utf-8-sig')
    print(f"💾 原始分数已保存到: {csv_raw}")
    
    # 保存百分比
    csv_percentage = f"{base_name}_percentages.csv"
    df_percentage.to_csv(csv_percentage, index=False, encoding='utf-8-sig')
    print(f"💾 百分比数据已保存到: {csv_percentage}")
    
    # 保存排名
    csv_ranking = f"{base_name}_rankings.csv"
    df_ranking.to_csv(csv_ranking, index=False, encoding='utf-8-sig')
    print(f"💾 排名数据已保存到: {csv_ranking}")
    
    # 也可以保存一个综合的CSV
    csv_combined = f"{base_name}_combined.csv"
    with open(csv_combined, 'w', encoding='utf-8-sig') as f:
        f.write("原始分数\n")
        df_raw.to_csv(f, index=False)
        f.write("\n\n百分比\n")
        df_percentage.to_csv(f, index=False)
        f.write("\n\n排名\n")
        df_ranking.to_csv(f, index=False)
    print(f"💾 综合报告已保存到: {csv_combined}")

def main():
    parser = argparse.ArgumentParser(description="自动提取多个模型的三城市评估结果")
    parser.add_argument("--base_dir", type=str, 
                       default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/hehang03/tuansou/Localplayground/evaluation_toolkit/evaluation_results",
                       help="评估结果基础目录")
    parser.add_argument("--models", nargs='+', 
                       default=['deepseek-v31-meituan', 'Qwen-Plus-Latest', 'gpt-4.1', 'glm-4.5', 'gemini-2.5-pro', 'hunyuan-t1-latest',
                                'LongCat-Large-32K-Chat-0626', 'Qwen3-235B-A22B-Meituan', 'Qwen3-32B-Meituan', 'Qwen3-14B-Meituan'],
                       help="要处理的模型列表")
    parser.add_argument("--cities", nargs='+', 
                       default=['beijing', 'shanghai', 'guangzhou'],
                       help="要处理的城市列表")
    parser.add_argument("--output", type=str, help="输出结果到JSON文件")
    parser.add_argument("--csv", type=str, default="model_comparison_results.csv",
                       help="输出CSV报告文件（默认: model_comparison_results.csv）")
    
    args = parser.parse_args()
    
    print(f"🚀 开始自动提取模型评估结果...")
    print(f"基础目录: {args.base_dir}")
    print(f"模型列表: {args.models}")
    print(f"城市列表: {args.cities}")
    
    all_models_results = {}
    
    # 处理每个模型
    for model_name in args.models:
        print(f"\n🔍 处理模型: {model_name}")
        
        cities_data = {}
        
        # 收集三个城市的数据
        for city in args.cities:
            print(f"  正在查找 {city} 的数据...")
            result_file = find_result_files(args.base_dir, model_name, city)
            
            if result_file:
                judge_averages = extract_judge_averages(result_file)
                if judge_averages:
                    cities_data[city] = judge_averages
                    print(f"  ✅ {city} 数据提取成功")
                else:
                    print(f"  ❌ {city} 数据提取失败")
            else:
                print(f"  ❌ {city} 未找到结果文件")
        
        # 计算该模型的加权平均
        if cities_data:
            weighted_avg = calculate_weighted_averages(cities_data, model_name)
            if weighted_avg:
                all_models_results[model_name] = weighted_avg
                print(f"✅ {model_name} 处理完成")
            else:
                print(f"❌ {model_name} 计算失败")
        else:
            print(f"❌ {model_name} 没有有效数据")
    
    # 生成对比报告
    if all_models_results:
        generate_comparison_report(all_models_results)
        
        # 保存JSON结果到文件
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_models_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 JSON结果已保存到: {args.output}")
        
        # 生成CSV报告
        if args.csv:
            print(f"\n📊 正在生成CSV报告...")
            generate_csv_report(all_models_results, args.csv)
        
        print(f"\n✅ 所有模型处理完成！")
    else:
        print(f"\n❌ 没有成功处理任何模型")

if __name__ == "__main__":
    main()

