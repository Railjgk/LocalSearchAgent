#!/usr/bin/env python3
"""
三城市平均指标计算脚本
从汇总日志中提取各城市的统计数据，计算平均指标
"""

import argparse
import re
import sys
from collections import defaultdict


def parse_log_file(log_file_path):
    """解析汇总日志文件，提取统计数据"""
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按模型分割内容
    model_sections = re.split(r'^模型: ([^\n]+)$', content, flags=re.MULTILINE)[1:]  # 去掉第一个空元素
    
    results = {}
    
    for i in range(0, len(model_sections), 2):
        if i + 1 >= len(model_sections):
            break
            
        model_name = model_sections[i].strip()
        model_content = model_sections[i + 1]
        
        # 提取城市数据 - 修复正则表达式
        city_sections = re.split(r'^\s*([^:\-]+):\s*$', model_content, flags=re.MULTILINE)
        
        model_data = {}
        
        for j in range(1, len(city_sections), 2):
            if j + 1 >= len(city_sections):
                break
                
            city_name = city_sections[j].strip()
            
            # 跳过分隔线
            if city_name.startswith('-') or len(city_name) == 0:
                continue
                
            city_content = city_sections[j + 1]
            
            # 提取各项指标
            metrics = {}
            
            # 基础统计
            if match := re.search(r'总问题数[:：]\s*(\d+)', city_content):
                metrics['total_questions'] = int(match.group(1))
            if match := re.search(r'成功处理[:：]\s*(\d+)', city_content):
                metrics['success_count'] = int(match.group(1))
            if match := re.search(r'处理失败[:：]\s*(\d+)', city_content):
                metrics['failure_count'] = int(match.group(1))
            if match := re.search(r'平均处理时间[:：]\s*([\d.]+)秒', city_content):
                metrics['avg_processing_time'] = float(match.group(1))
            if match := re.search(r'总处理时间[:：]\s*([\d.]+)秒', city_content):
                metrics['total_processing_time'] = float(match.group(1))
            
            # 工具调用统计
            if match := re.search(r'总工具调用次数[:：]\s*(\d+)', city_content):
                metrics['total_tool_calls'] = int(match.group(1))
            if match := re.search(r'平均工具调用次数[:：]\s*([\d.]+)', city_content):
                metrics['avg_tool_calls'] = float(match.group(1))
            if match := re.search(r'Web搜索调用[:：]\s*(\d+)', city_content):
                metrics['web_search_calls'] = int(match.group(1))
            if match := re.search(r'RAG检索调用[:：]\s*(\d+)', city_content):
                metrics['rag_calls'] = int(match.group(1))
            
            # 成功率统计
            if match := re.search(r'Web搜索成功率[:：]\s*\d+/\d+\s*\(([\d.]+)%\)', city_content):
                metrics['web_search_success_rate'] = float(match.group(1))
            if match := re.search(r'RAG检索成功率[:：]\s*\d+/\d+\s*\(([\d.]+)%\)', city_content):
                metrics['rag_success_rate'] = float(match.group(1))
            if match := re.search(r'平均检索商户数[:：]\s*([\d.]+)', city_content):
                metrics['avg_merchants'] = float(match.group(1))
            if match := re.search(r'平均对话轮数[:：]\s*([\d.]+)', city_content):
                metrics['avg_conversation_rounds'] = float(match.group(1))
            
            # Judge评分
            if match := re.search(r'正确性\s*\(Correctness\)[:：]\s*([\d.]+)/1\.0', city_content):
                metrics['correctness'] = float(match.group(1))
            if match := re.search(r'完整性\s*\(Completeness\)[:：]\s*([\d.]+)/1\.0', city_content):
                metrics['completeness'] = float(match.group(1))
            if match := re.search(r'流畅度\s*\(Fluency\)[:：]\s*([\d.]+)/1\.0', city_content):
                metrics['fluency'] = float(match.group(1))
            if match := re.search(r'安全性\s*\(Safety\)[:：]\s*([\d.]+)/1\.0', city_content):
                metrics['safety'] = float(match.group(1))
            if match := re.search(r'总分\s*\(Total Score\)[:：]\s*([\d.]+)/([\d.]+)', city_content):
                metrics['total_score'] = float(match.group(1))
                metrics['max_score'] = float(match.group(2))
            
            if metrics:  # 只有当找到指标时才添加城市数据
                model_data[city_name] = metrics
        
        if model_data:  # 只有当找到城市数据时才添加模型数据
            results[model_name] = model_data
    
    return results


def calculate_city_averages(results):
    """计算三城市平均指标"""
    
    city_averages = {}
    
    for model_name, model_data in results.items():
        if len(model_data) == 0:
            continue
            
        # 收集所有城市的指标
        all_metrics = defaultdict(list)
        
        for city_name, metrics in model_data.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[metric_name].append(value)
        
        # 计算平均值
        averages = {}
        for metric_name, values in all_metrics.items():
            if values:
                averages[metric_name] = sum(values) / len(values)
        
        city_averages[model_name] = averages
    
    return city_averages


def print_city_averages(city_averages, output_file):
    """输出三城市平均指标"""
    
    output_lines = []
    output_lines.append("\n" + "=" * 80)
    output_lines.append("🌍 三城市平均指标统计")
    output_lines.append("=" * 80)
    
    for model_name, averages in city_averages.items():
        output_lines.append(f"\n📊 模型: {model_name}")
        output_lines.append("-" * 60)
        
        # 基础统计
        if 'total_questions' in averages:
            output_lines.append(f"平均总问题数: {averages['total_questions']:.1f}")
        if 'success_count' in averages:
            output_lines.append(f"平均成功处理: {averages['success_count']:.1f}")
        if 'failure_count' in averages:
            output_lines.append(f"平均处理失败: {averages['failure_count']:.1f}")
        if 'avg_processing_time' in averages:
            output_lines.append(f"平均处理时间: {averages['avg_processing_time']:.2f}秒")
        if 'total_processing_time' in averages:
            output_lines.append(f"平均总处理时间: {averages['total_processing_time']:.2f}秒")
        
        # 工具调用统计
        if 'total_tool_calls' in averages:
            output_lines.append(f"平均总工具调用次数: {averages['total_tool_calls']:.1f}")
        if 'avg_tool_calls' in averages:
            output_lines.append(f"平均工具调用次数: {averages['avg_tool_calls']:.2f}")
        if 'web_search_calls' in averages:
            output_lines.append(f"平均Web搜索调用: {averages['web_search_calls']:.1f}")
        if 'rag_calls' in averages:
            output_lines.append(f"平均RAG检索调用: {averages['rag_calls']:.1f}")
        
        # 成功率统计
        if 'web_search_success_rate' in averages:
            output_lines.append(f"平均Web搜索成功率: {averages['web_search_success_rate']:.1f}%")
        if 'rag_success_rate' in averages:
            output_lines.append(f"平均RAG检索成功率: {averages['rag_success_rate']:.1f}%")
        if 'avg_merchants' in averages:
            output_lines.append(f"平均检索商户数: {averages['avg_merchants']:.2f}")
        if 'avg_conversation_rounds' in averages:
            output_lines.append(f"平均对话轮数: {averages['avg_conversation_rounds']:.2f}")
        
        # Judge评分
        if 'correctness' in averages:
            output_lines.append(f"平均正确性: {averages['correctness']:.3f}/1.0")
        if 'completeness' in averages:
            output_lines.append(f"平均完整性: {averages['completeness']:.3f}/1.0")
        if 'fluency' in averages:
            output_lines.append(f"平均流畅度: {averages['fluency']:.3f}/1.0")
        if 'safety' in averages:
            output_lines.append(f"平均安全性: {averages['safety']:.3f}/1.0")
        if 'total_score' in averages and 'max_score' in averages:
            output_lines.append(f"平均总分: {averages['total_score']:.3f}/{averages['max_score']:.1f}")
    
    output_lines.append("\n" + "=" * 80)
    
    # 输出到控制台
    for line in output_lines:
        print(line)
    
    # 追加到文件
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="计算三城市平均指标")
    parser.add_argument(
        "--summary-log",
        type=str,
        required=True,
        help="汇总日志文件路径"
    )
    
    args = parser.parse_args()
    
    try:
        # 解析日志文件
        results = parse_log_file(args.summary_log)
        
        if not results:
            print("❌ 未找到有效的评测数据")
            sys.exit(1)
        
        # 计算平均指标
        city_averages = calculate_city_averages(results)
        
        # 输出结果
        print_city_averages(city_averages, args.summary_log)
        
        print(f"\n✅ 三城市平均指标已计算完成并追加到: {args.summary_log}")
        
    except Exception as e:
        print(f"❌ 计算平均指标时出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
