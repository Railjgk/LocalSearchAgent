#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用示例 - 演示如何在代码中使用评测工具包
"""

import os
from evaluation_pipeline import EvaluationPipeline
from system_prompts import get_prompt

def example_1_basic_usage():
    """示例1: 基础使用 - 单数据集评测"""
    print("\n" + "=" * 60)
    print("示例1: 基础使用 - 单数据集评测")
    print("=" * 60 + "\n")
    
    # 创建评测流程
    pipeline = EvaluationPipeline(
        model_name="deepseek-r1-friday",
        use_web_search=True,
        use_rag=True,
        rag_top_k=5,
        output_dir="./example_results"
    )
    
    # 运行评测
    df = pipeline.run_evaluation(
        dataset_path="/Users/hehang03/code/tuansou/ai-search-benchmark/bj_query_1.txt",
        max_questions=5,  # 只处理前5个问题
        parallel=False
    )
    
    # 查看结果
    print("\n评测完成！")
    print(f"处理问题数: {len(df)}")
    print(f"成功率: {df['success'].sum() / len(df) * 100:.1f}%")
    
    # 清理资源
    pipeline.cleanup()

def example_2_rag_only():
    """示例2: 仅使用RAG"""
    print("\n" + "=" * 60)
    print("示例2: 仅使用RAG - 商户推荐场景")
    print("=" * 60 + "\n")
    
    # 创建评测流程，只启用RAG
    pipeline = EvaluationPipeline(
        model_name="deepseek-r1-friday",
        use_web_search=False,  # 不使用Web搜索
        use_rag=True,
        rag_top_k=10,
        output_dir="./example_results"
    )
    
    # 使用商户推荐的系统提示词
    merchant_prompt = get_prompt("merchant_recommendation")
    
    # 运行评测
    df = pipeline.run_evaluation(
        dataset_path="/Users/hehang03/code/tuansou/ai-search-benchmark/bj_query_1.txt",
        system_prompt=merchant_prompt,
        max_questions=3,
        parallel=False
    )
    
    print("\n评测完成！")
    
    # 清理资源
    pipeline.cleanup()

def example_3_web_search_only():
    """示例3: 仅使用Web搜索"""
    print("\n" + "=" * 60)
    print("示例3: 仅使用Web搜索 - 信息查询场景")
    print("=" * 60 + "\n")
    
    # 创建评测流程，只启用Web搜索
    pipeline = EvaluationPipeline(
        model_name="deepseek-r1-friday",
        use_web_search=True,
        use_rag=False,  # 不使用RAG
        output_dir="./example_results"
    )
    
    # 使用信息查询的系统提示词
    info_prompt = get_prompt("information_query")
    
    # 运行评测
    df = pipeline.run_evaluation(
        dataset_path="/Users/hehang03/code/tuansou/ai-search-benchmark/sh_query_1.txt",
        system_prompt=info_prompt,
        max_questions=3,
        parallel=False
    )
    
    print("\n评测完成！")
    
    # 清理资源
    pipeline.cleanup()

def example_4_parallel_processing():
    """示例4: 并行处理"""
    print("\n" + "=" * 60)
    print("示例4: 并行处理 - 提高评测速度")
    print("=" * 60 + "\n")
    
    # 创建评测流程
    pipeline = EvaluationPipeline(
        model_name="deepseek-r1-friday",
        use_web_search=True,
        use_rag=True,
        output_dir="./example_results"
    )
    
    # 运行并行评测
    df = pipeline.run_evaluation(
        dataset_path="/Users/hehang03/code/tuansou/ai-search-benchmark/gz_query_1.txt",
        max_questions=10,
        parallel=True,  # 启用并行
        max_workers=3   # 使用3个工作线程
    )
    
    print("\n评测完成！")
    
    # 清理资源
    pipeline.cleanup()

def example_5_custom_prompt():
    """示例5: 自定义提示词"""
    print("\n" + "=" * 60)
    print("示例5: 自定义提示词")
    print("=" * 60 + "\n")
    
    # 自定义系统提示词
    custom_prompt = """你是一个专业的地方向导，熟悉各个城市的商户和景点。

你的任务：
1. 根据用户的需求推荐最合适的商户
2. 考虑位置、类型、评价等多个因素
3. 给出详细的理由和建议
4. 如果有多个选择，按优先级排序

请用友好、专业的语气回答。"""
    
    # 创建评测流程
    pipeline = EvaluationPipeline(
        model_name="deepseek-r1-friday",
        use_web_search=True,
        use_rag=True,
        output_dir="./example_results"
    )
    
    # 使用自定义提示词运行评测
    df = pipeline.run_evaluation(
        dataset_path="/Users/hehang03/code/tuansou/ai-search-benchmark/bj_query_1.txt",
        system_prompt=custom_prompt,
        max_questions=3
    )
    
    print("\n评测完成！")
    
    # 清理资源
    pipeline.cleanup()

def example_6_analyze_results():
    """示例6: 分析评测结果"""
    print("\n" + "=" * 60)
    print("示例6: 分析评测结果")
    print("=" * 60 + "\n")
    
    import pandas as pd
    import json
    
    # 假设已经有评测结果文件
    results_dir = "./example_results"
    
    # 查找最新的结果文件
    import glob
    result_files = glob.glob(f"{results_dir}/*_results_*.json")
    
    if not result_files:
        print("没有找到评测结果文件，请先运行评测。")
        return
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"分析文件: {os.path.basename(latest_file)}")
    
    # 读取结果
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    # 统计分析
    print(f"\n📊 统计信息:")
    print(f"  总问题数: {len(df)}")
    print(f"  成功处理: {df['success'].sum()}")
    print(f"  失败数: {(~df['success']).sum()}")
    print(f"  成功率: {df['success'].sum() / len(df) * 100:.1f}%")
    print(f"  平均处理时间: {df['processing_time'].mean():.2f}秒")
    print(f"  最快: {df['processing_time'].min():.2f}秒")
    print(f"  最慢: {df['processing_time'].max():.2f}秒")
    
    # Token统计（如果有）
    if 'token_info' in df.columns:
        total_input = sum([t.get('input_tokens', 0) for t in df['token_info'] if isinstance(t, dict)])
        total_output = sum([t.get('output_tokens', 0) for t in df['token_info'] if isinstance(t, dict)])
        print(f"\n💰 Token使用:")
        print(f"  输入Token: {total_input:,}")
        print(f"  输出Token: {total_output:,}")
        print(f"  总Token: {total_input + total_output:,}")
    
    # 查看失败的问题（如果有）
    failed = df[~df['success']]
    if len(failed) > 0:
        print(f"\n❌ 失败的问题:")
        for idx, row in failed.iterrows():
            print(f"  - Q{row['question_id']}: {row['question'][:50]}...")
            print(f"    错误: {row.get('error', 'Unknown')}")

def main():
    """主函数 - 运行所有示例"""
    print("\n" + "=" * 60)
    print("评测工具包 - 使用示例")
    print("=" * 60)
    
    examples = [
        ("基础使用", example_1_basic_usage),
        ("仅使用RAG", example_2_rag_only),
        ("仅使用Web搜索", example_3_web_search_only),
        ("并行处理", example_4_parallel_processing),
        ("自定义提示词", example_5_custom_prompt),
        ("分析结果", example_6_analyze_results),
    ]
    
    print("\n可用的示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. 运行所有示例")
    print()
    
    choice = input("请选择要运行的示例 [1-6, 0=全部]: ").strip()
    
    if choice == "0":
        for name, func in examples:
            try:
                func()
            except KeyboardInterrupt:
                print("\n\n⚠️ 用户中断")
                break
            except Exception as e:
                print(f"\n❌ 示例 '{name}' 运行失败: {str(e)}")
                import traceback
                traceback.print_exc()
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        idx = int(choice) - 1
        name, func = examples[idx]
        try:
            func()
        except Exception as e:
            print(f"\n❌ 示例 '{name}' 运行失败: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ 无效的选择")

if __name__ == "__main__":
    main()


