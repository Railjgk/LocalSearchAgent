#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测流程主程序 - 整合Web Search、RAG和LLM进行评测
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

# 导入各个Agent
from web_search_agent import WebSearchAgent
from rag_agent import RAGAgent
from llm_utils import get_friday_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """评测流程管道 - 整合Web Search、RAG和LLM"""
    
    def __init__(
        self,
        model_name: str = "deepseek-v31-meituan",
        model_rpm: int = None,
        use_web_search: bool = True,
        use_rag: bool = True,
        rag_top_k: int = 5,
        web_search_api_url: str = "https://api.example.com/v1/friday/api/search",
        web_search_token: str = "xxxxxxxxxxxxxxxxxxxx",
        rag_index_path: str = None,
        output_dir: str = "./evaluation_results"
    ):
        """
        初始化评测流程
        
        Args:
            model_name: LLM模型名称
            model_rpm: 模型RPM限制
            use_web_search: 是否使用Web搜索
            use_rag: 是否使用RAG
            rag_top_k: RAG返回结果数量
            web_search_api_url: Web搜索API URL
            web_search_token: Web搜索API令牌
            rag_index_path: RAG索引路径
            output_dir: 输出目录
        """
        self.model_name = model_name
        self.model_rpm = model_rpm
        self.use_web_search = use_web_search
        self.use_rag = use_rag
        self.rag_top_k = rag_top_k
        self.output_dir = output_dir
        
        # 创建 RAG 调用锁 - 确保 VLLM 模型串行访问
        # VLLM 的内部调度器在高并发时可能出现竞态条件
        self._rag_lock = threading.Lock()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化LLM客户端
        logger.info(f"🚀 初始化LLM客户端: {model_name}")
        self.llm_client = get_friday_client(model_name, model_rpm)
        
        # 初始化Web Search Agent（如果启用）
        self.web_search_agent = None
        if use_web_search:
            logger.info("🌐 初始化Web Search Agent")
            self.web_search_agent = WebSearchAgent(
                api_url=web_search_api_url,
                api_token=web_search_token
            )
        
        # 初始化RAG Agent（如果启用）
        self.rag_agent = None
        if use_rag:
            logger.info("📚 初始化RAG Agent")
            self.rag_agent = RAGAgent(
                index_path=rag_index_path,
                use_reranker=True
            )
        
        logger.info("✅ 评测流程初始化完成")
    
    def _process_single_question(
        self, 
        question: str, 
        question_id: int,
        system_prompt: str
    ) -> Dict[str, Any]:
        """
        处理单个问题
        
        Args:
            question: 问题文本
            question_id: 问题ID
            system_prompt: 系统提示词
            
        Returns:
            处理结果字典
        """
        start_time = time.time()
        result = {
            "question_id": question_id,
            "question": question,
            "web_search_context": "",
            "web_search_result": None,  # 完整的Web搜索结果
            "rag_context": "",
            "rag_result": None,  # 完整的RAG检索结果
            "llm_response": "",
            "processing_time": 0,
            "success": False,
            "error": None
        }
        
        try:
            # 1. 收集上下文
            contexts = []
            
            # 1.1 Web Search
            if self.use_web_search and self.web_search_agent:
                logger.info(f"[Q{question_id}] 🌐 执行Web搜索")
                web_result = self.web_search_agent.search(question)
                result["web_search_result"] = web_result  # 保存完整结果
                
                if web_result["success"]:
                    result["web_search_context"] = web_result["context"]
                    contexts.append(f"=== Web搜索结果 ===\n{web_result['context']}")
                    
                    # 打印详细信息
                    logger.info(f"[Q{question_id}] ✅ Web搜索成功")
                    if "result" in web_result:
                        raw_result = web_result["result"]
                        if isinstance(raw_result, dict):
                            if "baiduSearchResults" in raw_result:
                                num_results = len(raw_result.get("baiduSearchResults", []))
                                logger.info(f"[Q{question_id}]   📊 找到 {num_results} 条百度搜索结果")
                            elif "results" in raw_result:
                                num_results = len(raw_result.get("results", []))
                                logger.info(f"[Q{question_id}]   📊 找到 {num_results} 条搜索结果")
                    logger.info(f"[Q{question_id}]   📄 上下文预览:\n{web_result['context'][:300]}...")
                else:
                    logger.warning(f"[Q{question_id}] ⚠️ Web搜索失败: {web_result.get('error', 'Unknown')}")
            
            # 1.2 RAG检索
            if self.use_rag and self.rag_agent:
                logger.info(f"[Q{question_id}] 📚 执行RAG检索")
                
                # 使用锁确保 VLLM 模型串行访问
                # VLLM 的内部调度器在高并发时可能出现竞态条件（AssertionError）
                with self._rag_lock:
                    rag_result = self.rag_agent.search(question, top_k=self.rag_top_k)
                
                result["rag_result"] = rag_result  # 保存完整结果
                
                if rag_result["success"]:
                    result["rag_context"] = rag_result["context"]
                    contexts.append(f"=== RAG检索结果 ===\n{rag_result['context']}")
                    
                    # 打印详细信息
                    logger.info(f"[Q{question_id}] ✅ RAG检索成功")
                    total_results = rag_result.get("total_results", 0)
                    logger.info(f"[Q{question_id}]   📊 找到 {total_results} 个商户")
                    
                    # 打印前3个商户
                    results = rag_result.get("results", [])
                    for i, merchant in enumerate(results[:3], 1):
                        name = merchant.get("name", "未知")
                        score = merchant.get("score", 0)
                        logger.info(f"[Q{question_id}]     {i}. {name} (相似度: {score:.4f})")
                    
                    logger.info(f"[Q{question_id}]   📄 上下文预览:\n{rag_result['context'][:300]}...")
                else:
                    logger.warning(f"[Q{question_id}] ⚠️ RAG检索失败: {rag_result.get('error', 'Unknown')}")
            
            # 2. 构建完整的上下文
            full_context = "\n\n".join(contexts) if contexts else "未找到相关上下文信息。"
            
            # 3. 构建LLM消息
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"问题: {question}\n\n参考信息:\n{full_context}"
                }
            ]
            
            # 4. 调用LLM
            logger.info(f"[Q{question_id}] 🤖 调用LLM生成回答")
            llm_response, cost_time, token_info = self.llm_client.single_request(messages)
            
            result["llm_response"] = llm_response
            result["token_info"] = token_info
            result["llm_cost_time"] = cost_time
            result["success"] = True
            
            logger.info(f"[Q{question_id}] ✅ 处理成功")
            
        except Exception as e:
            logger.error(f"[Q{question_id}] ❌ 处理失败: {str(e)}")
            result["error"] = str(e)
        
        finally:
            result["processing_time"] = time.time() - start_time
        
        return result
    
    def load_dataset(self, file_path: str) -> List[str]:
        """
        加载评测数据集
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            问题列表
        """
        logger.info(f"📂 加载数据集: {file_path}")
        
        questions = []
        
        try:
            if file_path.lower().endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                
                entries = []
                if isinstance(raw, list):
                    entries = raw
                elif isinstance(raw, dict):
                    for key in ("data", "questions", "items", "records"):
                        if isinstance(raw.get(key), list):
                            entries = raw[key]
                            break
                
                for entry in entries:
                    if isinstance(entry, dict):
                        question = str(entry.get('query') or entry.get('question') or '').strip()
                    else:
                        question = str(entry).strip()
                    if question:
                        questions.append(question)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 按分隔符分割问题
                sections = content.split('-' * 100)  # 分隔符是100个减号
                
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                    
                    # 提取"问题："后的内容
                    lines = section.split('\n')
                    for line in lines:
                        if line.startswith('问题：') or line.startswith('问题:'):
                            question = (
                                line.replace('问题：', '').replace('问题:', '').strip()
                            )
                            if question:
                                questions.append(question)
                            break
            
            logger.info(f"✅ 成功加载 {len(questions)} 个问题")
            
        except Exception as e:
            logger.error(f"❌ 加载数据集失败: {str(e)}")
            raise
        
        return questions
    
    def run_evaluation(
        self,
        dataset_path: str,
        system_prompt: str = None,
        max_questions: int = None,
        parallel: bool = False,
        max_workers: int = 3
    ) -> pd.DataFrame:
        """
        运行评测
        
        Args:
            dataset_path: 数据集路径
            system_prompt: 系统提示词
            max_questions: 最多处理的问题数量（None表示全部）
            parallel: 是否并行处理
            max_workers: 并行工作线程数
            
        Returns:
            评测结果DataFrame
        """
        # 默认系统提示词
        if system_prompt is None:
            system_prompt = """你是一个智能助手，需要根据提供的参考信息回答用户的问题。

回答要求：
1. 基于提供的参考信息（Web搜索结果和RAG检索结果）回答问题
2. 如果参考信息中有相关内容，优先使用这些信息
3. 如果参考信息不足，可以结合你的知识回答
4. 回答要准确、简洁、有条理
5. 如果无法确定答案，请明确说明

请根据上述要求回答用户的问题。"""
        
        # 加载数据集
        questions = self.load_dataset(dataset_path)
        
        # 限制问题数量
        if max_questions:
            questions = questions[:max_questions]
            logger.info(f"⚠️ 限制处理前 {max_questions} 个问题")
        
        # 处理问题
        results = []
        
        if parallel and max_workers > 1:
            # 并行处理
            logger.info(f"🚀 开始并行评测 (工作线程数: {max_workers})")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                futures = {
                    executor.submit(
                        self._process_single_question, 
                        question, 
                        idx + 1,
                        system_prompt
                    ): idx 
                    for idx, question in enumerate(questions)
                }
                
                # 使用tqdm显示进度
                with tqdm(total=len(questions), desc="评测进度") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
        else:
            # 顺序处理
            logger.info(f"🚀 开始顺序评测")
            
            for idx, question in enumerate(tqdm(questions, desc="评测进度")):
                result = self._process_single_question(question, idx + 1, system_prompt)
                results.append(result)
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        
        output_file = os.path.join(
            self.output_dir, 
            f"{dataset_name}_results_{timestamp}.json"
        )
        
        # 保存为JSON（完整数据）
        df.to_json(output_file, orient='records', force_ascii=False, indent=2)
        logger.info(f"💾 评测结果已保存: {output_file}")
        
        # 保存详细的搜索结果到单独的文件
        search_results_file = output_file.replace('.json', '_search_details.json')
        self._save_search_details(results, search_results_file)
        logger.info(f"💾 搜索详情已保存: {search_results_file}")
        
        # 保存为CSV（可选）
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"💾 CSV结果已保存: {csv_file}")
        
        # 打印统计信息
        self._print_statistics(df)
        
        return df
    
    def _save_search_details(self, results: List[Dict[str, Any]], output_file: str):
        """
        保存详细的搜索结果到单独的JSON文件
        
        Args:
            results: 评测结果列表
            output_file: 输出文件路径
        """
        search_details = []
        
        for result in results:
            question_id = result.get("question_id")
            question = result.get("question")
            
            detail = {
                "question_id": question_id,
                "question": question,
                "web_search": None,
                "rag": None
            }
            
            # 提取Web搜索结果
            web_result = result.get("web_search_result")
            if web_result:
                web_detail = {
                    "success": web_result.get("success", False),
                    "context": web_result.get("context", ""),
                    "error": web_result.get("error", "")
                }
                
                # 提取原始结果和结果数量
                raw_result = web_result.get("result", {})
                if isinstance(raw_result, dict):
                    web_detail["raw_result"] = raw_result
                    if "baiduSearchResults" in raw_result:
                        web_detail["num_results"] = len(raw_result.get("baiduSearchResults", []))
                    elif "results" in raw_result:
                        web_detail["num_results"] = len(raw_result.get("results", []))
                
                detail["web_search"] = web_detail
            
            # 提取RAG检索结果
            rag_result = result.get("rag_result")
            if rag_result:
                rag_detail = {
                    "success": rag_result.get("success", False),
                    "total_results": rag_result.get("total_results", 0),
                    "context": rag_result.get("context", ""),
                    "merchants": rag_result.get("results", []),
                    "error": rag_result.get("error", "")
                }
                detail["rag"] = rag_detail
            
            search_details.append(detail)
        
        # 保存为格式化的JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(search_details, f, ensure_ascii=False, indent=2)
    
    def _print_statistics(self, df: pd.DataFrame):
        """打印评测统计信息"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 评测统计信息")
        logger.info("=" * 60)
        logger.info(f"总问题数: {len(df)}")
        logger.info(f"成功处理: {df['success'].sum()}")
        logger.info(f"处理失败: {(~df['success']).sum()}")
        logger.info(f"平均处理时间: {df['processing_time'].mean():.2f}秒")
        logger.info(f"总处理时间: {df['processing_time'].sum():.2f}秒")
        
        if 'token_info' in df.columns:
            total_input_tokens = sum([t.get('input_tokens', 0) for t in df['token_info'] if isinstance(t, dict)])
            total_output_tokens = sum([t.get('output_tokens', 0) for t in df['token_info'] if isinstance(t, dict)])
            logger.info(f"总输入Token: {total_input_tokens}")
            logger.info(f"总输出Token: {total_output_tokens}")
            logger.info(f"总Token: {total_input_tokens + total_output_tokens}")
        
        # 搜索结果统计
        if 'web_search_result' in df.columns or 'rag_result' in df.columns:
            logger.info("\n" + "=" * 60)
            logger.info("🔍 搜索结果统计")
            logger.info("=" * 60)
            
            # Web搜索统计
            if 'web_search_result' in df.columns:
                web_success = sum([1 for r in df['web_search_result'] if r and r.get('success')])
                logger.info(f"🌐 Web搜索成功: {web_success}/{len(df)}")
            
            # RAG检索统计
            if 'rag_result' in df.columns:
                rag_success = sum([1 for r in df['rag_result'] if r and r.get('success')])
                rag_total_merchants = sum([r.get('total_results', 0) for r in df['rag_result'] if r and r.get('success')])
                avg_merchants = rag_total_merchants / rag_success if rag_success > 0 else 0
                logger.info(f"📚 RAG检索成功: {rag_success}/{len(df)}")
                logger.info(f"📊 平均检索商户数: {avg_merchants:.2f}")
        
        logger.info("=" * 60 + "\n")
    
    def cleanup(self):
        """清理资源"""
        if self.rag_agent:
            self.rag_agent.cleanup()
        logger.info("✅ 资源已清理")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评测工具 - 整合Web Search、RAG和LLM")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="数据集文件路径"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-v31-meituan",
        help="LLM模型名称 (默认: deepseek-v31-meituan)"
    )
    
    parser.add_argument(
        "--rpm", 
        type=int, 
        default=None,
        help="模型RPM限制"
    )
    
    parser.add_argument(
        "--use-web-search", 
        action="store_true",
        help="启用Web搜索"
    )
    
    parser.add_argument(
        "--use-rag", 
        action="store_true",
        help="启用RAG检索"
    )
    
    parser.add_argument(
        "--rag-top-k", 
        type=int, 
        default=5,
        help="RAG返回结果数量 (默认: 5)"
    )
    
    parser.add_argument(
        "--rag-index-path", 
        type=str, 
        default=None,
        help="RAG索引路径（例如：/path/to/faiss_merchant_index_vllm_shanghai）"
    )
    
    parser.add_argument(
        "--max-questions", 
        type=int, 
        default=None,
        help="最多处理的问题数量"
    )
    
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="并行处理"
    )
    
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=3,
        help="并行工作线程数 (默认: 3)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./evaluation_results",
        help="输出目录 (默认: ./evaluation_results)"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建评测流程
        pipeline = EvaluationPipeline(
            model_name=args.model,
            model_rpm=args.rpm,
            use_web_search=args.use_web_search,
            use_rag=args.use_rag,
            rag_top_k=args.rag_top_k,
            rag_index_path=args.rag_index_path,
            output_dir=args.output_dir
        )
        
        # 运行评测
        pipeline.run_evaluation(
            dataset_path=args.dataset,
            max_questions=args.max_questions,
            parallel=args.parallel,
            max_workers=args.max_workers
        )
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ 用户中断评测")
    except Exception as e:
        logger.error(f"❌ 评测失败: {str(e)}")
        raise
    finally:
        if 'pipeline' in locals():
            pipeline.cleanup()


if __name__ == "__main__":
    main()