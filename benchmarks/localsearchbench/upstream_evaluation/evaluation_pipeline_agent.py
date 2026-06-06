#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent模式评测流程 - LLM自主决定何时使用工具
基于ReAct (Reasoning + Acting) 模式
"""

import os
import re
import sys
import json
import time
import logging
import argparse
import threading
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import pandas as pd
from tqdm import tqdm

# 导入各个Agent
from web_search_agent import WebSearchAgent
from rag_agent import RAGAgent
from llm_utils import get_friday_client
from system_prompts import get_prompt
from llm_judge import LLMJudge

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AgentEvaluationPipeline:
    """Agent模式评测流程 - LLM自主使用工具"""
    
    def __init__(
        self,
        model_name: str = "deepseek-v31-meituan",
        model_rpm: int = None,
        web_search_api_url: str = "https://api.example.com/v1/friday/api/search",
        web_search_token: str = "xxxxxxxxxxxxxxxxxxxx",
        web_search_tokens: List[str] = None,
        use_token_manager: bool = False,
        rag_index_path: str = None,
        output_dir: str = "./evaluation_results",
        max_tool_rounds: int = 10,  # 最大工具调用轮数
        use_judge: bool = False,  # 是否使用LLM Judge评分
        judge_model: str = "deepseek-chat",  # Judge使用的模型
        judge_api_url: str = None,  # Judge API地址（如果为None则使用主模型的API）
        judge_api_key: str = None  # Judge API密钥（如果为None则使用主模型的密钥）
    ):
        """
        初始化Agent评测流程
        
        Args:
            model_name: LLM模型名称
            model_rpm: 模型RPM限制
            web_search_api_url: Web搜索API URL
            web_search_token: Web搜索API令牌（单token模式）
            web_search_tokens: Web搜索API令牌列表（多token模式）
            use_token_manager: 是否使用token管理器
            rag_index_path: RAG索引路径
            output_dir: 输出目录
            max_tool_rounds: 最大工具调用轮数（防止无限循环）
            use_judge: 是否使用LLM Judge进行评分
            judge_model: Judge使用的模型名称
            judge_api_url: Judge API地址
            judge_api_key: Judge API密钥
        """
        resolved_model_name = (model_name or "").strip()
        if not resolved_model_name:
            raise ValueError("model_name 不能为空，请通过 --model 提供有效的模型名称")
        self.model_name = resolved_model_name
        self.model_rpm = model_rpm
        self.output_dir = output_dir
        self.max_tool_rounds = max_tool_rounds
        self.use_token_manager = use_token_manager
        self.use_judge = use_judge
        self.hop_similarity_threshold = 0.72  # 最低相似度阈值，判定Hop是否匹配
        
        # 创建 RAG 调用锁 - 确保 VLLM 模型串行访问
        # VLLM 的内部调度器在高并发时可能出现竞态条件
        self._rag_lock = threading.Lock()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化LLM客户端
        logger.info(f"🚀 初始化LLM客户端: {self.model_name}")
        self.llm_client = get_friday_client(
            self.model_name, 
            model_rpm,
            use_api_key_manager=use_token_manager
        )
        
        # 初始化Web Search Agent
        logger.info("🌐 初始化Web Search Agent")
        self.web_search_agent = WebSearchAgent(
            api_url=web_search_api_url,
            api_token=web_search_token,
            api_tokens=web_search_tokens,
            use_token_manager=use_token_manager
        )
        
        # 初始化RAG Agent
        logger.info("📚 初始化RAG Agent")
        self.rag_agent = RAGAgent(
            index_path=rag_index_path,
            use_reranker=True,
            gpu_memory_utilization=0.65  # 降低到65%，避免CUDA内存分配失败
        )
        
        # 初始化LLM Judge（如果启用）
        self.judge = None
        if use_judge:
            logger.info("🎯 初始化LLM Judge")
            # 如果没有指定Judge的API，使用主模型的API配置
            if judge_api_url is None:
                from llm_utils import ConfigManager
                config_mgr = ConfigManager()
                judge_api_url = config_mgr.get_base_url()
            
            # 支持多个API key进行负载均衡
            judge_api_keys = None
            if judge_api_key is None:
                from llm_utils import ConfigManager
                config_mgr = ConfigManager()
                # 获取所有API keys用于负载均衡
                judge_api_keys = config_mgr.get_api_keys()
                if not judge_api_keys:
                    judge_api_keys = ["default_key"]
                logger.info(f"🔑 从配置中获取到 {len(judge_api_keys)} 个Judge API Keys")
            else:
                # 如果指定了单个key，也转为列表格式
                judge_api_keys = [judge_api_key]
            
            self.judge = LLMJudge(
                api_url=judge_api_url,
                api_keys=judge_api_keys,
                model_name=judge_model
            )
            logger.info(f"✅ LLM Judge初始化完成 (模型: {judge_model})")
        
        logger.info("✅ Agent评测流程初始化完成")
    
    def _parse_tool_calls(self, text: str) -> List[Tuple[str, str]]:
        """
        解析LLM响应中的工具调用标签
        
        Args:
            text: LLM响应文本
            
        Returns:
            工具调用列表 [(tool_type, query), ...]
            tool_type: "web_search" 或 "rag"
        """
        tool_calls = []
        
        # 解析 <web_search>query</web_search>
        web_search_pattern = r'<web_search>(.*?)</web_search>'
        for match in re.finditer(web_search_pattern, text, re.DOTALL):
            query = match.group(1).strip()
            if query:
                tool_calls.append(("web_search", query))
        
        # 解析 <rag>query</rag>
        rag_pattern = r'<rag>(.*?)</rag>'
        for match in re.finditer(rag_pattern, text, re.DOTALL):
            query = match.group(1).strip()
            if query:
                tool_calls.append(("rag", query))
        
        return tool_calls
    
    def _execute_tool(self, tool_type: str, query: str, question_id: int = None) -> Dict[str, Any]:
        """
        执行工具调用
        
        Args:
            tool_type: 工具类型 ("web_search" 或 "rag")
            query: 查询内容
            question_id: 问题ID（用于日志记录）
            
        Returns:
            工具执行结果
        """
        prefix = f"[Q{question_id}] " if question_id else ""
        
        if tool_type == "web_search":
            logger.info(f"{prefix}  🌐 执行Web搜索: {query[:50]}...")
            result = self.web_search_agent.search(query)
            
            # 打印详细的Web搜索结果
            if result.get("success"):
                logger.info(f"{prefix}  ✅ Web搜索成功")
                context = result.get("context", "")
                logger.info(f"{prefix}  📄 搜索结果预览:\n{context[:500]}...")
                
                # 如果有原始结果，打印摘要
                if "result" in result:
                    raw_result = result["result"]
                    if isinstance(raw_result, dict):
                        if "baiduSearchResults" in raw_result:
                            num_results = len(raw_result.get("baiduSearchResults", []))
                            logger.info(f"{prefix}  📊 找到 {num_results} 条百度搜索结果")
                        elif "results" in raw_result:
                            num_results = len(raw_result.get("results", []))
                            logger.info(f"{prefix}  📊 找到 {num_results} 条搜索结果")
            else:
                logger.error(f"{prefix}  ❌ Web搜索失败: {result.get('error', 'Unknown')}")
            
            return result
        
        elif tool_type == "rag":
            logger.info(f"{prefix}  📚 执行RAG检索: {query[:50]}...")
            
            # 使用锁确保 VLLM 模型串行访问
            # VLLM 的内部调度器在高并发时可能出现竞态条件（AssertionError）
            with self._rag_lock:
                # 从候选池100个中rerank选出20个
                try:
                    result = self.rag_agent.search(
                        query,
                        rerank_top_k=20,
                        candidate_multiplier=5.0
                    )
                except TypeError as e:
                    if "candidate_multiplier" in str(e):
                        logger.warning(
                            f"{prefix}  ⚠️ RAGAgent.search 不支持 candidate_multiplier 参数，回退到默认行为: {e}"
                        )
                        result = self.rag_agent.search(query, rerank_top_k=20)
                    else:
                        raise
            
            # 打印详细的RAG检索结果
            if result.get("success"):
                logger.info(f"{prefix}  ✅ RAG检索成功")
                total_results = result.get("total_results", 0)
                logger.info(f"{prefix}  📊 找到 {total_results} 个商户")
                
                # 打印每个商户的基本信息
                results = result.get("results", [])
                for i, merchant in enumerate(results[:3], 1):  # 只打印前3个
                    name = merchant.get("name", "未知")
                    score = merchant.get("score", 0)
                    logger.info(f"{prefix}    {i}. {name} (相似度: {score:.4f})")
                
                # 打印上下文预览
                context = result.get("context", "")
                logger.info(f"{prefix}  📄 上下文预览:\n{context[:500]}...")
            else:
                logger.error(f"{prefix}  ❌ RAG检索失败: {result.get('error', 'Unknown')}")
            
            return result
        
        else:
            return {"success": False, "error": f"未知工具类型: {tool_type}"}
    
    def _format_tool_results(self, tool_calls: List[Tuple[str, str]], 
                            tool_results: List[Dict[str, Any]]) -> str:
        """
        格式化工具结果为文本
        
        Args:
            tool_calls: 工具调用列表
            tool_results: 工具结果列表
            
        Returns:
            格式化的结果文本
        """
        formatted_results = []
        
        for (tool_type, query), result in zip(tool_calls, tool_results):
            if tool_type == "web_search":
                if result.get("success"):
                    formatted_results.append(
                        f"【Web搜索结果】\n"
                        f"查询: {query}\n"
                        f"结果:\n{result.get('context', '无结果')}\n"
                    )
                else:
                    formatted_results.append(
                        f"【Web搜索失败】\n"
                        f"查询: {query}\n"
                        f"错误: {result.get('error', '未知错误')}\n"
                    )
            
            elif tool_type == "rag":
                if result.get("success"):
                    formatted_results.append(
                        f"【RAG检索结果】\n"
                        f"查询: {query}\n"
                        f"结果:\n{result.get('context', '无结果')}\n"
                    )
                else:
                    formatted_results.append(
                        f"【RAG检索失败】\n"
                        f"查询: {query}\n"
                        f"错误: {result.get('error', '未知错误')}\n"
                    )
        
        return "\n".join(formatted_results)
    
    def _process_single_question(
        self, 
        question: str, 
        question_id: int,
        system_prompt: str,
        ground_truth: str = None,
        search_path: str = None,
        reference_answer: str = None,
        difficulty: Optional[str] = None,
        expected_hops: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        使用Agent模式处理单个问题
        
        Args:
            question: 问题文本
            question_id: 问题ID
            system_prompt: 系统提示词
            ground_truth: 简短参考答案（用于Judge评分）
            search_path: 多跳搜索路径（用于推理评估）
            reference_answer: 完整参考答案（用于全面评估）
            
        Returns:
            处理结果字典
        """
        start_time = time.time()
        result = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,  # 简短参考答案
            "search_path": search_path,  # 多跳搜索路径
            "reference_answer": reference_answer,  # 完整参考答案
            "difficulty": difficulty,
            "conversation_history": [],  # 记录完整对话历史
            "tool_calls": [],  # 记录所有工具调用
            "final_response": "",
            "processing_time": 0,
            "success": False,
            "error": None,
            "judge_scores": None,  # LLM Judge评分结果
            "expected_hops": expected_hops,
            "trajectory_evaluation": None
        }
        
        try:
            # 初始化对话
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            # 多轮工具调用
            for round_idx in range(self.max_tool_rounds):
                logger.info(f"[Q{question_id}] 🔄 第{round_idx + 1}轮对话")
                
                # 调用LLM
                llm_response, cost_time, token_info = self.llm_client.single_request(messages)
                
                # 记录对话
                result["conversation_history"].append({
                    "round": round_idx + 1,
                    "llm_response": llm_response,
                    "token_info": token_info,
                    "cost_time": cost_time
                })
                
                # 解析工具调用
                tool_calls = self._parse_tool_calls(llm_response)
                
                if not tool_calls:
                    # 没有工具调用，说明LLM给出了最终答案
                    logger.info(f"[Q{question_id}] ✅ LLM给出最终答案（无工具调用）")
                    result["final_response"] = llm_response
                    result["success"] = True
                    break
                
                logger.info(f"[Q{question_id}] 🛠️  检测到 {len(tool_calls)} 个工具调用")
                
                # 执行工具调用
                tool_results = []
                for tool_type, query in tool_calls:
                    tool_result = self._execute_tool(tool_type, query, question_id)
                    tool_results.append(tool_result)
                    
                    # 记录工具调用（包含完整搜索结果）
                    result["tool_calls"].append({
                        "round": round_idx + 1,
                        "tool_type": tool_type,
                        "query": query,
                        "result": tool_result,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # 构建工具结果反馈
                tool_feedback = self._format_tool_results(tool_calls, tool_results)
                
                # 将工具结果添加到对话中
                messages.append({"role": "assistant", "content": llm_response})
                
                # 如果是最后一轮，强制要求LLM给出最终答案
                if round_idx == self.max_tool_rounds - 1:
                    logger.info(f"[Q{question_id}] ⚠️ 达到最大轮数，强制要求最终答案")
                    messages.append({
                        "role": "user", 
                        "content": f"工具执行结果：\n\n{tool_feedback}\n\n这是最后一轮，请不要再调用任何工具，直接基于以上结果给出最终答案。"
                    })
                    
                    # 立即获取最终答案
                    final_response, cost_time, token_info = self.llm_client.single_request(messages)
                    result["final_response"] = final_response
                    result["conversation_history"].append({
                        "round": round_idx + 2,  # 工具调用是round_idx+1，这次是+2
                        "role": "assistant",
                        "llm_response": final_response,
                        "token_info": token_info,
                        "cost_time": cost_time
                    })
                    result["success"] = True
                    break
                else:
                    # 非最后一轮，正常添加工具结果反馈
                    messages.append({
                        "role": "user", 
                        "content": f"工具执行结果：\n\n{tool_feedback}\n\n请基于这些结果回答原问题。"
                    })
            
            # 如果循环结束还没有最终答案，使用最后一次响应
            if not result["success"]:
                logger.warning(f"[Q{question_id}] ⚠️ 达到最大轮数，使用最后一次响应")
                final_response, cost_time, token_info = self.llm_client.single_request(messages)
                result["final_response"] = final_response
                result["conversation_history"].append({
                    "round": self.max_tool_rounds + 1,
                    "llm_response": final_response,
                    "token_info": token_info,
                    "cost_time": cost_time
                })
                result["success"] = True
            
            logger.info(f"[Q{question_id}] ✅ 处理成功")
            
            # 如果启用了Judge且有参考答案，进行评分
            if self.use_judge and self.judge and ground_truth:
                logger.info(f"[Q{question_id}] 🎯 开始LLM Judge评分...")
                try:
                    # 提取RAG上下文用于幻觉检测
                    rag_context = self._extract_rag_context(result["tool_calls"])
                    
                    judge_result = self.judge.evaluate_all(
                        query=question,
                        model_output=result["final_response"],
                        ground_truth=ground_truth,
                        conversation_history=result["conversation_history"],
                        tool_calls=result["tool_calls"],
                        rag_context=rag_context,
                        enable_hallucination=True
                    )
                    result["judge_scores"] = judge_result
                    
                    # 根据是否有幻觉评分显示不同的日志
                    if "hallucination" in judge_result:
                        logger.info(f"[Q{question_id}] ✅ Judge评分完成: {judge_result['total_score']}/4, 幻觉: {judge_result['hallucination']['score']}/10")
                    else:
                        logger.info(f"[Q{question_id}] ✅ Judge评分完成: {judge_result['total_score']}/4")
                except Exception as judge_error:
                    logger.error(f"[Q{question_id}] ❌ Judge评分失败: {str(judge_error)}")
                    result["judge_scores"] = {"error": str(judge_error)}
            
        except Exception as e:
            logger.error(f"[Q{question_id}] ❌ 处理失败: {str(e)}")
            result["error"] = str(e)
        
        finally:
            result["processing_time"] = time.time() - start_time
        
        if expected_hops:
            result["trajectory_evaluation"] = self._evaluate_trajectory_accuracy(
                expected_hops, 
                result.get("tool_calls", [])
            )
        
        return result
    
    def load_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载评测数据集
        
        Args:
            file_path: 数据集文件路径
            
        Returns:
            数据列表，每个元素包含:
            {
                "question": "...",           # 问题
                "search_path": "...",        # 多跳搜索路径（可选）
                "reference_answer": "...",   # 完整参考答案（可选）
                "ground_truth": "...",       # boxed中的简短答案（用于评分）
                "difficulty": "L3/L4/..."    # 难度标签（可选）
            }
        """
        logger.info(f"📂 加载数据集: {file_path}")
        
        dataset = []
        
        try:
            file_lower = file_path.lower()
            if file_lower.endswith('.json'):
                dataset = self._load_json_dataset(file_path)
            elif file_lower.endswith('.csv'):
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    question = str(row.get('query', row.get('question', ''))).strip()
                    if not question:
                        continue
                    ground_truth = None
                    if 'ground_truth' in row and pd.notna(row['ground_truth']):
                        ground_truth = self._extract_ground_truth(str(row['ground_truth']))
                    dataset.append({
                        "question": question,
                        "ground_truth": ground_truth,
                        "search_path": None,
                        "reference_answer": None,
                        "difficulty": row.get('difficulty')
                    })
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                sections = re.split(r'-{80,}', content)
                
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                    
                    question = None
                    question_match = re.search(r'问题[：:](.*?)(?=多跳搜索路径[：:]|多跳推理路径[：:]|$)', section, re.DOTALL)
                    if question_match:
                        question = question_match.group(1).strip().replace('\n', ' ').replace('\r', '')
                    else:
                        no_label_match = re.search(r'^(.*?)(?=多跳搜索路径[：:]|多跳推理路径[：:]|$)', section, re.DOTALL)
                        if no_label_match:
                            question = no_label_match.group(1).strip().replace('\n', ' ').replace('\r', '')
                    
                    search_path = None
                    search_path_match = re.search(
                        r'(?:多跳搜索路径|多跳推理路径)[：:](.*?)(?=参考答案[：:]|最终参考答案[：:]|$)', 
                        section, 
                        re.DOTALL
                    )
                    if search_path_match:
                        search_path = search_path_match.group(1).strip()
                    
                    reference_answer = None
                    ref_answer_match = re.search(
                        r'(?:参考答案|最终参考答案)[：:](.*?)$',
                        section,
                        re.DOTALL
                    )
                    if ref_answer_match:
                        reference_answer = ref_answer_match.group(1).strip()
                    
                    ground_truth = self._extract_ground_truth(section if reference_answer is None else reference_answer)
                    
                    if question:
                        dataset.append({
                            "question": question,
                            "search_path": search_path,
                            "reference_answer": reference_answer,
                            "ground_truth": ground_truth,
                            "difficulty": None
                        })
            
            logger.info(f"✅ 成功加载 {len(dataset)} 个问题")
            if dataset:
                first = dataset[0]
                if first.get("ground_truth"):
                    logger.info(f"  📋 数据集包含参考答案（可用于Judge评分）")
                if first.get("search_path"):
                    logger.info(f"  🔍 数据集包含多跳搜索路径（可用于推理评估）")
                logger.info(f"  📄 第一个样例:")
                logger.info(f"     问题: {first['question'][:100]}...")
                if first.get("search_path"):
                    search_preview = first['search_path'][:150]
                    logger.info(f"     多跳路径: {search_preview}...")
                logger.info(f"     参考答案: {first.get('ground_truth', 'N/A')}")
                if first.get("difficulty"):
                    logger.info(f"     难度: {first.get('difficulty')}")
            
        except Exception as e:
            logger.error(f"❌ 加载数据集失败: {str(e)}")
            raise
        
        return dataset

    def _load_json_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """加载JSON格式的数据集（支持新版data_cons结构）"""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        entries = self._normalize_json_entries(raw)
        dataset: List[Dict[str, Any]] = []
        
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            question = str(
                self._get_entry_value(
                    entry,
                    "query",
                    "question",
                    "title"
                ) or ""
            ).strip()
            if not question:
                continue
            
            reference_answer = self._get_entry_value(
                entry,
                "reference_answer",
                "answer",
                "response"
            )
            if isinstance(reference_answer, str):
                reference_answer = reference_answer.strip()
            else:
                reference_answer = None
            
            ground_truth = (
                self._get_entry_value(entry, "box", "ground_truth", "final_answer")
                or reference_answer
            )
            ground_truth = self._extract_ground_truth(ground_truth)
            
            # 标准化Hop计划
            hop_queries = self._sanitize_hop_queries(
                self._get_entry_value(entry, "hop_used_queries", "%", "hop_queries")
            )
            entry_for_path = dict(entry)
            entry_for_path["hop_used_queries"] = hop_queries
            
            search_path_text = self._get_entry_value(
                entry,
                "search_path",
                "multi-hop search path",
                "planning_path",
                "trajectory"
            )
            if isinstance(search_path_text, str) and search_path_text.strip():
                entry_for_path["search_path"] = search_path_text.strip()
            
            dataset.append({
                "question": question,
                "search_path": self._build_search_path_from_entry(entry_for_path),
                "reference_answer": reference_answer,
                "ground_truth": ground_truth,
                "difficulty": entry.get("difficulty"),
                "hop_used_queries": hop_queries
            })
        
        return dataset

    def _normalize_json_entries(self, raw: Any) -> List[Any]:
        """规范化JSON数据结构，兼容不同包裹格式"""
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ("data", "questions", "items", "records"):
                value = raw.get(key)
                if isinstance(value, list):
                    return value
        return []

    def _build_search_path_from_entry(self, entry: Dict[str, Any]) -> Optional[str]:
        """根据hop数据构建多跳搜索路径文本"""
        hop_queries = self._sanitize_hop_queries(
            self._get_entry_value(entry, "hop_used_queries", "%", "hop_queries")
        )
        hop_answers = self._get_entry_value(entry, "hop_reference_answers")
        
        if not hop_queries:
            raw_path = self._get_entry_value(
                entry,
                "search_path",
                "multi-hop search path",
                "planning_path",
                "trajectory"
            )
            return raw_path.strip() if isinstance(raw_path, str) and raw_path.strip() else None
        
        path_blocks = []
        for idx, hop_query in enumerate(hop_queries):
            if not hop_query:
                continue
            block = [f"Hop {idx + 1}: {hop_query}"]
            if isinstance(hop_answers, list) and idx < len(hop_answers):
                answer_text = hop_answers[idx]
                if isinstance(answer_text, str) and answer_text.strip():
                    block.append(answer_text.strip())
            path_blocks.append("\n".join(block))
        
        return "\n\n".join(path_blocks) if path_blocks else None

    def _sanitize_hop_queries(self, hop_queries: Any) -> List[str]:
        """标准化hop查询列表，过滤空值并展开字符串"""
        if not hop_queries:
            return []
        sanitized: List[str] = []
        if isinstance(hop_queries, list):
            iterator = hop_queries
        elif isinstance(hop_queries, str):
            iterator = re.split(r'[,\n→]+', hop_queries)
        else:
            return []
        
        for hop in iterator:
            if not isinstance(hop, str):
                continue
            text = hop.strip()
            if text:
                sanitized.append(text)
        return sanitized

    def _normalize_field_name(self, key: str) -> str:
        """Normalize key names for loose matching (lowercase, underscores)."""
        if not isinstance(key, str):
            return ""
        return re.sub(r'[^a-z0-9]+', '_', key.strip().lower())

    def _get_entry_value(self, entry: Dict[str, Any], *keys: str) -> Any:
        """Fetch value from entry ignoring case and separators."""
        if not isinstance(entry, dict):
            return None
        normalized_cache = None
        
        for key in keys:
            if key in entry and entry[key] not in (None, ""):
                return entry[key]
        
        for key in keys:
            normalized_key = self._normalize_field_name(key)
            if not normalized_key:
                continue
            if normalized_cache is None:
                normalized_cache = {
                    self._normalize_field_name(k): v
                    for k, v in entry.items()
                }
            if normalized_cache.get(normalized_key) not in (None, ""):
                return normalized_cache[normalized_key]
        
        return None

    def _extract_ground_truth(self, text: Optional[str]) -> Optional[str]:
        """提取\\boxed{}中的简短答案，如果不存在则返回去除标记的原文本"""
        if not text:
            return None
        text = str(text).strip()
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1).strip()
        if '\\boxed' in text:
            text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
        return text.strip() if text else None
    
    def run_evaluation(
        self,
        dataset_path: str,
        system_prompt: str = None,
        max_questions: int = None,
        parallel: bool = False,
        max_workers: int = 3
    ) -> pd.DataFrame:
        """
        运行Agent模式评测
        
        Args:
            dataset_path: 数据集路径
            system_prompt: 系统提示词（默认使用agent模式）
            max_questions: 最多处理的问题数量
            parallel: 是否并行处理
            max_workers: 并行工作线程数
            
        Returns:
            评测结果DataFrame
        """
        # 默认使用Agent模式提示词
        if system_prompt is None:
            system_prompt = get_prompt("agent")
        
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
                futures = {
                    executor.submit(
                        self._process_single_question, 
                        item["question"], 
                        idx + 1,
                        system_prompt,
                        item.get("ground_truth"),
                        item.get("search_path"),
                        item.get("reference_answer"),
                        item.get("difficulty"),
                        item.get("hop_used_queries")
                    ): idx 
                    for idx, item in enumerate(questions)
                }
                
                with tqdm(total=len(questions), desc="评测进度") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
        else:
            # 顺序处理
            logger.info(f"🚀 开始顺序评测")
            
            for idx, item in enumerate(tqdm(questions, desc="评测进度")):
                result = self._process_single_question(
                    item["question"], 
                    idx + 1, 
                    system_prompt,
                    item.get("ground_truth"),
                    item.get("search_path"),
                    item.get("reference_answer"),
                    item.get("difficulty"),
                    item.get("hop_used_queries")
                )
                results.append(result)
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        
        output_file = os.path.join(
            self.output_dir, 
            f"{dataset_name}_agent_results_{timestamp}.json"
        )
        
        # 计算Judge指标平均值
        judge_averages = self._calculate_judge_averages(results)
        
        # 获取有效样本数量（有Judge评分的样本数）
        sample_count = judge_averages.get('count', len(results)) if judge_averages else len(results)
        
        # 计算工具调用和对话轮数统计，使用有效样本数量
        tool_conversation_stats = self._calculate_tool_and_conversation_stats(results, sample_count)
        
        # 计算Hop轨迹准确率
        trajectory_stats = self._summarize_trajectory_accuracy(results)
        
        # 构建输出数据结构
        output_data = {
            "metadata": {
                "dataset": dataset_name,
                "timestamp": timestamp,
                "total_questions": len(results),
                "model": self.model_name
            },
            "results": results
        }
        
        # 添加工具调用和对话统计到metadata中
        output_data["metadata"].update(tool_conversation_stats)
        if trajectory_stats:
            output_data["metadata"]["trajectory_accuracy"] = trajectory_stats
        
        # 如果有Judge评分，添加到metadata中
        if judge_averages:
            output_data["metadata"]["judge_averages"] = judge_averages
        
        # 保存为JSON（完整数据，包含所有搜索结果和统计信息）
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 评测结果已保存: {output_file}")
        
        # 保存详细的搜索结果到单独的文件
        search_results_file = output_file.replace('.json', '_search_details.json')
        self._save_search_details(results, search_results_file)
        logger.info(f"💾 搜索详情已保存: {search_results_file}")
        
        # 保存为CSV（简化版，只包含主要字段）
        csv_data = []
        for _, row in df.iterrows():
            csv_row = {
                "question_id": row["question_id"],
                "question": row["question"],
                "final_response": row["final_response"],
                "num_tool_calls": len(row["tool_calls"]),
                "num_rounds": len(row["conversation_history"]),
                "processing_time": row["processing_time"],
                "success": row["success"],
                "error": row.get("error", "")
            }
            traj_eval = row.get("trajectory_evaluation")
            if isinstance(traj_eval, dict):
                csv_row["trajectory_accuracy"] = traj_eval.get("accuracy")
                csv_row["missed_hops"] = "|".join(traj_eval.get("missing_hops", []))
            csv_data.append(csv_row)
        
        csv_df = pd.DataFrame(csv_data)
        csv_file = output_file.replace('.json', '.csv')
        csv_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"💾 CSV结果已保存: {csv_file}")
        
        # 打印统计信息
        self._print_statistics(df, trajectory_stats)
        
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
            
            # 提取所有搜索调用
            for tool_call in result.get("tool_calls", []):
                tool_type = tool_call.get("tool_type")
                query = tool_call.get("query")
                tool_result = tool_call.get("result", {})
                timestamp = tool_call.get("timestamp", "")
                round_num = tool_call.get("round")
                
                detail = {
                    "question_id": question_id,
                    "question": question,
                    "round": round_num,
                    "tool_type": tool_type,
                    "query": query,
                    "timestamp": timestamp,
                    "success": tool_result.get("success", False)
                }
                
                # 根据工具类型提取详细信息
                if tool_type == "web_search":
                    detail["web_search"] = {
                        "context": tool_result.get("context", ""),
                        "raw_result": tool_result.get("result", {}),
                        "error": tool_result.get("error", "")
                    }
                    
                    # 提取百度搜索结果数量
                    raw_result = tool_result.get("result", {})
                    if isinstance(raw_result, dict):
                        if "baiduSearchResults" in raw_result:
                            detail["num_results"] = len(raw_result.get("baiduSearchResults", []))
                        elif "results" in raw_result:
                            detail["num_results"] = len(raw_result.get("results", []))
                
                elif tool_type == "rag":
                    detail["rag"] = {
                        "total_results": tool_result.get("total_results", 0),
                        "context": tool_result.get("context", ""),
                        "merchants": tool_result.get("results", []),
                        "error": tool_result.get("error", "")
                    }
                    detail["num_results"] = tool_result.get("total_results", 0)
                
                search_details.append(detail)
        
        # 保存为格式化的JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(search_details, f, ensure_ascii=False, indent=2)
    
    def _calculate_judge_averages(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算Judge评分的平均值
        
        Args:
            results: 评测结果列表
            
        Returns:
            包含所有指标平均分和总分平均的字典，如果没有Judge评分则返回None
        """
        if not self.use_judge:
            return None
        
        # 收集所有有效的Judge评分
        grouped_scores: Dict[str, List[Dict[str, Any]]] = {"__overall__": []}
        for result in results:
            judge_scores = result.get("judge_scores")
            if judge_scores and "error" not in judge_scores:
                grouped_scores["__overall__"].append(judge_scores)
                difficulty = result.get("difficulty")
                if isinstance(difficulty, str) and difficulty.strip():
                    grouped_scores.setdefault(difficulty.strip(), []).append(judge_scores)
        
        if not grouped_scores["__overall__"]:
            return None
        
        averages = self._summarize_judge_scores(grouped_scores["__overall__"])
        
        # 计算不同难度的拆分指标
        difficulty_breakdown = {}
        for diff, scores in grouped_scores.items():
            if diff == "__overall__" or not scores:
                continue
            difficulty_breakdown[diff] = self._summarize_judge_scores(scores)
        
        averages["difficulty_breakdown"] = difficulty_breakdown
        return averages

    def _summarize_judge_scores(self, scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """对一组Judge评分求平均"""
        if not scores:
            return {}
        
        summary = {
            "correctness": sum(s["correctness"]["score"] for s in scores) / len(scores),
            "completeness": sum(s["completeness"]["score"] for s in scores) / len(scores),
            "fluency": sum(s["fluency"]["score"] for s in scores) / len(scores),
            "safety": sum(s["safety"]["score"] for s in scores) / len(scores),
            "total_score": sum(s["total_score"] for s in scores) / len(scores),
            "max_score": sum(s["max_score"] for s in scores) / len(scores),
            "count": len(scores)
        }
        
        hallucination_scores = [
            s.get("hallucination", {}).get("score")
            for s in scores
            if isinstance(s.get("hallucination"), dict) and s["hallucination"].get("score") is not None
        ]
        if hallucination_scores:
            summary["hallucination"] = sum(hallucination_scores) / len(hallucination_scores)
        
        return summary
    
    def _normalize_hop_text(self, text: Optional[str]) -> str:
        """将Hop查询标准化用于模糊匹配"""
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", str(text)).strip().lower()
        if not normalized:
            return ""
        normalized = re.sub(r'[^\w\u4e00-\u9fff]+', '', normalized)
        return normalized

    def _evaluate_trajectory_accuracy(
        self,
        expected_hops: List[str],
        tool_calls: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        根据hop计划与实际工具调用记录计算轨迹准确率
        规则：
            1. 必须覆盖所有hop（顺序可调）
            2. 每个hop需要有对应的local RAG检索，查询语义需匹配
            3. 检索调用必须成功且返回商户信息
            4. 有任何缺失或失败即判0分
        """
        sanitized_hops = [
            {"raw": hop, "normalized": self._normalize_hop_text(hop)}
            for hop in expected_hops
            if isinstance(hop, str) and hop.strip()
        ]
        if not sanitized_hops:
            return None
        
        rag_steps: List[Dict[str, Any]] = []
        other_steps: List[Dict[str, Any]] = []
        for idx, call in enumerate(tool_calls):
            step = {
                "index": idx,
                "round": call.get("round"),
                "tool_type": call.get("tool_type"),
                "query": call.get("query") or "",
                "normalized_query": self._normalize_hop_text(call.get("query")),
                "result": call.get("result", {}) or {}
            }
            if step["tool_type"] == "rag":
                rag_steps.append(step)
            else:
                other_steps.append(step)
        
        used_indices = set()
        missing_hops: List[str] = []
        invalid_hops: List[Dict[str, Any]] = []
        matched_hops: List[Dict[str, Any]] = []
        
        for hop in sanitized_hops:
            matched_idx, similarity = self._find_best_rag_step(
                hop["normalized"], rag_steps, used_indices
            )
            if matched_idx is None:
                missing_hops.append(hop["raw"])
                continue
            
            used_indices.add(matched_idx)
            step = rag_steps[matched_idx]
            issues = []
            query_text = step["query"].strip()
            if not query_text:
                issues.append("查询为空")
            result_payload = step.get("result", {})
            if not result_payload.get("success"):
                issues.append("检索失败")
            if not result_payload.get("results"):
                issues.append("未返回商家信息")
            
            if issues:
                invalid_hops.append({
                    "hop": hop["raw"],
                    "query": step["query"],
                    "issues": issues
                })
            else:
                matched_hops.append({
                    "hop": hop["raw"],
                    "query": step["query"],
                    "round": step.get("round"),
                    "similarity": round(similarity, 3),
                    "total_results": result_payload.get("total_results", 0)
                })
        
        extra_steps = [
            {
                "round": rag_steps[i].get("round"),
                "tool_type": "rag",
                "query": rag_steps[i].get("query")
            }
            for i in range(len(rag_steps))
            if i not in used_indices
        ]
        extra_steps.extend([
            {
                "round": step.get("round"),
                "tool_type": step.get("tool_type"),
                "query": step.get("query")
            }
            for step in other_steps
        ])
        
        coverage_passed = len(missing_hops) == 0
        hop_quality_passed = len(invalid_hops) == 0
        accuracy = 1.0 if coverage_passed and hop_quality_passed else 0.0
        failure_reasons = []
        if not coverage_passed:
            failure_reasons.append("覆盖不完整")
        if not hop_quality_passed:
            failure_reasons.append("步骤校验失败")
        
        evaluation = {
            "expected_hops_count": len(sanitized_hops),
            "matched_hops": matched_hops,
            "matched_ratio": len(matched_hops) / len(sanitized_hops) if sanitized_hops else 0,
            "missing_hops": missing_hops,
            "invalid_hops": invalid_hops,
            "extra_steps": extra_steps,
            "coverage_passed": coverage_passed,
            "accuracy": accuracy,
            "failure_reasons": failure_reasons
        }
        return evaluation

    def _find_best_rag_step(
        self,
        target: str,
        rag_steps: List[Dict[str, Any]],
        used_indices: set
    ) -> Tuple[Optional[int], float]:
        """根据相似度为hop找到最匹配的RAG调用"""
        if not target:
            return None, 0.0
        best_idx = None
        best_score = 0.0
        for idx, step in enumerate(rag_steps):
            if idx in used_indices:
                continue
            candidate = step.get("normalized_query") or ""
            if not candidate:
                continue
            if target in candidate or candidate in target:
                score = 1.0
            else:
                score = SequenceMatcher(None, target, candidate).ratio()
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= self.hop_similarity_threshold:
            return best_idx, best_score
        return None, 0.0

    def _summarize_trajectory_accuracy(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """汇总轨迹准确率统计"""
        evaluations = [
            result.get("trajectory_evaluation")
            for result in results
            if isinstance(result.get("trajectory_evaluation"), dict)
        ]
        if not evaluations:
            return None
        
        total = len(evaluations)
        passed = sum(1 for ev in evaluations if ev.get("accuracy") == 1.0)
        coverage_failures = sum(1 for ev in evaluations if not ev.get("coverage_passed", False))
        invalid_failures = sum(1 for ev in evaluations if ev.get("invalid_hops"))
        avg_extra_steps = sum(len(ev.get("extra_steps", [])) for ev in evaluations) / total if total else 0
        
        return {
            "total_evaluated": total,
            "passed": passed,
            "failed": total - passed,
            "accuracy_rate": passed / total if total else 0,
            "coverage_failures": coverage_failures,
            "invalid_step_failures": invalid_failures,
            "avg_extra_steps": round(avg_extra_steps, 3)
        }
    
    def _extract_rag_context(self, tool_calls: List[Dict[str, Any]]) -> str:
        """
        从工具调用记录中提取RAG上下文
        
        Args:
            tool_calls: 工具调用记录列表
            
        Returns:
            合并后的RAG上下文字符串
        """
        rag_contexts = []
        
        for tool_call in tool_calls:
            if tool_call.get("tool_type") == "rag":
                result = tool_call.get("result", {})
                context = result.get("context", "")
                if context:
                    rag_contexts.append(context)
        
        return "\n\n".join(rag_contexts) if rag_contexts else ""
    
    def _calculate_tool_and_conversation_stats(self, results: List[Dict[str, Any]], sample_count: int = None) -> Dict[str, Any]:
        """
        计算工具调用和对话轮数的统计信息
        
        Args:
            results: 评测结果列表
            sample_count: 有效样本数量（用于计算平均值）。如果为None，使用len(results)
            
        Returns:
            包含工具调用和对话统计的字典
        """
        total_tool_calls = 0
        total_conversation_rounds = 0
        
        for result in results:
            # 统计工具调用次数
            tool_calls = result.get("tool_calls", [])
            total_tool_calls += len(tool_calls)
            
            # 统计对话轮数
            conversation_history = result.get("conversation_history", [])
            total_conversation_rounds += len(conversation_history)
        
        # 使用传入的sample_count（有效样本数），如果没有则使用结果总数
        num_samples = sample_count if sample_count is not None else len(results)
        
        stats = {
            "total_tool_calls": total_tool_calls,
            "avg_tool_calls": total_tool_calls / num_samples if num_samples > 0 else 0,
            "total_conversation_rounds": total_conversation_rounds,
            "avg_conversation_rounds": total_conversation_rounds / num_samples if num_samples > 0 else 0
        }
        
        return stats
    
    def _print_statistics(self, df: pd.DataFrame, trajectory_stats: Optional[Dict[str, Any]] = None):
        """打印评测统计信息"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 Agent模式评测统计")
        logger.info("=" * 60)

        if df.empty or "success" not in df.columns:
            logger.warning("暂无有效评测数据（可能未读取到任何样本），跳过统计。")
            return

        logger.info(f"总问题数: {len(df)}")
        logger.info(f"成功处理: {df['success'].sum()}")
        logger.info(f"处理失败: {(~df['success']).sum()}")
        logger.info(f"平均处理时间: {df['processing_time'].mean():.2f}秒")
        logger.info(f"总处理时间: {df['processing_time'].sum():.2f}秒")
        
        # 工具调用统计
        total_tool_calls = sum([len(row['tool_calls']) for _, row in df.iterrows()])
        avg_tool_calls = total_tool_calls / len(df) if len(df) > 0 else 0
        logger.info(f"总工具调用次数: {total_tool_calls}")
        logger.info(f"平均工具调用次数: {avg_tool_calls:.2f}")
        
        # 工具类型统计
        web_search_count = 0
        rag_count = 0
        for _, row in df.iterrows():
            for tool_call in row['tool_calls']:
                if tool_call['tool_type'] == 'web_search':
                    web_search_count += 1
                elif tool_call['tool_type'] == 'rag':
                    rag_count += 1
        
        logger.info(f"Web搜索调用: {web_search_count}")
        logger.info(f"RAG检索调用: {rag_count}")
        
        # 搜索成功率统计
        web_success = 0
        rag_success = 0
        rag_total_merchants = 0
        for _, row in df.iterrows():
            for tool_call in row['tool_calls']:
                if tool_call['tool_type'] == 'web_search':
                    if tool_call.get('result', {}).get('success'):
                        web_success += 1
                elif tool_call['tool_type'] == 'rag':
                    result = tool_call.get('result', {})
                    if result.get('success'):
                        rag_success += 1
                        rag_total_merchants += result.get('total_results', 0)
        
        if web_search_count > 0:
            logger.info(f"Web搜索成功率: {web_success}/{web_search_count} ({web_success/web_search_count*100:.1f}%)")
        if rag_count > 0:
            logger.info(f"RAG检索成功率: {rag_success}/{rag_count} ({rag_success/rag_count*100:.1f}%)")
            avg_merchants = rag_total_merchants / rag_success if rag_success > 0 else 0
            logger.info(f"平均检索商户数: {avg_merchants:.2f}")
        
        if trajectory_stats:
            logger.info("\n" + "-" * 60)
            logger.info("🧭 多跳轨迹准确率")
            logger.info("-" * 60)
            logger.info(f"可评估样本: {trajectory_stats.get('total_evaluated')}")
            logger.info(f"完全覆盖: {trajectory_stats.get('passed')}")
            logger.info(f"覆盖失败: {trajectory_stats.get('coverage_failures')}")
            logger.info(f"步骤校验失败: {trajectory_stats.get('invalid_step_failures')}")
            logger.info(f"轨迹准确率: {trajectory_stats.get('accuracy_rate', 0)*100:.1f}%")
            logger.info(f"平均多余步骤: {trajectory_stats.get('avg_extra_steps', 0):.2f}")
        
        # 对话轮数统计
        avg_rounds = sum([len(row['conversation_history']) for _, row in df.iterrows()]) / len(df)
        logger.info(f"平均对话轮数: {avg_rounds:.2f}")
        
        # Judge评分统计
        if self.use_judge:
            valid_judge_scores = []
            for _, row in df.iterrows():
                judge_scores = row.get("judge_scores")
                if judge_scores and isinstance(judge_scores, dict) and "error" not in judge_scores:
                    valid_judge_scores.append(judge_scores)
            
            if valid_judge_scores:
                logger.info("\n" + "-" * 60)
                logger.info("🎯 LLM Judge 评分统计")
                logger.info("-" * 60)
                logger.info(f"有效评分数: {len(valid_judge_scores)}/{len(df)}")
                
                avg_correctness = sum(s["correctness"]["score"] for s in valid_judge_scores) / len(valid_judge_scores)
                avg_completeness = sum(s["completeness"]["score"] for s in valid_judge_scores) / len(valid_judge_scores) / 10
                avg_fluency = sum(s["fluency"]["score"] for s in valid_judge_scores) / len(valid_judge_scores) / 10
                avg_safety = sum(s["safety"]["score"] for s in valid_judge_scores) / len(valid_judge_scores) / 10
                avg_total = sum(s["total_score"] for s in valid_judge_scores) / len(valid_judge_scores)
                avg_max = sum(s["max_score"] for s in valid_judge_scores) / len(valid_judge_scores)
                
                logger.info(f"正确性 (Correctness): {avg_correctness:.3f}/1.0")
                logger.info(f"完整性 (Completeness): {avg_completeness:.3f}/1.0")
                logger.info(f"流畅度 (Fluency): {avg_fluency:.3f}/1.0")
                logger.info(f"安全性 (Safety): {avg_safety:.3f}/1.0")
                
                # 幻觉检测评分（如果存在）
                hallucination_scores = [s["hallucination"]["score"] for s in valid_judge_scores if "hallucination" in s]
                if hallucination_scores:
                    avg_hallucination = sum(hallucination_scores) / len(hallucination_scores) / 10
                    logger.info(f"幻觉检测 (Hallucination): {avg_hallucination:.3f}/1.0 (评分数: {len(hallucination_scores)})")
                
                logger.info(f"总分 (Total Score): {avg_total:.3f}/{avg_max:.1f}")
                logger.info("-" * 60)
        
        logger.info("=" * 60 + "\n")
    
    def cleanup(self):
        """清理资源"""
        if self.rag_agent:
            self.rag_agent.cleanup()
        logger.info("✅ 资源已清理")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Agent模式评测工具 - LLM自主使用工具")
    
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
        "--max-tool-rounds", 
        type=int, 
        default=10,
        help="最大工具调用轮数 (默认: 10)"
    )
    
    parser.add_argument(
        "--rag-index-path", 
        type=str, 
        default=None,
        help="RAG索引路径（例如：/path/to/faiss_merchant_index_vllm_shanghai）"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./evaluation_results",
        help="输出目录 (默认: ./evaluation_results)"
    )
    
    parser.add_argument(
        "--use-multi-tokens",
        action="store_true",
        help="使用多API tokens进行并行请求（从config/model_rpm.yaml读取）"
    )
    
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="用户当前所在城市（可选），支持：beijing/上海/北京/sh/shanghai/guangzhou等"
    )
    
    parser.add_argument(
        "--use-judge",
        action="store_true",
        help="使用LLM Judge对模型输出进行评分（需要数据集包含ground_truth）"
    )
    
    parser.add_argument(
        "--judge-model",
        type=str,
        default="anthropic.claude-opus-4.1",
        help="Judge使用的模型名称 (默认: anthropic.claude-opus-4.1)"
    )
    
    args = parser.parse_args()
    
    default_model = parser.get_default("model") or "deepseek-v31-meituan"
    if not args.model or not args.model.strip():
        logger.warning(f"⚠️ 未提供有效的模型名称，自动回退到默认模型: {default_model}")
        args.model = default_model
    
    try:
        # 读取配置（如果使用多tokens）
        web_search_tokens = None
        use_token_manager = False
        
        if args.use_multi_tokens:
            from llm_utils import ConfigManager
            config_mgr = ConfigManager()
            web_search_tokens = config_mgr.get_api_keys()
            use_token_manager = True
            logger.info(f"📋 使用多token模式，共 {len(web_search_tokens)} 个tokens")
        
        # 创建Agent评测流程
        pipeline = AgentEvaluationPipeline(
            model_name=args.model,
            model_rpm=args.rpm,
            web_search_tokens=web_search_tokens,
            use_token_manager=use_token_manager,
            rag_index_path=args.rag_index_path,
            output_dir=args.output_dir,
            max_tool_rounds=args.max_tool_rounds,
            use_judge=args.use_judge,
            judge_model=args.judge_model
        )
        
        # 生成系统提示词（根据城市参数）
        system_prompt = get_prompt("agent", city=args.city)
        if args.city:
            logger.info(f"📍 用户当前位置: {args.city}")
        
        # 运行评测
        pipeline.run_evaluation(
            dataset_path=args.dataset,
            system_prompt=system_prompt,
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

