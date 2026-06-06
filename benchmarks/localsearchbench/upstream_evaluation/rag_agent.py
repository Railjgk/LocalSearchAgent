#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Agent - 负责使用本地RAG系统检索商户信息
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
import faiss

# 添加rag_gpu路径
rag_gpu_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag_gpu")
if rag_gpu_path not in sys.path:
    sys.path.insert(0, rag_gpu_path)

# 导入VLLM加速版本的搜索系统
from interactive_merchant_search_vllm import InteractiveMerchantSearchVLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAgent:
    """RAG代理，使用本地向量数据库检索商户信息"""
    
    def __init__(
        self,
        index_path: str = None,
        embedding_model_path: str = None,
        reranker_model_path: str = None,
        use_reranker: bool = True,
        gpu_memory_utilization: float = 0.65
    ):
        """
        初始化RAG代理
        
        Args:
            index_path: FAISS索引路径前缀
            embedding_model_path: 嵌入模型路径（默认使用Qwen3-Embedding-8B）
            reranker_model_path: 重排序模型路径
            use_reranker: 是否使用重排序
            gpu_memory_utilization: GPU内存使用率，会同时应用于embedding和reranker（默认0.35）
        """
        # 设置默认路径
        if index_path is None:
            index_path = os.path.join(rag_gpu_path, "enhanced_merchant_index")
        
        # 使用固定的8B模型（4096维）
        if embedding_model_path is None:
            embedding_model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/hehang03/tuansou/Localplayground/rag_gpu/Qwen3-Embedding-8B"
            logger.info("✅ 使用固定模型: Qwen3-Embedding-8B (4096维)")
        
        if reranker_model_path is None and use_reranker:
            reranker_model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/hehang03/tuansou/Localplayground/rag_gpu/Qwen3-Reranker-8B"
            logger.info("✅ 使用固定模型: Qwen3-Reranker-8B")
        
        self.index_path = index_path
        self.embedding_model_path = embedding_model_path
        self.reranker_model_path = reranker_model_path
        self.use_reranker = use_reranker
        
        # 初始化搜索系统
        logger.info(f"🚀 正在初始化RAG系统...")
        logger.info(f"  索引路径: {index_path}")
        logger.info(f"  嵌入模型: {embedding_model_path}")
        logger.info(f"  重排序模型: {reranker_model_path if use_reranker else '未启用'}")
        logger.info(f"  GPU内存使用率: {gpu_memory_utilization}")
        
        self.search_system = InteractiveMerchantSearchVLLM(
            index_path=index_path,
            embedding_model_path=embedding_model_path,
            reranker_model_path=reranker_model_path,
            use_reranker=use_reranker,
            gpu_memory_utilization=gpu_memory_utilization
        )
        
        # 尝试在初始化后设置候选数量（如果底层系统支持）
        # 有些系统可能在初始化后可以通过属性设置
        if hasattr(self.search_system, 'max_candidates'):
            self.search_system.max_candidates = 100
            logger.info(f"  ✅ 设置底层系统 max_candidates=100")
        if hasattr(self.search_system, 'default_top_k'):
            self.search_system.default_top_k = 100
            logger.info(f"  ✅ 设置底层系统 default_top_k=100")
        
        # 初始化系统
        if not self.search_system.initialize():
            raise RuntimeError("RAG系统初始化失败")
        
        logger.info("✅ RAG系统初始化完成")
    
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        use_reranker: bool = None,
        rerank_top_k: int = 20,
        candidate_multiplier: float = 5.0
    ) -> Dict[str, Any]:
        """
        执行RAG检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量（未使用，保留以兼容）
            use_reranker: 是否使用重排序（None则使用默认设置）
            rerank_top_k: 重排序后返回的数量（默认20）
            candidate_multiplier: 候选池倍数（默认5.0，即100个候选）
            
        Returns:
            检索结果字典
        """
        try:
            # 计算候选池大小：rerank_top_k * candidate_multiplier = 20 * 5.0 = 100
            candidate_pool_size = int(rerank_top_k * candidate_multiplier)
            
            logger.info(f"🔍 执行RAG检索: {query}")
            logger.info(f"  📊 检索参数: 候选池={candidate_pool_size}, 重排序返回={rerank_top_k}")
            logger.info(f"  📊 参数详情: rerank_top_k={rerank_top_k}, candidate_multiplier={candidate_multiplier}")
            
            # 执行搜索 - 尝试多种参数名以确保底层系统能正确接收
            search_kwargs = {
                "query": query,
                "top_k": candidate_pool_size,  # 从候选池中检索100个
                "use_reranker": use_reranker,
                "rerank_top_k": rerank_top_k,  # 重排序后返回20个
                "candidate_multiplier": 1  # 设为1，因为我们已经通过top_k传入了计算好的候选池大小
            }
            
            # 尝试传递候选池大小参数（如果底层系统支持）
            # 有些系统可能使用 max_candidates 或 candidate_pool_size
            if hasattr(self.search_system, 'max_candidates'):
                search_kwargs["max_candidates"] = candidate_pool_size
            if hasattr(self.search_system, 'candidate_pool_size'):
                search_kwargs["candidate_pool_size"] = candidate_pool_size
                
            logger.debug(f"  🔧 传递给底层系统的参数: {search_kwargs}")
            
            results = self.search_system.search(**search_kwargs)
            
            # 检查底层系统是否使用了我们期望的候选数量
            # 如果底层系统日志显示使用了500而不是100，说明底层系统可能忽略了top_k参数
            # 这需要修改底层 InteractiveMerchantSearchVLLM 代码
            
            # 构建上下文
            context = self._build_context(results)
            
            logger.info(f"✅ RAG检索成功，找到 {len(results)} 个结果")
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "context": context,
                "total_results": len(results)
            }
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"❌ RAG检索失败: {error_msg}")
            logger.error(f"完整错误信息:\n{traceback.format_exc()}")
            return {
                "success": False,
                "query": query,
                "error": error_msg,
                "context": "",
                "results": []
            }
    
    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """
        从检索结果构建上下文
        
        Args:
            results: 检索结果列表
            
        Returns:
            格式化的上下文字符串
        """
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            # 尝试从 metadata 或 result 本身获取数据
            metadata = result.get("metadata", {})
            if not metadata:
                # 如果 metadata 为空，尝试直接使用 result
                metadata = result
            
            # 构建商户信息
            merchant_info = []
            merchant_info.append(f"[商户{i}]")
            
            # 基本信息 - 使用实际的字段名
            if "name" in metadata:
                merchant_info.append(f"商户名称: {metadata['name']}")
            
            # 商户ID/POI ID - 优先使用id字段（实际返回的字段名），然后才是poi_id和merchant_id
            if "id" in metadata:
                merchant_info.append(f"poi_id: {metadata['id']}")
            elif "poi_id" in metadata:
                merchant_info.append(f"poi_id: {metadata['poi_id']}")
            elif "merchant_id" in metadata:
                merchant_info.append(f"商户ID: {metadata['merchant_id']}")
            
            if "category" in metadata:
                merchant_info.append(f"类别: {metadata['category']}")
            
            if "subcategory" in metadata:
                merchant_info.append(f"子类别: {metadata['subcategory']}")
            
            if "address" in metadata:
                merchant_info.append(f"地址: {metadata['address']}")
            
            if "district" in metadata:
                merchant_info.append(f"区域: {metadata['district']}")
            
            if "business_area" in metadata:
                merchant_info.append(f"商圈: {metadata['business_area']}")
            
            if "business_hours" in metadata:
                merchant_info.append(f"营业时间: {metadata['business_hours']}")
            
            # 评分和价格
            if "rating" in metadata:
                merchant_info.append(f"评分: {metadata['rating']}")
            
            if "avg_price" in metadata:
                merchant_info.append(f"人均: {metadata['avg_price']}元")
            
            if "price_range" in metadata:
                merchant_info.append(f"价格区间: {metadata['price_range']}")
            
            # 特色和标签
            if "specialties" in metadata and metadata["specialties"]:
                merchant_info.append(f"特色菜品: {metadata['specialties']}")
            
            if "tags" in metadata and metadata["tags"]:
                merchant_info.append(f"标签: {metadata['tags']}")
            
            if "description" in metadata and metadata["description"]:
                merchant_info.append(f"描述: {metadata['description']}")
            
            # 设施和促销
            if "facilities" in metadata and metadata["facilities"]:
                merchant_info.append(f"设施: {metadata['facilities']}")
            
            if "promotions" in metadata and metadata["promotions"]:
                merchant_info.append(f"优惠活动: {metadata['promotions']}")
            
            # 联系方式
            if "phone" in metadata and metadata["phone"]:
                merchant_info.append(f"电话: {metadata['phone']}")
            
            # 相似度分数
            if "similarity_score" in result:
                merchant_info.append(f"相似度: {result['similarity_score']:.4f}")
            
            # 如果只有 [商户{i}]，说明没有提取到任何字段，记录警告并显示可用字段
            if len(merchant_info) == 1:
                logger.warning(f"⚠️ 商户{i}未提取到任何字段，可用字段: {list(metadata.keys())[:10]}")
                # 尝试直接显示所有非空字段
                for key, value in metadata.items():
                    if key not in ["metadata", "similarity_score", "rerank_score"] and value:
                        merchant_info.append(f"{key}: {value}")
            
            context_parts.append("\n".join(merchant_info))
        
        return "\n\n".join(context_parts)
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self.search_system, 'cleanup_resources'):
            self.search_system.cleanup_resources()
            logger.info("✅ RAG系统资源已清理")


if __name__ == "__main__":
    # 测试RAG代理
    agent = RAGAgent()
    
    # 测试查询
    test_query = "望京附近有什么好吃的餐厅"
    result = agent.search(test_query, top_k=3)
    
    print(f"\n查询: {test_query}")
    print(f"成功: {result['success']}")
    print(f"找到结果: {result['total_results']}")
    print(f"\n上下文:\n{result['context']}")
    
    # 清理资源
    agent.cleanup()


