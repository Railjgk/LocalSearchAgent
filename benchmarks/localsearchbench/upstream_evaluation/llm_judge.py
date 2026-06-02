"""
LLM as Judge - LLM评估器
使用LLM对模型输出进行多维度评分
"""

import json
import logging
import re
import random
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from judge_prompts import get_judge_prompt, get_all_dimensions
from llm_utils import ApiKeyManager, ConfigManager

logger = logging.getLogger(__name__)


class LLMJudge:
    """LLM评估器 - 使用LLM对模型输出进行评分"""
    
    def __init__(
        self,
        api_url: str,
        api_key: str = None,
        api_keys: List[str] = None,
        model_name: str = "deepseek-v32-meituan",
        timeout: int = 300,  # 从120秒增加到300秒，适应长文本评估
        rpm_per_key: Optional[int] = None,
        max_tokens: int = 18000
    ):
        """
        初始化LLM评估器
        
        Args:
            api_url: LLM API地址
            api_key: API密钥（单个，兼容旧版）
            api_keys: API密钥列表（多个，用于负载均衡）
            model_name: 评估使用的模型名称
            timeout: 超时时间（秒）
            rpm_per_key: 每个API key的RPM限制
            max_tokens: 最大token数（默认18000，与轨迹评估一致）
        """
        # 支持多API key配置
        if api_keys is not None and len(api_keys) > 0:
            self.api_keys = api_keys
            logger.info(f"🔑 使用多个API Key进行负载均衡 (数量: {len(api_keys)})")
        elif api_key is not None:
            self.api_keys = [api_key]
            logger.info(f"🔑 使用单个API Key")
        else:
            raise ValueError("必须提供 api_key 或 api_keys 参数")
        
        # 设置API URL（与FridayClient保持一致）
        if not api_url.endswith('/chat/completions'):
            self.api_url = f"{api_url}/chat/completions"
        else:
            self.api_url = api_url
        
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.current_key_index = 0
        self.exhausted_clients: Dict[int, str] = {}  # 存储用尽的key索引
        cfg = ConfigManager()
        self.rpm_per_key = rpm_per_key or cfg.get_default_rpm()
        self.max_retries_per_request = 5  # 与 FridayClient 保持一致
        
        logger.info(f"🎯 LLM Judge初始化完成 (模型: {model_name}, API Keys: {len(self.api_keys)})")
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """
        从响应中提取JSON
        
        Args:
            response: LLM响应文本
            
        Returns:
            提取的JSON字典，如果提取失败返回None
        """
        # 尝试直接解析
        try:
            return json.loads(response)
        except:
            pass
        
        # 尝试提取 {} 之间的内容
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                # 验证是否包含评估维度的键
                if any(key in data for key in ["答案正确性", "输出完整性", "内容流畅度", "安全性"]):
                    return data
            except:
                continue
        
        return None
    
    def _mask_api_key(self, api_key: str) -> str:
        """Mask API key for logging."""
        if not api_key or len(api_key) < 8:
            return "***"
        return f"{api_key[:4]}...{api_key[-4:]}"
    
    def _extract_app_id(self, message: str) -> Optional[str]:
        """Try to extract AppId from an error message."""
        match = re.search(r"AppId:[\*]*([0-9A-Za-z]+)", message)
        if match:
            return match.group(1)
        return None
    
    def _is_rate_limit_error(self, message: str) -> bool:
        """Detect transient rate limit (应当重试或等待，而不是标记 key 用尽)."""
        lowered = message.lower()
        return any(
            keyword in lowered
            for keyword in [
                "429",
                "rate limit",
                "too many requests",
                "每分钟请求次数超过限制",
                "请求次数超过限制"
            ]
        )

    def _is_quota_exhausted_error(self, message: str) -> bool:
        """Detect quota exhausted / hard usage cap (可标记 key 用尽)."""
        lowered = message.lower()
        return any(
            keyword in lowered
            for keyword in [
                "达到使用量上限",
                "quota",
                "usage reached",
                "exceeded your current quota",
                "billing hard limit"
            ]
        )
    
    def _is_timeout_error(self, message: str) -> bool:
        """Detect timeout error (可重试，可能是临时网络问题或服务器负载高)."""
        lowered = message.lower()
        return any(
            keyword in lowered
            for keyword in [
                "timeout",
                "timed out",
                "request timeout",
                "read timeout",
                "连接超时",
                "请求超时"
            ]
        )
    
    def _mark_client_exhausted(self, key_index: int, message: str):
        """Mark the api key as exhausted so it will be skipped later."""
        if key_index in self.exhausted_clients:
            return
        
        app_id = self._extract_app_id(message) or "未知"
        masked_key = self._mask_api_key(self.api_keys[key_index])
        self.exhausted_clients[key_index] = message
        
        logger.warning(
            f"🪫 API Key已耗尽 (AppId: {app_id}, Key: {masked_key})，后续将跳过该Key"
        )
    
    def _get_next_api_key(self, tried_keys: set) -> Optional[Tuple[str, int]]:
        """
        轮询获取下一个未尝试过的API key（负载均衡）
        
        Args:
            tried_keys: 已尝试过的key集合
            
        Returns:
            (api_key, key_index) 或 None
        """
        if len(self.exhausted_clients) == len(self.api_keys):
            raise RuntimeError("所有API Key都已达到使用上限，无法继续调用LLM Judge")
        
        for _ in range(len(self.api_keys)):
            key_index = self.current_key_index
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
            if key_index not in self.exhausted_clients:
                api_key = self.api_keys[key_index]
                if api_key not in tried_keys:
                    return api_key, key_index
        
        return None
    
    def _call_llm(self, prompt: str, user_input: str) -> Optional[Dict]:
        """
        调用LLM进行评估（带多次重试、按 key 限速、配额轮询）。
        使用与轨迹生成一致的 requests 直接调用方式。
        """
        # 构建请求payload
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.0,  # 降低随机性，保证评估一致性
            "max_tokens": self.max_tokens,
            "stream": False,
            "thinking": {"type": "disabled"}  # 关闭 thinking 模式
        }
        
        # 重试逻辑：优先在同一轮尝试其他可用 key；若全部限流则等待后继续，直到成功或遇到非限流错误累计超限
        last_error = None
        non_rate_limit_failures = 0
        
        while True:
            tried_keys = set()
            rate_limited_all = False
            
            while True:
                key_result = self._get_next_api_key(tried_keys)
                if not key_result:
                    rate_limited_all = True
                    break
                
                current_api_key, key_index = key_result
                tried_keys.add(current_api_key)

                # 针对当前 key 做节流（与生成轨迹保持一致）
                ApiKeyManager.rate_limit_api_key(current_api_key, self.rpm_per_key)

                headers = {
                    "Authorization": f"Bearer {current_api_key}",
                    "Content-Type": "application/json"
                }

                try:
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )

                    # 仅在非2xx且命中限流关键词时认定为限流；200 响应不视为限流
                    if response.status_code == 429 or (response.status_code >= 400 and self._is_rate_limit_error(response.text)):
                        masked_key = f"...{current_api_key[-6:]}" if current_api_key else "***"
                        logger.warning(
                            f"⏱️ API限流 (状态码: {response.status_code}, key: {masked_key})，尝试切换其他可用key"
                        )
                        continue  # 换下一个 key

                    response.raise_for_status()
                    
                    result = response.json()
                    
                    response_text = ""
                    if "choices" in result and len(result["choices"]) > 0:
                        choice = result["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            response_text = choice["message"]["content"]
                    
                    if not response_text or not response_text.strip():
                        logger.error(f"❌ 响应内容为空，完整响应: {result}")
                        non_rate_limit_failures += 1
                        if non_rate_limit_failures >= self.max_retries_per_request:
                            logger.error(f"所有重试都失败: 响应内容为空")
                            return None
                        break  # 跳出内层，进入退避

                    # 提取JSON结果
                    json_result = self._extract_json_from_response(response_text)
                    if json_result:
                        return json_result
                    
                    logger.warning(f"⚠️  无法从响应中提取JSON: {response_text[:200]}")
                    return None
                    
                except requests.exceptions.RequestException as e:
                    last_error = e
                    error_detail = str(e)
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            error_json = e.response.json()
                            error_detail = f"{str(e)} | 响应内容: {json.dumps(error_json, ensure_ascii=False)}"
                        except:
                            error_detail = f"{str(e)} | 响应文本: {e.response.text[:200]}"

                    if self._is_rate_limit_error(error_detail):
                        logger.warning(
                            f"请求失败-限流 (key ...{current_api_key[-6:] if current_api_key else '***'})，尝试其他key"
                        )
                        continue  # 换下一个 key

                    # 检查配额用尽
                    if key_index is not None and self._is_quota_exhausted_error(error_detail):
                        self._mark_client_exhausted(key_index, error_detail)
                        continue  # 换下一个 key

                    # 非限流错误计数
                    non_rate_limit_failures += 1
                    logger.warning(
                        f"请求失败 (非限流，第{non_rate_limit_failures}/{self.max_retries_per_request}次): {error_detail}"
                    )
                    if non_rate_limit_failures >= self.max_retries_per_request:
                        logger.error(f"所有重试都失败: {last_error}")
                        return None
                    break  # 跳出内层，进入退避

            # 如果本轮所有 key 都限流，等待后继续（不限轮次，直到成功或非限流错误超限）
            base_wait = 4.0  # 基础等待时间
            exponential_wait = (2 ** min(non_rate_limit_failures, 6))  # 最大指数从4增加到6（64秒）
            sleep_time = base_wait + exponential_wait + random.uniform(0, 2)
            logger.info(
                f"所有可用key均限流，等待 {sleep_time:.2f} 秒后继续尝试"
            )
            time.sleep(sleep_time)
    
    def evaluate_correctness(
        self,
        query: str,
        model_answer: str,
        ground_truth: str,
        rag_context: str = None
    ) -> Dict:
        """
        评估答案正确性
        
        Args:
            query: 用户查询
            model_answer: 模型答案（【答案】部分）
            ground_truth: 参考答案
            rag_context: LocalRAG检索结果（商店信息的唯一真实来源）
            
        Returns:
            评估结果：{"score": 0或1, "reason": "评分理由"}
        """
        prompt = get_judge_prompt("correctness")
        
        user_data = {
            "query": query,
            "model_answer": model_answer,
            "ground_truth": ground_truth
        }
        
        # 如果提供了rag_context，添加到输入中
        if rag_context:
            user_data["rag_context"] = rag_context
        
        user_input = json.dumps(user_data, ensure_ascii=False, indent=2)
        
        result = self._call_llm(prompt, user_input)
        
        if result and "答案正确性" in result:
            score_data = result["答案正确性"]
            return {
                "score": score_data.get("score", 0),
                "reason": score_data.get("reason", "")
            }
        else:
            logger.warning("⚠️  答案正确性评估失败，返回默认分数0")
            return {"score": 0, "reason": "评估失败"}
    
    def evaluate_completeness(
        self,
        query: str,
        model_output: str,
        conversation_history: Optional[List] = None,
        tool_calls: Optional[List] = None
    ) -> Dict:
        """
        评估输出完整性
        
        Args:
            query: 用户查询
            model_output: 模型完整输出
            conversation_history: 对话历史（可选）
            tool_calls: 工具调用记录（可选）
            
        Returns:
            评估结果：{"score": 0或1, "reason": "评分理由"}
        """
        prompt = get_judge_prompt("completeness")
        
        user_data = {
            "query": query,
            "model_output": model_output
        }
        
        if conversation_history:
            user_data["conversation_history"] = conversation_history
        if tool_calls:
            user_data["tool_calls"] = tool_calls
        
        user_input = json.dumps(user_data, ensure_ascii=False, indent=2)
        
        result = self._call_llm(prompt, user_input)
        
        if result and "输出完整性" in result:
            score_data = result["输出完整性"]
            return {
                "score": score_data.get("score", 0),
                "reason": score_data.get("reason", "")
            }
        else:
            logger.warning("⚠️  输出完整性评估失败，返回默认分数0")
            return {"score": 0, "reason": "评估失败"}
    
    def evaluate_fluency(
        self,
        query: str,
        model_output: str
    ) -> Dict:
        """
        评估内容流畅度
        
        Args:
            query: 用户查询
            model_output: 模型完整输出
            
        Returns:
            评估结果：{"score": 0或1, "reason": "评分理由"}
        """
        prompt = get_judge_prompt("fluency")
        
        user_input = json.dumps({
            "query": query,
            "model_output": model_output
        }, ensure_ascii=False, indent=2)
        
        result = self._call_llm(prompt, user_input)
        
        if result and "内容流畅度" in result:
            score_data = result["内容流畅度"]
            return {
                "score": score_data.get("score", 0),
                "reason": score_data.get("reason", "")
            }
        else:
            logger.warning("⚠️  内容流畅度评估失败，返回默认分数0")
            return {"score": 0, "reason": "评估失败"}
    
    def evaluate_safety(
        self,
        query: str,
        model_output: str
    ) -> Dict:
        """
        评估安全性
        
        Args:
            query: 用户查询
            model_output: 模型完整输出
            
        Returns:
            评估结果：{"score": 0或1, "reason": "评分理由"}
        """
        prompt = get_judge_prompt("safety")
        
        user_input = json.dumps({
            "query": query,
            "model_output": model_output
        }, ensure_ascii=False, indent=2)
        
        result = self._call_llm(prompt, user_input)
        
        if result and "安全性" in result:
            score_data = result["安全性"]
            return {
                "score": score_data.get("score", 0),
                "reason": score_data.get("reason", "")
            }
        else:
            logger.warning("⚠️  安全性评估失败，返回默认分数0")
            return {"score": 0, "reason": "评估失败"}
    
    def evaluate_hallucination(
        self,
        query: str,
        model_output: str,
        rag_context: str
    ) -> Dict:
        """
        评估幻觉检测
        
        Args:
            query: 用户查询
            model_output: 模型完整输出
            rag_context: RAG检索到的上下文
            
        Returns:
            评估结果：{"score": 0-10, "reason": "评分理由"}
        """
        prompt = get_judge_prompt("hallucination")
        
        user_input = json.dumps({
            "query": query,
            "model_output": model_output,
            "rag_context": rag_context
        }, ensure_ascii=False, indent=2)
        
        result = self._call_llm(prompt, user_input)
        
        if result and "幻觉检测" in result:
            score_data = result["幻觉检测"]
            return {
                "score": score_data.get("score", 0),
                "reason": score_data.get("reason", "")
            }
        else:
            logger.warning("⚠️  幻觉检测评估失败，返回默认分数0")
            return {"score": 0, "reason": "评估失败"}
    
    def evaluate_all(
        self,
        query: str,
        model_output: str,
        ground_truth: str,
        model_answer: Optional[str] = None,
        conversation_history: Optional[List] = None,
        tool_calls: Optional[List] = None,
        rag_context: Optional[str] = None,
        enable_hallucination: bool = True
    ) -> Dict:
        """
        对模型输出进行全维度评估
        
        Args:
            query: 用户查询
            model_output: 模型完整输出
            ground_truth: 参考答案
            model_answer: 模型答案（【答案】部分），如果为None则从model_output中提取
            conversation_history: 对话历史（可选）
            tool_calls: 工具调用记录（可选）
            rag_context: RAG检索上下文（可选，用于幻觉检测）
            enable_hallucination: 是否启用幻觉检测（默认True）
            
        Returns:
            评估结果字典：
            {
                "correctness": {"score": 0或1, "reason": "..."},
                "completeness": {"score": 0-10, "reason": "..."},
                "fluency": {"score": 0-10, "reason": "..."},
                "safety": {"score": 0-10, "reason": "..."},
                "hallucination": {"score": 0-10, "reason": "..."} (可选),
                "total_score": 总分,
                "max_score": 满分
            }
        """
        # 如果没有提供model_answer，尝试从model_output中提取
        if model_answer is None:
            model_answer = self._extract_answer_from_output(model_output)
        
        logger.info(f"🎯 开始LLM Judge评估（一次性获取多维度）...")

        # 如果 model_answer 未指定，已在上方提取
        # 构造一次性调用的 prompt（包含所有维度，包括幻觉检测，见 judge_prompts.ALL_DIMENSIONS_PROMPT）
        prompt = get_judge_prompt("all")
        user_data = {
            "query": query,
            "model_output": model_output,
            "model_answer": model_answer,
            "ground_truth": ground_truth
        }
        if rag_context:
            user_data["rag_context"] = rag_context
        if conversation_history:
            user_data["conversation_history"] = conversation_history
        if tool_calls:
            user_data["tool_calls"] = tool_calls

        user_input = json.dumps(user_data, ensure_ascii=False, indent=2)
        json_result = self._call_llm(prompt, user_input)

        # 如果一次性调用失败或无法解析结果，直接返回默认结果（不再回退到逐项调用，保证单次调用语义）
        if not json_result:
            logger.warning("⚠️  一次性评估失败，返回默认分数（不回退到逐项评估）")
            return {
                "correctness": {"score": 0, "reason": "评估失败"},
                "completeness": {"score": 0, "reason": "评估失败"},
                "fluency": {"score": 0, "reason": "评估失败"},
                "safety": {"score": 0, "reason": "评估失败"},
                "hallucination": {"score": None, "reason": "未返回幻觉评估"},
                "total_score": 0,
                "max_score": 4
            }

        # 解析一次性调用返回的 JSON 结果（优先使用同一次调用的幻觉评估）
        correctness_data = json_result.get("答案正确性", {})
        completeness_data = json_result.get("输出完整性", {})
        fluency_data = json_result.get("内容流畅度", {})
        safety_data = json_result.get("安全性", {})
        hallucination_data = json_result.get("幻觉检测", {})

        correctness = {
            "score": correctness_data.get("score", 0),
            "reason": correctness_data.get("reason", "")
        }
        completeness = {
            "score": completeness_data.get("score", 0),
            "reason": completeness_data.get("reason", "")
        }
        fluency = {
            "score": fluency_data.get("score", 0),
            "reason": fluency_data.get("reason", "")
        }
        safety = {
            "score": safety_data.get("score", 0),
            "reason": safety_data.get("reason", "")
        }

        total_score = (
            correctness["score"] +
            completeness["score"] / 10 +
            fluency["score"] / 10 +
            safety["score"] / 10
        )

        result = {
            "correctness": correctness,
            "completeness": completeness,
            "fluency": fluency,
            "safety": safety,
            "total_score": total_score,
            "max_score": 4
        }

        if hallucination_data:
            result["hallucination"] = {
                "score": hallucination_data.get("score", 0),
                "reason": hallucination_data.get("reason", "")
            }
            logger.info(f"  ✅ 评估完成！总分: {total_score}/4, 幻觉检测(from same call): {result['hallucination'].get('score')}/10")
            return result

        # 若一次性调用未返回幻觉评估，设置为未返回但不再进行单独调用（保持单次API语义）
        result["hallucination"] = {"score": None, "reason": "未返回幻觉评估"}
        logger.info(f"  ✅ 评估完成（未返回幻觉评估）！总分: {total_score}/4")
        return result
    
    def _extract_answer_from_output(self, model_output: str) -> str:
        """
        从完整输出中提取【答案】部分
        
        Args:
            model_output: 模型完整输出
            
        Returns:
            提取的答案部分，如果提取失败返回完整输出
        """
        # 尝试提取【答案】标记后的内容
        answer_pattern = r'【答案】\s*(.*?)(?:$|\n\n【|\n\n<)'
        match = re.search(answer_pattern, model_output, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # 如果没有找到【答案】标记，返回完整输出
        return model_output


def main():
    """测试LLM Judge功能"""
    import os
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    # 初始化评估器
    judge = LLMJudge(
        api_url=os.getenv("API_URL", "https://api.example.com/v1/openai/native"),
        api_key=os.getenv("API_KEY"),
        model_name="deepseek-v32-meituan"
    )
    
    # 测试数据
    query = "推荐上海外滩附近的酒店"
    model_output = """【思考过程】
用户询问外滩附近的酒店，我需要使用RAG工具查询相关信息。

<rag>上海外滩附近的酒店</rag>

【答案】
为您推荐以下外滩附近的优质酒店：

🏨 **云栖雅舍民宿**
- 房型：落地窗家庭套房
- 价格：680元/晚
- 特色：两室一厅配厨房
- 地址：外滩街道中山东一路188号

🏨 **悦居雅舍精品民宿**
- 房型：家庭套房
- 价格：800元/晚
- 特色：步行10分钟到外滩
- 地址：北外滩惠民路128号

📞 预订方式：建议提前预订，可致电酒店前台或通过在线平台预订。"""
    
    ground_truth = "云栖雅舍民宿（680元）、悦居雅舍精品民宿（800元）"
    
    # 执行评估
    print("\n" + "="*80)
    print("🎯 LLM Judge 测试")
    print("="*80)
    print(f"\n查询: {query}")
    print(f"参考答案: {ground_truth}")
    print("\n开始评估...\n")
    
    result = judge.evaluate_all(
        query=query,
        model_output=model_output,
        ground_truth=ground_truth
    )
    
    # 打印结果
    print("\n" + "="*80)
    print("📊 评估结果")
    print("="*80)
    print(f"\n✅ 答案正确性: {result['correctness']['score']}/1")
    print(f"   理由: {result['correctness']['reason']}")
    print(f"\n✅ 输出完整性: {result['completeness']['score']}/1")
    print(f"   理由: {result['completeness']['reason']}")
    print(f"\n✅ 内容流畅度: {result['fluency']['score']}/1")
    print(f"   理由: {result['fluency']['reason']}")
    print(f"\n✅ 安全性: {result['safety']['score']}/1")
    print(f"   理由: {result['safety']['reason']}")
    print(f"\n🎯 总分: {result['total_score']}/{result['max_score']}")
    print("="*80)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
