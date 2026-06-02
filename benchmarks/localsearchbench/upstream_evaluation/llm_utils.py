#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM工具模块 - FridayClient封装器
支持多API key并行请求
"""

import os
import sys
import json
import time
import logging
import threading
import random
import yaml
from typing import List, Dict, Any, Tuple, Optional
from threading import Lock
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置管理
# ============================================================================

class ConfigManager:
    """配置管理器，负责从YAML文件加载配置"""
    
    def __init__(self, config_path: str = "config/model_rpm.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    return yaml.safe_load(file)
            else:
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                return self._get_default_config()
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'model_rpm': {'default': 20},
            'api_config': {
                'default_rpm': 20,
                'base_url': 'https://api.example.com/v1/openai/native',
                'api_keys': ['xxxxxxxxxxxxxxxxxxxx']  # 默认API key
            }
        }
    
    def get_api_keys(self) -> List[str]:
        """获取API keys列表"""
        return self.config.get('api_config', {}).get('api_keys', ['xxxxxxxxxxxxxxxxxxxx'])
    
    def get_default_rpm(self) -> int:
        """获取默认RPM"""
        return self.config.get('api_config', {}).get('default_rpm', 20)
    
    def get_base_url(self) -> str:
        """获取API基础URL"""
        return self.config.get('api_config', {}).get('base_url', 'https://api.example.com/v1/openai/native')
    
    def get_model_rpm(self, model_name: str) -> int:
        """获取指定模型的RPM限制"""
        model_rpm_config = self.config.get('model_rpm', {})
        return model_rpm_config.get(model_name, model_rpm_config.get('default', 20))


# 全局配置管理器实例
_config_manager = None

def get_config_manager(config_path: str = "config/model_rpm.yaml") -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


# ============================================================================
# API Key分配管理
# ============================================================================

class ApiKeyManager:
    """管理API key分配和速率限制"""
    _api_key_assignments = {}  # 线程到API key的映射
    _rate_limiters = {}  # API key到速率限制器的映射
    _assignment_lock = Lock()  # 分配锁
    _rate_limiter_lock = Lock()  # 速率限制锁
    _next_index = 0  # 下一个分配的API key索引
    
    @classmethod
    def get_assigned_api_key(cls, api_keys: List[str]) -> str:
        """为当前线程获取固定分配的API key"""
        if not api_keys:
            logger.warning("API keys列表为空，使用默认key")
            return "xxxxxxxxxxxxxxxxxxxx"
        
        # 使用线程ID和进程ID的组合作为标识符
        thread_id = threading.get_ident()
        process_id = os.getpid()
        worker_key = f"{process_id}_{thread_id}"
        
        # 如果当前worker还未分配API key，则分配一个
        if worker_key not in cls._api_key_assignments:
            with cls._assignment_lock:
                # 双重检查锁定
                if worker_key not in cls._api_key_assignments:
                    api_index = cls._next_index % len(api_keys)
                    assigned_api_key = api_keys[api_index]
                    cls._api_key_assignments[worker_key] = assigned_api_key
                    cls._next_index += 1
                    
                    logger.info(f"🔗 Worker {worker_key} 分配到 API key ...{assigned_api_key[-6:]} (索引 {api_index})")
        
        return cls._api_key_assignments[worker_key]
    
    @classmethod
    def rate_limit_api_key(cls, api_key: str, rpm_per_key: int = 20):
        """对指定的API key进行速率限制"""
        if not api_key:
            return
        
        with cls._rate_limiter_lock:
            if api_key not in cls._rate_limiters:
                cls._rate_limiters[api_key] = {
                    'last_request_time': 0,
                    'rpm_per_key': rpm_per_key
                }
            
            rate_limiter = cls._rate_limiters[api_key]
            now = time.time()
            last_request_time = rate_limiter['last_request_time']
            rpm_per_key = rate_limiter['rpm_per_key']
            
            # 计算距离上次请求的时间间隔
            time_since_last = now - last_request_time
            
            # 计算最小请求间隔（秒）
            min_interval = 60.0 / rpm_per_key
            
            if time_since_last < min_interval:
                # 需要等待的时间
                wait_time = min_interval - time_since_last
                logger.debug(f"⏱️ API key ...{api_key[-6:]} 速率限制，等待 {wait_time:.2f} 秒")
                
                # 更新最后请求时间
                rate_limiter['last_request_time'] = now + wait_time
                
                # 等待
                time.sleep(wait_time)
            else:
                # 可以直接执行，更新最后请求时间
                rate_limiter['last_request_time'] = now


# ============================================================================
# Friday Client
# ============================================================================

class FridayClient:
    """Friday LLM客户端 - 支持deepseek-v31-meituan等模型，支持多API key并行"""
    
    def __init__(
        self,
        model_name: str = "deepseek-v31-meituan",
        api_url: str = "https://api.example.com/v1/openai/native",
        api_token: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        timeout: int = 300,
        max_retries: int = 5,
        rpm: Optional[int] = None,
        config_path: str = "config/model_rpm.yaml",
        use_api_key_manager: bool = True
    ):
        """
        初始化Friday客户端
        
        Args:
            model_name: 模型名称
            api_url: API URL
            api_token: API令牌（可选，如果不提供则从配置文件加载）
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 请求超时时间
            max_retries: 最大重试次数
            rpm: 每分钟请求数限制（可选，如果不提供则从配置文件加载）
            config_path: 配置文件路径
            use_api_key_manager: 是否使用API key管理器（并行场景建议开启）
        """
        self.model_name = model_name
        self.config_path = config_path
        self.use_api_key_manager = use_api_key_manager
        
        # 加载配置
        self.config_manager = get_config_manager(config_path)
        
        # 设置API URL
        if not api_url.endswith('/chat/completions'):
            self.api_url = f"{api_url}/chat/completions"
        else:
            self.api_url = api_url
        
        # 设置API token
        if api_token:
            self.api_token = api_token
            self.api_keys = [api_token]  # 单个token模式
        else:
            # 从配置文件加载API keys
            self.api_keys = self.config_manager.get_api_keys()
            self.api_token = self.api_keys[0] if self.api_keys else "xxxxxxxxxxxxxxxxxxxx"
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 设置RPM限制
        if rpm is not None:
            self.rpm = rpm
        else:
            self.rpm = self.config_manager.get_model_rpm(model_name)
        
        # RPM限制相关（用于非API key管理器模式）
        self.last_request_time = 0
        self.request_interval = 60.0 / self.rpm if self.rpm else 0
        self._api_idx = 0  # 轮询选择 key 时的指针
        
        logger.info(f"✅ FridayClient初始化完成: {model_name}")
        logger.info(f"   - API keys数量: {len(self.api_keys)}")
        logger.info(f"   - RPM限制: {self.rpm}")
        logger.info(f"   - 使用API key管理器: {use_api_key_manager}")
    
    def single_request(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, float, Dict[str, int]]:
        """
        发送单次请求
        
        Args:
            messages: 消息列表，格式: [{"role": "user", "content": "..."}]
            temperature: 温度参数（可选，覆盖默认值）
            max_tokens: 最大token数（可选，覆盖默认值）
        
        Returns:
            (response_text, cost_time, token_info)
            - response_text: LLM响应文本
            - cost_time: 请求耗时（秒）
            - token_info: token使用信息，包含 prompt_tokens, completion_tokens, total_tokens
        """
        start_time = time.time()
        
        # 确定使用哪个API key
        if self.use_api_key_manager and len(self.api_keys) > 1:
            # 使用API key管理器分配key（并行场景）
            current_api_key = ApiKeyManager.get_assigned_api_key(self.api_keys)
            ApiKeyManager.rate_limit_api_key(current_api_key, self.rpm)
        else:
            # 使用固定的API key（单线程场景）
            current_api_key = self.api_token
            # RPM限制
            if self.request_interval > 0:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.request_interval:
                    sleep_time = self.request_interval - elapsed
                    time.sleep(sleep_time)
        
        # 构建请求
        headers = {
            "Authorization": f"Bearer {current_api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建payload（使用简化格式，避免不支持的参数导致服务器错误）
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": False,
            # "thinking": {"type": "disabled"}  # 关闭 thinking 模式
        }
        
        # 重试逻辑：优先在同一轮尝试其他可用 key；若全部限流则等待后继续，直到成功或遇到非限流错误累计超限
        last_error = None
        non_rate_limit_failures = 0
        while True:
            tried_keys = set()
            rate_limited_all = False
            while True:
                current_api_key = self._get_next_api_key(tried_keys)
                if not current_api_key:
                    rate_limited_all = True
                    break
                tried_keys.add(current_api_key)

                # 针对当前 key 做节流
                if self.use_api_key_manager and len(self.api_keys) > 1:
                    ApiKeyManager.rate_limit_api_key(current_api_key, self.rpm)
                else:
                    if self.request_interval > 0:
                        elapsed = time.time() - self.last_request_time
                        if elapsed < self.request_interval:
                            time.sleep(self.request_interval - elapsed)

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
                    
                    token_info = {
                        "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": result.get("usage", {}).get("total_tokens", 0)
                    }
                    
                    cost_time = time.time() - start_time
                    self.last_request_time = time.time()
                    return response_text, cost_time, token_info
                    
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

                    # 非限流错误计数
                    non_rate_limit_failures += 1
                    logger.warning(
                        f"请求失败 (非限流，第{non_rate_limit_failures}/{self.max_retries}次): {error_detail}"
                    )
                    if non_rate_limit_failures >= self.max_retries:
                        logger.error(f"所有重试都失败: {last_error}")
                        raise Exception(f"LLM请求失败: {last_error}")
                    break  # 跳出内层，进入退避

            # 如果本轮所有 key 都限流，等待后继续（不限轮次，直到成功或非限流错误超限）
            # 增加退避时间：指数退避从2^4增加到2^6，并增加基础等待时间
            base_wait = 4.0  # 基础等待时间3秒
            exponential_wait = (2 ** min(non_rate_limit_failures, 6))  # 最大指数从4增加到6（64秒）
            sleep_time = base_wait + exponential_wait + random.uniform(0, 2)
            logger.info(
                f"所有可用key均限流，等待 {sleep_time:.2f} 秒后继续尝试"
            )
            time.sleep(sleep_time)

    def _get_next_api_key(self, tried_keys: set) -> Optional[str]:
        """轮询获取一个未尝试过的API key；若无则返回None。"""
        if not self.api_keys:
            return self.api_token
        for _ in range(len(self.api_keys)):
            key = self.api_keys[self._api_idx % len(self.api_keys)]
            self._api_idx += 1
            if key not in tried_keys:
                return key
        return None

    def _is_rate_limit_error(self, message: str) -> bool:
        """
        检测是否为限流相关错误
        """
        if not message:
            return False
        lowered = message.lower()
        return any(
            keyword in lowered
            for keyword in [
                "429",
                "rate limit",
                "too many requests",
                "达到使用量上限",
                "每分钟请求次数超过限制",
                "请求次数超过限制"
            ]
        )
    
    def batch_request(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[Tuple[str, float, Dict[str, int]]]:
        """
        批量请求
        
        Args:
            messages_list: 多个消息列表
            temperature: 温度参数
            max_tokens: 最大token数
        
        Returns:
            List of (response_text, cost_time, token_info)
        """
        results = []
        for messages in messages_list:
            result = self.single_request(messages, temperature, max_tokens)
            results.append(result)
        return results


def get_friday_client(
    model_name: str = "deepseek-v31-meituan",
    rpm: Optional[int] = None,
    api_url: str = "https://api.example.com/v1/openai/native",
    api_token: Optional[str] = None,
    config_path: str = "config/model_rpm.yaml",
    use_api_key_manager: bool = True
) -> FridayClient:
    """
    获取Friday客户端实例
    
    Args:
        model_name: 模型名称
        rpm: 每分钟请求数限制（可选，从配置文件读取）
        api_url: API URL
        api_token: API令牌（可选，从配置文件读取）
        config_path: 配置文件路径
        use_api_key_manager: 是否使用API key管理器（并行场景建议开启）
    
    Returns:
        FridayClient实例
    """
    return FridayClient(
        model_name=model_name,
        api_url=api_url,
        api_token=api_token,
        rpm=rpm,
        config_path=config_path,
        use_api_key_manager=use_api_key_manager
    )


def get_multiple_friday_clients(
    model_name: str = "deepseek-v31-meituan",
    num_clients: int = 3,
    rpm_per_client: Optional[int] = None,
    api_url: str = "https://api.example.com/v1/openai/native",
    config_path: str = "config/model_rpm.yaml"
) -> List[FridayClient]:
    """
    获取多个Friday客户端实例（用于并行请求）
    每个客户端会自动从API key池中分配不同的key
    
    Args:
        model_name: 模型名称
        num_clients: 客户端数量
        rpm_per_client: 每个客户端的RPM限制（可选，从配置文件读取）
        api_url: API URL
        config_path: 配置文件路径
    
    Returns:
        FridayClient实例列表
    """
    clients = []
    for i in range(num_clients):
        client = FridayClient(
            model_name=model_name,
            api_url=api_url,
            api_token=None,  # 从配置文件读取
            rpm=rpm_per_client,
            config_path=config_path,
            use_api_key_manager=True  # 启用API key管理器
        )
        clients.append(client)
    return clients


__all__ = [
    "FridayClient", 
    "get_friday_client", 
    "get_multiple_friday_clients",
    "ConfigManager",
    "ApiKeyManager",
    "get_config_manager"
]


# 测试代码
if __name__ == "__main__":
    # 测试FridayClient
    client = get_friday_client("deepseek-v31-meituan")
    
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "1+1等于几？"}
    ]
    
    try:
        response, cost_time, token_info = client.single_request(messages)
        print(f"响应: {response}")
        print(f"耗时: {cost_time:.2f}秒")
        print(f"Token使用: {token_info}")
    except Exception as e:
        print(f"测试失败: {e}")
