#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Search Agent - 负责调用搜索API获取信息
"""

import requests
import json
import logging
from typing import Dict, Any, Optional, List
import urllib3
import subprocess
import os

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchAgent:
    """Web搜索代理，通过API调用搜索引擎"""
    
    def __init__(
        self, 
        api_url: str = "https://api.example.com/v1/friday/api/search",
        api_token: str = "xxxxxxxxxxxxxxxxxxxx",
        api_tokens: List[str] = None,
        use_token_manager: bool = False,
        search_engine: str = "search_pro",
        api_type: str = "baidu-search",
        use_mcp: bool = True,
        mcp_tool_name: str = "search_web",
        mcp_script_path: Optional[str] = None,
        mcp_server_url: str = "https://api.example.com/sse",
        node_path: Optional[str] = None
    ):
        """
        初始化Web搜索代理
        
        Args:
            api_url: 搜索API的URL
            api_token: API认证令牌（单token模式）
            api_tokens: API认证令牌列表（多token模式）
            use_token_manager: 是否使用token管理器进行分发
            search_engine: 搜索引擎类型
            api_type: API类型
        """
        self.api_url = api_url
        self.search_engine = search_engine
        self.api_type = api_type
        self.use_token_manager = use_token_manager
        self.use_mcp = use_mcp
        self.mcp_tool_name = mcp_tool_name
        self.mcp_server_url = mcp_server_url
        
        # Node.js 路径配置
        self.node_path = node_path or self._find_node_path()
        
        # 多token模式
        if use_token_manager and api_tokens:
            self.api_tokens = api_tokens
            self.api_token = None
            logger.info(f"✅ WebSearchAgent初始化完成（多token模式，{len(api_tokens)}个tokens）")
        else:
            # 单token模式
            self.api_token = api_token
            self.api_tokens = None
            logger.info(f"✅ WebSearchAgent初始化完成（单token模式）")
        
        logger.info(f"   API URL: {api_url}")
        logger.info(f"   MCP mode: {'ON' if use_mcp else 'OFF'} (tool: {mcp_tool_name})")
        if use_mcp:
            logger.info(f"   MCP server: {mcp_server_url}")

        # MCP 脚本路径，默认放在当前目录
        if self.use_mcp:
            if mcp_script_path:
                self.mcp_script_path = mcp_script_path
            else:
                self.mcp_script_path = os.path.join(
                    os.path.dirname(__file__),
                    "mcp_call_tool.js"
                )
            logger.info(f"   MCP script: {self.mcp_script_path}")
            
            # 检查 Node.js 是否可用
            if self.node_path:
                try:
                    result = subprocess.run(
                        [self.node_path, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        logger.info(f"   Node.js version: {result.stdout.strip()} (路径: {self.node_path})")
                    else:
                        logger.warning("⚠️  Node.js 检查失败，Web Search 功能可能无法使用")
                except Exception as e:
                    logger.warning(f"⚠️  检查 Node.js 时出错: {e}")
            else:
                logger.error("❌ 未找到 Node.js，Web Search 功能将无法使用")
                logger.error("   请安装 Node.js 或通过 node_path 参数指定路径")
    
    def _find_node_path(self) -> Optional[str]:
        """
        自动查找 Node.js 可执行文件路径（支持 Node.js 16+）
        
        Returns:
            Node.js 可执行文件的完整路径，如果未找到或不可用则返回 None
        """
        def _test_node_executable(node_path: str, min_version: int = 16) -> bool:
            """测试 Node.js 是否真的可以运行，并检查版本"""
            try:
                result = subprocess.run(
                    [node_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    return False
                # 检查版本号
                version_str = result.stdout.strip().lstrip('v')
                try:
                    major = int(version_str.split('.')[0])
                    return major >= min_version
                except:
                    return False
            except:
                return False
        
        # 1. 先检查 PATH 中是否有 node（要求版本 >= 16）
        try:
            result = subprocess.run(
                ["which", "node"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                node_path = result.stdout.strip()
                if os.path.exists(node_path) and _test_node_executable(node_path, min_version=16):
                    return node_path
        except:
            pass
        
        # 2. 检查常见的安装路径
        base_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/hehang03"
        common_paths = [
            # 检查用户提供的路径（包括 v16.20.2）
            f"{base_dir}/node-v16.20.2-linux-x64/bin/node",
            f"{base_dir}/node-v20.10.0-linux-x64/bin/node",
            f"{base_dir}/node-v18/bin/node",
            f"{base_dir}/node-v16/bin/node",
            f"{base_dir}/node/bin/node",
            # 检查 conda 环境
            "/usr/local/conda/envs/swift/bin/node",
            "/usr/local/conda/bin/node",
            "/usr/local/conda/envs/base/bin/node",
            os.path.expanduser("~/anaconda3/bin/node"),
            os.path.expanduser("~/miniconda3/bin/node"),
            os.path.expanduser("~/conda/bin/node"),
            "/opt/conda/bin/node",
            "/opt/miniconda3/bin/node",
            # 检查 nvm 安装（优先使用 v18+，但也允许 v16+）
            os.path.expanduser("~/.nvm/versions/node/v18.20.8/bin/node"),
            os.path.expanduser("~/.nvm/versions/node/v16.20.2/bin/node"),
            os.path.expanduser("~/.nvm/versions/node/v16.20.0/bin/node"),
            # 检查其他常见路径
            os.path.expanduser("~/node-v20.10.0-linux-x64/bin/node"),
            os.path.expanduser("~/node-v18.20.0-linux-x64/bin/node"),
            os.path.expanduser("~/node-v16.20.2-linux-x64/bin/node"),
            os.path.expanduser("~/node-v16.20.0-linux-x64/bin/node"),
            "/usr/local/bin/node",
            "/usr/bin/node",
        ]
        
        # 3. 动态搜索 base_dir 下的所有 node-* 目录
        if os.path.exists(base_dir):
            try:
                for item in os.listdir(base_dir):
                    if item.startswith("node-") and os.path.isdir(os.path.join(base_dir, item)):
                        node_path = os.path.join(base_dir, item, "bin", "node")
                        if node_path not in common_paths:
                            common_paths.append(node_path)
            except:
                pass
        
        # 4. 检查 nvm 目录下的所有版本
        nvm_base = os.path.expanduser("~/.nvm/versions/node")
        if os.path.exists(nvm_base):
            try:
                for item in os.listdir(nvm_base):
                    version_dir = os.path.join(nvm_base, item)
                    if os.path.isdir(version_dir):
                        node_path = os.path.join(version_dir, "bin", "node")
                        if node_path not in common_paths:
                            common_paths.append(node_path)
            except:
                pass
        
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                # 验证是否真的可以运行，且版本 >= 16
                if _test_node_executable(path, min_version=16):
                    return path
        
        return None
    
    def _get_current_token(self) -> str:
        """获取当前线程/worker应该使用的token"""
        if self.use_token_manager and self.api_tokens:
            # 使用ApiKeyManager分配token
            from llm_utils import ApiKeyManager
            return ApiKeyManager.get_assigned_api_key(self.api_tokens)
        else:
            # 使用单个token
            return self.api_token

    def _search_via_mcp(self, query: str, timeout: int = 60) -> Dict[str, Any]:
        """
        通过 MCP search_web 工具执行搜索
        """
        if not os.path.exists(self.mcp_script_path):
            return {
                "success": False,
                "query": query,
                "error": f"MCP脚本不存在: {self.mcp_script_path}",
                "context": ""
            }

        args_json = json.dumps({"query": query}, ensure_ascii=False)
        cmd = [
            self.node_path or "node",
            self.mcp_script_path,
            self.mcp_tool_name,
            args_json
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "query": query,
                "error": "MCP调用超时",
                "context": ""
            }
        except FileNotFoundError:
            return {
                "success": False,
                "query": query,
                "error": "未找到 node，请安装 Node.js 或检查 PATH",
                "context": ""
            }
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": f"MCP调用异常: {e}",
                "context": ""
            }

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0:
            err_msg = stderr if stderr else "MCP调用失败"
            return {
                "success": False,
                "query": query,
                "error": err_msg[:500],
                "context": ""
            }

        if not stdout:
            return {
                "success": False,
                "query": query,
                "error": "MCP返回为空",
                "context": ""
            }

        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "query": query,
                "error": f"解析MCP返回失败: {e}: {stdout[:200]}",
                "context": ""
            }

        if result.get("isError"):
            return {
                "success": False,
                "query": query,
                "error": result.get("error", "MCP返回错误"),
                "context": ""
            }

        # MCP search_web 预期返回 content 列表
        raw_content = result.get("content") or []
        text_parts = []
        if isinstance(raw_content, list):
            for item in raw_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text") or ""
                    if text:
                        text_parts.append(text)
        context = "\n".join(text_parts) if text_parts else json.dumps(result, ensure_ascii=False)

        return {
            "success": True,
            "query": query,
            "result": result,
            "context": context
        }
    
    def search(self, query: str, timeout: int = 60) -> Dict[str, Any]:
        """
        执行搜索查询
        
        Args:
            query: 搜索查询
            timeout: 请求超时时间（默认60秒）
            
        Returns:
            搜索结果字典
        """
        try:
            logger.info(f"🔍 执行Web搜索: {query}")

            if self.use_mcp:
                return self._search_via_mcp(query, timeout)
            
            # Fallback: 旧 HTTP 搜索
            # 获取当前token
            current_token = self._get_current_token()
            
            # 构建请求头
            headers = {
                'Authorization': f'Bearer {current_token}',
                'Content-Type': 'application/json;charset=UTF-8',
            }
            
            # 构建请求数据
            data = {
                "query": query,
                "api": self.api_type,
                "search_engine": self.search_engine
            }
            
            # 发送POST请求（禁用SSL验证以避免某些环境下的SSL错误）
            response = requests.post(
                self.api_url, 
                headers=headers, 
                data=json.dumps(data),
                timeout=timeout,
                verify=False  # 禁用SSL证书验证
            )
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Web搜索成功")
                
                # 安全提取上下文
                try:
                    context = self._extract_context(result)
                except Exception as e:
                    logger.warning(f"⚠️ 提取上下文失败: {e}")
                    context = json.dumps(result, ensure_ascii=False) if result else ""
                
                return {
                    "success": True,
                    "query": query,
                    "result": result,
                    "context": context
                }
            else:
                logger.error(f"❌ Web搜索失败，状态码: {response.status_code}")
                return {
                    "success": False,
                    "query": query,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "context": ""
                }
                
        except Exception as e:
            logger.error(f"❌ Web搜索异常: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "context": ""
            }
    
    def _extract_context(self, result: Any) -> str:
        """
        从搜索结果中提取上下文信息
        
        Args:
            result: 搜索API返回的结果
            
        Returns:
            提取的上下文字符串
        """
        try:
            # 处理None或空值
            if result is None:
                logger.debug("⚠️ 搜索结果为None")
                return ""
            
            # 记录原始结果类型（用于调试）
            logger.debug(f"📦 原始结果类型: {type(result)}")
            if isinstance(result, dict):
                logger.debug(f"📦 结果包含的键: {list(result.keys())[:10]}")  # 只显示前10个键
            
            # 如果结果是字典类型
            if isinstance(result, dict):
                # 尝试提取常见的搜索结果字段
                contexts = []
                
                # 提取data字段（可能包含实际结果）
                if "data" in result:
                    data = result.get("data")
                    if data is None:
                        logger.debug("⚠️ data字段为None")
                    elif isinstance(data, dict):
                        result = data
                        logger.debug(f"📦 使用data字段，包含键: {list(data.keys())[:10]}")
                
                # 提取answer字段
                if "answer" in result:
                    answer = result.get("answer")
                    if answer:
                        contexts.append(f"答案: {answer}")
                
                # 提取baiduSearchResults（官方百度搜索API返回格式）
                baidu_results = result.get("baiduSearchResults")
                if baidu_results is not None and isinstance(baidu_results, list):
                    for i, item in enumerate(baidu_results[:5]):  # 只取前5个结果
                        if item is not None and isinstance(item, dict):
                            title = item.get("title", "")
                            content = item.get("content", "")
                            url = item.get("url", "")
                            if title or content:
                                contexts.append(f"[结果{i+1}] {title}: {content}")
                
                # 提取snippets或results（通用格式）
                results_list = result.get("results")
                if results_list is not None and isinstance(results_list, list):
                    for i, item in enumerate(results_list[:5]):  # 只取前5个结果
                        if item is not None and isinstance(item, dict):
                            title = item.get("title", "")
                            snippet = item.get("snippet", item.get("content", ""))
                            if title or snippet:
                                contexts.append(f"[结果{i+1}] {title}: {snippet}")
                
                # 提取items字段（另一种常见格式）
                items_list = result.get("items")
                if items_list is not None and isinstance(items_list, list):
                    for i, item in enumerate(items_list[:5]):
                        if item is not None and isinstance(item, dict):
                            title = item.get("title", "")
                            snippet = item.get("snippet", item.get("description", ""))
                            if title or snippet:
                                contexts.append(f"[结果{i+1}] {title}: {snippet}")
                
                # 提取其他可能的字段
                content = result.get("content")
                if content:
                    contexts.append(f"内容: {content}")
                
                summary = result.get("summary")
                if summary:
                    contexts.append(f"摘要: {summary}")
                
                # 如果没有提取到任何内容，返回完整的JSON
                if contexts:
                    return "\n".join(contexts)
                else:
                    logger.debug("⚠️ 未能提取到任何已知字段，返回完整JSON")
                    return json.dumps(result, ensure_ascii=False)[:5000]  # 限制长度
            
            # 如果是字符串，直接返回
            elif isinstance(result, str):
                return result if result else ""
            
            # 其他类型转为JSON字符串
            else:
                return json.dumps(result, ensure_ascii=False)[:5000] if result else ""
                
        except Exception as e:
            logger.warning(f"⚠️ 提取上下文失败: {str(e)}")
            logger.debug(f"⚠️ 失败时的result类型: {type(result)}")
            
            # 尝试返回原始字符串
            try:
                return str(result)[:5000] if result else ""
            except:
                return ""


if __name__ == "__main__":
    # 测试Web搜索代理
    agent = WebSearchAgent()
    
    # 测试查询
    test_query = "上海最高的建筑是什么"
    result = agent.search(test_query)
    
    print(f"\n查询: {test_query}")
    print(f"成功: {result['success']}")
    print(f"上下文:\n{result['context']}")

# 兜底：如果在某些旧缓存环境下类缺少 _find_node_path，动态补上
if not hasattr(WebSearchAgent, "_find_node_path"):
    def _find_node_path(self) -> Optional[str]:
        """
        自动查找 Node.js 可执行文件路径（支持 Node.js 16+）
        """
        def _test_node_executable(node_path: str, min_version: int = 16) -> bool:
            try:
                result = subprocess.run(
                    [node_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    return False
                version_str = result.stdout.strip().lstrip('v')
                try:
                    major = int(version_str.split('.')[0])
                    return major >= min_version
                except:
                    return False
            except:
                return False

        try:
            result = subprocess.run(["which", "node"], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                node_path = result.stdout.strip()
                if os.path.exists(node_path) and _test_node_executable(node_path, 16):
                    return node_path
        except:
            pass

        base_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/hehang03"
        common_paths = [
            f"{base_dir}/node-v16.20.2-linux-x64/bin/node",
            f"{base_dir}/node-v20.10.0-linux-x64/bin/node",
            f"{base_dir}/node-v18/bin/node",
            f"{base_dir}/node-v16/bin/node",
            f"{base_dir}/node/bin/node",
            "/usr/local/conda/envs/swift/bin/node",
            "/usr/local/conda/bin/node",
            "/usr/local/conda/envs/base/bin/node",
            os.path.expanduser("~/anaconda3/bin/node"),
            os.path.expanduser("~/miniconda3/bin/node"),
            os.path.expanduser("~/conda/bin/node"),
            "/opt/conda/bin/node",
            "/opt/miniconda3/bin/node",
            os.path.expanduser("~/.nvm/versions/node/v20.10.0/bin/node"),
            os.path.expanduser("~/.nvm/versions/node/v18.20.8/bin/node"),
            os.path.expanduser("~/.nvm/versions/node/v16.20.2/bin/node"),
            os.path.expanduser("~/node-v20.10.0-linux-x64/bin/node"),
            os.path.expanduser("~/node-v18.20.0-linux-x64/bin/node"),
            os.path.expanduser("~/node-v16.20.2-linux-x64/bin/node"),
            os.path.expanduser("~/node-v16.20.0-linux-x64/bin/node"),
            "/usr/local/bin/node",
            "/usr/bin/node",
        ]

        if os.path.exists(base_dir):
            try:
                for item in os.listdir(base_dir):
                    if item.startswith("node-") and os.path.isdir(os.path.join(base_dir, item)):
                        node_path = os.path.join(base_dir, item, "bin", "node")
                        if node_path not in common_paths:
                            common_paths.append(node_path)
            except:
                pass

        nvm_base = os.path.expanduser("~/.nvm/versions/node")
        if os.path.exists(nvm_base):
            try:
                for item in os.listdir(nvm_base):
                    version_dir = os.path.join(nvm_base, item)
                    if os.path.isdir(version_dir):
                        node_path = os.path.join(version_dir, "bin", "node")
                        if node_path not in common_paths:
                            common_paths.append(node_path)
            except:
                pass

        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                if _test_node_executable(path, 16):
                    return path
        return None

    WebSearchAgent._find_node_path = _find_node_path


