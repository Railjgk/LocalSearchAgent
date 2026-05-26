import os
import requests
from typing import Dict, Any, Optional


class IPLocator:
    """A 模块：通过 IP 定位识别用户城市"""
    
    def __init__(self, api_key: str = None):
        # 优先从环境变量读取，否则用默认值（开发测试用）
        self.api_key = api_key or os.getenv("GAODE_API_KEY", "自己的key")
        self.base_url = "https://restapi.amap.com/v3/ip"
    
    def locate(self, ip: Optional[str] = None) -> Dict[str, Any]:
        """
        根据 IP 获取城市信息
        :param ip: 用户 IP，不传则定位当前网络出口
        :return: {"status": "1", "info": "OK", "province": "北京", "city": "北京市", "adcode": "110000"}
        """
        params = {
            "key": self.api_key,
            "output": "JSON"
        }
        if ip:
            params["ip"] = ip
        
        try:
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1":
                raise RuntimeError(f"高德 API 错误: {data.get('info', '未知错误')}")
            
            return {
                "province": data.get("province"),
                "city": data.get("city"),
                "adcode": data.get("adcode"),
                "rectangle": data.get("rectangle"),  # 经纬度范围
                "raw": data  # 保留原始数据备用
            }
            
        except requests.exceptions.Timeout:
            raise RuntimeError("请求超时，请检查网络")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败: {e}")
