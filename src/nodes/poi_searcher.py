# src/nodes/poi_searcher.py
import os
import requests
from typing import Dict, Any, List, Optional


class POISearcher:
    """B 模块：根据关键词搜索真实 POI（兴趣点）"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GAODE_API_KEY", "自己的key")
        self.base_url = "https://restapi.amap.com/v5/place/text"
    
    def search(
        self,
        keywords: str,
        city: Optional[str] = None,
        citylimit: bool = True,
        page: int = 1,
        offset: int = 10
    ) -> List[Dict[str, Any]]:
        """
        根据关键词搜索 POI
        
        :param keywords: 搜索关键词，如"肯德基"、"北京大学"
        :param city: 城市名或 adcode，如"北京"或"110000"
        :param citylimit: 是否强制城市内搜索
        :param page: 页码，默认第1页
        :param offset: 每页条数，默认10条（最大25）
        :return: POI 列表，包含名称、地址、经纬度等信息
        """
        params = {
            "key": self.api_key,
            "keywords": keywords,
            "output": "JSON",
            "page": page,
            "offset": min(offset, 25)  # 高德限制最大25
        }
        
        if city:
            params["city"] = city
            params["citylimit"] = "true" if citylimit else "false"
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1":
                raise RuntimeError(f"高德 API 错误: {data.get('info', '未知错误')}")
            
            pois = data.get("pois", [])
            if not pois:
                return []
            
            # 提取关键字段，简化返回结构
            results = []
            for poi in pois:
                results.append({
                    "id": poi.get("id"),
                    "name": poi.get("name"),
                    "type": poi.get("type"),
                    "address": poi.get("address"),
                    "location": poi.get("location"),  # "经度,纬度"
                    "tel": poi.get("tel"),
                    "pcode": poi.get("pcode"),      # 省份编码
                    "citycode": poi.get("citycode"), # 城市编码
                    "adcode": poi.get("adcode"),    # 区域编码
                    "distance": poi.get("distance"),  # 距离（需要传入中心点）
                    "biz_ext": poi.get("biz_ext", {}), # 扩展信息（评分、价格等）
                    "raw": poi  # 保留完整原始数据
                })
            
            return results
            
        except requests.exceptions.Timeout:
            raise RuntimeError("请求超时，请检查网络")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败: {e}")
    
    def search_single(self, keywords: str, city: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        搜索并返回第一个最匹配的结果
        """
        results = self.search(keywords, city, offset=1)
        return results[0] if results else None


