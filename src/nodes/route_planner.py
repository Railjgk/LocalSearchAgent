# src/nodes/route_planner.py
import os
import requests
from typing import Dict, Any, Optional, Tuple
from urllib.parse import quote


class RoutePlanner:
    """B 模块：路径规划，验证两个 POI 之间的时间可行性"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GAODE_API_KEY", "自己的key")
        # 支持多种交通方式
        self.base_urls = {
            "driving": "https://restapi.amap.com/v3/direction/driving",      # 驾车
            "transit": "https://restapi.amap.com/v3/direction/transit/integrated",  # 公交
            "walking": "https://restapi.amap.com/v3/direction/walking",     # 步行
            "bicycling": "https://restapi.amap.com/v3/direction/bicycling"  # 骑行
        }
    
    def plan(
        self,
        origin: str,
        destination: str,
        mode: str = "transit",
        city: Optional[str] = None,
        cityd: Optional[str] = None,
        strategy: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        规划两点之间的路线
        
        :param origin: 起点，格式 "经度,纬度" 或 "名称,城市"
        :param destination: 终点，格式同上
        :param mode: 交通方式，driving/transit/walking/bicycling
        :param city: 起点城市（公交必填）
        :param cityd: 终点城市（跨城时填）
        :param strategy: 策略，如 0=最快，2=少换乘，5=不坐地铁
        :return: 包含时间、距离、路线详情
        """
        if mode not in self.base_urls:
            raise ValueError(f"不支持的交通方式: {mode}，可选: {list(self.base_urls.keys())}")
        
        url = self.base_urls[mode]
        
        params = {
            "key": self.api_key,
            "origin": origin,
            "destination": destination,
            "output": "JSON"
        }
        
        # 公交需要城市参数
        if mode == "transit":
            if not city:
                raise ValueError("公交模式必须提供 city 参数")
            params["city"] = city
            if cityd:
                params["cityd"] = cityd
            if strategy is not None:
                params["strategy"] = strategy
        
        # 驾车策略
        elif mode == "driving" and strategy is not None:
            params["strategy"] = strategy
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1":
                raise RuntimeError(f"高德 API 错误: {data.get('info', '未知错误')}")
            
            # 解析不同模式的返回结构
            if mode == "transit":
                return self._parse_transit(data)
            elif mode == "driving":
                return self._parse_driving(data)
            else:
                return self._parse_walking_bicycling(data, mode)
                
        except requests.exceptions.Timeout:
            raise RuntimeError("请求超时，请检查网络")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败: {e}")
    
    def _parse_transit(self, data: Dict) -> Dict[str, Any]:
        """解析公交路线"""
        route = data.get("route", {})
        transits = route.get("transits", [])
        
        if not transits:
            return {"feasible": False, "reason": "未找到公交路线"}
        
        # 取第一条推荐路线
        best = transits[0]
        
        return {
            "feasible": True,
            "mode": "transit",
            "duration": int(best.get("duration", 0)),  # 秒
            "duration_text": self._format_duration(int(best.get("duration", 0))),
            "distance": int(best.get("distance", 0)),  # 米
            "distance_text": self._format_distance(int(best.get("distance", 0))),
            "cost": best.get("cost", {}),  # 费用
            "segments": len(best.get("segments", [])),  # 换乘段数
            "nightflag": best.get("nightflag", "0") == "1",  # 是否夜间公交
            "walking_distance": int(best.get("walking_distance", 0)),
            "raw": best
        }
    
    def _parse_driving(self, data: Dict) -> Dict[str, Any]:
        """解析驾车路线"""
        route = data.get("route", {})
        paths = route.get("paths", [])
        
        if not paths:
            return {"feasible": False, "reason": "未找到驾车路线"}
        
        best = paths[0]
        
        return {
            "feasible": True,
            "mode": "driving",
            "duration": int(best.get("duration", 0)),
            "duration_text": self._format_duration(int(best.get("duration", 0))),
            "distance": int(best.get("distance", 0)),
            "distance_text": self._format_distance(int(best.get("distance", 0))),
            "tolls": int(best.get("tolls", 0)),  # 过路费（元）
            "traffic_lights": int(best.get("traffic_lights", 0)),
            "strategy": best.get("strategy"),
            "steps": len(best.get("steps", [])),
            "raw": best
        }
    
    def _parse_walking_bicycling(self, data: Dict, mode: str) -> Dict[str, Any]:
        """解析步行/骑行路线"""
        route = data.get("route", {})
        paths = route.get("paths", [])
        
        if not paths:
            return {"feasible": False, "reason": f"未找到{mode}路线"}
        
        best = paths[0]
        
        return {
            "feasible": True,
            "mode": mode,
            "duration": int(best.get("duration", 0)),
            "duration_text": self._format_duration(int(best.get("duration", 0))),
            "distance": int(best.get("distance", 0)),
            "distance_text": self._format_distance(int(best.get("distance", 0))),
            "steps": len(best.get("steps", [])),
            "raw": best
        }
    
    def check_time_feasible(
        self,
        origin: str,
        destination: str,
        max_duration_minutes: int,
        mode: str = "transit",
        city: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        检查时间是否可行（核心功能：验证两个 POI 之间的时间可行性）
        
        :param max_duration_minutes: 最大可接受时间（分钟）
        :return: (是否可行, 路线详情)
        """
        try:
            result = self.plan(origin, destination, mode=mode, city=city)
            
            if not result.get("feasible"):
                return False, result
            
            duration_minutes = result["duration"] / 60
            
            feasible = duration_minutes <= max_duration_minutes
            
            return feasible, {
                **result,
                "max_duration": max_duration_minutes,
                "actual_duration_minutes": round(duration_minutes, 1),
                "within_limit": feasible
            }
            
        except Exception as e:
            return False, {"feasible": False, "reason": str(e)}
    
    @staticmethod
    def _format_duration(seconds: int) -> str:
        """格式化时长"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        
        if hours > 0:
            return f"{hours}小时{minutes}分钟"
        return f"{minutes}分钟"
    
    @staticmethod
    def _format_distance(meters: int) -> str:
        """格式化距离"""
        if meters >= 1000:
            return f"{meters / 1000:.1f}公里"
        return f"{meters}米"

