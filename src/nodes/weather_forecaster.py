import os
import requests
from typing import Dict, Any, Optional, List


class WeatherForecaster:
    """天气查询：根据城市编码或坐标获取实时天气和预报"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GAODE_API_KEY", "自己的key")
        self.base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
    
    def get_weather(
        self,
        city: str,
        extensions: str = "base"
    ) -> Dict[str, Any]:
        """
        获取天气信息
        
        :param city: 城市编码（adcode），如 "110000"（北京）
        :param extensions: base=实时天气，all=实时+预报
        :return: 天气数据
        """
        params = {
            "key": self.api_key,
            "city": city,
            "extensions": extensions,
            "output": "JSON"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1":
                raise RuntimeError(f"高德 API 错误: {data.get('info', '未知错误')}")
            
            lives = data.get("lives", [])
            if not lives:
                return {"feasible": False, "reason": "未获取到天气数据"}
            
            current = lives[0]
            
            result = {
                "feasible": True,
                "city": current.get("city"),
                "adcode": current.get("adcode"),
                "weather": current.get("weather"),          # 天气现象，如"晴"
                "temperature": current.get("temperature"),   # 实时温度
                "winddirection": current.get("winddirection"), # 风向
                "windpower": current.get("windpower"),      # 风力
                "humidity": current.get("humidity"),        # 湿度
                "reporttime": current.get("reporttime"),    # 数据发布时间
            }
            
            # 如果有预报数据
            if extensions == "all":
                forecasts = data.get("forecasts", [])
                if forecasts:
                    forecast_data = forecasts[0]
                    casts = forecast_data.get("casts", [])
                    result["forecast"] = [
                        {
                            "date": c.get("date"),
                            "week": c.get("week"),
                            "dayweather": c.get("dayweather"),
                            "nightweather": c.get("nightweather"),
                            "daytemp": c.get("daytemp"),
                            "nighttemp": c.get("nighttemp"),
                            "daywind": c.get("daywind"),
                            "nightwind": c.get("nightwind"),
                            "daypower": c.get("daypower"),
                            "nightpower": c.get("nightpower"),
                        }
                        for c in casts
                    ]
            
            return result
            
        except requests.exceptions.Timeout:
            raise RuntimeError("请求超时，请检查网络")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败: {e}")
    
    def get_forecast(self, city: str) -> List[Dict[str, Any]]:
        """
        获取未来天气预报（简化接口）
        """
        result = self.get_weather(city, extensions="all")
        return result.get("forecast", [])
    
    def is_rainy(self, city: str) -> bool:
        """
        快速判断当前是否下雨
        """
        result = self.get_weather(city, extensions="base")
        weather = result.get("weather", "")
        return any(kw in weather for kw in ["雨", "雪", "雹", "霰"])


