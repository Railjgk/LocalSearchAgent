"""
Mock API Layer - 模拟真实美团接口
符合B模块的API规范要求
"""

import time
import math
import random
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime


DEFAULT_LATITUDE = 31.308
DEFAULT_LONGITUDE = 121.508


def _estimate_distance_km(
    info: Dict[str, Any],
    latitude: float = None,
    longitude: float = None,
) -> float:
    """Estimate a stable demo distance from the user's origin to a POI."""

    origin_latitude = DEFAULT_LATITUDE if latitude is None else latitude
    origin_longitude = DEFAULT_LONGITUDE if longitude is None else longitude
    poi_latitude = info.get("latitude", origin_latitude)
    poi_longitude = info.get("longitude", origin_longitude)

    lat_delta_km = (poi_latitude - origin_latitude) * 111.0
    lon_delta_km = (
        (poi_longitude - origin_longitude)
        * 111.0
        * math.cos(math.radians(origin_latitude))
    )
    distance = math.hypot(lat_delta_km, lon_delta_km)
    return round(max(0.5, distance), 1)

# ========== 1. 内存数据库（模拟真实库存）==========

class MockDatabase:
    """模拟真实业务数据库"""

    def __init__(self):
        # POI基础信息库
        self.pois = {
            # 活动类
            "act_001": {
                "name": "亲子陶艺体验馆",
                "type": "activity",
                "tags": ["kid_friendly", "indoor", "low_intensity", "creative"],
                "price": 198,
                "duration_min": 90,
                "rating": 4.7,
                "location": "杨浦区大学路",
                "latitude": 31.308,
                "longitude": 121.508,
                "time_slots": ["14:00", "15:30", "17:00"],
                "capacity_per_slot": 20
            },
            "act_002": {
                "name": "欢乐谷亲子乐园",
                "type": "activity",
                "tags": ["kid_friendly", "outdoor", "high_intensity", "amusement"],
                "price": 150,
                "duration_min": 120,
                "rating": 4.5,
                "location": "松江区林湖路",
                "latitude": 31.088,
                "longitude": 121.328,
                "time_slots": ["10:00", "13:00", "15:00"],
                "capacity_per_slot": 100
            },
            "act_003": {
                "name": "上海自然博物馆",
                "type": "activity",
                "tags": ["kid_friendly", "indoor", "educational", "low_intensity"],
                "price": 30,
                "duration_min": 120,
                "rating": 4.8,
                "location": "静安区北京西路",
                "latitude": 31.233,
                "longitude": 121.458,
                "time_slots": ["09:00", "11:00", "13:00", "15:00"],
                "capacity_per_slot": 500
            },

            # 餐厅类
            "res_001": {
                "name": "轻食日料餐厅",
                "type": "restaurant",
                "tags": ["low_calorie", "light_food", "family_friendly", "japanese"],
                "price": 220,
                "duration_min": 90,
                "rating": 4.6,
                "location": "五角场商圈",
                "latitude": 31.308,
                "longitude": 121.518,
                "time_slots": ["11:30", "12:30", "13:30", "17:30", "18:30", "19:30"],
                "capacity_per_slot": 30
            },
            "res_002": {
                "name": "绿色有机轻食馆",
                "type": "restaurant",
                "tags": ["low_calorie", "organic", "vegan_friendly", "healthy"],
                "price": 180,
                "duration_min": 60,
                "rating": 4.4,
                "location": "徐汇区衡山路",
                "latitude": 31.198,
                "longitude": 121.448,
                "time_slots": ["11:00", "12:00", "13:00", "17:00", "18:00", "19:00"],
                "capacity_per_slot": 25
            },
            "res_003": {
                "name": "海底捞火锅",
                "type": "restaurant",
                "tags": ["hotpot", "family_friendly", "service_good"],
                "price": 150,
                "duration_min": 120,
                "rating": 4.9,
                "location": "黄浦区南京东路",
                "latitude": 31.238,
                "longitude": 121.478,
                "time_slots": ["11:00", "12:00", "13:00", "17:00", "18:00", "19:00", "20:00"],
                "capacity_per_slot": 50
            }
        }

        # 库存状态：{poi_id: {time_slot: remaining_capacity}}
        self.inventory = {}
        # 订单记录：{order_id: order_info}
        self.orders = {}
        # 支付记录：{payment_id: payment_info}
        self.payments = {}
        # 排队记录：{poi_id: {time_slot: [queue_entries]}}
        self.queues = {}

        # 初始化库存
        self._init_inventory()

    def _init_inventory(self):
        """初始化所有POI的库存"""
        for poi_id, info in self.pois.items():
            self.inventory[poi_id] = {}
            for slot in info["time_slots"]:
                self.inventory[poi_id][slot] = info["capacity_per_slot"]
            self.queues[poi_id] = {}

    def get_poi(self, poi_id: str) -> Optional[Dict]:
        """获取POI基础信息"""
        return self.pois.get(poi_id)

    def check_availability(self, poi_id: str, time_slot: str) -> Dict:
        """检查库存"""
        if poi_id not in self.inventory:
            return {"available": False, "reason": "poi_not_found"}

        if time_slot not in self.inventory[poi_id]:
            return {"available": False, "reason": "invalid_time_slot"}

        remaining = self.inventory[poi_id][time_slot]
        queue_length = len(self.queues.get(poi_id, {}).get(time_slot, []))

        return {
            "available": remaining > 0,
            "remaining": remaining,
            "queue_length": queue_length,
            "capacity_total": self.pois[poi_id]["capacity_per_slot"]
        }

    def reserve(self, poi_id: str, time_slot: str, quantity: int) -> Dict:
        """预订（扣减库存）"""
        # 检查库存
        avail = self.check_availability(poi_id, time_slot)

        if not avail["available"]:
            return {
                "success": False,
                "error": "slot_full",
                "message": f"{time_slot} 时段已满",
                "queue_length": avail["queue_length"]
            }

        if self.inventory[poi_id][time_slot] < quantity:
            return {
                "success": False,
                "error": "insufficient_capacity",
                "message": f"库存不足，剩余{self.inventory[poi_id][time_slot]}个位"
            }

        # 扣减库存
        self.inventory[poi_id][time_slot] -= quantity
        order_id = f"{poi_id}_{time_slot}_{uuid.uuid4().hex[:8]}"

        self.orders[order_id] = {
            "poi_id": poi_id,
            "time_slot": time_slot,
            "quantity": quantity,
            "status": "confirmed",
            "payment_status": "unpaid",
            "created_at": time.time()
        }

        return {
            "success": True,
            "order_id": order_id,
            "remaining": self.inventory[poi_id][time_slot],
            "message": f"预订成功！剩余{self.inventory[poi_id][time_slot]}个位"
        }

    def add_to_queue(self, poi_id: str, time_slot: str, people: int) -> Dict:
        """加入排队"""
        if poi_id not in self.queues:
            self.queues[poi_id] = {}
        if time_slot not in self.queues[poi_id]:
            self.queues[poi_id][time_slot] = []

        queue_number = len(self.queues[poi_id][time_slot]) + 1
        queue_entry = {
            "queue_number": queue_number,
            "people": people,
            "joined_at": time.time()
        }
        self.queues[poi_id][time_slot].append(queue_entry)

        # 预估等待时间：每桌20分钟
        estimated_wait = queue_number * 20

        return {
            "success": True,
            "queue_number": queue_number,
            "estimated_wait_minutes": estimated_wait,
            "ahead_count": queue_number - 1
        }

    def cancel_order(self, order_id: str) -> Dict:
        """取消订单，释放库存"""
        if order_id not in self.orders:
            return {"success": False, "error": "order_not_found"}

        order = self.orders[order_id]
        if order["status"] != "confirmed":
            return {"success": False, "error": "order_cannot_be_cancelled"}

        # 释放库存
        poi_id = order["poi_id"]
        time_slot = order["time_slot"]
        quantity = order["quantity"]

        if poi_id in self.inventory and time_slot in self.inventory[poi_id]:
            self.inventory[poi_id][time_slot] += quantity

        order["status"] = "cancelled"

        return {
            "success": True,
            "message": f"已取消订单{order_id}，库存已释放"
        }

    def pay_order(self, order_id: str, amount: float, method: str = "mock_pay") -> Dict:
        """模拟支付订单，记录支付流水。"""
        if order_id not in self.orders:
            return {
                "success": False,
                "error": "order_not_found",
                "message": f"未找到订单{order_id}"
            }

        order = self.orders[order_id]
        if order.get("status") == "cancelled":
            return {
                "success": False,
                "error": "order_cancelled",
                "message": f"订单{order_id}已取消，无法支付"
            }

        if order.get("payment_status") == "paid":
            return {
                "success": True,
                "payment_id": order.get("payment_id"),
                "order_id": order_id,
                "amount": order.get("paid_amount", amount),
                "payment_status": "paid",
                "message": "订单已支付"
            }

        payment_id = f"PAY_{uuid.uuid4().hex[:10]}"
        paid_at = time.time()

        order["payment_status"] = "paid"
        order["payment_id"] = payment_id
        order["paid_amount"] = amount
        order["paid_at"] = paid_at
        order["payment_method"] = method

        self.payments[payment_id] = {
            "payment_id": payment_id,
            "order_id": order_id,
            "amount": amount,
            "method": method,
            "status": "paid",
            "paid_at": paid_at
        }

        return {
            "success": True,
            "payment_id": payment_id,
            "order_id": order_id,
            "amount": amount,
            "payment_status": "paid",
            "message": "支付成功"
        }


# 全局数据库实例
db = MockDatabase()


# ========== 2. API 接口实现 ==========

def search_activities(
    latitude: float = None,
    longitude: float = None,
    radius: int = 5000,
    kid_friendly: bool = False,
    low_intensity: bool = False,
    indoor: bool = False,
    limit: int = 10
) -> Dict[str, Any]:
    """
    搜索活动
    符合B的/api/activities/search规范
    """
    time.sleep(0.1)  # 模拟网络延迟

    results = []

    for poi_id, info in db.pois.items():
        if info["type"] != "activity":
            continue

        # 标签筛选
        tags = info["tags"]
        if kid_friendly and "kid_friendly" not in tags:
            continue
        if low_intensity and "low_intensity" not in tags:
            continue
        if indoor and "indoor" not in tags:
            continue

        # 计算距离（模拟），保持稳定，避免同一 demo 输入随机失败。
        distance_km = _estimate_distance_km(info, latitude, longitude)
        if distance_km > radius / 1000:
            continue

        # 获取可用时段
        available_slots = []
        for slot in info["time_slots"]:
            avail = db.check_availability(poi_id, slot)
            if avail["available"]:
                available_slots.append({
                    "time": slot,
                    "remaining": avail["remaining"]
                })

        if not available_slots:
            continue

        results.append({
            "poi_id": poi_id,
            "name": info["name"],
            "type": info["type"],
            "tags": info["tags"],
            "price": info["price"],
            "distance_km": round(distance_km, 1),
            "duration_min": info["duration_min"],
            "rating": info["rating"],
            "queue_time_min": 0,
            "available": len(available_slots) > 0,
            "available_slots": available_slots,
            "location": info["location"]
        })

    # 按评分排序
    results.sort(key=lambda x: x["rating"], reverse=True)

    return {
        "items": results[:limit],
        "total": len(results)
    }


def search_restaurants(
    latitude: float = None,
    longitude: float = None,
    radius: int = 5000,
    low_calorie: bool = False,
    family_friendly: bool = False,
    cuisine_type: str = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    搜索餐厅
    符合B的/api/restaurants/search规范
    """
    time.sleep(0.1)

    results = []

    for poi_id, info in db.pois.items():
        if info["type"] != "restaurant":
            continue

        # 标签筛选
        tags = info["tags"]
        if low_calorie and "low_calorie" not in tags:
            continue
        if family_friendly and "family_friendly" not in tags:
            continue

        # 计算距离，保持稳定，避免同一 demo 输入随机失败。
        distance_km = _estimate_distance_km(info, latitude, longitude)
        if distance_km > radius / 1000:
            continue

        # 获取可用时段
        available_slots = []
        for slot in info["time_slots"]:
            avail = db.check_availability(poi_id, slot)
            if avail["available"]:
                available_slots.append({
                    "time": slot,
                    "remaining": avail["remaining"]
                })

        results.append({
            "poi_id": poi_id,
            "name": info["name"],
            "type": info["type"],
            "tags": info["tags"],
            "price": info["price"],
            "distance_km": round(distance_km, 1),
            "duration_min": info["duration_min"],
            "rating": info["rating"],
            "queue_time_min": 0,
            "available": len(available_slots) > 0,
            "available_slots": available_slots,
            "location": info["location"]
        })

    results.sort(key=lambda x: x["rating"], reverse=True)

    return {
        "items": results[:limit],
        "total": len(results)
    }


def reserve_restaurant(
    poi_id: str,
    time_slot: str,
    people: int,
    notes: List[str] = None
) -> Dict[str, Any]:
    """
    预订餐厅
    对应B的action_type: reserve_restaurant
    """
    time.sleep(0.15)

    # 验证POI存在
    poi = db.get_poi(poi_id)
    if not poi or poi["type"] != "restaurant":
        return {
            "success": False,
            "error": "invalid_poi",
            "message": "餐厅不存在"
        }

    # 验证时段
    if time_slot not in poi["time_slots"]:
        return {
            "success": False,
            "error": "invalid_time_slot",
            "message": f"{time_slot} 不在营业时间内"
        }

    # 预订
    result = db.reserve(poi_id, time_slot, people)

    if result["success"]:
        db.orders[result["order_id"]]["payment_status"] = "not_required"
        return {
            "success": True,
            "order_id": result["order_id"],
            "poi_name": poi["name"],
            "time_slot": time_slot,
            "people": people,
            "payment_required": False,
            "message": f"已预订{poi['name']} {time_slot}，{people}位"
        }
    else:
        # 满位时自动取号
        queue_result = db.add_to_queue(poi_id, time_slot, people)
        return {
            "success": False,
            "error": result["error"],
            "message": f"{time_slot}已满，已为你取号排队",
            "queue_number": queue_result["queue_number"],
            "estimated_wait_minutes": queue_result["estimated_wait_minutes"]
        }


def order_activity_ticket(
    poi_id: str,
    time_slot: str,
    quantity: int,
    notes: List[str] = None
) -> Dict[str, Any]:
    """
    购买活动门票
    对应B的action_type: order_activity_ticket
    """
    time.sleep(0.15)

    poi = db.get_poi(poi_id)
    if not poi or poi["type"] != "activity":
        return {
            "success": False,
            "error": "invalid_poi",
            "message": "活动不存在"
        }

    if time_slot not in poi["time_slots"]:
        return {
            "success": False,
            "error": "invalid_time_slot",
            "message": f"{time_slot} 场次不存在"
        }

    result = db.reserve(poi_id, time_slot, quantity)

    if result["success"]:
        return {
            "success": True,
            "order_id": result["order_id"],
            "poi_name": poi["name"],
            "time_slot": time_slot,
            "quantity": quantity,
            "total_price": poi["price"] * quantity,
            "payment_required": True,
            "message": f"已购{poi['name']} {time_slot}场次，{quantity}张票"
        }
    else:
        return {
            "success": False,
            "error": result["error"],
            "message": f"{time_slot}场次已售罄"
        }


def take_queue_number(poi_id: str, time_slot: str, people: int) -> Dict[str, Any]:
    """
    取号排队
    用于满位时的降级方案
    """
    time.sleep(0.1)

    poi = db.get_poi(poi_id)
    if not poi:
        return {
            "success": False,
            "error": "poi_not_found"
        }

    result = db.add_to_queue(poi_id, time_slot, people)

    return {
        "success": True,
        "queue_number": result["queue_number"],
        "estimated_wait_minutes": result["estimated_wait_minutes"],
        "ahead_count": result["ahead_count"],
        "poi_name": poi["name"],
        "time_slot": time_slot,
        "message": f"已取号{result['queue_number']}号，前方{result['ahead_count']}桌，预计等待{result['estimated_wait_minutes']}分钟"
    }


def check_route(start_poi_id: str, end_poi_id: str, mode: str = "drive") -> Dict[str, Any]:
    """
    检查路线（可选，MVP可简化）
    """
    # 简化版：随机生成合理距离
    duration = random.randint(10, 45)
    distance = round(duration * 0.8, 1)

    return {
        "success": True,
        "duration_minutes": duration,
        "distance_km": distance,
        "mode": mode,
        "feasible": duration <= 60,
        "suggestion": f"建议{mode}，约{duration}分钟"
    }


def check_availability(poi_id: str, time_slot: str) -> Dict[str, Any]:
    """
    查询库存（可选）
    """
    return db.check_availability(poi_id, time_slot)


def cancel_order(order_id: str) -> Dict[str, Any]:
    """
    取消订单（加分项）
    """
    return db.cancel_order(order_id)


def pay_order(order_id: str, amount: float, method: str = "mock_pay") -> Dict[str, Any]:
    """
    支付订单（模拟支付网关）。
    """
    time.sleep(0.1)
    return db.pay_order(order_id, amount, method)


# ========== 3. 辅助函数：获取数据库状态（用于调试）==========

def get_inventory_status() -> Dict:
    """获取当前库存状态，用于调试"""
    status = {}
    for poi_id, slots in db.inventory.items():
        poi_name = db.pois.get(poi_id, {}).get("name", poi_id)
        status[poi_name] = {}
        for slot, remaining in slots.items():
            status[poi_name][slot] = remaining
    return status


def get_queue_status() -> Dict:
    """获取排队状态"""
    status = {}
    for poi_id, slots in db.queues.items():
        poi_name = db.pois.get(poi_id, {}).get("name", poi_id)
        status[poi_name] = {}
        for slot, queue in slots.items():
            status[poi_name][slot] = len(queue)
    return status


# ========== 4. 附加服务与交通接口 ==========

def order_addon_service(
        addon_type: str,
        poi_id: str = None,
        delivery_time: str = None,
        special_requests: List[str] = None
) -> Dict[str, Any]:
    """
    附加服务下单（蛋糕、鲜花等）
    对应B的action_type: order_addon_service
    """
    time.sleep(0.1)

    addon_menu = {
        "cake": {"name": "庆祝蛋糕", "price": 88, "default": "草莓口味"},
        "flowers": {"name": "鲜花礼盒", "price": 68, "default": "红玫瑰"},
        "gift": {"name": "伴手礼", "price": 48, "default": "零食礼包"}
    }

    if addon_type not in addon_menu:
        return {
            "success": False,
            "error": "invalid_addon_type",
            "message": f"不支持的附加服务类型: {addon_type}，支持: cake, flowers, gift"
        }

    addon_info = addon_menu[addon_type]
    order_id = f"ADDON_{addon_type}_{uuid.uuid4().hex[:8]}"

    # 存储订单
    db.orders[order_id] = {
        "type": "addon_service",
        "addon_type": addon_type,
        "delivery_time": delivery_time,
        "special_requests": special_requests,
        "status": "confirmed",
        "payment_status": "unpaid",
        "created_at": time.time()
    }

    return {
        "success": True,
        "order_id": order_id,
        "addon_type": addon_type,
        "name": addon_info["name"],
        "price": addon_info["price"],
        "delivery_time": delivery_time,
        "payment_required": True,
        "message": f"已下单{addon_info['name']}（{addon_info['default']}），订单号{order_id}，预计{delivery_time}送达"
    }


def call_taxi(
        start: str,
        end: str,
        time_slot: str = None
) -> Dict[str, Any]:
    """
    叫车预约
    对应B的action_type: call_taxi
    """
    time.sleep(0.1)

    # 模拟叫车（90%成功率）
    if random.random() < 0.9:
        estimated_price = random.randint(30, 80)
        estimated_duration = random.randint(10, 35)
        driver_id = f"DRV_{random.randint(100, 999)}"

        return {
            "success": True,
            "order_id": f"TAXI_{uuid.uuid4().hex[:8]}",
            "driver": f"师傅_{driver_id[-3:]}",
            "plate": f"沪A{random.randint(10000, 99999)}",
            "price": estimated_price,
            "duration_minutes": estimated_duration,
            "eta_minutes": random.randint(3, 8),
            "message": f"已为您叫车，{estimated_price}元左右，{estimated_duration}分钟可达"
        }
    else:
        return {
            "success": False,
            "error": "no_car_available",
            "message": "附近暂无可用车辆，请稍后再试"
        }
