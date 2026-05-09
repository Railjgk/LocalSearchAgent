"""
Mock API Layer - 模拟API调用
C负责
"""

from src.state import PlanState
from src.tools.mock_apis import (
    check_availability,
    reserve_restaurant,
    order_activity_ticket,
    order_addon_service,
    call_taxi
)
from typing import Dict, Any


def mock_api_layer_node(state: PlanState) -> Dict[str, Any]:
    """
    执行Tool Router生成的动作序列，调用对应的Mock API
    """
    print("📡 [9] Mock API Layer: 调用API...")

    execution_log = state.get("execution_log", [])
    action_sequence = state.get("action_sequence", [])
    raw_results = {}

    for action in action_sequence:
        # 统一使用 action_type 字段（与 tool_router 输出对齐）
        action_type = action.get("action_type") or action.get("action")
        step = action.get("step")
        name = action.get("name", "")

        execution_log.append(f"   📞 调用API: {action_type} - {name}")

        # 根据action_type调用对应的Mock API函数
        if action_type == "reserve_restaurant":
            result = reserve_restaurant(
                poi_id=action.get("poi_id", ""),
                time_slot=action.get("time", ""),
                people=action.get("people", 3),
                notes=action.get("notes", [])
            )
        elif action_type == "order_activity_ticket":
            result = order_activity_ticket(
                poi_id=action.get("poi_id", ""),
                time_slot=action.get("time", ""),
                quantity=action.get("quantity", 1),
                notes=action.get("notes", [])
            )
        elif action_type == "order_addon_service":
            # 附加服务：蛋糕、鲜花等
            addon_type = action.get("addon_type", "cake")
            result = order_addon_service(
                addon_type=addon_type,
                poi_id=action.get("poi_id", ""),
                delivery_time=action.get("time", "18:00"),
                special_requests=action.get("notes", [])
            )
        elif action_type == "call_taxi":
            result = call_taxi(
                start=action.get("start", "家"),
                end=action.get("end", "目的地"),
                time_slot=action.get("time", "")
            )
        elif action_type == "check_availability":
            result = check_availability(
                poi_id=action.get("poi_id", ""),
                time_slot=action.get("time", "")
            )
        else:
            result = {
                "success": False,
                "error": f"未知动作类型: {action_type}",
                "message": f"不支持的动作类型: {action_type}"
            }

        # 记录结果
        raw_results[f"{action_type}_{step}"] = {
            "action": action_type,
            "name": name,
            "result": result
        }

        if result.get("success"):
            execution_log.append(f"      ✅ {action_type} 成功: {result.get('message', result.get('order_id', ''))}")
        else:
            execution_log.append(f"      ❌ {action_type} 失败: {result.get('error', result.get('message', '未知错误'))}")

    execution_log.append(f"✅ Mock API Layer: 完成{len(action_sequence)}个API调用")

    return {
        "raw_api_results": raw_results,
        "execution_log": execution_log
    }