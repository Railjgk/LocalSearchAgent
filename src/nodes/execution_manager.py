"""
Execution Manager - 管理执行过程、重试、状态追踪
C负责
"""

from src.state import PlanState
from src.tools.mock_apis import (
    reserve_restaurant,
    order_activity_ticket,
    call_taxi,
    order_addon_service      # ← 改成 order_addon_service
)
from typing import Dict, Any


def execution_manager_node(state: PlanState) -> Dict[str, Any]:
    """
    执行action_sequence中的动作，调用对应的Mock API
    """
    print("⚙️ [9-10] Execution Manager: 执行动作序列...")

    execution_log = state.get("execution_log", [])
    action_sequence = state.get("action_sequence", [])
    tool_results = {}
    raw_results = {}

    for action in action_sequence:
        action_type = action.get("action_type")
        step = action.get("step")
        name = action.get("name", "")

        execution_log.append(f"   ▶ 执行: {action_type} - {name}")

        # 根据action_type调用对应的函数
        if action_type == "reserve_restaurant":
            result = reserve_restaurant(
                poi_id=action.get("poi_id"),
                time_slot=action.get("time"),
                people=action.get("people", 2),
                notes=action.get("notes", [])
            )
        elif action_type == "order_activity_ticket":
            result = order_activity_ticket(
                poi_id=action.get("poi_id"),
                time_slot=action.get("time"),
                quantity=action.get("quantity", 1),
                notes=action.get("notes", [])
            )
        elif action_type == "order_addon_service":
            # 直接使用已导入的 order_addon_service
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
        else:
            result = {"success": False, "error": f"未知动作类型: {action_type}"}

        # 记录结果
        raw_results[f"{action_type}_{step}"] = {
            "action": action_type,
            "name": name,
            "result": result
        }

        tool_results[f"{action_type}_{step}"] = {
            "success": result.get("success", False),
            "data": result,
            "action": action_type,
            "name": name
        }

        if result.get("success"):
            execution_log.append(f"      ✅ {action_type} 成功: {result.get('message', result.get('order_id', ''))}")
        else:
            execution_log.append(f"      ❌ {action_type} 失败: {result.get('error', '未知错误')}")

    # 统计执行结果
    success_count = sum(1 for v in tool_results.values() if v["success"])
    fail_count = len(tool_results) - success_count

    if fail_count == 0:
        execution_status = "success"
    elif success_count > 0:
        execution_status = "partial"
    else:
        execution_status = "failed"

    execution_log.append(f"📊 执行统计: 成功{success_count}/{len(tool_results)}, 失败{fail_count}")
    execution_log.append(f"📊 执行状态: {execution_status}")

    return {
        "tool_results": tool_results,
        "raw_api_results": raw_results,
        "execution_status": execution_status,
        "execution_log": execution_log
    }