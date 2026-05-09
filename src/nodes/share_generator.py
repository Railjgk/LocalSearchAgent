"""
Share Message Generator - 生成分享文案
C负责
"""

from src.state import PlanState
from typing import Dict, Any


def share_generator_node(state: PlanState) -> Dict[str, Any]:
    """
    根据执行结果生成分享消息
    """
    print("📱 [11] Share Generator: 生成分享消息...")

    execution_log = state.get("execution_log", [])
    selected_plan = state.get("selected_plan", {})
    tool_results = state.get("tool_results", {})
    execution_status = state.get("execution_status", "pending")
    scene_type = state.get("scene_type", "family")
    explanation_text = state.get("explanation_text", "")

    timeline = selected_plan.get("timeline", [])

    if not timeline:
        share_msg = explanation_text or "暂时没有找到可执行方案，可以放宽距离、预算或时间约束后再试。"
        execution_log.append("✅ 分享消息已生成")
        execution_log.append(f"📨 消息内容: {share_msg[:100]}...")
        return {
            "final_share_message": share_msg,
            "execution_log": execution_log
        }

    # 提取成功预订的信息
    booked_items = []
    failed_items = []

    for key, value in tool_results.items():
        if value.get("success"):
            booked_items.append(value.get("name", "未知"))
        else:
            failed_items.append(value.get("name", "未知"))

    # 构建时间线描述
    timeline_desc = ""
    for item in timeline:
        timeline_desc += f"{item.get('time', '')} {item.get('activity', '')} → "
    timeline_desc = timeline_desc.rstrip(" → ")

    # 根据场景和状态生成分享文案
    if execution_status == "success":
        if scene_type == "family":
            share_msg = f"🎉 搞定了！下午安排好了：{timeline_desc}。已经帮你订好了，蛋糕也会准时送到家～祝你们玩得开心！❤️"
        else:
            share_msg = f"🎉 安排好了！{timeline_desc}。位置已经订好了，大家直接去就行～下午见！🍻"

    elif execution_status == "partial":
        success_str = "、".join(booked_items) if booked_items else "部分项目"
        fail_str = "、".join(failed_items) if failed_items else "个别项目"

        if scene_type == "family":
            share_msg = f"⚠️ 大部分安排好了：{timeline_desc}。但{fail_str}遇到点问题（可能满位了），其他{success_str}已确认。建议早点去现场看看～"
        else:
            share_msg = f"⚠️ {success_str}已经订好了，但{fail_str}暂时没订上（可能人太多）。到了现场再看吧，不行我们可以换地方～"

    elif execution_status == "failed":
        share_msg = f"😅 抱歉，刚才尝试预订时遇到了一些问题（{failed_items[0] if failed_items else '系统繁忙'}）。要不我们换个时间或地方？"

    else:
        share_msg = f"📝 帮你们看了下：{timeline_desc}。需要我帮忙预订吗？确认的话说一声～"

    execution_log.append(f"✅ 分享消息已生成")
    execution_log.append(f"📨 消息内容: {share_msg[:100]}...")

    return {
        "final_share_message": share_msg,
        "execution_log": execution_log
    }
