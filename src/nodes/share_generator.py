"""
Share Message Generator - 生成分享文案
C负责
"""

from src.state import PlanState
from typing import Dict, Any


def _clock_to_minutes(value: Any) -> int | None:
    text = str(value or "").strip()
    if ":" not in text:
        return None
    hour_text, minute_text = text.split(":", 1)
    try:
        hour = int(hour_text)
        minute = int(minute_text[:2])
    except ValueError:
        return None
    if hour < 0 or minute < 0 or minute >= 60:
        return None
    return hour * 60 + minute


def _minutes_to_clock(minutes: int) -> str:
    hour = (minutes // 60) % 24
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def _retime_range(original: Any, executed_start: Any) -> str:
    start = _clock_to_minutes(executed_start)
    if start is None:
        return str(original or "")

    text = str(original or "").strip()
    if "-" not in text:
        return _minutes_to_clock(start)

    original_start_text, original_end_text = text.split("-", 1)
    original_start = _clock_to_minutes(original_start_text)
    original_end = _clock_to_minutes(original_end_text)
    if original_start is None or original_end is None:
        return _minutes_to_clock(start)

    duration = original_end - original_start
    if duration <= 0:
        duration = 60
    return f"{_minutes_to_clock(start)}-{_minutes_to_clock(start + duration)}"


def _range_bounds(value: Any) -> tuple[int | None, int | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    if "-" not in text:
        start = _clock_to_minutes(text)
        return start, None

    start_text, end_text = text.split("-", 1)
    return _clock_to_minutes(start_text), _clock_to_minutes(end_text)


def _retime_preserves_order(
    original: Any,
    executed_start: Any,
    previous_end: int | None,
    next_start: int | None,
) -> str | None:
    retimed = _retime_range(original, executed_start)
    retimed_start, retimed_end = _range_bounds(retimed)
    if retimed_start is None:
        return None
    if previous_end is not None and retimed_start < previous_end:
        return None
    if next_start is not None and retimed_end is not None and retimed_end > next_start:
        return None
    return retimed


def _executed_start_times(tool_results: dict[str, Any]) -> dict[str, str]:
    executed_times: dict[str, str] = {}
    for value in tool_results.values():
        if not value.get("success"):
            continue
        data = value.get("data") if isinstance(value.get("data"), dict) else {}
        raw_step = data.get("raw_step") if isinstance(data.get("raw_step"), dict) else {}
        executed_time = raw_step.get("time") or data.get("time")
        if not executed_time:
            continue
        poi_id = raw_step.get("poi_id")
        name = value.get("name")
        if poi_id:
            executed_times[str(poi_id)] = str(executed_time)
        if name:
            executed_times[str(name)] = str(executed_time)
    return executed_times


def _append_repair_guidance(share_msg: str, state: PlanState) -> str:
    repair_plan = state.get("b_repair_plan") or {}
    if not isinstance(repair_plan, dict):
        return share_msg

    user_message = _sanitize_repair_guidance_message(
        str(repair_plan.get("user_message") or "").strip(),
        state,
    )
    if not user_message or user_message in share_msg:
        return share_msg

    return f"{share_msg} 调整建议：{user_message}"


def _repair_guidance_needs_cautious_copy(state: PlanState) -> bool:
    execution_status = str(state.get("execution_status") or "").strip().lower()
    if execution_status == "failed":
        return True
    if execution_status != "partial":
        return False
    tool_results = state.get("tool_results", {}) or {}
    return any(not value.get("success") for value in tool_results.values())


def _sanitize_repair_guidance_message(message: str, state: PlanState) -> str:
    if not message or not _repair_guidance_needs_cautious_copy(state):
        return message

    replacements = (
        ("已为您保留", "可尝试保留"),
        ("已经为您保留", "可尝试保留"),
        ("已为您替换为", "建议替换为"),
        ("已经为您替换为", "建议替换为"),
        ("已同步替换为", "建议替换为"),
        ("已替换为", "可考虑替换为"),
        ("已经替换为", "可考虑替换为"),
        ("已替换", "可考虑替换"),
        ("已经替换", "可考虑替换"),
        ("已保留", "可尝试保留"),
        ("已经保留", "可尝试保留"),
        ("已确认", "需重新确认"),
        ("已经确认", "需重新确认"),
        ("已安排好", "可尝试安排"),
        ("已经安排好", "可尝试安排"),
        ("已安排", "可尝试安排"),
        ("已经安排", "可尝试安排"),
        ("已订好", "需重新预订"),
        ("已经订好", "需重新预订"),
        ("已订", "需重新预订"),
        ("已经订", "需重新预订"),
        ("现在为您寻找", "建议继续寻找"),
        ("正在为您寻找", "建议继续寻找"),
    )
    sanitized = message
    for old, new in replacements:
        sanitized = sanitized.replace(old, new)
    sanitized = sanitized.replace("请稍候。", "").replace("请稍候", "").strip()
    sanitized = sanitized.rstrip("，,；; ")

    overclaim_terms = (
        "已为您",
        "已替换",
        "已保留",
        "已确认",
        "已安排",
        "已订",
        "已经替换",
        "已经保留",
        "已经确认",
        "已经安排",
        "已经订",
    )
    if any(term in sanitized for term in overclaim_terms):
        return "后续调整需重新确认：" + sanitized
    return sanitized


def _text_contains_any(value: Any, terms: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        return any(_text_contains_any(item, terms) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_text_contains_any(item, terms) for item in value)
    text = str(value or "")
    return any(term in text for term in terms)


def _partial_failure_tail(state: PlanState) -> str:
    constraints = state.get("constraints", {}) or {}
    scenario_facets = state.get("scenario_facets", {}) or {}
    risk_text = {
        "avoid": constraints.get("avoid") or state.get("intent", {}).get("avoid"),
        "scenario_facets": scenario_facets,
        "scenario_activities": state.get("scenario_activities"),
        "weather_context": state.get("weather_context"),
        "raw_text": constraints.get("raw_text") or state.get("user_input"),
    }
    rain_sensitive = _text_contains_any(risk_text, ("下雨", "雨天", "rain"))
    low_wait_sensitive = _text_contains_any(
        risk_text,
        ("排队", "等待", "少排队", "别排", "不用排", "low_wait", "long_queue"),
    )

    if rain_sensitive and low_wait_sensitive:
        return "先不要直接到现场等位，雨天和排队风险都不适合；建议改约其他可订时段，或换一家可预约、等待更短的备选。"
    if low_wait_sensitive:
        return "先不要直接到现场等位；建议改约其他可订时段，或换一家可预约、等待更短的备选。"
    if rain_sensitive:
        return "先不要直接到现场碰运气；建议改约可确认的时段，或换一个雨天更稳妥的备选。"
    return "建议先改约其他可订时段，或换一家可确认的备选。"


def _success_arrival_tail(state: PlanState) -> str:
    constraints = state.get("constraints", {}) or {}
    intent = state.get("intent", {}) or {}
    people_count = constraints.get("people_count") or intent.get("people_count") or 0
    try:
        people_count = int(people_count)
    except (TypeError, ValueError):
        people_count = 0

    if people_count <= 1:
        return "你按确认时间过去即可。"
    return "大家按确认时间过去即可。"


def _partial_success_copy(
    *,
    scene_type: str,
    timeline_desc: str,
    success_str: str,
    fail_str: str,
    success_count: int,
    failure_count: int,
    recovery_tail: str,
) -> str:
    if scene_type != "family":
        return f"⚠️ {success_str}已经订好了，但{fail_str}暂时没订上（可能人太多）。{recovery_tail}"

    if success_count > failure_count:
        return (
            f"⚠️ 可执行部分多数已确认：{timeline_desc}。但{fail_str}遇到点问题"
            f"（可能满位了），其他{success_str}已确认。{recovery_tail}"
        )

    if success_count > 0:
        return (
            f"⚠️ 目前只确认了{success_str}：{timeline_desc}。"
            f"{fail_str}还没订上（可能满位了）。{recovery_tail}"
        )

    return (
        f"⚠️ 目前还没有确认成功的预订：{timeline_desc}。"
        f"{fail_str}还没订上（可能满位了）。{recovery_tail}"
    )


def _dedupe_texts(values: list[Any], *, limit: int = 5) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _role_label_lookup(state: PlanState, timeline: list[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item in timeline:
        role = str(item.get("role") or "").strip()
        activity = str(item.get("activity") or item.get("label") or "").strip()
        if role and activity:
            lookup.setdefault(role, activity)

    blueprint_sources = [
        state.get("b_itinerary_blueprint"),
        (state.get("constraints") or {}).get("b_itinerary_blueprint"),
    ]
    for blueprint in blueprint_sources:
        if not isinstance(blueprint, dict):
            continue
        for node in blueprint.get("node_intents") or []:
            if not isinstance(node, dict):
                continue
            role = str(node.get("role") or "").strip()
            label = str(node.get("label") or "").strip()
            if role and label:
                lookup.setdefault(role, label)
    return lookup


def _non_executable_plan_tail(state: PlanState, timeline: list[dict[str, Any]]) -> str:
    selected_plan = state.get("selected_plan", {}) or {}
    coverage = (
        state.get("b_rag_candidate_coverage")
        or state.get("candidate_recall_diagnostics", {}).get("rag_candidate_coverage")
        or {}
    )
    blueprint = (
        state.get("b_itinerary_blueprint")
        or (state.get("constraints") or {}).get("b_itinerary_blueprint")
        or {}
    )

    placeholder_nodes = _dedupe_texts(
        [
            item.get("activity") or item.get("name")
            for item in timeline
            if not item.get("poi_id") or item.get("type") == "planning_intent"
        ]
    )

    missing_roles = _dedupe_texts(
        list(selected_plan.get("partial_missing_roles") or [])
        + list(coverage.get("unsupported_missing_roles") or [])
        + list(blueprint.get("unsupported_roles") or [])
    )
    role_labels = _role_label_lookup(state, timeline)
    missing_role_labels = _dedupe_texts(
        [role_labels.get(role, role.replace("_", "/")) for role in missing_roles]
    )

    details: list[str] = []
    if placeholder_nodes:
        details.append(f"还只是行程意图、不是可预订商家的节点：{'、'.join(placeholder_nodes)}")
    if missing_role_labels:
        details.append(f"当前执行层不支持直接下单/确认的节点：{'、'.join(missing_role_labels)}")

    missing_node_ids = _dedupe_texts(coverage.get("missing_node_ids") or [], limit=3)
    if missing_node_ids and not missing_role_labels:
        details.append(f"还有{len(missing_node_ids)}个规划节点缺少可执行候选")

    if not details:
        return ""

    return (
        " "
        + "；".join(details)
        + "。这些内容只能作为待补充建议，不能说已经订好；需要先补到具体可预约商家、场次或门票后再执行。"
    )


def share_generator_node(state: PlanState) -> Dict[str, Any]:
    """
    根据执行结果生成分享消息
    """
    print("📱 [12] Share Generator: 生成分享消息...")

    execution_log = state.get("execution_log", [])
    selected_plan = state.get("selected_plan", {})
    tool_results = state.get("tool_results", {})
    execution_status = state.get("execution_status", "pending")
    payment_status = state.get("payment_status", "not_required")
    payment_order = state.get("payment_order", {})
    scene_type = state.get("scene_type", "family")
    explanation_text = state.get("explanation_text", "")
    action_sequence = state.get("action_sequence", [])

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
    booked_addons = [
        value.get("name", "附加服务")
        for value in tool_results.values()
        if value.get("success") and value.get("data", {}).get("action") == "order_addon_service"
    ]

    # 构建时间线描述
    executed_times = _executed_start_times(tool_results)
    timeline_desc = ""
    previous_end: int | None = None
    for index, item in enumerate(timeline):
        executed_start = executed_times.get(str(item.get("poi_id"))) or executed_times.get(
            str(item.get("activity"))
        )
        item_time = item.get("time", "")
        next_item = timeline[index + 1] if index + 1 < len(timeline) else {}
        next_start, _ = _range_bounds(next_item.get("time", ""))
        if executed_start:
            item_time = (
                _retime_preserves_order(item_time, executed_start, previous_end, next_start)
                or item_time
            )
        _, item_end = _range_bounds(item_time)
        if item_end is not None:
            previous_end = item_end
        timeline_desc += f"{item_time} {item.get('activity', '')} → "
    timeline_desc = timeline_desc.rstrip(" → ")

    payment_tail = ""
    if payment_status == "success":
        payment_name = payment_order.get("name", "需要支付的项目")
        payment_tail = f"{payment_name}已支付成功。"
    elif payment_status in {"pending", "cancelled", "failed"}:
        payment_name = payment_order.get("name", "需要支付的项目")
        payment_tail = f"{payment_name}订单已创建，但支付还未完成。"

    # 根据场景和状态生成分享文案
    if execution_status == "success":
        non_executable_tail = _non_executable_plan_tail(state, timeline)
        if scene_type == "family":
            addon_tail = ""
            if booked_addons:
                addon_tail = f"{'、'.join(booked_addons)}也已安排好。"
            if non_executable_tail:
                share_msg = (
                    f"🎉 可执行的部分已经订好了：{timeline_desc}。"
                    f"{addon_tail}{payment_tail}{non_executable_tail}"
                )
            else:
                share_msg = f"🎉 搞定了！下午安排好了：{timeline_desc}。已经帮你订好了。{addon_tail}{payment_tail}祝你们玩得开心！❤️"
        else:
            arrival_tail = _success_arrival_tail(state)
            if non_executable_tail:
                share_msg = (
                    f"🎉 可执行的部分已经订好了：{timeline_desc}。"
                    f"{payment_tail}{arrival_tail}{non_executable_tail}"
                )
            else:
                share_msg = f"🎉 安排好了！{timeline_desc}。位置已经订好了，{payment_tail}{arrival_tail}"

    elif execution_status == "partial":
        success_str = "、".join(booked_items) if booked_items else "部分项目"
        fail_str = "、".join(failed_items) if failed_items else "个别项目"
        recovery_tail = _partial_failure_tail(state)
        share_msg = _partial_success_copy(
            scene_type=scene_type,
            timeline_desc=timeline_desc,
            success_str=success_str,
            fail_str=fail_str,
            success_count=len(booked_items),
            failure_count=len(failed_items),
            recovery_tail=recovery_tail,
        )
        share_msg = _append_repair_guidance(share_msg, state)

    elif execution_status == "failed":
        if not failed_items and not action_sequence:
            non_executable_tail = _non_executable_plan_tail(state, timeline)
            share_msg = (
                f"暂时没有形成可执行的预订动作：{timeline_desc}。"
                f"{non_executable_tail}"
                "建议换一个时间段、增加可预订节点，或放宽部分约束后再试。"
            )
        else:
            failure_names = "、".join(failed_items) if failed_items else "当前方案"
            share_msg = (
                f"😅 刚才尝试预订没有成功：{timeline_desc}。"
                f"未订上：{failure_names}。"
                "建议先改约其他可订时段，或换成可确认的备选。"
            )
        share_msg = _append_repair_guidance(share_msg, state)

    else:
        share_msg = f"📝 帮你们看了下：{timeline_desc}。需要我帮忙预订吗？确认的话说一声～"

    execution_log.append(f"✅ 分享消息已生成")
    execution_log.append(f"📨 消息内容: {share_msg[:100]}...")

    return {
        "final_share_message": share_msg,
        "execution_log": execution_log
    }
