"""
Share Message Generator - 生成分享文案
C负责
"""

import re

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
        raw_step = (
            data.get("raw_step") if isinstance(data.get("raw_step"), dict) else {}
        )
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
        return (
            "先不要直接到现场碰运气；建议改约可确认的时段，或换一个雨天更稳妥的备选。"
        )
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


def _role_label_lookup(
    state: PlanState, timeline: list[dict[str, Any]]
) -> dict[str, str]:
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


def _blueprint_sources(
    state: PlanState, selected_plan: dict[str, Any]
) -> list[dict[str, Any]]:
    sources = [
        selected_plan.get("b_itinerary_blueprint"),
        state.get("b_itinerary_blueprint"),
        (state.get("constraints") or {}).get("b_itinerary_blueprint"),
    ]
    return [source for source in sources if isinstance(source, dict)]


def _slot_time_label(slot: dict[str, Any]) -> str:
    start = str(slot.get("start_time") or "").strip()
    end = str(slot.get("end_time") or "").strip()
    if start and end:
        return f"{start}-{end}"
    return str(slot.get("time") or "").strip()


def _slot_display_label(slot: dict[str, Any], *, include_time: bool = False) -> str:
    label = str(
        slot.get("label") or slot.get("activity") or slot.get("role") or ""
    ).strip()
    if not include_time:
        return label
    time_label = _slot_time_label(slot)
    if time_label and label:
        return f"{time_label} {label}"
    return label or time_label


def _is_protected_guidance(value: dict[str, Any]) -> bool:
    return (
        value.get("execution_status") == "protected_non_executable"
        or value.get("protected") is True
        or value.get("anchor_type") == "rest"
        or value.get("supply_domain") == "planning_guidance"
    )


def _blueprint_node_lookup(
    state: PlanState,
    selected_plan: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_node_id: dict[str, dict[str, Any]] = {}
    slots: list[dict[str, Any]] = []
    for blueprint in _blueprint_sources(state, selected_plan):
        for day in (blueprint.get("time_skeleton") or {}).get("days") or []:
            if not isinstance(day, dict):
                continue
            for slot in day.get("slots") or []:
                if not isinstance(slot, dict):
                    continue
                slots.append(slot)
                node_id = str(slot.get("node_id") or "").strip()
                if node_id:
                    by_node_id.setdefault(node_id, slot)
        for node in blueprint.get("node_intents") or []:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("node_id") or "").strip()
            if node_id:
                by_node_id.setdefault(node_id, node)
    return by_node_id, slots


def _timeline_slot_metadata(
    item: dict[str, Any],
    slots: list[dict[str, Any]],
) -> dict[str, Any]:
    item_node_id = str(item.get("node_id") or "").strip()
    item_role = str(item.get("role") or "").strip()
    item_label = str(
        item.get("activity") or item.get("label") or item.get("name") or ""
    ).strip()
    item_time = str(item.get("time") or "").strip()

    for slot in slots:
        slot_node_id = str(slot.get("node_id") or "").strip()
        if item_node_id and slot_node_id == item_node_id:
            return slot

    for slot in slots:
        if item_time and item_time != _slot_time_label(slot):
            continue
        slot_label = _slot_display_label(slot)
        slot_role = str(slot.get("role") or "").strip()
        if item_label and item_label == slot_label:
            return slot
        if item_role and item_role == slot_role:
            return slot

    for slot in slots:
        slot_label = _slot_display_label(slot)
        slot_role = str(slot.get("role") or "").strip()
        if item_label and item_label == slot_label:
            return slot
        if item_role and item_role == slot_role:
            return slot

    return {}


def _missing_node_labels(
    missing_node_ids: list[str],
    by_node_id: dict[str, dict[str, Any]],
) -> list[str]:
    labels: list[str] = []
    for node_id in missing_node_ids:
        metadata = by_node_id.get(node_id) or {}
        labels.append(_slot_display_label(metadata) or node_id)
    return _dedupe_texts(labels, limit=5)


def _candidate_blocker_text(state: PlanState, selected_plan: dict[str, Any]) -> str:
    filter_reasons = state.get("filter_reasons") or {}
    detail = (
        filter_reasons.get("_summary_detail")
        if isinstance(filter_reasons, dict)
        else {}
    )
    reason_counts = detail.get("reason_counts") if isinstance(detail, dict) else {}
    if isinstance(reason_counts, dict) and reason_counts:
        reasons = sorted(
            (
                (str(reason or "").strip(), count)
                for reason, count in reason_counts.items()
                if str(reason or "").strip()
            ),
            key=lambda item: item[1] if isinstance(item[1], (int, float)) else 0,
            reverse=True,
        )
        if reasons:
            return reasons[0][0]

    source_texts = [
        selected_plan.get("candidate_generation_summary"),
        filter_reasons.get("_summary") if isinstance(filter_reasons, dict) else "",
    ]
    signal_terms = (
        "缺少",
        "证据",
        "不可用",
        "不满足",
        "过敏",
        "宠物",
        "儿童",
        "预算",
        "距离",
    )
    for value in source_texts:
        for part in re.split(r"[。；;]", str(value or "")):
            text = re.sub(r"（\d+\s*个）", "", part).strip()
            if 4 <= len(text) <= 40 and any(term in text for term in signal_terms):
                return text
    return ""


def _candidate_evidence_grade_nodes(selected_plan: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_grade = selected_plan.get("candidate_evidence_grade") or {}
    nodes = candidate_grade.get("nodes") if isinstance(candidate_grade, dict) else []
    if not isinstance(nodes, list):
        return []
    return [node for node in nodes if isinstance(node, dict)]


def _grade_node_label(node: dict[str, Any], *, include_time: bool = False) -> str:
    label = str(node.get("label") or node.get("node_id") or "").strip()
    time_label = str(node.get("time") or "").strip()
    if include_time and time_label and label:
        return f"{time_label} {label}"
    return label


def _non_executable_tail_from_grade(
    state: PlanState,
    selected_plan: dict[str, Any],
) -> str:
    grade_nodes = _candidate_evidence_grade_nodes(selected_plan)
    if not grade_nodes:
        return ""

    protected_labels = _dedupe_texts(
        [
            _grade_node_label(node, include_time=True)
            for node in grade_nodes
            if node.get("grade") == "protected_guidance"
        ],
        limit=5,
    )
    missing_labels = _dedupe_texts(
        [
            _grade_node_label(node)
            for node in grade_nodes
            if node.get("grade") == "missing_node_evidence"
        ],
        limit=7,
    )
    raw_unconfirmed_labels = _dedupe_texts(
        [
            _grade_node_label(node)
            for node in grade_nodes
            if node.get("grade") in {"all_candidates_filtered", "raw_candidates_unconfirmed"}
        ],
        limit=7,
    )

    details: list[str] = []
    if protected_labels:
        details.append(f"已保留为不需要预订的行程约束：{'、'.join(protected_labels)}")
    if missing_labels:
        details.append(
            f"还只是行程意图、缺少可执行候选证据的规划节点：{'、'.join(missing_labels)}"
        )
    if raw_unconfirmed_labels:
        details.append(
            f"已有初始候选但未通过硬约束/可预订确认的节点：{'、'.join(raw_unconfirmed_labels)}"
        )

    if state.get("execution_status") == "failed" and not state.get("action_sequence"):
        blocker = (
            str((selected_plan.get("candidate_evidence_grade") or {}).get("hard_filter_reason") or "").strip()
            or _candidate_blocker_text(state, selected_plan)
        )
        if blocker:
            details.append(f"当前主要阻塞：{blocker}")

    if not details:
        return ""
    return (
        " "
        + "；".join(details)
        + "。这些内容只能作为待补充建议，不能说已经订好；需要先补到具体可预约商家、场次或门票后再执行。"
    )


def _non_executable_plan_tail(state: PlanState, timeline: list[dict[str, Any]]) -> str:
    selected_plan = state.get("selected_plan", {}) or {}
    graded_tail = _non_executable_tail_from_grade(state, selected_plan)
    if graded_tail:
        return graded_tail

    coverage = (
        state.get("b_rag_candidate_coverage")
        or state.get("candidate_recall_diagnostics", {}).get("rag_candidate_coverage")
        or {}
    )
    blueprint_sources = _blueprint_sources(state, selected_plan)
    blueprint = blueprint_sources[0] if blueprint_sources else {}
    by_node_id, slots = _blueprint_node_lookup(state, selected_plan)

    missing_roles = _dedupe_texts(
        list(selected_plan.get("partial_missing_roles") or [])
        + list(coverage.get("unsupported_missing_roles") or [])
        + list(blueprint.get("unsupported_roles") or [])
    )
    role_labels = _role_label_lookup(state, timeline)
    missing_role_labels = _dedupe_texts(
        [role_labels.get(role, role.replace("_", "/")) for role in missing_roles]
    )

    missing_node_ids = _dedupe_texts(coverage.get("missing_node_ids") or [], limit=5)
    missing_node_id_set = set(missing_node_ids)
    covered_node_ids = _dedupe_texts(coverage.get("covered_node_ids") or [], limit=20)
    covered_node_id_set = set(covered_node_ids)

    protected_guidance: list[str] = []
    covered_unconfirmed: list[str] = []
    unclassified_placeholders: list[str] = []
    for item in timeline:
        if item.get("poi_id") and item.get("type") != "planning_intent":
            continue

        metadata = _timeline_slot_metadata(item, slots)
        label = str(
            item.get("activity") or item.get("name") or ""
        ).strip() or _slot_display_label(metadata)
        if not label:
            continue

        if _is_protected_guidance(item) or _is_protected_guidance(metadata):
            protected_guidance.append(
                _slot_display_label(metadata or item, include_time=True)
                or f"{item.get('time', '')} {label}".strip()
            )
            continue

        node_id = str(item.get("node_id") or metadata.get("node_id") or "").strip()
        if node_id and node_id in missing_node_id_set:
            continue
        if node_id and node_id in covered_node_id_set:
            covered_unconfirmed.append(label)
            continue
        if coverage.get("all_nodes_covered") and not missing_node_id_set:
            covered_unconfirmed.append(label)
            continue
        unclassified_placeholders.append(label)

    missing_node_labels = _missing_node_labels(missing_node_ids, by_node_id)

    details: list[str] = []
    protected_labels = _dedupe_texts(protected_guidance, limit=5)
    if protected_labels:
        details.append(f"已保留为不需要预订的行程约束：{'、'.join(protected_labels)}")
    if missing_node_labels and not missing_role_labels:
        details.append(
            f"缺少可执行候选证据的规划节点：{'、'.join(missing_node_labels)}"
        )
    covered_labels = _dedupe_texts(covered_unconfirmed, limit=7)
    if covered_labels:
        details.append(
            f"已有候选/规划证据但还未完成可预订确认的节点：{'、'.join(covered_labels)}"
        )
    placeholder_nodes = _dedupe_texts(unclassified_placeholders, limit=5)
    if placeholder_nodes:
        details.append(
            f"还只是行程意图、尚未完成可预订确认的节点：{'、'.join(placeholder_nodes)}"
        )
    if missing_role_labels:
        details.append(
            f"当前执行层不支持直接下单/确认的节点：{'、'.join(missing_role_labels)}"
        )

    if state.get("execution_status") == "failed" and not state.get("action_sequence"):
        blocker = _candidate_blocker_text(state, selected_plan)
        if blocker:
            details.append(f"当前主要阻塞：{blocker}")

    if not details:
        return ""

    return (
        " "
        + "；".join(details)
        + "。这些内容只能作为待补充建议，不能说已经订好；需要先补到具体可预约商家、场次或门票后再执行。"
    )


def _completed_bounded_schedule_repair_request(state: PlanState) -> dict[str, Any]:
    def request_is_completed(request: Any) -> bool:
        if not isinstance(request, dict):
            return False
        return (
            request.get("status") == "completed"
            and request.get("trace_only") is True
            and request.get("post_replan_trace_status") == "completed"
        )

    constraints = state.get("constraints") or {}
    trace = constraints.get("b_replan_trace") if isinstance(constraints, dict) else {}
    sources: list[Any] = []
    if isinstance(trace, dict) and trace.get("status") == "completed":
        sources.append(trace.get("last_completed_request"))

    diagnostics = state.get("candidate_recall_diagnostics") or {}
    if isinstance(diagnostics, dict):
        sources.append(diagnostics.get("last_replan_request"))

    for request in sources:
        if not request_is_completed(request):
            continue
        repair = request.get("schedule_repair_request")
        if not isinstance(repair, dict):
            continue
        if repair.get("request_type") != "bounded_schedule_repair":
            continue
        target_nodes = repair.get("target_nodes")
        if isinstance(target_nodes, list) and target_nodes:
            return repair
    return {}


def _schedule_repair_node_text(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    label = str(node.get("label") or "").strip()
    if not label:
        return ""
    slot_window = str(node.get("slot_window") or "").strip()
    if slot_window:
        return f"{slot_window} {label}"
    return label


def _bounded_schedule_repair_guidance(state: PlanState) -> str:
    if state.get("execution_status") != "failed" or state.get("action_sequence"):
        return ""

    repair = _completed_bounded_schedule_repair_request(state)
    if not repair:
        return ""

    target_texts = _dedupe_texts(
        [_schedule_repair_node_text(node) for node in repair.get("target_nodes") or []],
        limit=5,
    )
    if not target_texts:
        return ""

    details = [f"下一步应优先按原时间窗重查候选证据：{'、'.join(target_texts)}"]

    protected_texts = _dedupe_texts(
        [
            _schedule_repair_node_text(slot)
            for slot in repair.get("protected_slots") or []
        ],
        limit=3,
    )
    if protected_texts:
        details.append(
            f"继续保留 {'、'.join(protected_texts)} 作为非执行约束"
        )

    preserved_texts = _dedupe_texts(
        [
            _schedule_repair_node_text(node)
            for node in repair.get("preserve_existing_slot_fit_nodes") or []
        ],
        limit=3,
    )
    if preserved_texts:
        details.append(
            f"{'、'.join(preserved_texts)} 仅作为元数据里的时段匹配证据保留，不代表可执行"
        )

    return "；".join(details) + "。"


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
        share_msg = (
            explanation_text
            or "暂时没有找到可执行方案，可以放宽距离、预算或时间约束后再试。"
        )
        execution_log.append("✅ 分享消息已生成")
        execution_log.append(f"📨 消息内容: {share_msg[:100]}...")
        return {"final_share_message": share_msg, "execution_log": execution_log}

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
        if value.get("success")
        and value.get("data", {}).get("action") == "order_addon_service"
    ]

    # 构建时间线描述
    executed_times = _executed_start_times(tool_results)
    timeline_desc = ""
    previous_end: int | None = None
    for index, item in enumerate(timeline):
        executed_start = executed_times.get(
            str(item.get("poi_id"))
        ) or executed_times.get(str(item.get("activity")))
        item_time = item.get("time", "")
        next_item = timeline[index + 1] if index + 1 < len(timeline) else {}
        next_start, _ = _range_bounds(next_item.get("time", ""))
        if executed_start:
            item_time = (
                _retime_preserves_order(
                    item_time, executed_start, previous_end, next_start
                )
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
            guidance_tail = (
                _bounded_schedule_repair_guidance(state)
                or "建议换一个时间段、增加可预订节点，或放宽部分约束后再试。"
            )
            share_msg = (
                f"暂时没有形成可执行的预订动作：{timeline_desc}。"
                f"{non_executable_tail}"
                f"{guidance_tail}"
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
        share_msg = (
            f"📝 帮你们看了下：{timeline_desc}。需要我帮忙预订吗？确认的话说一声～"
        )

    execution_log.append(f"✅ 分享消息已生成")
    execution_log.append(f"📨 消息内容: {share_msg[:100]}...")

    return {"final_share_message": share_msg, "execution_log": execution_log}
