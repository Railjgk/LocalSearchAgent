"""
Execution Manager - 管理执行过程、重试、状态追踪
C负责
"""

from src.state import PlanState
from src.tools.mock_apis import (
    reserve_restaurant,
    order_activity_ticket,
    call_taxi,
    order_addon_service
)
from typing import Dict, Any, List


def _summarize_tool_results(tool_results: Dict[str, Any]) -> tuple[int, int, str]:
    success_count = sum(1 for value in tool_results.values() if value["success"])
    fail_count = len(tool_results) - success_count

    if not tool_results:
        execution_status = "failed"
    elif fail_count == 0:
        execution_status = "success"
    elif success_count > 0:
        execution_status = "partial"
    else:
        execution_status = "failed"

    return success_count, fail_count, execution_status


def _tool_result_from_raw_api(value: Dict[str, Any]) -> Dict[str, Any]:
    result = value.get("result", {})
    return {
        "success": result.get("success", False),
        "data": result,
        "action": value.get("action", ""),
        "name": value.get("name", ""),
    }


def _tool_result_from_commit_result(execution_commit_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": execution_commit_result.get("success", False),
        "data": execution_commit_result,
        "action": "execution_commit",
        "name": execution_commit_result.get("execution_id", ""),
    }


def _run_fallback_action(action: Dict[str, Any]) -> Dict[str, Any]:
    action_type = action.get("action_type")
    if action_type == "reserve_restaurant":
        return reserve_restaurant(
            poi_id=action.get("poi_id"),
            time_slot=action.get("time"),
            people=action.get("people", 2),
            notes=action.get("notes", []),
        )
    if action_type == "order_activity_ticket":
        return order_activity_ticket(
            poi_id=action.get("poi_id"),
            time_slot=action.get("time"),
            quantity=action.get("quantity", 1),
            notes=action.get("notes", []),
        )
    if action_type == "order_addon_service":
        return order_addon_service(
            addon_type=action.get("addon_type", "cake"),
            poi_id=action.get("poi_id", ""),
            delivery_time=action.get("time", "18:00"),
            special_requests=action.get("notes", []),
        )
    if action_type == "call_taxi":
        return call_taxi(
            start=action.get("start", "家"),
            end=action.get("end", "目的地"),
            time_slot=action.get("time", ""),
        )
    return {"success": False, "error": f"未知动作类型: {action_type}"}


def _append_tool_result_log(
    execution_log: List[str],
    *,
    action: str,
    name: str,
    result: Dict[str, Any],
    prefix: str,
) -> None:
    status_icon = "✅" if result.get("success") else "❌"
    message = result.get("message", result.get("error", ""))
    execution_log.append(f"   {status_icon} {prefix}: {action} - {name} {message}")


def _collect_existing_raw_results(
    existing_raw_results: Dict[str, Any],
    execution_log: List[str],
) -> Dict[str, Any]:
    tool_results: Dict[str, Any] = {}
    for key, value in existing_raw_results.items():
        result = value.get("result", {})
        tool_results[key] = _tool_result_from_raw_api(value)
        _append_tool_result_log(
            execution_log,
            action=value.get("action", ""),
            name=value.get("name", ""),
            result=result,
            prefix="汇总",
        )
    return tool_results


def _collect_execution_commit_result(
    execution_commit_result: Dict[str, Any],
    execution_log: List[str],
) -> Dict[str, Any]:
    execution_log.append(
        "   ✅ 汇总: execution_commit"
        if execution_commit_result.get("success")
        else "   ❌ 汇总: execution_commit"
    )
    return {
        "execution_commit": _tool_result_from_commit_result(execution_commit_result)
    }


def _record_fallback_action_result(
    *,
    action: Dict[str, Any],
    result: Dict[str, Any],
    raw_results: Dict[str, Any],
    tool_results: Dict[str, Any],
    execution_log: List[str],
) -> None:
    action_type = action.get("action_type")
    step = action.get("step")
    name = action.get("name", "")
    result_key = f"{action_type}_{step}"

    raw_results[result_key] = {
        "action": action_type,
        "name": name,
        "result": result,
    }
    tool_results[result_key] = {
        "success": result.get("success", False),
        "data": result,
        "action": action_type,
        "name": name,
    }
    if result.get("success"):
        execution_log.append(
            f"      ✅ {action_type} 成功: {result.get('message', result.get('order_id', ''))}"
        )
    else:
        execution_log.append(
            f"      ❌ {action_type} 失败: {result.get('error', '未知错误')}"
        )


def _execute_action_sequence_fallback(
    action_sequence: List[Dict[str, Any]],
    execution_log: List[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    tool_results: Dict[str, Any] = {}
    raw_results: Dict[str, Any] = {}
    for action in action_sequence:
        action_type = action.get("action_type")
        name = action.get("name", "")
        execution_log.append(f"   ▶ 执行: {action_type} - {name}")
        result = _run_fallback_action(action)
        _record_fallback_action_result(
            action=action,
            result=result,
            raw_results=raw_results,
            tool_results=tool_results,
            execution_log=execution_log,
        )
    return tool_results, raw_results


def _as_text_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _format_timeline_item_zh(item: Dict[str, Any]) -> str:
    time_text = str(item.get("time") or "").strip()
    activity = str(item.get("activity") or item.get("name") or "").strip()
    if time_text and activity:
        return f"{time_text} {activity}"
    return activity or time_text


def _is_protected_non_executable_anchor(item: Dict[str, Any]) -> bool:
    text = " ".join(
        str(part or "")
        for part in (
            item.get("activity"),
            item.get("name"),
            item.get("role"),
            item.get("itinerary_role"),
            " ".join(str(note) for note in item.get("notes", []) if note),
        )
    )
    protected_terms = ("午睡", "休息", "自由活动", "缓冲", "到达", "返程")
    return any(term in text for term in protected_terms)


def _collect_upstream_blocker_reasons_zh(
    selected_plan: Dict[str, Any],
    b_replan_request: Dict[str, Any],
) -> List[str]:
    reasons: List[str] = []

    for key in (
        "execution_blockers",
        "blocking_reasons",
        "blocker_reasons",
        "non_executable_reasons",
    ):
        reasons.extend(_as_text_list(selected_plan.get(key)))

    execution_contract = selected_plan.get("execution_contract") or {}
    reasons.extend(_as_text_list(execution_contract.get("blocking_reasons")))

    candidate_evidence = selected_plan.get("candidate_evidence_grade") or {}
    reasons.extend(_as_text_list(candidate_evidence.get("hard_filter_reason")))

    for key in (
        "reason_zh",
        "blocker_summary_zh",
        "reason",
        "summary_zh",
        "blocking_reasons",
    ):
        reasons.extend(_as_text_list(b_replan_request.get(key)))

    deduped = []
    seen = set()
    for reason in reasons:
        if reason not in seen:
            deduped.append(reason)
            seen.add(reason)
    return deduped


def _is_completed_replan_trace(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("status") != "completed":
        return False
    return value.get("trace_only") is True or (
        value.get("post_replan_trace_status") == "completed"
    )


def _completed_replan_metadata(state: PlanState) -> Dict[str, Any]:
    attempt = state.get("b_replan_attempt") or {}
    completed_attempt = (
        isinstance(attempt, dict) and attempt.get("status") == "completed"
    )

    constraints = state.get("constraints") or {}
    trace = constraints.get("b_replan_trace") if isinstance(constraints, dict) else {}
    last_completed = (
        trace.get("last_completed_request")
        if isinstance(trace, dict) and trace.get("status") == "completed"
        else {}
    )
    completed_trace = _is_completed_replan_trace(last_completed)

    if not completed_attempt and not completed_trace:
        return {}

    source = ""
    if completed_attempt:
        source = str(attempt.get("source_request") or "").strip()
    if not source and isinstance(last_completed, dict):
        source = str(
            last_completed.get("source")
            or (last_completed.get("candidate_generation_hints") or {}).get(
                "evidence_source"
            )
            or ""
        ).strip()

    candidate_count = attempt.get("candidate_count") if completed_attempt else None
    filtered_count = attempt.get("filtered_count") if completed_attempt else None

    result_zh = "已完成一次有界候选证据/时段修复，但仍未形成可执行动作。"
    if isinstance(candidate_count, int) and isinstance(filtered_count, int):
        result_zh = (
            "已完成一次有界候选证据/时段修复，"
            f"候选计数为{candidate_count}、过滤后计数为{filtered_count}，"
            "仍未形成可执行动作。"
        )

    return {
        "completed_replan_attempt": True,
        "replan_source": source or None,
        "replan_candidate_count": candidate_count,
        "replan_filtered_count": filtered_count,
        "replan_result_zh": result_zh,
    }


def _build_no_action_execution_blocker(state: PlanState) -> Dict[str, Any]:
    selected_plan = state.get("selected_plan", {}) or {}
    b_replan_request = (
        state.get("b_replan_request")
        or selected_plan.get("b_replan_request")
        or {}
    )
    timeline = selected_plan.get("timeline", []) or []
    candidate_evidence = selected_plan.get("candidate_evidence_grade") or {}

    upstream_reasons = _collect_upstream_blocker_reasons_zh(
        selected_plan,
        b_replan_request,
    )
    protected_anchors = [
        formatted
        for item in timeline
        if isinstance(item, dict)
        for formatted in [_format_timeline_item_zh(item)]
        if formatted and _is_protected_non_executable_anchor(item)
    ]
    executable_intent_nodes = [
        formatted
        for item in timeline
        if isinstance(item, dict)
        for formatted in [_format_timeline_item_zh(item)]
        if formatted and not _is_protected_non_executable_anchor(item)
    ]

    reason_parts = list(upstream_reasons)
    plan_status = selected_plan.get("plan_status")
    if plan_status == "needs_rag_candidate_evidence":
        reason_parts.append("计划缺少可执行候选证据，暂不能进入预订或购票。")
    if selected_plan.get("execution_ready") is False:
        reason_parts.append("上游计划标记为不可执行，C阶段未收到可执行动作。")
    if not reason_parts:
        reason_parts.append(
            "上游计划未提供可执行动作，且没有工具/API执行结果；当前只能作为非可执行行程意图处理，不能声称已预订。"
        )

    blocker_summary_zh = "；".join(reason_parts)
    node_reasons_zh = []
    for item in candidate_evidence.get("nodes", []) or []:
        if not isinstance(item, dict):
            continue
        node = _format_timeline_item_zh(
            {"time": item.get("time"), "activity": item.get("label")}
        )
        if not node:
            continue
        node_status = str(item.get("execution_status") or item.get("grade") or "")
        if (
            node_status == "protected_non_executable"
            or item.get("grade") == "protected_guidance"
        ):
            continue
        reason = str(item.get("hard_filter_reason") or "").strip()
        if reason:
            node_reasons_zh.append(f"{node} 缺少可执行证据：{reason}")
        else:
            node_reasons_zh.append(f"{node} 缺少可执行候选或动作证据。")

    if not node_reasons_zh:
        node_reasons_zh = [
            f"{node} 缺少可执行候选或动作证据。"
            for node in executable_intent_nodes
        ]

    blocker = {
        "source": "execution_handoff",
        "reason_code": "no_executable_actions",
        "reason_zh": reason_parts[0],
        "blocker_summary_zh": blocker_summary_zh,
        "node_reasons_zh": node_reasons_zh,
        "protected_non_executable_anchors_zh": protected_anchors,
        "selected_plan_status": plan_status,
        "execution_ready": selected_plan.get("execution_ready"),
        "upstream_replan_request": bool(b_replan_request),
        "candidate_evidence_counts": {
            "raw_candidate_count": candidate_evidence.get("raw_candidate_count"),
            "normalized_candidate_count": candidate_evidence.get(
                "normalized_candidate_count"
            ),
        },
    }
    blocker.update(_completed_replan_metadata(state))
    return blocker


def execution_manager_node(state: PlanState) -> Dict[str, Any]:
    """
    汇总 action_sequence 的执行结果，必要时兜底调用 Mock API。
    """
    print("⚙️ [10] Execution Manager: 汇总执行结果...")

    execution_log = state.get("execution_log", [])
    action_sequence = state.get("action_sequence", [])
    existing_raw_results = state.get("raw_api_results", {})
    execution_commit_result = state.get("execution_commit_result", {})
    tool_results = {}
    raw_results = existing_raw_results or {}

    if existing_raw_results:
        tool_results = _collect_existing_raw_results(existing_raw_results, execution_log)
    elif execution_commit_result:
        tool_results = _collect_execution_commit_result(execution_commit_result, execution_log)
    elif not action_sequence:
        execution_log.append("⚠️ Execution Manager: 没有可执行动作")
    else:
        tool_results, raw_results = _execute_action_sequence_fallback(
            action_sequence,
            execution_log,
        )

    # 统计执行结果
    success_count, fail_count, execution_status = _summarize_tool_results(tool_results)

    execution_log.append(f"📊 执行统计: 成功{success_count}/{len(tool_results)}, 失败{fail_count}")
    execution_log.append(f"📊 执行状态: {execution_status}")

    result = {
        "tool_results": tool_results,
        "raw_api_results": raw_results,
        "execution_status": execution_status,
        "execution_log": execution_log
    }

    if not action_sequence and not existing_raw_results and not execution_commit_result:
        result["execution_failure_type"] = "no_executable_actions"
        result["execution_blocker"] = _build_no_action_execution_blocker(state)

    return result
