"""One-pass B replan loop driven by LongCat critic output.

The loop is deliberately bounded.  LongCat can ask B to replan, but it cannot
invent POIs or execute tools.  Candidate generation still uses local supply or
future RAG-provided ``b_rag_node_candidates``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from src.state import PlanState
except ImportError:  # pragma: no cover
    PlanState = dict

from src.nodes.b_poi_rag import b_poi_rag_node
from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.plan_optimizer import plan_optimizer_node


MAX_REPLAN_ATTEMPTS = 1

PREFER_TERM_RULES = (
    (("social", "group", "chat", "lively", "朋友", "社交", "热闹", "聊天"), ["社交", "热闹", "适合聊天", "朋友聚会"]),
    (("escape room", "密室"), ["密室", "沉浸式游戏"]),
    (("board game", "桌游"), ["桌游", "棋牌桌游"]),
    (("karaoke", "ktv", "唱歌"), ["KTV", "唱歌"]),
    (("citywalk", "local", "本地", "市集"), ["citywalk", "市集", "本地生活"]),
    (("family", "kid", "亲子", "孩子"), ["亲子", "儿童友好", "低强度"]),
)

AVOID_TERM_RULES = (
    (("quiet", "安静"), ["安静"]),
    (("kid_friendly", "child-oriented", "亲子", "儿童"), ["亲子", "儿童"]),
    (("low_intensity", "educational", "低强度", "教育"), ["低强度", "教育"]),
    (("crowded", "拥挤"), ["拥挤"]),
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _dedupe_keep_order(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _rule_terms(text: str, rules: tuple[tuple[tuple[str, ...], list[str]], ...]) -> list[str]:
    lowered = text.lower()
    terms: list[str] = []
    for needles, values in rules:
        if any(needle.lower() in lowered for needle in needles):
            terms.extend(values)
    return _dedupe_keep_order(terms)


def _replan_text(request: dict) -> str:
    parts: list[str] = []
    parts.extend(str(item) for item in _as_list(request.get("guidance")))
    parts.extend(str(item) for item in _as_list(request.get("global_notes")))
    hints = request.get("candidate_generation_hints") or {}
    parts.extend(str(item) for item in _as_list(hints.get("prefer_terms_from_guidance")))
    return "\n".join(parts)


def _prepare_replan_state(state: PlanState, request: dict) -> PlanState:
    replan_state = deepcopy(dict(state))
    constraints = dict(replan_state.get("constraints") or {})
    user_profile = dict(replan_state.get("user_profile") or {})
    planning_preferences = dict(constraints.get("planning_preferences") or {})
    text = _replan_text(request)
    prefer_terms = _rule_terms(text, PREFER_TERM_RULES)
    avoid_terms = _rule_terms(text, AVOID_TERM_RULES)

    if prefer_terms:
        planning_preferences["activity_type"] = _dedupe_keep_order(
            _as_list(planning_preferences.get("activity_type")) + prefer_terms
        )
        planning_preferences["experience_type"] = _dedupe_keep_order(
            _as_list(planning_preferences.get("experience_type")) + prefer_terms
        )
    if avoid_terms:
        user_profile["avoid"] = _dedupe_keep_order(_as_list(user_profile.get("avoid")) + avoid_terms)

    constraints["planning_preferences"] = planning_preferences
    constraints["b_replan_request"] = request
    constraints["replan_mode"] = "ai_guided"
    replan_state["constraints"] = constraints
    replan_state["user_profile"] = user_profile
    replan_state["scenario_activities"] = _dedupe_keep_order(
        _as_list(replan_state.get("scenario_activities")) + prefer_terms
    )

    for key in (
        "selected_plan",
        "optimization_score",
        "alternative_plans",
        "b_ai_planning_review",
        "b_ai_plan_critic",
        "action_sequence",
        "raw_api_results",
        "execution_commit_result",
        "retry_history",
        "b_replan_request",
    ):
        replan_state.pop(key, None)
    replan_state["b_replan_attempt_count"] = int(state.get("b_replan_attempt_count") or 0) + 1
    return replan_state


def b_replan_loop_node(state: PlanState) -> dict[str, Any]:
    """Run one bounded B replan pass when ``b_replan_request`` is present."""

    request = state.get("b_replan_request") or (state.get("selected_plan", {}) or {}).get("b_replan_request")
    if not isinstance(request, dict) or not request:
        return {}

    attempt_count = int(state.get("b_replan_attempt_count") or 0)
    execution_log = list(state.get("execution_log", []) or [])
    if attempt_count >= MAX_REPLAN_ATTEMPTS:
        execution_log.append("[B] b_replan_loop_node skipped; max replan attempts reached")
        return {
            "execution_log": execution_log,
            "b_replan_attempt": {
                "status": "skipped",
                "reason": "max_attempts_reached",
                "attempt_count": attempt_count,
            },
        }

    replan_state = _prepare_replan_state(state, request)
    replan_state["execution_log"] = execution_log + [
        "[B] b_replan_loop_node started one-pass AI-guided replan"
    ]

    for node in (b_poi_rag_node, candidate_generator_node, constraint_filter_node, plan_optimizer_node):
        replan_state.update(node(replan_state))

    selected_plan = replan_state.get("selected_plan") or {}
    new_request = replan_state.get("b_replan_request") or selected_plan.get("b_replan_request") or {}
    updates = {
        "candidates": replan_state.get("candidates", []),
        "candidate_generation_issues": replan_state.get("candidate_generation_issues", []),
        "candidate_recall_diagnostics": replan_state.get("candidate_recall_diagnostics", {}),
        "filtered_candidates": replan_state.get("filtered_candidates", []),
        "filter_reasons": replan_state.get("filter_reasons", {}),
        "selected_plan": selected_plan,
        "optimization_score": replan_state.get("optimization_score", 0.0),
        "alternative_plans": replan_state.get("alternative_plans", []),
        "constraints": replan_state.get("constraints", state.get("constraints", {})),
        "user_profile": replan_state.get("user_profile", state.get("user_profile", {})),
        "scenario_activities": replan_state.get("scenario_activities", state.get("scenario_activities", [])),
        "b_requirement_contract": replan_state.get("b_requirement_contract", state.get("b_requirement_contract", {})),
        "b_itinerary_blueprint": replan_state.get("b_itinerary_blueprint", state.get("b_itinerary_blueprint", {})),
        "b_rag_candidate_evidence": replan_state.get(
            "b_rag_candidate_evidence",
            state.get("b_rag_candidate_evidence", {}),
        ),
        "b_poi_rag_metadata": replan_state.get("b_poi_rag_metadata", state.get("b_poi_rag_metadata", {})),
        "b_rag_candidate_metadata": replan_state.get("b_rag_candidate_metadata", state.get("b_rag_candidate_metadata", {})),
        "b_rag_candidate_coverage": replan_state.get("b_rag_candidate_coverage", state.get("b_rag_candidate_coverage", {})),
        "b_ai_semantic_hints": replan_state.get("b_ai_semantic_hints", state.get("b_ai_semantic_hints", {})),
        "b_ai_requirement_compiler": replan_state.get(
            "b_ai_requirement_compiler",
            state.get("b_ai_requirement_compiler", {}),
        ),
        "b_ai_plan_critic": replan_state.get("b_ai_plan_critic", {}),
        "b_ai_planning_review": replan_state.get("b_ai_planning_review", {}),
        "b_replan_request": new_request,
        "b_replan_attempt_count": int(replan_state.get("b_replan_attempt_count") or 1),
        "b_replan_attempt": {
            "status": "completed",
            "source_request": request.get("source"),
            "used_rag_candidates": bool(
                state.get("b_rag_node_candidates")
                or state.get("rag_node_candidates")
                or replan_state.get("b_rag_candidate_evidence")
            ),
            "candidate_count": len(replan_state.get("candidates", []) or []),
            "filtered_count": len(replan_state.get("filtered_candidates", []) or []),
            "selected_plan_id": selected_plan.get("plan_id"),
            "still_needs_replan": bool(new_request),
        },
        "execution_log": replan_state.get("execution_log", []),
    }
    if not new_request:
        updates["b_replan_request"] = {}
    return updates
