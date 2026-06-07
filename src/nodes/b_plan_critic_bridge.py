"""Bridge LongCat plan critic output into bounded B replanning requests."""
from __future__ import annotations

from .b_utils import to_float


AI_REPLAN_TRIGGER_TERMS = (
    "structural",
    "re-search",
    "research",
    "mismatch",
    "incoherent",
    "not coherent",
    "clash",
    "replace",
    "重新",
    "重排",
    "不匹配",
    "不连贯",
    "冲突",
    "替换",
)


def build_ai_planning_review(selected: dict, plan_critic_metadata: dict | None) -> dict | None:
    """Compact LongCat critic output into a planner-facing review object."""

    if not plan_critic_metadata or not plan_critic_metadata.get("enabled"):
        return None

    plan = selected.get("plan", {}) or {}
    selected_plan_id = str(plan.get("plan_id") or "").strip()
    applied_adjustments = plan_critic_metadata.get("applied_adjustments") or []
    selected_adjustment = next(
        (
            adjustment
            for adjustment in applied_adjustments
            if str(adjustment.get("plan_id") or "").strip() == selected_plan_id
        ),
        None,
    )

    global_notes = [
        str(note).strip()
        for note in (plan_critic_metadata.get("global_notes") or [])
        if str(note).strip()
    ][:4]
    selected_reasons = []
    selected_evidence = []
    if selected_adjustment:
        selected_reasons = [
            str(reason).strip()
            for reason in (selected_adjustment.get("reasons") or [])
            if str(reason).strip()
        ][:5]
        selected_evidence = [
            str(item).strip()
            for item in (selected_adjustment.get("evidence") or [])
            if str(item).strip()
        ][:5]

    review_text = " ".join(global_notes + selected_reasons + selected_evidence).lower()
    risk_delta = to_float((selected_adjustment or {}).get("risk_delta"), 0.0)
    score_delta = to_float((selected_adjustment or {}).get("score_delta"), 0.0)
    confidence = to_float((selected_adjustment or {}).get("confidence"), 0.0)
    term_trigger = any(term.lower() in review_text for term in AI_REPLAN_TRIGGER_TERMS)
    strong_negative_adjustment = confidence >= 0.6 and (risk_delta >= 0.03 or score_delta <= -0.02)
    needs_replan = bool(plan_critic_metadata.get("success") and (term_trigger or strong_negative_adjustment))
    replan_guidance = (selected_reasons or global_notes)[:3] if needs_replan else []

    return {
        "enabled": bool(plan_critic_metadata.get("enabled")),
        "success": bool(plan_critic_metadata.get("success")),
        "provider": plan_critic_metadata.get("provider"),
        "model": plan_critic_metadata.get("model"),
        "selected_after_critic": plan_critic_metadata.get("selected_after_critic"),
        "selected_adjustment": selected_adjustment or {},
        "global_notes": global_notes,
        "needs_replan": needs_replan,
        "replan_trigger": {
            "term_trigger": term_trigger,
            "strong_negative_adjustment": strong_negative_adjustment,
            "risk_delta": round(risk_delta, 4),
            "score_delta": round(score_delta, 4),
            "confidence": round(confidence, 3),
        },
        "replan_guidance": replan_guidance,
        "guardrails": plan_critic_metadata.get("guardrails", {}),
    }


def build_b_replan_request(
    selected_plan: dict,
    plan_base: dict,
    ai_planning_review: dict | None,
    constraints: dict,
) -> dict | None:
    """Create a bounded request for a future B replan loop."""

    if not ai_planning_review or not ai_planning_review.get("needs_replan"):
        return None

    planning_preferences = constraints.get("planning_preferences") or {}
    guidance = ai_planning_review.get("replan_guidance", [])
    global_notes = ai_planning_review.get("global_notes", [])
    guidance_text = " ".join(str(item) for item in guidance + global_notes).lower()
    supply_identity = selected_plan.get("supply_identity", {})
    target_domains = []
    if any(term in guidance_text for term in ("activity", "museum", "poi", "活动", "博物馆", "场所")):
        target_domains.append("activity")
    if any(term in guidance_text for term in ("restaurant", "hotpot", "餐厅", "火锅", "吃")):
        target_domains.append("restaurant")
    if not target_domains:
        target_domains = ["activity", "restaurant"]

    rag_target_nodes = []
    for domain in target_domains:
        avoid_poi_ids = []
        if domain == "activity" and supply_identity.get("activity_id"):
            avoid_poi_ids.append(supply_identity.get("activity_id"))
        if domain == "restaurant" and supply_identity.get("restaurant_id"):
            avoid_poi_ids.append(supply_identity.get("restaurant_id"))
        rag_target_nodes.append(
            {
                "supply_domain": domain,
                "query_terms": guidance[:3] or global_notes[:3],
                "avoid_poi_ids": avoid_poi_ids,
                "max_candidates": 12,
            }
        )

    return {
        "source": "longcat_plan_critic",
        "status": "needs_replan",
        "next_step": "rerun_candidate_generation",
        "reason": "LongCat critic found plan-level experience or coherence risks before execution.",
        "rejected_plan_id": selected_plan.get("plan_id"),
        "rejected_candidate_id": plan_base.get("plan_id"),
        "preserve_constraints": {
            "scene_type": selected_plan.get("scene_type"),
            "people_count": selected_plan.get("people_count"),
            "budget": constraints.get("budget"),
            "max_distance_km": constraints.get("max_distance_km"),
            "max_queue_time_min": constraints.get("max_queue_time_min")
            or constraints.get("max_queue_time"),
            "duration_range": constraints.get("duration_range"),
            "planning_preferences": planning_preferences,
            "b_requirement_contract": constraints.get("b_requirement_contract", {}),
        },
        "guidance": guidance,
        "global_notes": global_notes,
        "critic_trigger": ai_planning_review.get("replan_trigger", {}),
        "rag_request": {
            "request_type": "replacement_poi_candidates",
            "target_nodes": rag_target_nodes,
            "expected_output_key": "b_rag_candidate_evidence",
            "compatible_output_keys": ["b_rag_candidate_evidence", "b_rag_node_candidates"],
            "contract": "Return b_rag_candidate_evidence.node_evidence[].candidates[] or node-keyed b_rag_node_candidates compatible with src.nodes.b_rag_contract.normalize_rag_node_candidates",
        },
        "candidate_generation_hints": {
            "avoid_plan_ids": [plan_base.get("plan_id")] if plan_base.get("plan_id") else [],
            "avoid_supply_identity": supply_identity,
            "prefer_terms_from_guidance": guidance,
        },
    }
