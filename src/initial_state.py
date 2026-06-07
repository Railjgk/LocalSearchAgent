"""Initial state builders shared by CLI and debugging entry points."""

from __future__ import annotations

from src.state import PlanState


def build_user_initial_state(
    user_input: str,
    *,
    user_id: str = "u001",
    payment_ui_mode: str = "auto",
) -> PlanState:
    """Build a real-user initial state and let A-stage fill planning fields."""

    return {
        "user_id": user_id,
        "user_input": user_input.strip(),
        "scene_type": "unknown",
        "constraints": {},
        "user_profile": {},
        "short_term_memory": [],
        "scenario_activities": [],
        "candidates": [],
        "filtered_candidates": [],
        "filter_reasons": {},
        "selected_plan": {},
        "optimization_score": 0.0,
        "alternative_plans": [],
        "explanation_text": "",
        "action_sequence": [],
        "raw_api_results": {},
        "execution_status": "pending",
        "tool_results": {},
        "payment_order": {},
        "payment_results": {},
        "payment_status": "not_required",
        "retry_history": [],
        "final_share_message": "",
        "execution_log": [],
        "retry_count": 0,
        "need_confirm": False,
        "payment_ui_mode": payment_ui_mode,
        "payment_auto_confirm": True,
        "payment_auto_pay": True,
        "payment_method": "mock_pay",
    }
