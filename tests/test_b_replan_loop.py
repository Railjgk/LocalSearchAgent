from __future__ import annotations

from src.graph import WORKFLOW_NODES
from src.nodes import b_replan_loop


def _request() -> dict:
    return {
        "source": "longcat_plan_critic",
        "status": "needs_replan",
        "next_step": "rerun_candidate_generation",
        "guidance": [
            "Activity and restaurant feel incoherent; re-search social lively activity candidates",
            "Avoid quiet kid_friendly low_intensity educational museums",
        ],
        "global_notes": ["Prefer board game or escape room for friends chat"],
        "candidate_generation_hints": {
            "avoid_plan_ids": ["cand_old"],
            "prefer_terms_from_guidance": ["social", "lively", "board game"],
        },
    }


def test_b_replan_loop_prepares_rag_ready_second_pass(monkeypatch):
    calls = []

    def fake_candidate_generator(state):
        calls.append("candidate")
        prefs = state["constraints"]["planning_preferences"]
        assert state["constraints"]["replan_mode"] == "ai_guided"
        assert state["constraints"]["b_replan_request"]["source"] == "longcat_plan_critic"
        assert "社交" in prefs["activity_type"]
        assert "安静" in state["user_profile"]["avoid"]
        assert "社交" in state["scenario_activities"]
        return {
            "candidates": [{"plan_id": "cand_new"}],
            "candidate_recall_diagnostics": {"replan_request_active": True},
            "execution_log": state["execution_log"] + ["candidate ok"],
        }

    def fake_constraint_filter(state):
        calls.append("filter")
        return {
            "filtered_candidates": state["candidates"],
            "filter_reasons": {},
            "execution_log": state["execution_log"] + ["filter ok"],
        }

    def fake_plan_optimizer(state):
        calls.append("optimizer")
        return {
            "selected_plan": {
                "plan_id": "plan_new",
                "execution_ready": True,
                "action_hints": [{"action_type": "reserve_restaurant", "poi_id": "res_new"}],
            },
            "optimization_score": 88.0,
            "alternative_plans": [],
            "execution_log": state["execution_log"] + ["optimizer ok"],
        }

    monkeypatch.setattr(b_replan_loop, "candidate_generator_node", fake_candidate_generator)
    monkeypatch.setattr(b_replan_loop, "constraint_filter_node", fake_constraint_filter)
    monkeypatch.setattr(b_replan_loop, "plan_optimizer_node", fake_plan_optimizer)

    result = b_replan_loop.b_replan_loop_node(
        {
            "b_replan_request": _request(),
            "selected_plan": {"plan_id": "plan_old", "plan_status": "needs_ai_replan"},
            "constraints": {"planning_preferences": {}},
            "user_profile": {},
            "scenario_activities": [],
            "b_rag_node_candidates": {"intent_01": [{"poi_id": "rag_1", "name": "桌游店"}]},
            "execution_log": [],
        }
    )

    assert calls == ["candidate", "filter", "optimizer"]
    assert result["selected_plan"]["plan_id"] == "plan_new"
    assert result["b_replan_request"] == {}
    assert result["b_replan_attempt"]["status"] == "completed"
    assert result["b_replan_attempt"]["used_rag_candidates"] is True
    assert result["b_replan_attempt"]["still_needs_replan"] is False


def test_b_replan_loop_is_bounded():
    result = b_replan_loop.b_replan_loop_node(
        {
            "b_replan_request": _request(),
            "b_replan_attempt_count": 1,
            "execution_log": [],
        }
    )

    assert result["b_replan_attempt"]["status"] == "skipped"
    assert result["b_replan_attempt"]["reason"] == "max_attempts_reached"


def test_graph_runs_replan_loop_before_explainability():
    node_names = [node.__name__ for node in WORKFLOW_NODES]

    assert node_names.index("plan_optimizer_node") < node_names.index("b_replan_loop_node")
    assert node_names.index("b_replan_loop_node") < node_names.index("explainability_node")
