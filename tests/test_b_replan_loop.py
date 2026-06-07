from __future__ import annotations

from src.graph import WORKFLOW_NODES
from src.nodes import b_replan_loop
from src.nodes.plan_optimizer import plan_optimizer_node


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
    assert "b_replan_request" not in result["constraints"]
    assert result["constraints"]["b_replan_trace"]["status"] == "completed"
    assert result["constraints"]["b_replan_trace"]["last_completed_request"]["source"] == "longcat_plan_critic"
    assert result["constraints"]["b_replan_trace"]["last_completed_request"]["status"] == "completed"
    assert result["constraints"]["b_replan_trace"]["last_completed_request"]["trace_only"] is True
    assert result["constraints"]["b_replan_trace"]["last_completed_request"]["post_replan_trace_status"] == "completed"
    assert result["candidate_recall_diagnostics"]["replan_request_active"] is False
    assert result["candidate_recall_diagnostics"]["last_replan_request"]["source"] == "longcat_plan_critic"
    assert result["candidate_recall_diagnostics"]["last_replan_request"]["status"] == "completed"
    assert result["candidate_recall_diagnostics"]["last_replan_request"]["trace_only"] is True
    assert result["candidate_recall_diagnostics"]["last_replan_request"]["post_replan_trace_status"] == "completed"
    assert result["b_replan_attempt"]["status"] == "completed"
    assert result["b_replan_attempt"]["used_rag_candidates"] is True
    assert result["b_replan_attempt"]["still_needs_replan"] is False


def test_completed_skeleton_replan_trace_syncs_post_pass_evidence(monkeypatch):
    request = {
        "source": "skeleton_candidate_evidence",
        "status": "needs_replan",
        "next_step": "rerun_candidate_generation",
        "rag_request": {
            "target_nodes": [
                {
                    "node_id": "intent_01",
                    "label": "亲子活动",
                    "grade": "missing_node_evidence",
                    "execution_status": "missing_executable_evidence",
                    "coverage_status": "missing",
                    "has_raw_candidate_coverage": False,
                    "query_terms": ["亲子活动"],
                },
                {
                    "node_id": "legacy_intent_02",
                    "label": "晚餐",
                    "grade": "missing_node_evidence",
                    "execution_status": "missing_executable_evidence",
                    "coverage_status": "missing",
                    "has_raw_candidate_coverage": False,
                    "query_terms": ["晚餐", "清淡"],
                },
            ]
        },
    }
    hard_filter_reason = "已召回活动和餐饮候选，但尚未确认任何活动加晚餐组合同时满足硬时间窗、过敏、少步行等硬约束"
    selected_plan = {
        "plan_id": "plan_legacy_pair_skeleton",
        "execution_ready": False,
        "action_hints": [],
        "candidate_evidence_grade": {
            "version": "candidate_evidence_grade_v1",
            "applies_to": "non_executable_skeleton",
            "hard_filter_reason": hard_filter_reason,
            "nodes": [
                {
                    "node_id": "intent_01",
                    "label": "亲子活动",
                    "grade": "all_candidates_filtered",
                    "execution_status": "raw_candidates_failed_hard_constraints",
                    "has_raw_candidate_coverage": True,
                    "hard_filter_reason": hard_filter_reason,
                },
                {
                    "node_id": "intent_02",
                    "label": "晚餐",
                    "grade": "all_candidates_filtered",
                    "execution_status": "raw_candidates_failed_hard_constraints",
                    "has_raw_candidate_coverage": True,
                    "hard_filter_reason": hard_filter_reason,
                },
                {
                    "node_id": "intent_rest",
                    "label": "午睡/休息",
                    "grade": "protected_guidance",
                    "execution_status": "protected_non_executable",
                    "has_raw_candidate_coverage": False,
                },
            ],
        },
    }

    def fake_rag(state):
        return {"execution_log": state["execution_log"] + ["rag ok"]}

    def fake_candidate_generator(state):
        return {
            "candidates": [],
            "candidate_recall_diagnostics": {"replan_request_active": True},
            "constraints": state["constraints"],
            "execution_log": state["execution_log"] + ["candidate empty"],
        }

    def fake_constraint_filter(state):
        return {
            "filtered_candidates": [],
            "filter_reasons": {},
            "execution_log": state["execution_log"] + ["filter empty"],
        }

    def fake_plan_optimizer(state):
        return {
            "selected_plan": selected_plan,
            "optimization_score": 0.0,
            "alternative_plans": [],
            "execution_log": state["execution_log"] + ["optimizer skeleton"],
        }

    monkeypatch.setattr(b_replan_loop, "b_poi_rag_node", fake_rag)
    monkeypatch.setattr(b_replan_loop, "candidate_generator_node", fake_candidate_generator)
    monkeypatch.setattr(b_replan_loop, "constraint_filter_node", fake_constraint_filter)
    monkeypatch.setattr(b_replan_loop, "plan_optimizer_node", fake_plan_optimizer)

    result = b_replan_loop.b_replan_loop_node(
        {
            "b_replan_request": request,
            "selected_plan": {"plan_id": "plan_old"},
            "constraints": {},
            "execution_log": [],
        }
    )

    assert result["b_replan_request"] == {}
    assert result["candidate_recall_diagnostics"]["replan_request_active"] is False
    archives = [
        result["candidate_recall_diagnostics"]["last_replan_request"],
        result["constraints"]["b_replan_trace"]["last_completed_request"],
    ]
    for archive in archives:
        assert archive["status"] == "completed"
        assert archive["trace_only"] is True
        assert archive["post_replan_trace_status"] == "completed"
        targets = archive["rag_request"]["target_nodes"]
        assert [(node["label"], node["coverage_status"]) for node in targets] == [
            ("亲子活动", "filtered"),
            ("晚餐", "filtered"),
        ]
        assert all(node["grade"] == "all_candidates_filtered" for node in targets)
        assert all(node["execution_status"] == "raw_candidates_failed_hard_constraints" for node in targets)
        assert all(node["has_raw_candidate_coverage"] is True for node in targets)
        assert all(node["hard_filter_reason"] == hard_filter_reason for node in targets)
        assert [node["label"] for node in targets] == ["亲子活动", "晚餐"]
        assert "午睡/休息" not in {node["label"] for node in targets}
        assert all("poi_id" not in node for node in targets)
        assert targets[1]["node_id"] == "legacy_intent_02"
        assert targets[1]["query_terms"] == ["晚餐", "清淡"]

    assert result["selected_plan"]["execution_ready"] is False
    assert result["selected_plan"]["action_hints"] == []


def test_active_followup_replan_request_is_not_archived_or_rewritten(monkeypatch):
    request = {
        "source": "skeleton_candidate_evidence",
        "status": "needs_replan",
        "rag_request": {
            "target_nodes": [
                {
                    "node_id": "intent_01",
                    "label": "亲子活动",
                    "grade": "missing_node_evidence",
                    "coverage_status": "missing",
                }
            ]
        },
    }
    followup_request = {
        "source": "skeleton_candidate_evidence",
        "status": "needs_replan",
        "rag_request": {
            "target_nodes": [
                {
                    "node_id": "intent_02",
                    "label": "晚餐",
                    "grade": "missing_node_evidence",
                    "coverage_status": "missing",
                }
            ]
        },
    }
    selected_plan = {
        "plan_id": "plan_still_needs_replan",
        "b_replan_request": followup_request,
        "candidate_evidence_grade": {
            "applies_to": "non_executable_skeleton",
            "nodes": [
                {
                    "node_id": "intent_01",
                    "label": "亲子活动",
                    "grade": "all_candidates_filtered",
                    "execution_status": "raw_candidates_failed_hard_constraints",
                    "has_raw_candidate_coverage": True,
                }
            ],
        },
    }

    def fake_rag(state):
        return {"execution_log": state["execution_log"] + ["rag ok"]}

    def fake_candidate_generator(state):
        return {
            "candidates": [],
            "candidate_recall_diagnostics": {"replan_request_active": True},
            "constraints": state["constraints"],
            "execution_log": state["execution_log"] + ["candidate empty"],
        }

    def fake_constraint_filter(state):
        return {
            "filtered_candidates": [],
            "filter_reasons": {},
            "execution_log": state["execution_log"] + ["filter empty"],
        }

    def fake_plan_optimizer(state):
        return {
            "selected_plan": selected_plan,
            "optimization_score": 0.0,
            "alternative_plans": [],
            "execution_log": state["execution_log"] + ["optimizer followup"],
        }

    monkeypatch.setattr(b_replan_loop, "b_poi_rag_node", fake_rag)
    monkeypatch.setattr(b_replan_loop, "candidate_generator_node", fake_candidate_generator)
    monkeypatch.setattr(b_replan_loop, "constraint_filter_node", fake_constraint_filter)
    monkeypatch.setattr(b_replan_loop, "plan_optimizer_node", fake_plan_optimizer)

    result = b_replan_loop.b_replan_loop_node(
        {
            "b_replan_request": request,
            "selected_plan": {"plan_id": "plan_old"},
            "constraints": {},
            "execution_log": [],
        }
    )

    assert result["b_replan_request"] == followup_request
    assert result["b_replan_request"]["status"] == "needs_replan"
    assert "trace_only" not in result["b_replan_request"]
    assert "post_replan_trace_status" not in result["b_replan_request"]
    assert result["candidate_recall_diagnostics"] == {"replan_request_active": True}
    assert "b_replan_trace" not in result["constraints"]
    assert result["constraints"]["b_replan_request"] == request
    assert result["b_replan_attempt"]["still_needs_replan"] is True


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


def test_legacy_pair_skeleton_replan_targets_both_chinese_slots():
    blueprint = {
        "template_mode": "legacy_pair",
        "planning_horizon": "half_day",
        "planning_days": 1,
        "node_intents": [
            {"node_id": "intent_01", "role": "family_activity", "label": "亲子活动"},
            {"node_id": "intent_02", "role": "restaurant_dinner", "label": "晚餐"},
        ],
        "time_skeleton": {
            "days": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "label": "亲子活动",
                            "role": "family_activity",
                            "supply_domain": "activity",
                            "start_time": "16:20",
                            "end_time": "18:13",
                            "duration_min": 113,
                        },
                        {
                            "node_id": "intent_02",
                            "label": "晚餐",
                            "role": "restaurant_dinner",
                            "supply_domain": "restaurant",
                            "start_time": "18:13",
                            "end_time": "19:30",
                            "duration_min": 77,
                        },
                    ],
                }
            ]
        },
    }
    result = plan_optimizer_node(
        {
            "user_input": "16:20后亲子活动，再吃清淡晚餐，孩子坚果过敏，外婆少走路，19:30前结束。",
            "scene_type": "family",
            "constraints": {
                "raw_text": "16:20 19:30 亲子活动 晚餐 清淡 坚果过敏 少走路",
                "hard_tags": ["坚果过敏", "少走路"],
                "soft_tags": ["清淡", "室内"],
                "budget": 600,
            },
            "filtered_candidates": [],
            "candidates": [],
            "filter_reasons": {},
            "execution_log": [],
            "b_itinerary_blueprint": blueprint,
        }
    )

    request = result["b_replan_request"]
    targets = request["rag_request"]["target_nodes"]
    assert request["source"] == "skeleton_candidate_evidence"
    assert [(node["node_id"], node["label"]) for node in targets] == [
        ("intent_01", "亲子活动"),
        ("intent_02", "晚餐"),
    ]
    dinner_terms = targets[1]["query_terms"]
    for term in ("晚餐", "清淡", "坚果过敏", "少走路"):
        assert term in dinner_terms
    assert result["selected_plan"]["execution_ready"] is False
    assert result["selected_plan"]["action_hints"] == []


def test_schedule_drift_skeleton_replan_has_bounded_chinese_repair_targets():
    blueprint = {
        "template_mode": "multi_node",
        "planning_horizon": "full_day",
        "planning_days": 1,
        "node_intents": [
            {"node_id": "intent_01", "role": "citywalk_market", "label": "城市漫步/市集", "supply_domain": "activity"},
            {
                "node_id": "intent_rest",
                "role": "rest",
                "label": "午睡/休息",
                "supply_domain": "planning_guidance",
                "protected": True,
                "anchor_type": "rest",
            },
            {"node_id": "intent_02", "role": "restaurant_specific", "label": "指定餐饮", "supply_domain": "restaurant"},
        ],
        "time_skeleton": {
            "days": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "label": "城市漫步/市集",
                            "role": "citywalk_market",
                            "supply_domain": "activity",
                            "start_time": "10:30",
                            "end_time": "12:30",
                            "duration_min": 120,
                        },
                        {
                            "node_id": "intent_rest",
                            "label": "午睡/休息",
                            "role": "rest",
                            "supply_domain": "planning_guidance",
                            "start_time": "13:30",
                            "end_time": "15:00",
                            "duration_min": 90,
                            "execution_status": "protected_non_executable",
                            "protected": True,
                            "anchor_type": "rest",
                        },
                        {
                            "node_id": "intent_02",
                            "label": "指定餐饮",
                            "role": "restaurant_specific",
                            "supply_domain": "restaurant",
                            "start_time": "18:00",
                            "end_time": "19:15",
                            "duration_min": 75,
                        },
                    ],
                }
            ]
        },
    }
    schedule_reason = "时间骨架与候选营业/时段不匹配"
    result = plan_optimizer_node(
        {
            "user_input": "明天朋友 citywalk 拍照 晚餐，素食不辣，别跨太远。",
            "scene_type": "friends",
            "constraints": {
                "raw_text": "朋友 citywalk 晚餐 素食 不辣 10:30 20:30",
                "hard_tags": ["素食", "不辣"],
            },
            "filtered_candidates": [],
            "candidates": [
                {
                    "plan_id": "cand_multi_001",
                    "availability": {
                        "all_available": False,
                        "detail": {
                            "unavailable_poi_ids": [],
                            "time_window_feasible": False,
                            "schedule_feasibility": {
                                "time_window_feasible": False,
                                "drifted_nodes": [
                                    {
                                        "node_id": "intent_01",
                                        "label": "城市漫步/市集",
                                        "role": "citywalk_market",
                                        "day": 1,
                                        "slot_start": "10:30",
                                        "slot_end": "12:30",
                                        "scheduled_start": "14:00",
                                        "scheduled_end": "16:00",
                                        "reason": "slot_alignment_drift",
                                    },
                                    {
                                        "node_id": "intent_rest",
                                        "label": "午睡/休息",
                                        "role": "rest",
                                        "day": 1,
                                        "slot_start": "13:30",
                                        "slot_end": "15:00",
                                        "scheduled_start": "16:30",
                                        "scheduled_end": "17:30",
                                        "reason": "slot_alignment_drift",
                                    },
                                ],
                            },
                        },
                    },
                }
            ],
            "filter_reasons": {
                "cand_multi_001": schedule_reason,
                "_summary_detail": {"reason_counts": {schedule_reason: 1}},
            },
            "execution_log": [],
            "b_itinerary_blueprint": blueprint,
            "b_rag_candidate_metadata": {"normalized_candidate_count": 3},
            "b_rag_candidate_coverage": {
                "covered_node_ids": ["intent_01", "intent_02"],
                "missing_node_ids": [],
                "all_nodes_covered": True,
            },
        }
    )

    request = result["b_replan_request"]
    repair = request["schedule_repair_request"]
    repair_labels = [node["label"] for node in repair["target_nodes"]]
    assert repair_labels == ["城市漫步/市集"]
    assert repair["preserve_time_skeleton"] == blueprint["time_skeleton"]
    assert repair["protected_slots"][0]["label"] == "午睡/休息"
    assert result["selected_plan"]["execution_ready"] is False
    assert result["selected_plan"]["action_hints"] == []


def test_schedule_repair_preserves_same_window_fit_nodes_from_slot_diagnostics():
    blueprint = {
        "template_mode": "multi_node",
        "planning_horizon": "full_day",
        "planning_days": 1,
        "node_intents": [
            {"node_id": "intent_01", "role": "citywalk_market", "label": "城市漫步/市集", "supply_domain": "activity"},
            {"node_id": "intent_02", "role": "photo_activity", "label": "文化体验/拍照", "supply_domain": "activity"},
            {"node_id": "intent_03", "role": "afternoon_activity", "label": "下午补充活动", "supply_domain": "activity"},
            {"node_id": "intent_04", "role": "restaurant_specific", "label": "指定餐饮", "supply_domain": "restaurant"},
        ],
        "time_skeleton": {
            "days": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "label": "城市漫步/市集",
                            "role": "citywalk_market",
                            "supply_domain": "activity",
                            "start_time": "10:30",
                            "end_time": "12:30",
                            "duration_min": 120,
                        },
                        {
                            "node_id": "intent_02",
                            "label": "文化体验/拍照",
                            "role": "photo_activity",
                            "supply_domain": "activity",
                            "start_time": "14:00",
                            "end_time": "16:00",
                            "duration_min": 120,
                        },
                        {
                            "node_id": "intent_03",
                            "label": "下午补充活动",
                            "role": "afternoon_activity",
                            "supply_domain": "activity",
                            "start_time": "16:30",
                            "end_time": "18:00",
                            "duration_min": 90,
                        },
                        {
                            "node_id": "intent_04",
                            "label": "指定餐饮",
                            "role": "restaurant_specific",
                            "supply_domain": "restaurant",
                            "start_time": "18:00",
                            "end_time": "19:15",
                            "duration_min": 75,
                        },
                    ],
                }
            ]
        },
    }
    schedule_reason = "时间骨架与候选营业/时段不匹配"
    result = plan_optimizer_node(
        {
            "user_input": "明天4个朋友 citywalk 拍照 晚餐，素食不辣。",
            "scene_type": "friends",
            "constraints": {
                "raw_text": "朋友 citywalk 拍照 晚餐 素食 不辣 10:30 20:30",
                "hard_tags": ["素食", "不辣"],
                "budget": {"type": "per_person", "amount": 180},
            },
            "filtered_candidates": [],
            "candidates": [
                {
                    "plan_id": "cand_multi_001",
                    "availability": {
                        "all_available": False,
                        "detail": {
                            "time_window_feasible": False,
                            "schedule_feasibility": {
                                "time_window_feasible": False,
                                "drifted_nodes": [
                                    {
                                        "node_id": "intent_03",
                                        "label": "下午补充活动",
                                        "slot_start": "16:30",
                                        "slot_end": "18:00",
                                        "scheduled_start": "18:00",
                                        "scheduled_end": "19:30",
                                        "reason": "slot_alignment_drift",
                                    },
                                    {
                                        "node_id": "intent_04",
                                        "label": "指定餐饮",
                                        "slot_start": "18:00",
                                        "slot_end": "19:15",
                                        "scheduled_start": "20:00",
                                        "scheduled_end": "21:15",
                                        "reason": "slot_alignment_drift",
                                    },
                                ],
                            },
                        },
                    },
                }
            ],
            "filter_reasons": {
                "cand_multi_001": schedule_reason,
                "_summary_detail": {"reason_counts": {schedule_reason: 1}},
            },
            "candidate_recall_diagnostics": {
                "slot_window_diagnostics": [
                    {
                        "node_id": "intent_01",
                        "label": "城市漫步/市集",
                        "requested_start": "10:30",
                        "requested_end": "12:30",
                        "counts": {"slot_fit": 0, "slot_drift": 4, "slot_unknown": 0},
                    },
                    {
                        "node_id": "intent_02",
                        "label": "文化体验/拍照",
                        "requested_start": "14:00",
                        "requested_end": "16:00",
                        "counts": {"slot_fit": 0, "slot_drift": 0, "slot_unknown": 0},
                    },
                    {
                        "node_id": "intent_03",
                        "label": "下午补充活动",
                        "requested_start": "16:30",
                        "requested_end": "18:00",
                        "counts": {"slot_fit": 0, "slot_drift": 4, "slot_unknown": 0},
                    },
                    {
                        "node_id": "intent_04",
                        "label": "指定餐饮",
                        "requested_start": "18:00",
                        "requested_end": "19:15",
                        "counts": {"slot_fit": 2, "slot_drift": 9, "slot_unknown": 0},
                    },
                ]
            },
            "execution_log": [],
            "b_itinerary_blueprint": blueprint,
            "b_rag_candidate_metadata": {"normalized_candidate_count": 3},
            "b_rag_candidate_coverage": {
                "covered_node_ids": ["intent_01", "intent_03", "intent_04"],
                "missing_node_ids": ["intent_02"],
                "all_nodes_covered": False,
            },
        }
    )

    request = result["b_replan_request"]
    repair = request["schedule_repair_request"]
    assert [node["label"] for node in repair["target_nodes"]] == ["城市漫步/市集", "下午补充活动"]
    assert repair["target_nodes"][0]["reason"] == "slot_window_diagnostic_zero_fit"
    assert repair["target_nodes"][1]["reason"] == "slot_alignment_drift"
    assert [node["label"] for node in repair["preserve_existing_slot_fit_nodes"]] == ["指定餐饮"]
    preserved_restaurant = repair["preserve_existing_slot_fit_nodes"][0]
    assert preserved_restaurant == {
        "node_id": "intent_04",
        "label": "指定餐饮",
        "slot_window": "18:00-19:15",
        "supply_domain": "restaurant",
        "slot_fit": 2,
        "slot_drift": 9,
    }
    rag_targets = request["rag_request"]["target_nodes"]
    assert "文化体验/拍照" in {node["label"] for node in rag_targets}
    assert "指定餐饮" in {node["label"] for node in rag_targets}
    assert all("poi_id" not in node for node in rag_targets)
    assert result["selected_plan"]["execution_ready"] is False
    assert result["selected_plan"]["action_hints"] == []


def test_schedule_repair_keeps_fixed_anchor_protected_rest_as_guidance_only():
    blueprint = {
        "template_mode": "multi_node",
        "planning_horizon": "full_day",
        "planning_days": 1,
        "node_intents": [
            {"node_id": "intent_01", "role": "family_activity", "label": "亲子活动", "supply_domain": "activity"},
            {
                "node_id": "intent_rest",
                "role": "rest",
                "label": "午睡/休息",
                "supply_domain": "planning_guidance",
                "protected": True,
                "anchor_type": "rest",
            },
            {"node_id": "intent_03", "role": "afternoon_activity", "label": "下午补充活动", "supply_domain": "activity"},
            {"node_id": "intent_04", "role": "restaurant_specific", "label": "指定餐饮", "supply_domain": "restaurant"},
            {"node_id": "intent_05", "role": "show", "label": "脱口秀/演出", "supply_domain": "activity"},
        ],
        "time_skeleton": {
            "days": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "label": "亲子活动",
                            "role": "family_activity",
                            "supply_domain": "activity",
                            "start_time": "10:00",
                            "end_time": "12:00",
                            "duration_min": 120,
                        },
                        {
                            "node_id": "intent_rest",
                            "label": "午睡/休息",
                            "role": "rest",
                            "supply_domain": "planning_guidance",
                            "start_time": "13:30",
                            "end_time": "15:00",
                            "duration_min": 90,
                            "execution_status": "protected_non_executable",
                            "protected": True,
                            "anchor_type": "rest",
                        },
                        {
                            "node_id": "intent_03",
                            "label": "下午补充活动",
                            "role": "afternoon_activity",
                            "supply_domain": "activity",
                            "start_time": "16:30",
                            "end_time": "18:00",
                            "duration_min": 90,
                        },
                        {
                            "node_id": "intent_04",
                            "label": "指定餐饮",
                            "role": "restaurant_specific",
                            "supply_domain": "restaurant",
                            "start_time": "18:00",
                            "end_time": "19:00",
                            "duration_min": 60,
                        },
                        {
                            "node_id": "intent_05",
                            "label": "脱口秀/演出",
                            "role": "show",
                            "supply_domain": "activity",
                            "start_time": "19:00",
                            "end_time": "20:40",
                            "duration_min": 100,
                        },
                    ],
                }
            ]
        },
    }
    schedule_reason = "时间骨架与候选营业/时段不匹配"
    drifted_nodes = [
        {
            "node_id": "intent_01",
            "label": "亲子活动",
            "slot_start": "10:00",
            "slot_end": "12:00",
            "scheduled_start": "14:00",
            "scheduled_end": "16:00",
            "reason": "slot_alignment_drift",
        },
        {
            "node_id": "intent_rest",
            "label": "午睡/休息",
            "slot_start": "13:30",
            "slot_end": "15:00",
            "scheduled_start": "16:00",
            "scheduled_end": "17:30",
            "reason": "slot_alignment_drift",
        },
        {
            "node_id": "intent_03",
            "label": "下午补充活动",
            "slot_start": "16:30",
            "slot_end": "18:00",
            "scheduled_start": "18:30",
            "scheduled_end": "20:00",
            "reason": "slot_alignment_drift",
        },
        {
            "node_id": "intent_04",
            "label": "指定餐饮",
            "slot_start": "18:00",
            "slot_end": "19:00",
            "scheduled_start": "20:00",
            "scheduled_end": "21:00",
            "reason": "slot_alignment_drift",
        },
    ]
    result = plan_optimizer_node(
        {
            "user_input": "周日亲子一日，13:30-15:00 午睡，晚上演出。",
            "scene_type": "family",
            "constraints": {
                "raw_text": "亲子 午睡 午餐 清淡 演出 婴儿车 少楼梯",
                "hard_tags": ["低盐", "婴儿车友好", "少楼梯"],
                "budget": 1000,
            },
            "filtered_candidates": [],
            "candidates": [
                {
                    "plan_id": "cand_multi_001",
                    "availability": {
                        "all_available": False,
                        "detail": {
                            "time_window_feasible": False,
                            "schedule_feasibility": {
                                "time_window_feasible": False,
                                "drifted_nodes": drifted_nodes,
                            },
                        },
                    },
                }
            ],
            "filter_reasons": {
                "cand_multi_001": schedule_reason,
                "_summary_detail": {"reason_counts": {schedule_reason: 1}},
            },
            "candidate_recall_diagnostics": {
                "slot_window_diagnostics": [
                    {
                        "node_id": "intent_01",
                        "label": "亲子活动",
                        "requested_start": "10:00",
                        "requested_end": "12:00",
                        "counts": {"slot_fit": 0, "slot_drift": 6, "slot_unknown": 0},
                    },
                    {
                        "node_id": "intent_rest",
                        "label": "午睡/休息",
                        "requested_start": "13:30",
                        "requested_end": "15:00",
                        "counts": {"slot_fit": 0, "slot_drift": 1, "slot_unknown": 0},
                    },
                    {
                        "node_id": "intent_03",
                        "label": "下午补充活动",
                        "requested_start": "16:30",
                        "requested_end": "18:00",
                        "counts": {"slot_fit": 0, "slot_drift": 6, "slot_unknown": 0},
                    },
                    {
                        "node_id": "intent_04",
                        "label": "指定餐饮",
                        "requested_start": "18:00",
                        "requested_end": "19:00",
                        "counts": {"slot_fit": 0, "slot_drift": 8, "slot_unknown": 0},
                    },
                    {
                        "node_id": "intent_05",
                        "label": "脱口秀/演出",
                        "requested_start": "19:00",
                        "requested_end": "20:40",
                        "counts": {"slot_fit": 0, "slot_drift": 0, "slot_unknown": 0},
                    },
                ]
            },
            "execution_log": [],
            "b_itinerary_blueprint": blueprint,
            "b_rag_candidate_metadata": {"normalized_candidate_count": 3},
            "b_rag_candidate_coverage": {
                "covered_node_ids": ["intent_01", "intent_03", "intent_04"],
                "missing_node_ids": ["intent_05"],
                "all_nodes_covered": False,
            },
        }
    )

    request = result["b_replan_request"]
    repair = request["schedule_repair_request"]
    assert [node["label"] for node in repair["target_nodes"]] == ["亲子活动", "下午补充活动", "指定餐饮"]
    assert "脱口秀/演出" not in {node["label"] for node in repair["target_nodes"]}
    assert [slot["label"] for slot in repair["protected_slots"]] == ["午睡/休息"]
    assert "preserve_existing_slot_fit_nodes" not in repair
    assert repair["missing_evidence_target_nodes"] == [
        {
            "node_id": "intent_05",
            "label": "脱口秀/演出",
            "day": 1,
            "slot_window": "19:00-20:40",
            "compatibility_role": "show",
            "supply_domain": "activity",
            "grade": "missing_node_evidence",
            "execution_status": "missing_executable_evidence",
            "coverage_status": "missing",
            "reason": "missing_node_evidence",
        }
    ]
    rag_labels = {node["label"] for node in request["rag_request"]["target_nodes"]}
    assert "脱口秀/演出" in rag_labels
    assert "午睡/休息" not in rag_labels
    assert result["selected_plan"]["execution_ready"] is False
    assert result["selected_plan"]["action_hints"] == []


def test_skeleton_replan_loop_does_not_reemit_active_request(monkeypatch):
    request = {
        "source": "skeleton_candidate_evidence",
        "status": "needs_replan",
        "next_step": "rerun_candidate_generation",
        "rag_request": {"target_nodes": [{"node_id": "intent_01", "label": "亲子活动"}]},
    }

    def fake_rag(state):
        return {"execution_log": state["execution_log"] + ["rag ok"]}

    def fake_candidate_generator(state):
        return {
            "candidates": [],
            "candidate_recall_diagnostics": {"replan_request_active": True},
            "b_itinerary_blueprint": state["b_itinerary_blueprint"],
            "constraints": state["constraints"],
            "execution_log": state["execution_log"] + ["candidate empty"],
        }

    def fake_constraint_filter(state):
        return {
            "filtered_candidates": [],
            "filter_reasons": {},
            "execution_log": state["execution_log"] + ["filter empty"],
        }

    monkeypatch.setattr(b_replan_loop, "b_poi_rag_node", fake_rag)
    monkeypatch.setattr(b_replan_loop, "candidate_generator_node", fake_candidate_generator)
    monkeypatch.setattr(b_replan_loop, "constraint_filter_node", fake_constraint_filter)

    result = b_replan_loop.b_replan_loop_node(
        {
            "b_replan_request": request,
            "selected_plan": {"plan_id": "plan_old"},
            "constraints": {"raw_text": "亲子活动"},
            "b_itinerary_blueprint": {
                "template_mode": "multi_node",
                "planning_horizon": "half_day",
                "planning_days": 1,
                "time_skeleton": {
                    "days": [
                        {
                            "day": 1,
                            "slots": [
                                {
                                    "node_id": "intent_01",
                                    "label": "亲子活动",
                                    "role": "family_activity",
                                    "supply_domain": "activity",
                                    "start_time": "10:00",
                                    "end_time": "12:00",
                                    "duration_min": 120,
                                }
                            ],
                        }
                    ]
                },
            },
            "execution_log": [],
        }
    )

    assert result["b_replan_request"] == {}
    assert "b_replan_request" not in result["constraints"]
    assert result["constraints"]["b_replan_trace"]["last_completed_request"]["source"] == "skeleton_candidate_evidence"
    assert result["constraints"]["b_replan_trace"]["last_completed_request"]["status"] == "completed"
    assert result["constraints"]["b_replan_trace"]["last_completed_request"]["trace_only"] is True
    assert result["constraints"]["b_replan_trace"]["last_completed_request"]["post_replan_trace_status"] == "completed"
    assert result["candidate_recall_diagnostics"]["replan_request_active"] is False
    assert result["candidate_recall_diagnostics"]["last_replan_request"]["source"] == "skeleton_candidate_evidence"
    assert result["candidate_recall_diagnostics"]["last_replan_request"]["status"] == "completed"
    assert result["candidate_recall_diagnostics"]["last_replan_request"]["trace_only"] is True
    assert result["candidate_recall_diagnostics"]["last_replan_request"]["post_replan_trace_status"] == "completed"
    assert "b_replan_request" not in result["selected_plan"]
    assert result["selected_plan"]["plan_status"] == "needs_rag_candidate_evidence"
    assert result["b_replan_attempt"]["status"] == "completed"
    assert result["b_replan_attempt"]["still_needs_replan"] is False


def test_graph_runs_replan_loop_before_explainability():
    node_names = [node.__name__ for node in WORKFLOW_NODES]

    assert node_names.index("plan_optimizer_node") < node_names.index("b_replan_loop_node")
    assert node_names.index("b_replan_loop_node") < node_names.index("explainability_node")
