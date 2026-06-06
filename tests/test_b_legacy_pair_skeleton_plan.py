from src.nodes.execution_manager import execution_manager_node
from src.nodes.explainability import explainability_node
from src.nodes.payment_layer import payment_layer_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.share_generator import share_generator_node
from src.nodes.tool_router import tool_router_node


def test_urgent_legacy_pair_preserves_non_executable_chinese_skeleton():
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
    raw_request = (
        "临时改计划了，今天16:20以后从杨浦五角场出发，带6岁孩子和膝盖不太好的外婆"
        "出去透口气，最好有个不用排很久的室内活动，再吃个清淡晚饭；孩子坚果过敏，"
        "外婆不能走太多，19:30前要回到家附近，总预算600以内，能预约就先帮我稳住。"
    )
    state = {
        "user_input": raw_request,
        "scene_type": "family",
        "constraints": {
            "people_count": 3,
            "raw_text": raw_request,
            "hard_tags": ["儿童友好", "低强度", "少步行", "避开坚果"],
            "soft_tags": ["亲子", "室内", "少油", "轻食", "可预约"],
            "avoid": ["商场拥挤", "排队久", "太远"],
            "budget": {"type": "total", "amount": 600},
            "start_time": "16:20",
            "end_time": "19:30",
            "location": {"origin": "杨浦五角场", "city": "上海"},
            "route_pattern_hints": {
                "search_terms": ["室内", "低强度", "少步行", "轻食", "避开坚果"],
            },
        },
        "filtered_candidates": [],
        "candidates": [],
        "filter_reasons": {
            "_summary": "活动和餐饮候选未同时通过硬时间窗、过敏和少步行约束",
            "_summary_detail": {
                "reason_counts": {
                    "活动和餐饮候选未同时通过硬时间窗、过敏和少步行约束": 3
                }
            },
        },
        "execution_log": [],
        "b_itinerary_blueprint": blueprint,
    }

    state.update(plan_optimizer_node(state))

    selected_plan = state["selected_plan"]
    assert selected_plan["plan_id"] == "plan_legacy_pair_skeleton"
    assert selected_plan["plan_status"] == "needs_rag_candidate_evidence"
    assert (
        selected_plan["candidate_evidence_status"]
        == "legacy_pair_missing_candidate_evidence"
    )
    assert selected_plan["execution_ready"] is False
    assert selected_plan["action_hints"] == []
    assert selected_plan["objective_vector"] == {}
    assert selected_plan["weighted_score"] == 0.0
    grade = selected_plan["candidate_evidence_grade"]
    assert grade["version"] == "candidate_evidence_grade_v1"
    assert grade["has_raw_candidate_coverage"] is False
    assert {
        (node["label"], node["grade"], node["has_raw_candidate_coverage"])
        for node in grade["nodes"]
    } == {
        ("亲子活动", "missing_node_evidence", False),
        ("晚餐", "missing_node_evidence", False),
    }
    assert [(item["time"], item["activity"]) for item in selected_plan["timeline"]] == [
        ("16:20-18:13", "亲子活动"),
        ("18:13-19:30", "晚餐"),
    ]
    assert all(item["type"] == "planning_intent" for item in selected_plan["timeline"])
    assert all(item["poi_id"] is None for item in selected_plan["timeline"])
    state.update(explainability_node(state))
    explanation = state["explanation_text"]
    assert "半天计划" in explanation
    assert "亲子活动、晚餐" in explanation
    assert "规划意图" in explanation
    assert "具体候选与确认依据" in explanation
    assert "多节点行程" not in explanation
    assert "RAG/多城市" not in explanation

    state.update(tool_router_node(state))
    assert state["action_sequence"] == []

    state.update(execution_manager_node(state))
    assert state["execution_status"] == "failed"

    state.update(payment_layer_node(state))
    assert state["payment_status"] == "not_required"

    state.update(share_generator_node(state))
    share = state["final_share_message"]
    assert "16:20-18:13 亲子活动" in share
    assert "18:13-19:30 晚餐" in share
    assert "暂时没有形成可执行的预订动作" in share
    assert "还只是行程意图" in share
    assert "不能说已经订好" in share
    for forbidden in ("推荐这个方案", "可预约性好", "已订", "已确认", "已安排"):
        assert forbidden not in explanation
        assert forbidden not in share


def test_urgent_legacy_pair_with_rag_coverage_reports_unpaired_not_missing():
    blueprint = {
        "template_mode": "legacy_pair",
        "planning_horizon": "half_day",
        "planning_days": 1,
        "node_intents": [
            {
                "node_id": "intent_01",
                "role": "family_activity",
                "label": "亲子活动",
                "supply_domain": "activity",
                "search_terms": ["孩子", "儿童", "亲子"],
            },
            {
                "node_id": "intent_02",
                "role": "restaurant_dinner",
                "label": "晚餐",
                "supply_domain": "restaurant",
                "search_terms": ["轻食", "晚饭", "订座"],
            },
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
    state = {
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
        "filter_reasons": {
            "_summary": "未生成候选方案，因此没有可执行方案满足硬约束",
            "_summary_detail": {
                "total_candidates": 0,
                "valid_candidates": 0,
                "invalid_candidates": 0,
                "reason_counts": {},
            },
        },
        "execution_log": [],
        "b_itinerary_blueprint": blueprint,
        "b_rag_candidate_metadata": {
            "raw_candidate_count": 10,
            "normalized_candidate_count": 10,
            "covered_node_ids": ["intent_01", "intent_02"],
        },
        "b_rag_candidate_coverage": {
            "all_nodes_covered": True,
            "covered_node_ids": ["intent_01", "intent_02"],
            "missing_node_ids": [],
            "required_node_count": 2,
        },
        "candidate_recall_diagnostics": {
            "counts": {
                "activities_after_requirement_filter": 8,
                "restaurants_after_requirement_filter": 2,
                "raw_plan_candidates": 0,
                "plan_candidates": 0,
            }
        },
    }

    state.update(plan_optimizer_node(state))

    selected_plan = state["selected_plan"]
    grade = selected_plan["candidate_evidence_grade"]
    assert grade["hard_filter_reason"] == (
        "已召回活动和餐饮候选，但尚未确认任何活动加晚餐组合同时满足"
        "硬时间窗、过敏、少步行等硬约束"
    )
    assert {
        (node["label"], node["grade"], node["has_raw_candidate_coverage"])
        for node in grade["nodes"]
    } == {
        ("亲子活动", "all_candidates_filtered", True),
        ("晚餐", "all_candidates_filtered", True),
    }
    assert "missing_node_evidence" not in {
        node["grade"] for node in grade["nodes"]
    }

    request = state["b_replan_request"]
    targets = request["rag_request"]["target_nodes"]
    assert [(node["label"], node["coverage_status"]) for node in targets] == [
        ("亲子活动", "filtered"),
        ("晚餐", "filtered"),
    ]
    assert all(node["grade"] == "all_candidates_filtered" for node in targets)
    assert all(node["hard_filter_reason"] == grade["hard_filter_reason"] for node in targets)
    dinner_terms = targets[1]["query_terms"]
    for term in ("晚餐", "清淡", "坚果过敏", "少走路"):
        assert term in dinner_terms

    assert selected_plan["execution_ready"] is False
    assert selected_plan["action_hints"] == []
    assert all(item["poi_id"] is None for item in selected_plan["timeline"])

    state.update(explainability_node(state))
    explanation = state["explanation_text"]
    assert "已召回活动和餐饮候选" in explanation
    assert "未生成候选方案" not in explanation

    state.update(tool_router_node(state))
    assert state["action_sequence"] == []

    state.update(execution_manager_node(state))
    assert state["execution_status"] == "failed"

    state.update(payment_layer_node(state))
    assert state["payment_status"] == "not_required"

    state.update(share_generator_node(state))
    share = state["final_share_message"]
    assert "已有初始候选但未通过硬约束/可预订确认的节点：亲子活动、晚餐" in share
    assert "已召回活动和餐饮候选" in share
    assert "未生成候选方案" not in share
    for forbidden in ("推荐这个方案", "可预约性好", "已订", "已确认", "已安排"):
        assert forbidden not in explanation
        assert forbidden not in share
