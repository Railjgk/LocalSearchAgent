from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.explainability import explainability_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.tool_router import tool_router_node


def test_multinode_request_returns_non_executable_skeleton_without_legacy_pair():
    state = {
        "user_input": "我们一家四口想在外滩附近找个有厨房的住处，晚上吃蟹黄面，再找便利店买日用品，最后找停车场。",
        "scene_type": "family",
        "constraints": {"raw_text": "住宿 晚餐 便利店 停车"},
        "user_profile": {},
        "scenario_activities": [],
        "execution_log": [],
    }

    state.update(candidate_generator_node(state))
    assert state["candidates"]
    assert state["b_itinerary_blueprint"]["template_mode"] == "multi_node"
    assert state["b_itinerary_blueprint"]["planning_days"] == 2
    assert state["candidate_generation_issues"][0]["type"] == "multi_node_blueprint_not_yet_planned"

    state.update(constraint_filter_node(state))
    state.update(plan_optimizer_node(state))

    selected_plan = state["selected_plan"]
    assert selected_plan["plan_status"] == "needs_rag_candidate_evidence"
    assert selected_plan["execution_ready"] is False
    assert selected_plan["planning_days"] == 2
    assert selected_plan["timeline"]
    assert all(item.get("poi_id") is None for item in selected_plan["timeline"])
    assert selected_plan["risk_factors"]
    assert state["b_replan_request"]["source"] == "skeleton_candidate_evidence"

    state.update(explainability_node(state))
    assert state["explanation_text"]

    state.update(tool_router_node(state))
    assert state["action_sequence"] == []


def test_multinode_skeleton_timeline_is_chronological_within_day():
    state = {
        "scene_type": "friends",
        "constraints": {},
        "filtered_candidates": [],
        "candidates": [],
        "filter_reasons": {},
        "execution_log": [],
        "b_itinerary_blueprint": {
            "template_mode": "multi_node",
            "planning_horizon": "full_day",
            "planning_days": 1,
            "time_skeleton": {
                "days": [
                    {
                        "day": 1,
                        "slots": [
                            {
                                "node_id": "intent_03",
                                "label": "下午补充活动",
                                "role": "cultural_photo",
                                "supply_domain": "activity",
                                "start_time": "16:30",
                                "end_time": "18:00",
                                "duration_min": 90,
                            },
                            {
                                "node_id": "intent_04",
                                "label": "傍晚餐饮/休息",
                                "role": "cafe",
                                "supply_domain": "restaurant",
                                "start_time": "15:30",
                                "end_time": "16:30",
                                "duration_min": 60,
                            },
                        ],
                    }
                ]
            },
        },
    }

    result = plan_optimizer_node(state)

    assert [item["time"] for item in result["selected_plan"]["timeline"]] == [
        "15:30-16:30",
        "16:30-18:00",
    ]


def test_multinode_skeleton_adds_chinese_first_candidate_evidence_grade():
    blueprint = {
        "template_mode": "multi_node",
        "planning_horizon": "full_day",
        "planning_days": 1,
        "node_intents": [
            {
                "node_id": "intent_01",
                "role": "citywalk_market",
                "label": "城市漫步/市集",
                "supply_domain": "activity",
            },
            {
                "node_id": "intent_02",
                "role": "cultural_photo",
                "label": "文化体验/拍照",
                "supply_domain": "activity",
            },
            {
                "node_id": "intent_03",
                "role": "restaurant_specific",
                "label": "指定餐饮",
                "supply_domain": "restaurant",
            },
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
                            "role": "cultural_photo",
                            "supply_domain": "activity",
                            "start_time": "14:00",
                            "end_time": "16:00",
                            "duration_min": 120,
                        },
                        {
                            "node_id": "intent_03",
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
    state = {
        "scene_type": "friends",
        "constraints": {
            "raw_text": "下雨 室内 citywalk 拍照 素食 不辣 人均180 地铁方便",
            "hard_tags": ["素食", "不辣"],
            "soft_tags": ["下雨", "室内", "短距离", "地铁方便"],
            "avoid": ["商场拥挤", "儿童友好", "排队久", "亲子"],
            "budget": {"type": "per_person", "amount": 180},
        },
        "filtered_candidates": [],
        "candidates": [],
        "filter_reasons": {
            "_summary": "共生成 18 个候选方案，其中 0 个满足硬约束，18 个被过滤。",
            "_summary_detail": {"reason_counts": {"活动或餐厅当前不可用": 18}},
        },
        "execution_log": [],
        "b_itinerary_blueprint": blueprint,
        "b_rag_candidate_metadata": {"normalized_candidate_count": 20},
        "b_rag_candidate_coverage": {
            "covered_node_ids": ["intent_01", "intent_03"],
            "missing_node_ids": ["intent_02"],
            "all_nodes_covered": False,
        },
        "b_requirement_contract": {
            "version": "b_requirement_contract_v1",
            "hard_requirements": ["restaurant_reservation"],
            "forbidden_restaurant_groups": ["火锅"],
            "soft_preferences": ["短距离"],
            "needs_confirmation": ["雨天备选需要确认"],
        },
    }

    state.update(plan_optimizer_node(state))
    selected_plan = state["selected_plan"]
    grade = selected_plan["candidate_evidence_grade"]
    by_label = {node["label"]: node for node in grade["nodes"]}

    assert selected_plan["plan_status"] == "needs_rag_candidate_evidence"
    assert selected_plan["timeline"][0]["node_id"] == "intent_01"
    assert by_label["文化体验/拍照"]["grade"] == "missing_node_evidence"
    assert by_label["城市漫步/市集"]["grade"] == "all_candidates_filtered"
    assert by_label["指定餐饮"]["has_raw_candidate_coverage"] is True
    assert by_label["指定餐饮"]["hard_filter_reason"] == "活动或餐厅当前不可用"
    assert "缺少多节点候选池" not in selected_plan["risk_factors"]
    assert state["b_replan_request"]["source"] == "skeleton_candidate_evidence"
    assert selected_plan["b_replan_request"] == state["b_replan_request"]
    targets = state["b_replan_request"]["rag_request"]["target_nodes"]
    by_target_label = {node["label"]: node for node in targets}
    assert set(by_target_label) == {"城市漫步/市集", "文化体验/拍照", "指定餐饮"}
    assert by_target_label["文化体验/拍照"]["coverage_status"] == "missing"
    assert by_target_label["城市漫步/市集"]["coverage_status"] == "filtered"
    dinner_terms = by_target_label["指定餐饮"]["query_terms"]
    assert dinner_terms[0] == "指定餐饮"
    for term in ("下雨", "室内", "素食", "不辣", "人均180", "地铁方便"):
        assert term in dinner_terms
    structured_avoid_terms = {"儿童友好", "亲子", "商场拥挤", "排队久", "火锅"}
    for target in targets:
        assert structured_avoid_terms.isdisjoint(target["query_terms"])
    assert state["b_replan_request"]["rag_request"]["avoid_terms"] == [
        "商场拥挤",
        "儿童友好",
        "排队久",
        "亲子",
        "火锅",
    ]
    hints = state["b_replan_request"]["candidate_generation_hints"]
    assert hints["avoid_terms"] == state["b_replan_request"]["rag_request"]["avoid_terms"]
    assert structured_avoid_terms.isdisjoint(hints["prefer_terms_from_guidance"])
    assert selected_plan["execution_ready"] is False
    assert selected_plan["action_hints"] == []

    state.update(explainability_node(state))
    explanation = state["explanation_text"]
    assert "文化体验/拍照" in explanation
    assert "城市漫步/市集" in explanation
    assert "活动或餐厅当前不可用" in explanation
    assert "每个节点返回" not in explanation


def test_skeleton_replan_request_excludes_protected_rest_slot():
    blueprint = {
        "template_mode": "multi_node",
        "planning_horizon": "full_day",
        "planning_days": 1,
        "node_intents": [
            {
                "node_id": "intent_01",
                "role": "family_activity",
                "label": "亲子活动",
                "supply_domain": "activity",
            },
            {
                "node_id": "intent_rest",
                "role": "rest",
                "label": "午睡/休息",
                "supply_domain": "planning_guidance",
                "protected": True,
                "anchor_type": "rest",
            },
            {
                "node_id": "intent_02",
                "role": "show",
                "label": "脱口秀/演出",
                "supply_domain": "activity",
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
                            "node_id": "intent_02",
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
    state = {
        "scene_type": "family",
        "constraints": {
            "raw_text": "低盐清淡午饭 老人低强度 婴儿车 少楼梯 19点演出 1000以内",
            "hard_tags": ["儿童友好", "低强度", "低盐", "婴儿车友好", "少步行"],
            "soft_tags": ["亲子", "儿童友好", "低强度", "少步行"],
            "avoid": ["商场拥挤", "热闹", "排队久", "太远"],
        },
        "filtered_candidates": [],
        "candidates": [],
        "filter_reasons": {
            "_summary_detail": {"reason_counts": {"缺少婴儿车友好或低强度证据": 2}},
        },
        "execution_log": [],
        "b_itinerary_blueprint": blueprint,
        "b_rag_candidate_coverage": {
            "covered_node_ids": ["intent_01", "intent_rest"],
            "missing_node_ids": ["intent_02"],
            "all_nodes_covered": False,
        },
    }

    state.update(plan_optimizer_node(state))

    timeline_by_activity = {
        item["activity"]: item
        for item in state["selected_plan"]["timeline"]
    }
    labels = [item["activity"] for item in state["selected_plan"]["timeline"]]
    target_labels = [
        item["label"]
        for item in state["b_replan_request"]["rag_request"]["target_nodes"]
    ]
    assert "午睡/休息" in labels
    assert timeline_by_activity["午睡/休息"]["time"] == "13:30-15:00"
    assert timeline_by_activity["脱口秀/演出"]["time"].startswith("19:00-")
    assert "午睡/休息" not in target_labels
    assert target_labels == ["亲子活动", "脱口秀/演出"]

    request = state["b_replan_request"]
    family_positive_terms = {"儿童友好", "亲子", "低强度", "低盐", "婴儿车友好", "少步行"}
    family_avoid_terms = {"商场拥挤", "热闹", "排队久", "太远"}
    for target in request["rag_request"]["target_nodes"]:
        assert family_avoid_terms.isdisjoint(target["query_terms"])
        assert family_positive_terms.issubset(set(target["query_terms"]))
    assert request["rag_request"]["avoid_terms"] == ["商场拥挤", "热闹", "排队久", "太远"]
    assert request["candidate_generation_hints"]["avoid_terms"] == request["rag_request"]["avoid_terms"]
    assert family_avoid_terms.isdisjoint(request["candidate_generation_hints"]["prefer_terms_from_guidance"])
