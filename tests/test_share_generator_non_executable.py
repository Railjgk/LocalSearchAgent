from src.nodes.share_generator import share_generator_node


def _completed_schedule_repair_state(repair):
    request = {
        "status": "completed",
        "trace_only": True,
        "post_replan_trace_status": "completed",
        "schedule_repair_request": repair,
    }
    return {
        "constraints": {
            "b_replan_trace": {
                "status": "completed",
                "last_completed_request": request,
            }
        },
        "candidate_recall_diagnostics": {"last_replan_request": request},
    }


def test_failed_share_reports_non_executable_plan_without_system_busy():
    state = {
        "scene_type": "friends",
        "selected_plan": {
            "timeline": [
                {
                    "time": "12:00-13:30",
                    "activity": "禾间轻食日料",
                    "poi_id": "res_light_japanese",
                }
            ],
        },
        "tool_results": {},
        "action_sequence": [],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
        "explanation_text": "",
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "系统繁忙" not in message
    assert "没有形成可执行的预订动作" in message
    assert "禾间轻食日料" in message


def test_failed_non_executable_share_includes_repair_guidance():
    state = {
        "scene_type": "friends",
        "selected_plan": {
            "timeline": [
                {
                    "time": "12:00-13:15",
                    "activity": "云栖温泉轻餐茶室",
                    "poi_id": "res_spa_light_tea",
                }
            ],
        },
        "tool_results": {},
        "action_sequence": [],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
        "explanation_text": "",
        "b_repair_plan": {
            "repair_strategy": "ask_user_confirm",
            "user_message": "当前方案仅匹配到一个午餐节点，缺少上午和下午的活动节点，请确认是否重新规划完整行程。",
        },
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "没有形成可执行的预订动作" in message
    assert "调整建议" in message
    assert "缺少上午和下午的活动节点" in message


def test_failed_skeleton_share_names_non_executable_roles():
    state = {
        "scene_type": "couple",
        "selected_plan": {
            "timeline": [
                {
                    "time": "12:00-13:10",
                    "activity": "餐饮",
                    "poi_id": None,
                    "role": "restaurant_lunch",
                    "type": "planning_intent",
                },
                {
                    "time": "15:00-次日10:00",
                    "activity": "住宿",
                    "poi_id": None,
                    "role": "lodging",
                    "type": "planning_intent",
                },
                {
                    "time": "15:45-16:00",
                    "activity": "停车",
                    "poi_id": None,
                    "role": "parking",
                    "type": "planning_intent",
                },
            ],
        },
        "b_itinerary_blueprint": {
            "unsupported_roles": ["lodging", "parking"],
            "node_intents": [
                {"role": "lodging", "label": "住宿"},
                {"role": "parking", "label": "停车"},
            ],
        },
        "b_rag_candidate_coverage": {
            "unsupported_missing_roles": ["lodging", "parking"],
            "missing_node_ids": ["intent_02", "intent_03"],
        },
        "tool_results": {},
        "action_sequence": [],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
        "explanation_text": "",
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "没有形成可执行的预订动作" in message
    assert "还只是行程意图" in message
    assert "住宿" in message
    assert "停车" in message
    assert "不能说已经订好" in message


def test_failed_fixed_anchor_share_keeps_rest_as_protected_guidance():
    blueprint = {
        "node_intents": [
            {"node_id": "intent_01", "role": "family_activity", "label": "亲子活动"},
            {"node_id": "intent_02", "role": "talk_show", "label": "脱口秀/演出"},
            {
                "node_id": "intent_03",
                "role": "family_activity",
                "label": "下午补充活动",
            },
            {
                "node_id": "intent_04",
                "role": "restaurant_specific",
                "label": "指定餐饮",
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
                            "start_time": "10:00",
                            "end_time": "12:00",
                            "supply_domain": "activity",
                        },
                        {
                            "node_id": None,
                            "label": "午睡/休息",
                            "role": "rest",
                            "start_time": "13:30",
                            "end_time": "15:00",
                            "execution_status": "protected_non_executable",
                            "protected": True,
                            "anchor_type": "rest",
                            "supply_domain": "planning_guidance",
                        },
                        {
                            "node_id": "intent_03",
                            "label": "下午补充活动",
                            "role": "family_activity",
                            "start_time": "16:30",
                            "end_time": "18:00",
                            "supply_domain": "activity",
                        },
                        {
                            "node_id": "intent_04",
                            "label": "指定餐饮",
                            "role": "restaurant_specific",
                            "start_time": "18:00",
                            "end_time": "19:00",
                            "supply_domain": "restaurant",
                        },
                        {
                            "node_id": "intent_02",
                            "label": "脱口秀/演出",
                            "role": "talk_show",
                            "start_time": "19:00",
                            "end_time": "20:40",
                            "supply_domain": "activity",
                        },
                    ],
                }
            ]
        },
    }
    state = {
        "scene_type": "family",
        "selected_plan": {
            "b_itinerary_blueprint": blueprint,
            "timeline": [
                {
                    "time": "10:00-12:00",
                    "activity": "亲子活动",
                    "poi_id": None,
                    "role": "family_activity",
                    "type": "planning_intent",
                },
                {
                    "time": "13:30-15:00",
                    "activity": "午睡/休息",
                    "poi_id": None,
                    "role": "rest",
                    "supply_domain": "planning_guidance",
                    "type": "planning_intent",
                },
                {
                    "time": "16:30-18:00",
                    "activity": "下午补充活动",
                    "poi_id": None,
                    "role": "family_activity",
                    "type": "planning_intent",
                },
                {
                    "time": "18:00-19:00",
                    "activity": "指定餐饮",
                    "poi_id": None,
                    "role": "restaurant_specific",
                    "type": "planning_intent",
                },
                {
                    "time": "19:00-20:40",
                    "activity": "脱口秀/演出",
                    "poi_id": None,
                    "role": "talk_show",
                    "type": "planning_intent",
                },
            ],
            "candidate_generation_summary": "共生成 18 个候选方案，其中 0 个满足硬约束，18 个被过滤。活动或餐厅当前不可用（18 个）",
            "candidate_evidence_grade": {
                "version": "candidate_evidence_grade_v1",
                "hard_filter_reason": "活动或餐厅当前不可用",
                "nodes": [
                    {
                        "node_id": "intent_01",
                        "label": "亲子活动",
                        "supply_domain": "activity",
                        "execution_status": "raw_candidates_failed_hard_constraints",
                        "has_raw_candidate_coverage": True,
                        "hard_filter_reason": "活动或餐厅当前不可用",
                        "grade": "all_candidates_filtered",
                    },
                    {
                        "node_id": None,
                        "label": "午睡/休息",
                        "supply_domain": "planning_guidance",
                        "execution_status": "protected_non_executable",
                        "has_raw_candidate_coverage": False,
                        "hard_filter_reason": "",
                        "grade": "protected_guidance",
                        "time": "13:30-15:00",
                    },
                    {
                        "node_id": "intent_03",
                        "label": "下午补充活动",
                        "supply_domain": "activity",
                        "execution_status": "raw_candidates_failed_hard_constraints",
                        "has_raw_candidate_coverage": True,
                        "hard_filter_reason": "活动或餐厅当前不可用",
                        "grade": "all_candidates_filtered",
                    },
                    {
                        "node_id": "intent_04",
                        "label": "指定餐饮",
                        "supply_domain": "restaurant",
                        "execution_status": "raw_candidates_failed_hard_constraints",
                        "has_raw_candidate_coverage": True,
                        "hard_filter_reason": "活动或餐厅当前不可用",
                        "grade": "all_candidates_filtered",
                    },
                    {
                        "node_id": "intent_02",
                        "label": "脱口秀/演出",
                        "supply_domain": "activity",
                        "execution_status": "missing_executable_evidence",
                        "has_raw_candidate_coverage": False,
                        "hard_filter_reason": "",
                        "grade": "missing_node_evidence",
                    },
                ],
            },
        },
        "b_rag_candidate_coverage": {
            "covered_node_ids": ["intent_01", "intent_03", "intent_04"],
            "missing_node_ids": ["intent_02"],
        },
        "filter_reasons": {
            "_summary_detail": {"reason_counts": {"活动或餐厅当前不可用": 18}},
        },
        "tool_results": {},
        "action_sequence": [],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
    }

    message = share_generator_node(state)["final_share_message"]

    assert "已保留为不需要预订的行程约束：13:30-15:00 午睡/休息" in message
    assert "缺少可执行候选证据的规划节点：脱口秀/演出" in message
    assert (
        "已有初始候选但未通过硬约束/可预订确认的节点：亲子活动、下午补充活动、指定餐饮"
        in message
    )
    assert "缺少可执行候选证据的规划节点：午睡/休息" not in message
    assert "不是可预订商家的节点：亲子活动、午睡/休息" not in message


def test_failed_friends_share_names_only_missing_node_from_coverage():
    blueprint = {
        "node_intents": [
            {
                "node_id": "intent_01",
                "role": "citywalk_market",
                "label": "城市漫步/市集",
            },
            {
                "node_id": "intent_02",
                "role": "cultural_photo",
                "label": "文化体验/拍照",
            },
            {
                "node_id": "intent_03",
                "role": "citywalk_market",
                "label": "下午补充活动",
            },
            {
                "node_id": "intent_04",
                "role": "restaurant_specific",
                "label": "指定餐饮",
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
                            "start_time": "10:30",
                            "end_time": "12:30",
                        },
                        {
                            "node_id": "intent_02",
                            "label": "文化体验/拍照",
                            "role": "cultural_photo",
                            "start_time": "14:00",
                            "end_time": "16:00",
                        },
                        {
                            "node_id": "intent_03",
                            "label": "下午补充活动",
                            "role": "citywalk_market",
                            "start_time": "16:30",
                            "end_time": "18:00",
                        },
                        {
                            "node_id": "intent_04",
                            "label": "指定餐饮",
                            "role": "restaurant_specific",
                            "start_time": "18:00",
                            "end_time": "19:15",
                        },
                    ],
                }
            ]
        },
    }
    state = {
        "scene_type": "friends",
        "selected_plan": {
            "b_itinerary_blueprint": blueprint,
            "timeline": [
                {
                    "time": "10:30-12:30",
                    "activity": "城市漫步/市集",
                    "poi_id": None,
                    "role": "citywalk_market",
                    "type": "planning_intent",
                },
                {
                    "time": "14:00-16:00",
                    "activity": "文化体验/拍照",
                    "poi_id": None,
                    "role": "cultural_photo",
                    "type": "planning_intent",
                },
                {
                    "time": "16:30-18:00",
                    "activity": "下午补充活动",
                    "poi_id": None,
                    "role": "citywalk_market",
                    "type": "planning_intent",
                },
                {
                    "time": "18:00-19:15",
                    "activity": "指定餐饮",
                    "poi_id": None,
                    "role": "restaurant_specific",
                    "type": "planning_intent",
                },
            ],
            "candidate_evidence_grade": {
                "version": "candidate_evidence_grade_v1",
                "hard_filter_reason": "活动或餐厅当前不可用",
                "nodes": [
                    {
                        "node_id": "intent_01",
                        "label": "城市漫步/市集",
                        "supply_domain": "activity",
                        "execution_status": "raw_candidates_failed_hard_constraints",
                        "has_raw_candidate_coverage": True,
                        "hard_filter_reason": "活动或餐厅当前不可用",
                        "grade": "all_candidates_filtered",
                    },
                    {
                        "node_id": "intent_02",
                        "label": "文化体验/拍照",
                        "supply_domain": "activity",
                        "execution_status": "missing_executable_evidence",
                        "has_raw_candidate_coverage": False,
                        "hard_filter_reason": "",
                        "grade": "missing_node_evidence",
                    },
                    {
                        "node_id": "intent_03",
                        "label": "下午补充活动",
                        "supply_domain": "activity",
                        "execution_status": "raw_candidates_failed_hard_constraints",
                        "has_raw_candidate_coverage": True,
                        "hard_filter_reason": "活动或餐厅当前不可用",
                        "grade": "all_candidates_filtered",
                    },
                    {
                        "node_id": "intent_04",
                        "label": "指定餐饮",
                        "supply_domain": "restaurant",
                        "execution_status": "raw_candidates_failed_hard_constraints",
                        "has_raw_candidate_coverage": True,
                        "hard_filter_reason": "活动或餐厅当前不可用",
                        "grade": "all_candidates_filtered",
                    },
                ],
            },
        },
        "b_rag_candidate_coverage": {
            "covered_node_ids": ["intent_01", "intent_03", "intent_04"],
            "missing_node_ids": ["intent_02"],
        },
        "tool_results": {},
        "action_sequence": [],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
    }

    message = share_generator_node(state)["final_share_message"]

    assert "缺少可执行候选证据的规划节点：文化体验/拍照" in message
    assert (
        "已有初始候选但未通过硬约束/可预订确认的节点：城市漫步/市集、下午补充活动、指定餐饮"
        in message
    )
    assert "缺少可执行候选证据的规划节点：城市漫步/市集" not in message
    assert "还有1个规划节点缺少可执行候选" not in message


def test_failed_share_adds_completed_bounded_schedule_repair_guidance():
    repair = {
        "request_type": "bounded_schedule_repair",
        "target_nodes": [
            {
                "node_id": "intent_01",
                "label": "城市漫步/市集",
                "slot_window": "10:30-12:30",
                "scheduled_window": "14:00-16:00",
                "poi_id": "rejected_citywalk",
                "poi_name": "不应出现的城市漫步候选",
            },
            {
                "node_id": "intent_03",
                "label": "下午补充活动",
                "slot_window": "16:30-18:00",
                "scheduled_window": "17:00-19:00",
            },
        ],
        "preserve_existing_slot_fit_nodes": [
            {
                "node_id": "intent_04",
                "label": "指定餐饮",
                "slot_window": "18:00-19:15",
                "poi_name": "不应出现的餐厅候选",
            }
        ],
    }
    state = {
        "scene_type": "friends",
        "selected_plan": {
            "timeline": [
                {
                    "time": "10:30-12:30",
                    "activity": "城市漫步/市集",
                    "poi_id": None,
                    "type": "planning_intent",
                },
                {
                    "time": "14:00-16:00",
                    "activity": "文化体验/拍照",
                    "poi_id": None,
                    "type": "planning_intent",
                },
                {
                    "time": "16:30-18:00",
                    "activity": "下午补充活动",
                    "poi_id": None,
                    "type": "planning_intent",
                },
                {
                    "time": "18:00-19:15",
                    "activity": "指定餐饮",
                    "poi_id": None,
                    "type": "planning_intent",
                },
            ],
        },
        "tool_results": {},
        "action_sequence": [],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
        **_completed_schedule_repair_state(repair),
    }

    message = share_generator_node(state)["final_share_message"]
    guidance = message.split("这次修复覆盖的待补节点：", 1)[1]

    assert "已经尝试过一次有界候选证据/时段修复，但仍没有形成可执行动作" in message
    assert "下一步应优先按原时间窗重查候选证据" not in message
    assert "下一步需要改动约束、换一个时间窗，或拿到商家/场次的新供给确认后再执行" in message
    assert "10:30-12:30 城市漫步/市集" in guidance
    assert "16:30-18:00 下午补充活动" in guidance
    assert "18:00-19:15 指定餐饮 仅作为元数据里的时段匹配证据保留" in guidance
    assert "14:00-16:00 文化体验/拍照" not in guidance
    assert "不应出现" not in message
    assert "rejected_citywalk" not in message
    assert "已订" not in message
    assert "已确认" not in message
    assert "推荐这个方案" not in message
    assert "位置已经订好了" not in message


def test_failed_share_adds_protected_rest_from_schedule_repair_trace():
    repair = {
        "request_type": "bounded_schedule_repair",
        "target_nodes": [
            {
                "node_id": "intent_01",
                "label": "亲子活动",
                "slot_window": "10:00-12:00",
            },
            {
                "node_id": "intent_03",
                "label": "下午补充活动",
                "slot_window": "16:30-18:00",
            },
            {
                "node_id": "intent_04",
                "label": "指定餐饮",
                "slot_window": "18:00-19:00",
            },
        ],
        "protected_slots": [
            {
                "node_id": None,
                "label": "午睡/休息",
                "slot_window": "13:30-15:00",
                "reason": "rest",
            }
        ],
    }
    state = {
        "scene_type": "family",
        "selected_plan": {
            "timeline": [
                {
                    "time": "10:00-12:00",
                    "activity": "亲子活动",
                    "poi_id": None,
                    "type": "planning_intent",
                },
                {
                    "time": "13:30-15:00",
                    "activity": "午睡/休息",
                    "poi_id": None,
                    "type": "planning_intent",
                },
                {
                    "time": "16:30-18:00",
                    "activity": "下午补充活动",
                    "poi_id": None,
                    "type": "planning_intent",
                },
                {
                    "time": "18:00-19:00",
                    "activity": "指定餐饮",
                    "poi_id": None,
                    "type": "planning_intent",
                },
                {
                    "time": "19:00-20:40",
                    "activity": "脱口秀/演出",
                    "poi_id": None,
                    "type": "planning_intent",
                },
            ],
        },
        "tool_results": {},
        "action_sequence": [],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
        **_completed_schedule_repair_state(repair),
    }

    message = share_generator_node(state)["final_share_message"]
    guidance = message.split("这次修复覆盖的待补节点：", 1)[1]

    assert "已经尝试过一次有界候选证据/时段修复，但仍没有形成可执行动作" in message
    assert "10:00-12:00 亲子活动" in guidance
    assert "16:30-18:00 下午补充活动" in guidance
    assert "18:00-19:00 指定餐饮" in guidance
    assert "继续保留 13:30-15:00 午睡/休息 作为非执行约束" in guidance
    assert "19:00-20:40 脱口秀/演出" not in guidance


def test_failed_share_adds_completed_generic_replan_guidance_without_schedule_repair():
    request = {
        "status": "completed",
        "trace_only": True,
        "post_replan_trace_status": "completed",
        "source": "skeleton_candidate_evidence",
        "rag_request": {
            "request_type": "replacement_poi_candidates",
            "target_nodes": [
                {
                    "label": "亲子活动",
                    "slot_window": "16:20-18:13",
                    "poi_name": "不应出现的候选",
                    "query_terms": ["不应出现的查询词"],
                }
            ],
        },
        "schedule_repair_request": None,
    }
    state = {
        "scene_type": "family",
        "selected_plan": {
            "timeline": [
                {
                    "time": "16:20-18:13",
                    "activity": "亲子活动",
                    "poi_id": None,
                    "type": "planning_intent",
                }
            ],
        },
        "constraints": {
            "b_replan_trace": {
                "status": "completed",
                "last_completed_request": request,
            }
        },
        "candidate_recall_diagnostics": {"last_replan_request": request},
        "tool_results": {},
        "action_sequence": [],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
    }

    message = share_generator_node(state)["final_share_message"]

    assert "下一步应优先按原时间窗重查候选证据" not in message
    assert "已经尝试过一次有界候选证据/时段修复，但仍没有形成可执行动作" in message
    assert "下一步需要改动约束、换一个时间窗，或拿到商家/场次的新供给确认后再执行" in message
    assert "不应出现" not in message
    assert "已订" not in message


def test_failed_two_day_pet_share_surfaces_pet_friendly_blocker():
    request = {
        "status": "completed",
        "trace_only": True,
        "post_replan_trace_status": "completed",
        "source": "skeleton_candidate_evidence",
        "rag_request": {
            "request_type": "replacement_poi_candidates",
            "target_nodes": [
                {
                    "label": "住宿",
                    "poi_name": "不应出现的宠物友好住宿",
                    "query_terms": ["不应出现的宠物友好查询"],
                }
            ],
        },
        "schedule_repair_request": None,
    }
    blueprint = {
        "node_intents": [
            {"node_id": "intent_01", "role": "restaurant_lunch", "label": "餐饮"},
            {"node_id": "intent_02", "role": "family_activity", "label": "第一天活动"},
            {
                "node_id": "intent_03",
                "role": "souvenir_shopping",
                "label": "特产/伴手礼",
            },
            {"node_id": "intent_04", "role": "lodging", "label": "住宿"},
            {"node_id": "intent_05", "role": "family_activity", "label": "第二天活动"},
            {"node_id": "intent_06", "role": "cafe", "label": "宠物友好咖啡"},
            {"node_id": "intent_07", "role": "parking", "label": "停车"},
        ]
    }
    state = {
        "scene_type": "couple",
        "selected_plan": {
            "b_itinerary_blueprint": blueprint,
            "timeline": [
                {
                    "time": "12:00-13:10",
                    "activity": "餐饮",
                    "poi_id": None,
                    "role": "restaurant_lunch",
                    "type": "planning_intent",
                },
                {
                    "time": "16:30-18:00",
                    "activity": "第一天活动",
                    "poi_id": None,
                    "role": "family_activity",
                    "type": "planning_intent",
                },
                {
                    "time": "18:00-18:50",
                    "activity": "特产/伴手礼",
                    "poi_id": None,
                    "role": "souvenir_shopping",
                    "type": "planning_intent",
                },
                {
                    "time": "20:30-次日10:00",
                    "activity": "住宿",
                    "poi_id": None,
                    "role": "lodging",
                    "type": "planning_intent",
                },
                {
                    "time": "10:00-12:00",
                    "activity": "第二天活动",
                    "poi_id": None,
                    "role": "family_activity",
                    "type": "planning_intent",
                },
                {
                    "time": "12:00-13:00",
                    "activity": "宠物友好咖啡",
                    "poi_id": None,
                    "role": "cafe",
                    "type": "planning_intent",
                },
                {
                    "time": "15:45-16:00",
                    "activity": "停车",
                    "poi_id": None,
                    "role": "parking",
                    "type": "planning_intent",
                },
            ],
            "candidate_generation_summary": "共生成 18 个候选方案，其中 0 个满足硬约束，18 个被过滤。缺少活动和餐厅均宠物友好的证据（18 个）",
            "candidate_evidence_grade": {
                "version": "candidate_evidence_grade_v1",
                "hard_filter_reason": "缺少活动和餐厅均宠物友好的证据",
                "nodes": [
                    {
                        "node_id": node_id,
                        "label": label,
                        "supply_domain": supply_domain,
                        "execution_status": "raw_candidates_failed_hard_constraints",
                        "has_raw_candidate_coverage": True,
                        "hard_filter_reason": "缺少活动和餐厅均宠物友好的证据",
                        "grade": "all_candidates_filtered",
                    }
                    for node_id, label, supply_domain in [
                        ("intent_01", "餐饮", "restaurant"),
                        ("intent_02", "第一天活动", "activity"),
                        ("intent_03", "特产/伴手礼", "shopping"),
                        ("intent_04", "住宿", "lodging"),
                        ("intent_05", "第二天活动", "activity"),
                        ("intent_06", "宠物友好咖啡", "restaurant"),
                        ("intent_07", "停车", "transport_service"),
                    ]
                ],
            },
        },
        "b_rag_candidate_coverage": {
            "all_nodes_covered": True,
            "covered_node_ids": [
                "intent_01",
                "intent_02",
                "intent_03",
                "intent_04",
                "intent_05",
                "intent_06",
                "intent_07",
            ],
            "missing_node_ids": [],
        },
        "filter_reasons": {
            "_summary_detail": {
                "reason_counts": {"缺少活动和餐厅均宠物友好的证据": 18}
            },
        },
        "constraints": {
            "b_replan_trace": {
                "status": "completed",
                "last_completed_request": request,
            }
        },
        "candidate_recall_diagnostics": {"last_replan_request": request},
        "tool_results": {},
        "action_sequence": [],
        "execution_status": "failed",
        "payment_status": "not_required",
        "execution_log": [],
    }

    message = share_generator_node(state)["final_share_message"]

    assert "当前主要阻塞：缺少活动和餐厅均宠物友好的证据" in message
    assert "已有初始候选但未通过硬约束/可预订确认的节点" in message
    assert "宠物友好" in message
    assert "已经尝试过一次有界候选证据/时段修复，但仍没有形成可执行动作" in message
    assert "下一步需要改动约束、换一个时间窗，或拿到商家/场次的新供给确认后再执行" in message
    assert "不应出现" not in message
    assert "已订好" not in message
    assert "推荐这个方案" not in message
    assert "位置已经订好了" not in message


def test_success_share_still_reports_unsupported_dropped_role():
    state = {
        "scene_type": "solo",
        "constraints": {
            "people_count": 1,
            "raw_text": "想先找个20点以后还能处理的口腔/牙科看看，附近再吃点不辣、软一点的晚饭。",
        },
        "selected_plan": {
            "timeline": [
                {
                    "time": "18:30-20:00",
                    "activity": "禾间轻食日料",
                    "poi_id": "res_light_japanese",
                    "role": "restaurant_dinner",
                }
            ],
        },
        "b_itinerary_blueprint": {
            "unsupported_roles": ["dental_clinic"],
            "node_intents": [
                {"role": "dental_clinic", "label": "牙科/口腔诊所"},
                {"role": "restaurant_dinner", "label": "晚餐"},
            ],
        },
        "b_rag_candidate_coverage": {
            "unsupported_missing_roles": ["dental_clinic"],
            "missing_node_ids": ["intent_01"],
        },
        "tool_results": {
            "reserve_restaurant_1": {
                "success": True,
                "name": "禾间轻食日料",
                "data": {"action": "reserve_restaurant"},
            }
        },
        "execution_status": "success",
        "payment_status": "not_required",
        "execution_log": [],
    }

    shared = share_generator_node(state)

    message = shared["final_share_message"]
    assert "可执行的部分已经订好了" in message
    assert "禾间轻食日料" in message
    assert "牙科/口腔诊所" in message
    assert "不能说已经订好" in message
