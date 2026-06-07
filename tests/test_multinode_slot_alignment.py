from src.nodes.candidate_generator import _build_multinode_schedule, _combine_multinode_plan_candidates


def test_multinode_schedule_repairs_flexible_deal_time_inside_day_window():
    node = {
        "poi_id": "act_museum",
        "name": "Museum visit",
        "type": "activity",
        "itinerary_role": "cultural_photo",
        "_itinerary_intent": {
            "node_id": "intent_01",
            "role": "cultural_photo",
            "default_duration_min": 120,
        },
        "available_slots": [{"time": "10:00"}, {"time": "14:00"}],
        "deals": [
            {
                "deal_id": "deal_museum",
                "product_id": "prod_museum",
                "valid_time": ["14:00"],
            }
        ],
    }
    blueprint = {
        "time_skeleton": {
            "days": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "day": 1,
                            "start_time": "10:00",
                            "duration_min": 120,
                        }
                    ],
                }
            ]
        }
    }

    timeline, schedule = _build_multinode_schedule(
        [node],
        blueprint,
        {"start_time": "10:00", "end_time": "20:30"},
    )

    assert [item["poi_id"] for item in timeline] == ["act_museum"]
    assert timeline[0]["time"] == "14:00-16:00"
    assert schedule["time_window_feasible"] is True
    assert schedule["slot_alignment_violations"] == []
    assert schedule["flexible_slot_repairs"][0]["poi_id"] == "act_museum"
    assert schedule["flexible_slot_repairs"][0]["reason"] == "flexible_slot_drift_within_hard_window"


def test_multinode_schedule_marks_slots_after_deadline_infeasible():
    node = {
        "poi_id": "act_late",
        "name": "Late activity",
        "type": "activity",
        "itinerary_role": "cultural_photo",
        "duration_min": 90,
        "_itinerary_intent": {
            "node_id": "intent_01",
            "role": "cultural_photo",
            "default_duration_min": 90,
        },
        "available_slots": [{"time": "21:00"}],
    }
    blueprint = {
        "planning_days": 1,
        "time_skeleton": {
            "days": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "day": 1,
                            "start_time": "21:00",
                            "duration_min": 90,
                        }
                    ],
                }
            ]
        },
    }

    timeline, schedule = _build_multinode_schedule(
        [node],
        blueprint,
        {"start_time": "10:30", "end_time": "20:30"},
    )

    assert schedule["time_window_feasible"] is False
    assert timeline == []
    assert schedule["skipped_time_window_nodes"][0]["poi_id"] == "act_late"


def test_multinode_schedule_keeps_fit_nodes_and_skips_late_overflow_nodes():
    dinner = {
        "poi_id": "res_late_snack",
        "name": "Late snack",
        "type": "restaurant",
        "itinerary_role": "restaurant_specific",
        "duration_min": 80,
        "_itinerary_intent": {
            "node_id": "intent_01",
            "role": "restaurant_specific",
            "default_duration_min": 80,
        },
    }
    karaoke = {
        "poi_id": "act_karaoke",
        "name": "Karaoke",
        "type": "activity",
        "itinerary_role": "karaoke",
        "duration_min": 120,
        "_itinerary_intent": {
            "node_id": "intent_02",
            "role": "karaoke",
            "default_duration_min": 120,
        },
        "available_slots": [{"time": "23:30"}],
    }
    blueprint = {
        "planning_days": 1,
        "time_skeleton": {
            "days": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "day": 1,
                            "start_time": "21:00",
                            "duration_min": 80,
                        },
                        {
                            "node_id": "intent_02",
                            "day": 1,
                            "start_time": "23:30",
                            "duration_min": 120,
                        },
                    ],
                }
            ]
        },
    }

    timeline, schedule = _build_multinode_schedule(
        [dinner, karaoke],
        blueprint,
        {"start_time": "21:00", "end_time": "00:30"},
    )

    assert [item["poi_id"] for item in timeline] == ["res_late_snack"]
    assert timeline[0]["time"] == "21:00-22:20"
    assert schedule["time_window_feasible"] is False
    assert schedule["skipped_time_window_nodes"][0]["poi_id"] == "act_karaoke"


def test_multinode_schedule_does_not_make_lunch_role_hard_by_english_label():
    restaurant = {
        "poi_id": "res_evening_only",
        "name": "Evening-only bistro",
        "type": "restaurant",
        "itinerary_role": "restaurant_lunch",
        "duration_min": 80,
        "_itinerary_intent": {
            "node_id": "intent_01",
            "role": "restaurant_lunch",
            "default_duration_min": 80,
        },
        "available_slots": [{"time": "18:00"}],
    }
    blueprint = {
        "planning_days": 1,
        "time_skeleton": {
            "days": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "day": 1,
                            "start_time": "12:00",
                            "end_time": "13:20",
                            "duration_min": 80,
                            "part_of_day": "lunch",
                            "role": "restaurant_lunch",
                        }
                    ],
                }
            ]
        },
    }

    timeline, schedule = _build_multinode_schedule(
        [restaurant],
        blueprint,
        {"start_time": "10:00", "end_time": "21:00"},
    )

    assert [item["poi_id"] for item in timeline] == ["res_evening_only"]
    assert timeline[0]["time"] == "18:00-19:20"
    assert schedule["time_window_feasible"] is True
    assert schedule["slot_alignment_violations"] == []
    assert schedule["flexible_slot_repairs"][0]["role"] == "restaurant_lunch"


def test_multinode_schedule_rejects_flexible_drift_into_protected_rest_or_event():
    afternoon = {
        "poi_id": "act_overlap_nap",
        "name": "Nap-overlap activity",
        "type": "activity",
        "itinerary_role": "family_activity",
        "duration_min": 90,
        "_itinerary_intent": {
            "node_id": "intent_01",
            "role": "family_activity",
            "label": "亲子活动",
            "default_duration_min": 90,
        },
        "available_slots": [{"time": "14:00"}],
    }
    dinner = {
        "poi_id": "res_overlap_show",
        "name": "Show-overlap dinner",
        "type": "restaurant",
        "itinerary_role": "restaurant_specific",
        "duration_min": 75,
        "_itinerary_intent": {
            "node_id": "intent_02",
            "role": "restaurant_specific",
            "label": "指定餐饮",
            "default_duration_min": 75,
        },
        "available_slots": [{"time": "18:30"}],
    }
    blueprint = {
        "planning_days": 1,
        "time_skeleton": {
            "days": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "day": 1,
                            "start_time": "10:00",
                            "end_time": "11:30",
                            "duration_min": 90,
                            "role": "family_activity",
                            "label": "亲子活动",
                        },
                        {
                            "node_id": None,
                            "day": 1,
                            "start_time": "13:30",
                            "end_time": "15:00",
                            "duration_min": 90,
                            "role": "rest",
                            "label": "午睡/休息",
                            "part_of_day": "protected_rest",
                            "execution_status": "protected_non_executable",
                            "protected": True,
                            "anchor_type": "rest",
                        },
                        {
                            "node_id": "intent_02",
                            "day": 1,
                            "start_time": "18:00",
                            "end_time": "19:00",
                            "duration_min": 75,
                            "role": "restaurant_specific",
                            "label": "指定餐饮",
                            "part_of_day": "dinner",
                        },
                        {
                            "node_id": "intent_03",
                            "day": 1,
                            "start_time": "19:00",
                            "end_time": "20:40",
                            "duration_min": 100,
                            "role": "talk_show",
                            "label": "脱口秀/演出",
                            "anchor_type": "event",
                            "anchor_label": "亲子剧",
                        },
                    ],
                }
            ]
        },
    }

    timeline, schedule = _build_multinode_schedule(
        [afternoon, dinner],
        blueprint,
        {"start_time": "10:00", "end_time": "21:00"},
    )

    assert timeline == []
    assert schedule["time_window_feasible"] is False
    assert [item["reason"] for item in schedule["slot_alignment_violations"]] == [
        "protected_anchor_overlap",
        "protected_anchor_overlap",
    ]
    assert schedule["slot_alignment_violations"][0]["overlap_anchor"]["label"] == "午睡/休息"
    assert schedule["slot_alignment_violations"][1]["overlap_anchor"]["label"] == "脱口秀/演出"


def test_multinode_rag_pool_prefers_slot_fit_over_same_node_drift():
    blueprint = {
        "template_mode": "multi_node",
        "planning_horizon": "half_day",
        "planning_days": 1,
        "node_intents": [
            {
                "node_id": "intent_01",
                "role": "citywalk_market",
                "label": "城市漫步/市集",
                "supply_domain": "activity",
                "default_duration_min": 120,
            }
        ],
        "time_skeleton": {
            "days": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "day": 1,
                            "role": "citywalk_market",
                            "label": "城市漫步/市集",
                            "start_time": "10:30",
                            "end_time": "12:30",
                            "duration_min": 120,
                        }
                    ],
                }
            ],
        },
    }
    rag_candidates_by_node = {
        "intent_01": [
            {
                "poi_id": "act_late_citywalk",
                "name": "下午城市漫步",
                "type": "activity",
                "supply_domain": "activity",
                "itinerary_role": "citywalk_market",
                "category": "城市漫步",
                "rating": 5.0,
                "duration_min": 120,
                "available_slots": [{"time": "14:00"}],
                "available": True,
            },
            {
                "poi_id": "act_morning_citywalk",
                "name": "上午城市漫步",
                "type": "activity",
                "supply_domain": "activity",
                "itinerary_role": "citywalk_market",
                "category": "城市漫步",
                "rating": 4.0,
                "duration_min": 120,
                "available_slots": [{"time": "10:30"}],
                "available": True,
            },
        ]
    }

    candidates = _combine_multinode_plan_candidates(
        [],
        [],
        {"start_time": "10:30", "end_time": "20:30"},
        "friends",
        blueprint,
        rag_candidates_by_node=rag_candidates_by_node,
        rag_coverage={
            "covered_node_ids": ["intent_01"],
            "all_nodes_covered": True,
            "unsupported_roles_covered": True,
        },
        max_candidates=1,
    )

    assert candidates
    assert candidates[0]["nodes"][0]["poi_id"] == "act_morning_citywalk"
    assert candidates[0]["timeline"][0]["time"] == "10:30-12:30"


def test_multinode_schedule_uses_intent_day_index_when_skeleton_slot_missing():
    day_one = {
        "poi_id": "res_day_one",
        "name": "Day one dinner",
        "type": "restaurant",
        "itinerary_role": "restaurant_dinner",
        "duration_min": 80,
        "_itinerary_intent": {
            "node_id": "intent_01",
            "role": "restaurant_dinner",
            "day_index": 1,
            "default_duration_min": 80,
        },
    }
    day_two = {
        "poi_id": "act_day_two",
        "name": "Day two walk",
        "type": "activity",
        "itinerary_role": "park_scenic_walk",
        "duration_min": 75,
        "_itinerary_intent": {
            "node_id": "intent_02",
            "role": "park_scenic_walk",
            "day_index": 2,
            "default_duration_min": 75,
        },
    }
    blueprint = {
        "planning_days": 2,
        "planning_horizon": "two_day",
        "time_skeleton": {
            "days": [
                {
                    "day": 1,
                    "slots": [
                        {
                            "node_id": "intent_01",
                            "day": 1,
                            "start_time": "18:00",
                            "duration_min": 80,
                        }
                    ],
                },
                {
                    "day": 2,
                    "slots": [],
                },
            ]
        },
    }

    timeline, _schedule = _build_multinode_schedule(
        [day_one, day_two],
        blueprint,
        {"start_time": "15:00", "end_time": "16:00"},
    )

    assert timeline[0]["day"] == 1
    assert timeline[1]["day"] == 2
    assert timeline[1]["time"].startswith("09:")
