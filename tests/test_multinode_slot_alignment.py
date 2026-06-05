from src.nodes.candidate_generator import _build_multinode_schedule


def test_multinode_schedule_prefers_deal_valid_time_over_skeleton_time():
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

    timeline, _schedule = _build_multinode_schedule([node], blueprint, {"start_time": "10:00"})

    assert timeline[0]["time"] == "14:00-16:00"


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


def test_multinode_schedule_rejects_lunch_role_that_drifts_to_evening():
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

    assert timeline == []
    assert schedule["time_window_feasible"] is False
    assert schedule["slot_alignment_violations"][0]["poi_id"] == "res_evening_only"
    assert schedule["slot_alignment_violations"][0]["role"] == "restaurant_lunch"


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
