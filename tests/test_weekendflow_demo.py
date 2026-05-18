import json

from src.graph import get_graph
from src.nodes.intent_parser import constraints_from_intent, parse_intent
from src.nodes.memory_manager import (
    apply_value_memory,
    load_memory,
    MEMORY_STORE_VERSION,
    memory_manager_node,
    retrieve_relevant_memories,
    retrieve_relevant_memories_with_trace,
)
from src.nodes.payment_layer import _build_payable_items, payment_layer_node
from src.nodes.scenario_planner import build_scenario_plan
from src.memory.schema import build_memory_atom
from src.memory.storage import decay_short_term_items


def test_intent_parser_extracts_family_constraints() -> None:
    intent = parse_intent("今天下午想和老婆孩子出去玩，孩子5岁，老婆最近在减肥，别太远")

    assert intent["time"]["window"] == "today_afternoon"
    assert intent["scene"] == "family"
    assert intent["location"]["max_distance_km"] == 8.0
    assert "儿童友好" in intent["constraints"]["hard"]
    assert "低卡" in intent["constraints"]["soft"]
    assert "budget" in intent["missing_slots"]


def test_intent_parser_extracts_canonical_handoff_fields() -> None:
    intent = parse_intent(
        "今天下午2点从杨浦区大学路出发，开车和老婆孩子出去玩，"
        "孩子5岁，老婆最近在减脂，别太远，订个堂食。"
    )
    constraints = constraints_from_intent(intent)

    assert constraints["start_time"] == "14:00"
    assert constraints["route_mode"] == "driving"
    assert constraints["city"] == "上海"
    assert constraints["location"]["origin"] == "杨浦区大学路"
    assert constraints["mom_diet"] == "low_calorie"
    assert "堂食" in constraints["hard_tags"]
    assert "低卡" in constraints["planning_preferences"]["food_type"]
    assert "堂食" in constraints["planning_preferences"]["restaurant_type"]


def test_intent_parser_extracts_emotion_and_budget_type() -> None:
    couple_intent = parse_intent(
        "想和对象下午微度假放松一下，有点仪式感，吃得清爽一点。"
    )
    couple_constraints = constraints_from_intent(couple_intent)

    assert couple_constraints["scene"] == "couple"
    assert couple_constraints["ritual_need"] is True
    assert "微度假" in couple_constraints["planning_preferences"]["activity_type"]
    assert "放松" in couple_constraints["planning_preferences"]["emotion_type"]
    assert "轻食" in couple_constraints["planning_preferences"]["food_type"]

    budget_intent = parse_intent("我们三个人，人均200，附近少排队。")
    budget_constraints = constraints_from_intent(budget_intent)

    assert budget_constraints["people_count"] == 3
    assert budget_constraints["budget"] == 200
    assert budget_constraints["budget_type"] == "per_person"
    assert "人均预算" in budget_constraints["soft_tags"]


def test_intent_parser_handles_message_input_and_friends_scene() -> None:
    intent = parse_intent(
        [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "下午和朋友出去玩，4个人"},
        ]
    )

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 4
    assert "group_activity" in intent["planning_preferences"]["activity_type"]


def test_graph_accepts_messages_when_user_input_missing() -> None:
    graph = get_graph()
    result = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "下午和朋友出去玩，4个人"},
            ],
        }
    )

    assert result["user_input"] == "下午和朋友出去玩，4个人"
    assert result["scene_type"] == "friends"
    assert result["constraints"]["people_count"] == 4


def test_memory_does_not_apply_family_defaults_to_friends_request() -> None:
    intent = parse_intent("下午和朋友出去玩，4个人")
    constraints = constraints_from_intent(intent)
    merged = apply_value_memory(constraints, load_memory("u001"))

    assert merged["scene"] == "friends"
    assert merged["people_count"] == 4
    assert merged["child_age"] is None
    assert merged["mom_diet"] is None
    assert "kid_friendly" not in merged["hard_tags"]
    assert "low_calorie" not in merged["soft_tags"]
    assert "family_care" not in merged["active_value_ids"]
    assert "health" not in merged["active_value_ids"]


def test_memory_keeps_current_input_queue_limit() -> None:
    intent = parse_intent("下午和朋友出去玩，4个人，少排队")
    constraints = constraints_from_intent(intent)
    constraints["max_queue_time_min"] = 45

    merged = apply_value_memory(constraints, load_memory("u001"))

    assert merged["max_queue_time_min"] == 45
    assert merged["max_queue_time"] == 45


def test_memory_retrieval_keeps_family_memories_out_of_friends_scene() -> None:
    intent = parse_intent("下午和朋友出去玩，4个人")
    constraints = apply_value_memory(
        constraints_from_intent(intent), load_memory("u001")
    )
    retrieved = retrieve_relevant_memories(constraints, load_memory("u001"))
    retrieved_ids = {item["memory_id"] for item in retrieved}

    assert "companion_child" not in retrieved_ids
    assert "companion_wife" not in retrieved_ids
    assert "value_health" not in retrieved_ids
    assert "value_convenience" in retrieved_ids


def test_memory_manager_persists_current_turn_when_store_configured(
    tmp_path,
    monkeypatch,
) -> None:
    store_path = tmp_path / "memory.json"
    monkeypatch.setenv("WF_MEMORY_STORE_PATH", str(store_path))
    intent = parse_intent("今天下午想和老婆孩子出去玩，孩子6岁，老婆最近在减肥，别太远")

    result = memory_manager_node(
        {
            "user_id": "u_memory_test",
            "user_input": intent["raw_text"],
            "constraints": constraints_from_intent(intent),
            "short_term_memory": [],
        }
    )
    reloaded = load_memory("u_memory_test")
    stored = json.loads(store_path.read_text(encoding="utf-8"))

    assert store_path.exists()
    assert stored["version"] == MEMORY_STORE_VERSION
    assert {"atoms", "graph", "profile"} <= set(stored["users"]["u_memory_test"])
    assert any(
        atom["memory_id"] == "current_spouse_diet"
        for atom in stored["users"]["u_memory_test"]["atoms"]
    )
    assert reloaded["companion_profile"]["child"]["age"] == 6
    assert reloaded["companion_profile"]["wife"]["ttl_turns"] == 4
    assert result["memory_trace"]["policy"] == "explicit_current_input_first"
    assert {op["target"] for op in result["memory_updates"]} >= {
        "episodic_memory",
        "companion_profile.child",
        "companion_profile.wife",
    }

    second_intent = parse_intent("今天下午继续带孩子出去，孩子8岁，别太远")
    memory_manager_node(
        {
            "user_id": "u_memory_test",
            "user_input": second_intent["raw_text"],
            "constraints": constraints_from_intent(second_intent),
            "short_term_memory": [],
        }
    )
    updated_store = json.loads(store_path.read_text(encoding="utf-8"))
    assert (
        updated_store["users"]["u_memory_test"]["profile"]["companion_profile"][
            "child"
        ]["age"]
        == 8
    )


def test_memory_graph_expands_spouse_diet_relation(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "memory.json"
    monkeypatch.setenv("WF_MEMORY_STORE_PATH", str(store_path))
    intent = parse_intent("今天下午和老婆吃饭，老婆最近在减脂，吃得清爽一点")
    memory_manager_node(
        {
            "user_id": "u_graph_test",
            "user_input": intent["raw_text"],
            "constraints": constraints_from_intent(intent),
            "short_term_memory": [],
        }
    )
    memory = load_memory("u_graph_test")
    constraints = apply_value_memory(constraints_from_intent(intent), memory)

    retrieved, trace = retrieve_relevant_memories_with_trace(constraints, memory)
    spouse_memory = next(
        item for item in retrieved if item["memory_id"] == "current_spouse_diet"
    )

    assert "graph" in spouse_memory["retrieval_sources"]
    assert spouse_memory["graph_hop"] in {0, 1, 2}
    assert trace["sources"]["graph"]["hits"]["current_spouse_diet"]["path"]


def test_memory_duplicate_filter_refreshes_same_episode(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "memory.json"
    monkeypatch.setenv("WF_MEMORY_STORE_PATH", str(store_path))
    text = "今天下午和老婆孩子出去玩，孩子6岁，老婆最近在减肥，别太远"
    intent = parse_intent(text)
    state = {
        "user_id": "u_duplicate_test",
        "user_input": intent["raw_text"],
        "constraints": constraints_from_intent(intent),
        "short_term_memory": [],
    }

    memory_manager_node(state)
    second = memory_manager_node(state)
    stored = json.loads(store_path.read_text(encoding="utf-8"))
    atoms = stored["users"]["u_duplicate_test"]["atoms"]
    matching_episodes = [
        atom
        for atom in atoms
        if atom["kind"] == "episode" and atom["summary"] == intent["raw_text"]
    ]

    assert len(matching_episodes) == 1
    assert any(
        update["op"] == "refresh" and update["target"] == "episodic_memory"
        for update in second["memory_updates"]
    )


def test_short_term_ttl_decay_expires_spouse_diet() -> None:
    spouse_atom = build_memory_atom(
        "current_spouse_diet",
        "companion",
        "short_term",
        "伴侣近期偏低卡轻食",
        ["wife", "low_calorie", "light_food"],
        {"role": "wife", "state": "dieting", "needs": ["low_calorie"]},
        ttl_turns=1,
        entities=["wife"],
        relations=[
            {
                "source": "wife",
                "predicate": "prefers",
                "target": "low_calorie",
                "confidence": 0.9,
            }
        ],
    )
    memory = decay_short_term_items(
        load_memory(
            "u_ttl_test",
            {
                "user_id": "u_ttl_test",
                "atoms": [spouse_atom],
                "short_term_items": [spouse_atom],
                "companion_profile": {
                    "wife": {
                        "state": "dieting",
                        "needs": ["low_calorie"],
                        "ttl": "short_term",
                        "ttl_turns": 1,
                    }
                },
            },
        )
    )

    assert not any(
        atom["memory_id"] == "current_spouse_diet" for atom in memory["atoms"]
    )
    assert not any(
        item["memory_id"] == "current_spouse_diet"
        for item in memory["short_term_items"]
    )
    assert memory["companion_profile"]["wife"]["state"] is None


def test_memory_trace_records_scene_isolation_skip_reasons() -> None:
    intent = parse_intent("下午和朋友出去玩，4个人")
    constraints = apply_value_memory(
        constraints_from_intent(intent),
        load_memory("u001"),
    )
    child_atom = build_memory_atom(
        "current_child_profile",
        "companion",
        "long_term",
        "孩子年龄 6",
        ["child", "kid_friendly"],
        {"role": "child", "age": 6},
        entities=["child"],
    )
    retrieved, trace = retrieve_relevant_memories_with_trace(
        constraints,
        {
            **load_memory("u001"),
            "atoms": [child_atom],
            "graph": {
                "entities": {
                    "child": {
                        "entity_id": "child",
                        "memory_ids": ["current_child_profile"],
                    }
                },
                "relations": [],
            },
        },
    )

    assert "current_child_profile" not in {item["memory_id"] for item in retrieved}
    assert any(
        item["memory_id"] == "current_child_profile"
        and item["reason"] == "child_tag_requires_family_or_child_request"
        for item in trace["skipped"]
    )


def test_semantic_adapter_error_falls_back_to_sparse_retrieval() -> None:
    class BrokenStore:
        def search(self, *args, **kwargs):
            raise NotImplementedError("no embeddings")

    intent = parse_intent("下午和朋友出去玩，4个人，少排队")
    constraints = apply_value_memory(
        constraints_from_intent(intent),
        load_memory("u001"),
    )

    retrieved, trace = retrieve_relevant_memories_with_trace(
        constraints,
        load_memory("u001"),
        semantic_store=BrokenStore(),
    )

    assert retrieved
    assert trace["sources"]["semantic"]["status"] == "error"
    assert trace["sources"]["semantic"]["reason"] == "NotImplementedError"


def test_v2_memory_store_preserves_explicit_empty_profile_lists(
    tmp_path,
    monkeypatch,
) -> None:
    store_path = tmp_path / "cleared-memory.json"
    store_path.write_text(
        json.dumps(
            {
                "version": MEMORY_STORE_VERSION,
                "users": {
                    "cleared_user": {
                        "profile": {
                            "stable_profile": {"default_transport": "walk"},
                            "companion_profile": {
                                "wife": {
                                    "state": None,
                                    "needs": [],
                                    "ttl": "expired",
                                }
                            },
                            "preference_profile": {
                                "food": [],
                                "activity": [],
                                "avoid": [],
                            },
                            "history_feedback": [],
                            "derived_defaults": {},
                            "value_profile": [],
                        },
                        "atoms": [],
                        "graph": {"entities": {}, "relations": []},
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WF_MEMORY_STORE_PATH", str(store_path))

    memory = load_memory("cleared_user")

    assert memory["preference_profile"]["avoid"] == []
    assert memory["companion_profile"]["wife"]["needs"] == []
    assert memory["value_profile"] == []
    assert "long_queue" not in memory["preference_profile"]["avoid"]
    assert "low_calorie" not in memory["companion_profile"]["wife"]["needs"]


def test_v1_memory_store_migrates_to_v2_atoms_graph_profile(
    tmp_path,
    monkeypatch,
) -> None:
    store_path = tmp_path / "legacy-memory.json"
    store_path.write_text(
        json.dumps(
            {
                "version": 1,
                "users": {
                    "legacy_user": {
                        "user_id": "legacy_user",
                        "companion_profile": {
                            "child": {
                                "age": 7,
                                "needs": ["kid_friendly"],
                                "confidence": 0.88,
                                "source": "legacy_profile",
                            }
                        },
                        "short_term_items": [
                            {
                                "memory_id": "legacy_episode",
                                "kind": "episode",
                                "scope": "short_term",
                                "text": "昨天说想找亲子活动",
                                "tags": ["family", "kid_friendly"],
                                "ttl_turns": 3,
                            }
                        ],
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WF_MEMORY_STORE_PATH", str(store_path))

    memory = load_memory("legacy_user")
    migrated = json.loads(store_path.read_text(encoding="utf-8"))
    user_record = migrated["users"]["legacy_user"]

    assert migrated["version"] == MEMORY_STORE_VERSION
    assert {"atoms", "graph", "profile"} <= set(user_record)
    assert memory["companion_profile"]["child"]["age"] == 7
    assert any(atom["memory_id"] == "legacy_episode" for atom in user_record["atoms"])
    assert "child" in user_record["graph"]["entities"]


def test_scenario_planner_outputs_a_to_b_handoff() -> None:
    intent = parse_intent("今天下午想和老婆孩子出去玩，孩子5岁，老婆最近在减肥，别太远")
    constraints = apply_value_memory(
        constraints_from_intent(intent), load_memory("u001")
    )
    scenario_plan = build_scenario_plan(
        {
            "intent": intent,
            "constraints": constraints,
        }
    )

    assert scenario_plan["scene_type"] == "family"
    assert scenario_plan["scenario_subtype"] in {
        "family_health_food",
        "family_parent_child_light",
    }
    assert "亲子" in scenario_plan["scenario_activities"]
    assert "低卡" in scenario_plan["scenario_activities"]
    assert "scenario_facets" in scenario_plan
    assert scenario_plan["scenario_template"]["poi_mix"] == ["activity", "restaurant"]
    assert scenario_plan["route_pattern_hints"]["should_search"] is True


def test_weekendflow_a_stage_outputs_intent_and_value_memory() -> None:
    graph = get_graph()
    result = graph.invoke(
        {
            "user_id": "u001",
            "user_input": (
                "今天下午想和老婆孩子出去玩几个小时，"
                "别离家太远，孩子5岁，老婆最近在减肥。"
            ),
        }
    )

    assert result["scene_type"] == "family"
    assert result["intent"]["location"]["distance_preference"] == "nearby"
    assert result["constraints"]["max_queue_time_min"] == 15
    assert result["constraints"]["value_weights"]["family_care"] == 0.92
    assert result["memory"]["companion_profile"]["child"]["age"] == 5
    assert result["memory"]["value_profile"][0]["planning_effect"]
    assert result["constraints"]["memory_policy"] == "explicit_current_input_first"
    assert result["memory_trace"]["retrieved"]
    assert result["memory_updates"]
    assert result["scenario_activities"]
    assert result["constraints"]["scenario_activities"] == result["scenario_activities"]
    assert result["short_term_memory"]


def test_payment_layer_only_pays_itinerary_destinations() -> None:
    state = {
        "selected_plan": {
            "timeline": [
                {
                    "poi_id": "act_001",
                    "activity": "亲子陶艺体验馆",
                    "time": "14:00-15:30",
                },
                {
                    "poi_id": "rest_001",
                    "activity": "轻食餐厅",
                    "time": "17:00-18:00",
                },
            ]
        },
        "action_sequence": [
            {
                "step": 1,
                "action_type": "order_activity_ticket",
                "poi_id": "act_001",
                "time": "14:00",
            },
            {
                "step": 2,
                "action_type": "reserve_restaurant",
                "poi_id": "rest_001",
                "time": "17:00",
            },
            {
                "step": 3,
                "action_type": "order_addon_service",
                "time": "18:00",
            },
        ],
        "tool_results": {
            "order_activity_ticket_1": {
                "success": True,
                "action": "order_activity_ticket",
                "name": "亲子陶艺体验馆",
                "data": {
                    "success": True,
                    "order_id": "act_001_14:00_TEST",
                    "time_slot": "14:00",
                    "quantity": 3,
                    "total_price": 594,
                    "payment_required": True,
                },
            },
            "reserve_restaurant_2": {
                "success": True,
                "action": "reserve_restaurant",
                "name": "轻食餐厅",
                "data": {
                    "success": True,
                    "order_id": "rest_001_17:00_TEST",
                    "payment_required": False,
                },
            },
            "order_addon_service_3": {
                "success": True,
                "action": "order_addon_service",
                "name": "庆祝蛋糕",
                "data": {
                    "success": True,
                    "order_id": "ADDON_cake_TEST",
                    "price": 88,
                    "payment_required": True,
                },
            },
        },
        "execution_log": [],
    }

    payable_items = _build_payable_items(state)

    assert len(payable_items) == 1
    assert payable_items[0]["action_type"] == "order_activity_ticket"
    assert payable_items[0]["name"] == "亲子陶艺体验馆"


def test_payment_layer_marks_payable_destination_success() -> None:
    state = {
        "selected_plan": {
            "timeline": [
                {
                    "poi_id": "act_001",
                    "activity": "亲子陶艺体验馆",
                    "time": "14:00-15:30",
                }
            ]
        },
        "action_sequence": [
            {
                "step": 1,
                "action_type": "order_activity_ticket",
                "poi_id": "act_001",
                "time": "14:00",
            }
        ],
        "tool_results": {
            "order_activity_ticket_1": {
                "success": True,
                "action": "order_activity_ticket",
                "name": "亲子陶艺体验馆",
                "data": {
                    "success": True,
                    "order_id": "act_001_14:00_TESTPAY",
                    "time_slot": "14:00",
                    "quantity": 1,
                    "total_price": 198,
                    "payment_required": True,
                },
            }
        },
        "payment_ui_mode": "auto",
        "payment_auto_confirm": True,
        "payment_auto_pay": True,
        "execution_log": [],
    }

    from src.tools.mock_apis import db

    db.orders["act_001_14:00_TESTPAY"] = {
        "poi_id": "act_001",
        "time_slot": "14:00",
        "quantity": 1,
        "status": "confirmed",
        "payment_status": "unpaid",
    }

    result = payment_layer_node(state)

    assert result["payment_status"] == "success"
    assert result["payment_results"]["payment_status"] == "paid"
    assert db.orders["act_001_14:00_TESTPAY"]["payment_status"] == "paid"
