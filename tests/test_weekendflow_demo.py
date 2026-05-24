import json

from run import build_initial_state
from src.graph import get_graph
from src.nodes import intent_parser as intent_parser_module
from src.nodes.intent_parser import constraints_from_intent, intent_parser_node, parse_intent
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


def test_intent_parser_preserves_lively_bbq_handoff_keywords() -> None:
    raw_text = "我想和朋友一起玩半天，越热闹越好，晚上再吃个烤肉"
    intent = parse_intent(raw_text)
    constraints = constraints_from_intent(intent)

    assert intent["raw_text"] == raw_text
    assert constraints["raw_text"] == raw_text
    assert constraints["scene"] == "friends"
    assert constraints["time_window"] == "tonight"
    assert constraints["duration_range"] == [2, 4]
    assert constraints["people_count"] == 2
    assert "多人活动" in constraints["planning_preferences"]["activity_type"]
    assert "烤肉" in constraints["planning_preferences"]["food_type"]
    assert "烤肉" in constraints["planning_preferences"]["restaurant_type"]
    assert "热闹" in constraints["planning_preferences"]["emotion_type"]
    assert "氛围感" in constraints["planning_preferences"]["atmosphere_type"]
    assert "烤肉" in constraints["soft_tags"]
    assert "热闹" in constraints["soft_tags"]
    assert "社交" in constraints["soft_tags"]


def test_a_llm_prompt_uses_keyword_inventory_not_one_shot_examples() -> None:
    prompt = intent_parser_module.A_LLM_INTENT_SYSTEM_PROMPT

    assert "->" not in prompt
    assert "food:" in prompt
    assert "emotion:" in prompt
    assert "bbq" in prompt
    assert "lively" in prompt


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


def test_graph_accepts_messages_when_user_input_empty() -> None:
    graph = get_graph()
    result = graph.invoke(
        {
            "user_input": "",
            "messages": [
                {"role": "user", "content": "下午和朋友出去玩，4个人"},
            ],
        }
    )

    assert result["user_input"] == "下午和朋友出去玩，4个人"
    assert result["scene_type"] == "friends"
    assert result["constraints"]["people_count"] == 4


def _clear_a_llm_env(monkeypatch) -> None:
    for key in (
        "WF_A_LLM_ENABLED",
        "WF_A_AI_ENABLED",
        "WF_A_LLM_API_KEY",
        "WF_A_LLM_APP_KEY",
        "LONGCAT_API_KEY",
        "LONGCAT_APP_KEY",
        "WF_A_LLM_BASE_URL",
        "WF_A_LLM_MODEL",
        "WF_A_LLM_TIMEOUT_SECONDS",
        "WF_A_LLM_MAX_TOKENS",
        "WF_A_LLM_TEMPERATURE",
        "LONGCAT_BASE_URL",
        "LONGCAT_MODEL",
        "LONGCAT_TIMEOUT_SECONDS",
        "LONGCAT_MAX_TOKENS",
        "LONGCAT_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_a_llm_intent_is_default_off(monkeypatch) -> None:
    _clear_a_llm_env(monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("A-stage LLM should be default off")

    monkeypatch.setattr(intent_parser_module, "chat_completion", fail_if_called)

    result = intent_parser_node({"user_input": "下午和朋友出去玩，4个人"})

    assert result["scene_type"] == "friends"
    assert "a_llm_intent" not in result


def test_a_llm_intent_uses_longcat_when_enabled(monkeypatch) -> None:
    _clear_a_llm_env(monkeypatch)
    monkeypatch.setenv("WF_A_LLM_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fake_chat_completion(messages, *, config):
        payload = json.loads(messages[1]["content"])
        assert messages[0]["role"] == "system"
        assert config.base_url == "https://api.longcat.chat/openai"
        assert config.model == "LongCat-Flash-Chat"
        assert config.max_tokens == 900
        assert payload["user_input"] == "今晚想和对象散步吃轻食，预算500，别去太挤的商场"
        return {
            "content": json.dumps(
                {
                    "task_type": "local_life_plan",
                    "goal": "安排情侣轻量约会",
                    "scene": "couple",
                    "time": {
                        "window": "tonight",
                        "duration_range": [2, 4],
                        "start_time": "19:00",
                        "end_time": None,
                    },
                    "people": [
                        {"role": "self", "needs": []},
                        {"role": "partner", "needs": ["atmosphere", "light_food"]},
                    ],
                    "location": {
                        "origin": "静安寺",
                        "distance_preference": "nearby",
                        "max_distance_km": 6,
                        "transport_mode": "walking",
                        "route_mode": "walking",
                        "city": "上海",
                    },
                    "budget": {
                        "amount": 500,
                        "type": "total",
                        "sensitivity": "medium",
                    },
                    "planning_preferences": {
                        "activity_type": ["micro_vacation"],
                        "food_type": ["light_food"],
                        "emotion_type": ["ritual", "relaxation"],
                        "atmosphere_type": ["romantic"],
                        "experience_type": ["local_culture"],
                        "restaurant_type": ["dine_in"],
                        "pace": "relaxed",
                    },
                    "constraints": {
                        "hard": ["dine_in"],
                        "soft": ["nearby", "ritual"],
                        "avoid": ["crowded_mall"],
                    },
                    "people_count": 2,
                    "ritual_need": True,
                    "emotion_need": ["ritual"],
                    "missing_slots": [],
                    "confidence": {"scene": 0.91},
                    "raw_text": "今晚想和对象散步吃轻食，预算500，别去太挤的商场",
                }
            ),
            "model": config.model,
            "usage": {"total_tokens": 88},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(intent_parser_module, "chat_completion", fake_chat_completion)

    result = intent_parser_node(
        {"user_input": "今晚想和对象散步吃轻食，预算500，别去太挤的商场"}
    )

    assert result["scene_type"] == "couple"
    assert result["constraints"]["budget"] == 500
    assert result["constraints"]["route_mode"] == "walking"
    assert "微度假" in result["constraints"]["planning_preferences"]["activity_type"]
    assert "堂食" in result["constraints"]["hard_tags"]
    assert "商场拥挤" in result["constraints"]["avoid"]
    assert result["a_llm_intent"]["success"] is True
    assert result["a_llm_intent"]["provider"] == "longcat"
    assert result["a_llm_intent"]["api_format"] == "openai"
    assert result["a_llm_intent"]["base_url"] == "https://api.longcat.chat/openai"
    assert result["a_llm_intent"]["usage"] == {"total_tokens": 88}


def test_a_llm_normalization_keeps_bbq_and_lively_keywords() -> None:
    raw_text = "我想和朋友一起玩半天，越热闹越好，晚上再吃个烤肉"
    baseline_intent = parse_intent(raw_text)
    normalized = intent_parser_module._normalize_llm_intent(
        {
            "task_type": "local_life_plan",
            "scene": "friends",
            "planning_preferences": {
                "activity_type": ["group_activity"],
                "food_type": ["bbq"],
                "emotion_type": ["lively"],
                "atmosphere_type": ["atmosphere"],
                "experience_type": [],
                "restaurant_type": ["烤肉"],
                "pace": "lively",
            },
            "constraints": {
                "soft": ["bbq", "lively", "social"],
                "avoid": ["long_queue"],
            },
            "people_count": 2,
            "raw_text": raw_text,
        },
        baseline_intent,
        raw_text,
    )
    constraints = constraints_from_intent(normalized)

    assert "烤肉" in constraints["planning_preferences"]["food_type"]
    assert "烤肉" in constraints["planning_preferences"]["restaurant_type"]
    assert "热闹" in constraints["planning_preferences"]["emotion_type"]
    assert "热闹" in constraints["soft_tags"]


def test_a_llm_intent_prefers_a_specific_openai_config(monkeypatch) -> None:
    _clear_a_llm_env(monkeypatch)
    monkeypatch.setenv("WF_A_LLM_ENABLED", "1")
    monkeypatch.setenv("WF_A_LLM_API_KEY", "a-key")
    monkeypatch.setenv("LONGCAT_API_KEY", "shared-key")
    monkeypatch.setenv("WF_A_LLM_BASE_URL", "https://api.longcat.chat/openai")
    monkeypatch.setenv("WF_A_LLM_MODEL", "LongCat-Flash-Thinking-2601")
    monkeypatch.setenv("WF_A_LLM_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("WF_A_LLM_MAX_TOKENS", "1200")
    monkeypatch.setenv("WF_A_LLM_TEMPERATURE", "0.1")
    seen = {}

    def fake_chat_completion(messages, *, config):
        seen["api_key"] = config.api_key
        seen["base_url"] = config.base_url
        seen["model"] = config.model
        seen["timeout_seconds"] = config.timeout_seconds
        seen["max_tokens"] = config.max_tokens
        seen["temperature"] = config.temperature
        return {
            "content": json.dumps(
                {
                    "task_type": "local_life_plan",
                    "goal": "朋友聚餐",
                    "scene": "friends",
                    "people_count": 4,
                    "raw_text": "今晚和朋友吃饭，4个人",
                }
            ),
            "model": config.model,
            "usage": {},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(intent_parser_module, "chat_completion", fake_chat_completion)

    result = intent_parser_node({"user_input": "今晚和朋友吃饭，4个人"})

    assert seen == {
        "api_key": "a-key",
        "base_url": "https://api.longcat.chat/openai",
        "model": "LongCat-Flash-Thinking-2601",
        "timeout_seconds": 7.0,
        "max_tokens": 1200,
        "temperature": 0.1,
    }
    assert result["scene_type"] == "friends"
    assert result["a_llm_intent"]["api_format"] == "openai"


def test_a_llm_intent_falls_back_to_rules(monkeypatch) -> None:
    _clear_a_llm_env(monkeypatch)
    monkeypatch.setenv("WF_A_LLM_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")

    def fake_chat_completion(messages, *, config):
        raise RuntimeError("bad key test-key")

    monkeypatch.setattr(intent_parser_module, "chat_completion", fake_chat_completion)

    result = intent_parser_node(
        {"user_input": "今天下午想和老婆孩子出去玩，孩子5岁，老婆最近在减肥，别太远"}
    )

    assert result["scene_type"] == "family"
    assert result["a_llm_intent"]["success"] is False
    assert result["a_llm_intent"]["fallback"] is True
    assert "test-key" not in str(result["a_llm_intent"])


def test_run_build_initial_state_accepts_real_user_input() -> None:
    state = build_initial_state("今晚和朋友吃火锅，4个人，预算600", user_id="real_user")

    assert state["user_id"] == "real_user"
    assert state["user_input"] == "今晚和朋友吃火锅，4个人，预算600"
    assert state["constraints"] == {}
    assert state["scenario_activities"] == []
    assert state["payment_ui_mode"] == "auto"


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
