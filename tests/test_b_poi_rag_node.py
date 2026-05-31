import json

from src.nodes.b_poi_rag import (
    _forbidden_restaurant_groups_for_retrieval,
    _generic_poi_matches_domain,
    _item_has_negative_group_evidence,
    b_poi_rag_node,
)
from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.plan_optimizer import plan_optimizer_node


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_rag_negative_retrieval_ignores_generic_meat_for_hotpot():
    groups = _forbidden_restaurant_groups_for_retrieval(
        {"user_profile": {"avoid": ["\u70e4\u8089"]}},
        {"avoid": ["\u70e7\u70e4"]},
    )
    hotpot = {
        "name": "\u5df4\u5974\u6bdb\u809a\u706b\u9505",
        "restaurant_category": "\u706b\u9505",
        "tags": ["\u706b\u9505", "meat"],
    }
    bbq_hotpot = {
        "name": "\u725b\u4eba\u81ea\u52a9\u70e7\u70e4\u706b\u9505",
        "restaurant_category": "\u706b\u9505",
        "tags": ["\u706b\u9505", "bbq"],
    }

    assert groups == {"\u70e4\u8089"}
    assert _item_has_negative_group_evidence(hotpot, "\u70e4\u8089") is False
    assert _item_has_negative_group_evidence(bbq_hotpot, "\u70e4\u8089") is True


def test_multi_node_rag_skips_longcat_requirement_compiler_by_default(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "activities.json",
        [
            {
                "poi_id": "act_kid_craft",
                "name": "\u6811\u5c4b\u4eb2\u5b50\u624b\u4f5c\u9986",
                "type": "activity",
                "category": "\u4eb2\u5b50\u624b\u4f5c",
                "tags": ["\u4eb2\u5b50", "\u513f\u7ae5\u53cb\u597d", "\u624b\u4f5c"],
                "coordinates": "121.475,31.230",
                "price": 120,
                "rating": 4.6,
                "available": True,
            }
        ],
    )
    _write_json(
        tmp_path / "restaurants.json",
        [
            {
                "poi_id": "res_light_lunch",
                "name": "\u79be\u95f4\u8f7b\u98df\u9910\u5385",
                "type": "restaurant",
                "category": "\u8f7b\u98df",
                "restaurant_category": "\u4e2d\u5f0f\u8f7b\u98df",
                "tags": ["\u4e2d\u5348", "\u8f7b\u98df", "\u6e05\u6de1"],
                "coordinates": "121.477,31.231",
                "price": 88,
                "rating": 4.5,
                "available": True,
            }
        ],
    )

    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")
    monkeypatch.setenv("WF_B_RAG_ENABLED", "1")
    monkeypatch.setenv("WF_B_RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(tmp_path))

    from src.nodes import b_requirement_compiler

    def fail_if_called(messages, *, config):
        raise AssertionError("LongCat requirement compiler should be skipped for multi-node RAG")

    monkeypatch.setattr(b_requirement_compiler, "chat_completion", fail_if_called)

    result = b_poi_rag_node(
        {
            "user_input": "\u5468\u516d\u5e26\u5b69\u5b50\u4e0a\u5348\u505a\u4eb2\u5b50\u624b\u4f5c\uff0c\u4e2d\u5348\u5403\u8f7b\u98df\u3002",
            "scene_type": "family",
            "constraints": {
                "raw_text": "\u4eb2\u5b50\u624b\u4f5c \u4e2d\u5348 \u8f7b\u98df",
                "child_age": 5,
                "people_count": 3,
                "origin_coordinates": "121.475,31.230",
            },
            "user_profile": {},
            "scenario_activities": [],
            "execution_log": [],
        }
    )

    assert result["b_rag_candidate_evidence"]["coverage_summary"]["covered_node_ids"]
    assert any(
        "deterministic requirement compiler" in line
        for line in result["execution_log"]
    )


def test_candidate_generator_skips_semantic_hints_when_rag_evidence_exists(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "activities.json",
        [
            {
                "poi_id": "act_exhibition",
                "name": "\u57ce\u5e02\u827a\u672f\u5c55",
                "type": "activity",
                "category": "\u535a\u7269\u9986\u5c55\u89c8",
                "tags": ["\u770b\u5c55", "\u827a\u672f\u5c55"],
                "coordinates": "121.475,31.230",
                "price": 60,
                "rating": 4.6,
                "available": True,
            }
        ],
    )
    _write_json(
        tmp_path / "restaurants.json",
        [
            {
                "poi_id": "res_cafe",
                "name": "\u5c55\u9986\u5496\u5561",
                "type": "restaurant",
                "category": "\u5496\u5561",
                "restaurant_category": "\u5496\u5561\u751c\u54c1",
                "tags": ["\u5496\u5561", "\u5c0f\u5750"],
                "coordinates": "121.477,31.231",
                "price": 80,
                "rating": 4.4,
                "available": True,
            }
        ],
    )

    monkeypatch.setenv("WF_B_AI_ENABLED", "1")
    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(tmp_path))

    from src.nodes import candidate_generator as candidate_generator_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("semantic hints should be skipped when RAG evidence is present")

    monkeypatch.setattr(candidate_generator_module, "apply_b_semantic_hints", fail_if_called)
    state = {
        "user_input": "\u770b\u5c55\u540e\u627e\u5496\u5561\u5e97\u5750\u4e00\u4f1a",
        "scene_type": "solo",
        "constraints": {
            "budget": 300,
            "max_distance_km": 10,
            "origin_coordinates": "121.475,31.230",
        },
        "user_profile": {},
        "scenario_activities": [],
        "b_rag_candidate_evidence": {
            "version": "b_rag_candidate_evidence_v1",
            "node_evidence": [
                {
                    "node_id": "intent_01",
                    "role": "exhibition",
                    "coverage_status": "covered",
                    "candidates": [
                        {
                            "poi_id": "act_exhibition",
                            "name": "\u57ce\u5e02\u827a\u672f\u5c55",
                            "type": "activity",
                            "category": "\u535a\u7269\u9986\u5c55\u89c8",
                            "tags": ["\u770b\u5c55"],
                            "coordinates": "121.475,31.230",
                            "price": 60,
                            "available": True,
                        }
                    ],
                },
                {
                    "node_id": "intent_02",
                    "role": "cafe",
                    "coverage_status": "covered",
                    "candidates": [
                        {
                            "poi_id": "res_cafe",
                            "name": "\u5c55\u9986\u5496\u5561",
                            "type": "restaurant",
                            "restaurant_category": "\u5496\u5561\u751c\u54c1",
                            "tags": ["\u5496\u5561"],
                            "coordinates": "121.477,31.231",
                            "price": 80,
                            "available": True,
                        }
                    ],
                },
            ],
        },
        "execution_log": [],
    }

    result = candidate_generator_node(state)

    assert any("skipped LongCat semantic hints" in line for line in result["execution_log"])


def test_rag_uses_deduped_poi_fallback_for_missing_supply_domains(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "deduped_pois.json",
        [
            {
                "id": "B_SHOP_1",
                "name": "\u4e0a\u6d77\u7279\u4ea7\u4f34\u624b\u793c\u5e97",
                "type": "\u8d2d\u7269\u670d\u52a1;\u4e13\u5356\u5e97;\u571f\u7279\u4ea7\u4e13\u5356\u5e97",
                "address": "\u5357\u4eac\u4e1c\u8def",
                "location": "121.480,31.235",
            },
            {
                "id": "B_CONV_1",
                "name": "7-Eleven\u4fbf\u5229\u5e97",
                "type": "\u8d2d\u7269\u670d\u52a1;\u4fbf\u6c11\u5546\u5e97/\u4fbf\u5229\u5e97;\u4fbf\u6c11\u5546\u5e97/\u4fbf\u5229\u5e97",
                "address": "\u9759\u5b89\u5bfa\u9644\u8fd1",
                "location": "121.450,31.225",
            },
            {
                "id": "B_HOTEL_1",
                "name": "\u5916\u6ee9\u5bb6\u5ead\u516c\u5bd3\u9152\u5e97",
                "type": "\u4f4f\u5bbf\u670d\u52a1;\u5bbe\u9986\u9152\u5e97;\u7ecf\u6d4e\u578b\u8fde\u9501\u9152\u5e97",
                "address": "\u5916\u6ee9",
                "location": "121.485,31.240",
            },
            {
                "id": "B_PARK_1",
                "name": "\u5916\u6ee9\u505c\u8f66\u573a",
                "type": "\u4ea4\u901a\u8bbe\u65bd\u670d\u52a1;\u505c\u8f66\u573a;\u516c\u5171\u505c\u8f66\u573a",
                "address": "\u5916\u6ee9",
                "location": "121.486,31.241",
            },
        ],
    )

    monkeypatch.setenv("WF_B_RAG_ENABLED", "1")
    monkeypatch.setenv("WF_B_RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(tmp_path))

    state = {
        "user_input": (
            "\u5728\u5916\u6ee9\u9644\u8fd1\u627e\u4e2a\u4f4f\u5904\uff0c"
            "\u518d\u4e70\u4e0a\u6d77\u7279\u4ea7\u4f34\u624b\u793c\uff0c"
            "\u53bb\u4fbf\u5229\u5e97\u4e70\u96f6\u98df\uff0c\u6700\u540e\u627e\u505c\u8f66\u573a\u3002"
        ),
        "scene_type": "family",
        "constraints": {
            "raw_text": "\u5916\u6ee9 \u4f4f\u5904 \u7279\u4ea7 \u4f34\u624b\u793c \u4fbf\u5229\u5e97 \u505c\u8f66\u573a",
            "city": "\u4e0a\u6d77",
            "origin_coordinates": "121.480,31.235",
        },
        "user_profile": {},
        "scenario_activities": [],
        "execution_log": [],
    }

    evidence = b_poi_rag_node(state)["b_rag_candidate_evidence"]
    covered_roles = {
        block["role"]
        for block in evidence["node_evidence"]
        if block["coverage_status"] == "covered"
    }

    assert {"lodging", "souvenir_shopping", "convenience_store", "parking"} <= covered_roles


def test_generic_poi_fallback_rejects_parking_and_hotel_false_positives():
    restaurant_with_parking_address = {
        "name": "\u5f88\u4e45\u4ee5\u524d\u7f8a\u8089\u4e32",
        "type": "\u9910\u996e\u670d\u52a1;\u4e2d\u9910\u5385;\u4e2d\u9910\u5385",
        "address": "\u5546\u573a\u505c\u8f66\u53ef\u62b5\u6263",
    }
    spa_inside_hotel = {
        "name": "\u9152\u5e97\u5185SPA",
        "type": "\u751f\u6d3b\u670d\u52a1;\u6d17\u6d74\u63a8\u62ff\u573a\u6240;\u6d17\u6d74\u63a8\u62ff\u573a\u6240|\u4f4f\u5bbf\u670d\u52a1;\u4f4f\u5bbf\u670d\u52a1\u76f8\u5173;\u4f4f\u5bbf\u670d\u52a1\u76f8\u5173",
        "address": "\u65b0\u5929\u5730\u9152\u5e97B1\u5c42",
    }
    store_named_after_parking_lot = {
        "name": "\u5145\u7535\u7ad9(\u5730\u4e0b\u505c\u8f66\u573a\u5e97)",
        "type": "\u8d2d\u7269\u670d\u52a1;\u8d85\u7ea7\u5e02\u573a;\u8d85\u5e02",
    }
    real_parking_lot = {
        "name": "\u5916\u6ee9\u505c\u8f66\u573a",
        "type": "\u4ea4\u901a\u8bbe\u65bd\u670d\u52a1;\u505c\u8f66\u573a;\u516c\u5171\u505c\u8f66\u573a",
    }
    real_hotel = {
        "name": "\u5916\u6ee9\u5bb6\u5ead\u516c\u5bd3\u9152\u5e97",
        "type": "\u4f4f\u5bbf\u670d\u52a1;\u5bbe\u9986\u9152\u5e97;\u7ecf\u6d4e\u578b\u8fde\u9501\u9152\u5e97",
    }

    assert _generic_poi_matches_domain(restaurant_with_parking_address, "transport_service") is False
    assert _generic_poi_matches_domain(spa_inside_hotel, "hotel") is False
    assert _generic_poi_matches_domain(store_named_after_parking_lot, "transport_service") is False
    assert _generic_poi_matches_domain(real_parking_lot, "transport_service") is True
    assert _generic_poi_matches_domain(real_hotel, "hotel") is True


def test_parking_role_can_use_explicit_proxy_when_no_parking_poi_exists(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "restaurants.json",
        [
            {
                "poi_id": "res_with_parking",
                "name": "\u5546\u573a\u91cc\u7684\u8f7b\u98df\u9910\u5385",
                "type": "restaurant",
                "category": "\u8f7b\u98df",
                "gaode_type": "\u9910\u996e\u670d\u52a1;\u4e2d\u9910\u5385;\u4e2d\u9910\u5385",
                "coordinates": "121.480,31.235",
                "parking_available": True,
                "parking_fee_policy": "\u5546\u573a\u505c\u8f66\u53ef\u62b5\u6263",
                "available": True,
            }
        ],
    )

    monkeypatch.setenv("WF_B_RAG_ENABLED", "1")
    monkeypatch.setenv("WF_B_RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(tmp_path))

    state = {
        "user_input": "\u665a\u4e0a\u9700\u8981\u627e\u4e2a\u505c\u8f66\u573a\u505c\u8f66",
        "scene_type": "solo",
        "constraints": {
            "raw_text": "\u505c\u8f66 \u505c\u8f66\u573a",
            "city": "\u4e0a\u6d77",
            "origin_coordinates": "121.480,31.235",
        },
        "user_profile": {},
        "scenario_activities": [],
        "execution_log": [],
    }

    evidence = b_poi_rag_node(state)["b_rag_candidate_evidence"]
    parking_block = next(block for block in evidence["node_evidence"] if block["role"] == "parking")

    assert parking_block["coverage_status"] == "covered"
    assert parking_block["candidates"][0]["parking_proxy"] is True
    assert parking_block["candidates"][0]["type"] == "transport_service"


def test_location_anchor_boosts_nearby_generic_lodging(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "deduped_pois.json",
        [
            {
                "id": "hotel_hongqiao",
                "name": "\u4e0a\u6d77\u8679\u6865\u9152\u5e97",
                "type": "\u4f4f\u5bbf\u670d\u52a1;\u5bbe\u9986\u9152\u5e97;\u4e94\u661f\u7ea7\u5bbe\u9986",
                "address": "\u8679\u6865",
                "location": "121.395,31.202",
            },
            {
                "id": "hotel_bund",
                "name": "\u4e0a\u6d77\u5916\u6ee9\u5bb6\u5ead\u9152\u5e97",
                "type": "\u4f4f\u5bbf\u670d\u52a1;\u5bbe\u9986\u9152\u5e97;\u5bbe\u9986\u9152\u5e97",
                "address": "\u5916\u6ee9\u9644\u8fd1",
                "location": "121.490,31.240",
            },
        ],
    )

    monkeypatch.setenv("WF_B_RAG_ENABLED", "1")
    monkeypatch.setenv("WF_B_RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(tmp_path))

    state = {
        "user_input": "\u5728\u5916\u6ee9\u9644\u8fd1\u627e\u4e2a\u4f4f\u5904",
        "scene_type": "family",
        "constraints": {
            "raw_text": "\u5916\u6ee9 \u4f4f\u5904 \u9152\u5e97",
            "city": "\u4e0a\u6d77",
            "origin_coordinates": "121.490,31.240",
        },
        "user_profile": {},
        "scenario_activities": [],
        "execution_log": [],
    }

    evidence = b_poi_rag_node(state)["b_rag_candidate_evidence"]
    lodging_block = next(block for block in evidence["node_evidence"] if block["role"] == "lodging")

    assert lodging_block["candidates"][0]["poi_id"] == "hotel_bund"
    assert "\u5916\u6ee9" in lodging_block["retrieval_meta"]["location_anchor_terms"]


def test_local_poi_rag_emits_candidate_evidence_and_b_consumes_it(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "activities.json",
        [
            {
                "poi_id": "rag_act_lacquer",
                "name": "上海博物馆人民广场馆",
                "type": "activity",
                "category": "博物馆展览",
                "tags": ["展览", "漆器展", "文物展"],
                "coordinates": "121.475,31.230",
                "price": 0,
                "rating": 4.8,
                "duration_min": 120,
                "available": True,
                "business_hours": {"weekday": "09:00-17:00"},
            }
        ],
    )
    _write_json(
        tmp_path / "restaurants.json",
        [
            {
                "poi_id": "rag_lunch_benbang",
                "name": "人民广场本帮简餐",
                "type": "restaurant",
                "category": "本帮菜",
                "restaurant_category": "本帮菜",
                "tags": ["午餐", "中饭", "正餐"],
                "coordinates": "121.477,31.231",
                "price": 88,
                "rating": 4.5,
                "duration_min": 70,
                "queue_time_min": 10,
                "available": True,
            }
        ],
    )
    _write_json(
        tmp_path / "shopping.json",
        [
            {
                "poi_id": "rag_shop_souvenir",
                "name": "上海特产伴手礼店",
                "type": "shopping",
                "category": "伴手礼",
                "tags": ["上海特产", "伴手礼", "礼品"],
                "coordinates": "121.479,31.232",
                "price": 120,
                "rating": 4.3,
                "duration_min": 35,
                "available": True,
            }
        ],
    )
    _write_json(
        tmp_path / "deals.json",
        [
            {
                "deal_id": "deal_lunch",
                "poi_id": "rag_lunch_benbang",
                "title": "本帮午餐套餐",
                "sale_price": 88,
                "package_components": ["主食", "饮品"],
            }
        ],
    )

    monkeypatch.setenv("WF_B_RAG_ENABLED", "1")
    monkeypatch.setenv("WF_B_RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(tmp_path))

    state = {
        "user_input": "我想参加上海红翠斗芳菲漆器展，然后在附近吃个中饭，还想买点上海特产带回去。",
        "scene_type": "solo",
        "constraints": {
            "raw_text": "上海 红翠斗芳菲 漆器展 中饭 上海特产",
            "city": "上海",
            "people_count": 1,
            "budget": 500,
            "max_distance_km": 20,
            "max_queue_time_min": 45,
            "origin_coordinates": "121.475,31.230",
        },
        "user_profile": {},
        "scenario_activities": [],
        "execution_log": [],
    }

    state.update(b_poi_rag_node(state))

    evidence = state["b_rag_candidate_evidence"]
    assert evidence["version"] == "b_rag_candidate_evidence_v1"
    assert evidence["retrieval_trace"][0]["retriever"] == "local_poi_memory_bm25_v0+structured_rerank"
    assert {
        block["role"]
        for block in evidence["node_evidence"]
        if block["coverage_status"] == "covered"
    } >= {"exhibition", "restaurant_lunch", "souvenir_shopping"}
    assert all("retrieval_meta" in block for block in evidence["node_evidence"])

    state.update(candidate_generator_node(state))
    state.update(constraint_filter_node(state))
    state.update(plan_optimizer_node(state))

    selected = state["selected_plan"]
    assert selected["planner_mode"] == "multi_node_itinerary"
    assert selected["benchmark_ready"] is True
    assert {
        node.get("itinerary_role")
        for node in state["candidates"][0]["nodes"]
    } >= {"exhibition", "restaurant_lunch", "souvenir_shopping"}


def test_family_activity_rag_requires_child_friendly_identity(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "activities.json",
        [
            {
                "poi_id": "act_yoga",
                "name": "Yogaga优珈家·瑜伽普拉提",
                "type": "activity",
                "category": "健身中心",
                "tags": ["室内", "低强度", "放松", "运动体验"],
                "coordinates": "121.475,31.230",
                "price": 238,
                "rating": 4.8,
                "available": True,
            },
            {
                "poi_id": "act_kid_craft",
                "name": "树屋亲子手作乐园",
                "type": "activity",
                "category": "亲子手作",
                "tags": ["亲子", "儿童友好", "手作", "室内乐园"],
                "coordinates": "121.476,31.231",
                "price": 168,
                "rating": 4.6,
                "available": True,
            },
        ],
    )
    _write_json(
        tmp_path / "restaurants.json",
        [
            {
                "poi_id": "res_light_lunch",
                "name": "禾间轻食餐厅",
                "type": "restaurant",
                "category": "轻食",
                "tags": ["中餐", "午餐", "清淡"],
                "coordinates": "121.477,31.232",
                "price": 88,
                "rating": 4.5,
                "available": True,
            }
        ],
    )

    monkeypatch.setenv("WF_B_RAG_ENABLED", "1")
    monkeypatch.setenv("WF_B_RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(tmp_path))

    state = {
        "user_input": "周六想带孩子玩一整天，上午找亲子手作，中午吃清淡一点。",
        "scene_type": "family",
        "constraints": {
            "raw_text": "孩子 亲子手作 室内乐园 中午 清淡",
            "city": "上海",
            "people_count": 3,
            "child_age": 5,
            "budget": 800,
            "max_distance_km": 20,
            "origin_coordinates": "121.475,31.230",
        },
        "user_profile": {},
        "scenario_activities": [],
        "execution_log": [],
    }

    evidence = b_poi_rag_node(state)["b_rag_candidate_evidence"]
    family_block = next(
        block for block in evidence["node_evidence"] if block["role"] == "family_activity"
    )
    candidate_ids = {candidate["poi_id"] for candidate in family_block["candidates"]}

    assert "act_kid_craft" in candidate_ids
    assert "act_yoga" not in candidate_ids


def test_meal_nodes_use_local_meal_context_not_whole_request(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "restaurants.json",
        [
            {
                "poi_id": "res_light_lunch",
                "name": "喜禾年健康中式轻食",
                "type": "restaurant",
                "category": "轻食",
                "restaurant_category": "中式轻食",
                "tags": ["午餐", "清淡", "低卡", "少油"],
                "coordinates": "121.475,31.230",
                "price": 88,
                "rating": 4.6,
                "available": True,
            },
            {
                "poi_id": "res_benbang_dinner",
                "name": "阿金大白上海菜馆",
                "type": "restaurant",
                "category": "本帮菜",
                "restaurant_category": "上海菜",
                "tags": ["晚餐", "本帮菜", "家常菜"],
                "coordinates": "121.476,31.231",
                "price": 160,
                "rating": 4.7,
                "available": True,
            },
        ],
    )
    _write_json(
        tmp_path / "activities.json",
        [
            {
                "poi_id": "act_kid_craft",
                "name": "树屋亲子手作乐园",
                "type": "activity",
                "category": "亲子手作",
                "tags": ["亲子", "儿童友好", "手作"],
                "coordinates": "121.474,31.229",
                "price": 120,
                "rating": 4.5,
                "available": True,
            }
        ],
    )

    monkeypatch.setenv("WF_B_RAG_ENABLED", "1")
    monkeypatch.setenv("WF_B_RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(tmp_path))

    state = {
        "user_input": "周六带孩子玩一整天，中午吃清淡一点，晚上吃本帮菜。",
        "scene_type": "family",
        "constraints": {
            "raw_text": "周六 一整天 孩子 中午 清淡 晚上 本帮菜",
            "city": "上海",
            "people_count": 3,
            "budget": 800,
            "origin_coordinates": "121.475,31.230",
        },
        "user_profile": {},
        "scenario_activities": [],
        "execution_log": [],
    }

    evidence = b_poi_rag_node(state)["b_rag_candidate_evidence"]
    blocks = {block["role"]: block for block in evidence["node_evidence"]}

    assert blocks["restaurant_lunch"]["candidates"][0]["poi_id"] == "res_light_lunch"
    assert blocks["restaurant_dinner"]["candidates"][0]["poi_id"] == "res_benbang_dinner"
