import json

from src.nodes.b_poi_rag import (
    _fast_role_prefilter_pool,
    _forbidden_restaurant_groups_for_retrieval,
    _generic_poi_matches_domain,
    _inferred_location_anchor_terms,
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


def test_rag_negative_retrieval_excludes_heavy_food_for_current_low_calorie_need():
    groups = _forbidden_restaurant_groups_for_retrieval(
        {"user_input": "\u8001\u5a46\u6700\u8fd1\u5728\u51cf\u8102\uff0c\u60f3\u5403\u8f7b\u98df"},
        {"mom_diet": "low_calorie", "soft_tags": ["\u4f4e\u5361", "\u8f7b\u98df"]},
    )

    assert {"\u70e4\u8089", "\u706b\u9505", "\u70b8\u9e21\u5c0f\u5403"}.issubset(groups)


def test_rag_negative_retrieval_does_not_forbid_explicit_hotpot_request():
    groups = _forbidden_restaurant_groups_for_retrieval(
        {"user_input": "\u5468\u672b\u60f3\u548c\u670b\u53cb\u53bb\u9759\u5b89\u533a\u5403\u706b\u9505"},
        {"hard_tags": ["\u706b\u9505"], "soft_tags": ["\u805a\u9910"]},
    )

    assert "\u706b\u9505" not in groups


def test_rag_negative_retrieval_keeps_negated_bbq_when_hotpot_requested():
    groups = _forbidden_restaurant_groups_for_retrieval(
        {
            "user_input": "\u4eca\u665a\u670b\u53cb\u805a\u9910\u60f3\u5403\u706b\u9505\uff0c\u522b\u63a8\u8350\u70e4\u8089",
            "user_profile": {"food_preference": ["\u706b\u9505"], "avoid": ["\u70e4\u8089", "\u70e7\u70e4"]},
        },
        {"planning_preferences": {"food_type": ["\u706b\u9505"]}},
    )

    assert "\u706b\u9505" not in groups
    assert "\u70e4\u8089" in groups


def test_fast_prefilter_supports_explicit_restaurant_cuisine():
    bundle = {
        "root": "",
        "domain_items": {
            "restaurant": [
                {
                    "poi_id": "res_hotpot",
                    "name": "\u9759\u5b89\u706b\u9505",
                    "type": "restaurant",
                    "supply_domain": "restaurant",
                    "restaurant_category": "\u706b\u9505",
                    "tags": ["\u706b\u9505", "hotpot"],
                    "rating": 4.4,
                },
                {
                    "poi_id": "res_local",
                    "name": "\u672c\u5e2e\u5c0f\u9986",
                    "type": "restaurant",
                    "supply_domain": "restaurant",
                    "restaurant_category": "\u672c\u5e2e\u83dc",
                    "tags": ["\u672c\u5e2e\u83dc"],
                    "rating": 4.9,
                },
            ]
        },
    }

    pool, meta = _fast_role_prefilter_pool(
        bundle=bundle,
        domain="restaurant",
        role="restaurant_specific",
        node_terms=["\u706b\u9505", "hotpot"],
        global_terms=[],
        constraints={},
        limit=10,
    )

    assert meta["retriever"] == "local_poi_fast_role_prefilter_v1"
    assert [item["poi_id"] for item in pool] == ["res_hotpot"]


def test_local_service_generic_poi_prefilter_for_dental():
    bundle = {
        "root": "",
        "domain_items": {
            "local_service": [
                {
                    "poi_id": "svc_dental",
                    "name": "\u9759\u5b89\u53e3\u8154\u7259\u79d1\u8bca\u6240",
                    "type": "local_service",
                    "supply_domain": "local_service",
                    "category": "\u533b\u7597\u4fdd\u5065\u670d\u52a1",
                    "gaode_type": "\u533b\u7597\u4fdd\u5065\u670d\u52a1;\u4e13\u79d1\u533b\u9662;\u53e3\u8154\u533b\u9662",
                    "rating": 4.6,
                },
                {
                    "poi_id": "svc_travel",
                    "name": "\u9759\u5b89\u65c5\u884c\u793e",
                    "type": "local_service",
                    "supply_domain": "local_service",
                    "category": "\u751f\u6d3b\u670d\u52a1",
                    "gaode_type": "\u751f\u6d3b\u670d\u52a1;\u65c5\u884c\u793e;\u65c5\u884c\u793e",
                    "rating": 4.8,
                },
            ]
        },
    }

    pool, meta = _fast_role_prefilter_pool(
        bundle=bundle,
        domain="local_service",
        role="dental_clinic",
        node_terms=["\u7259\u79d1", "\u53e3\u8154", "\u6d17\u7259"],
        global_terms=[],
        constraints={},
        limit=10,
    )

    assert meta["retriever"] == "local_poi_fast_role_prefilter_v1"
    assert [item["poi_id"] for item in pool] == ["svc_dental"]


def test_local_service_prefilter_ignores_address_only_dental_match():
    bundle = {
        "root": "",
        "domain_items": {
            "local_service": [
                {
                    "poi_id": "svc_chess_above_dental",
                    "name": "\u4e0a\u8fb0\u68cb\u724c\u8336\u820d",
                    "type": "local_service",
                    "supply_domain": "local_service",
                    "category": "\u5eb7\u517b\u653e\u677e",
                    "gaode_type": "\u4f53\u80b2\u4f11\u95f2\u670d\u52a1;\u5a31\u4e50\u573a\u6240;\u68cb\u724c\u5ba4",
                    "location": "\u8001\u6caa\u592a\u8def29\u53f72\u5c42(\u7b11\u5965\u53e3\u8154\u697c\u4e0a)",
                    "raw": {
                        "address": "\u8001\u6caa\u592a\u8def29\u53f72\u5c42(\u7b11\u5965\u53e3\u8154\u697c\u4e0a)",
                    },
                }
            ]
        },
    }

    pool, meta = _fast_role_prefilter_pool(
        bundle=bundle,
        domain="local_service",
        role="dental_clinic",
        node_terms=["\u7259\u79d1", "\u53e3\u8154", "\u6d17\u7259"],
        global_terms=[],
        constraints={},
        limit=10,
    )

    assert meta["retriever"] == "local_poi_fast_role_prefilter_v1"
    assert pool == []


def test_sports_training_prefilter_prefers_training_over_football_club():
    bundle = {
        "root": "",
        "domain_items": {
            "local_service": [
                {
                    "poi_id": "svc_club",
                    "name": "\u4e0a\u6d77\u7533\u82b1\u8db3\u7403\u4ff1\u4e50\u90e8",
                    "type": "local_service",
                    "supply_domain": "local_service",
                    "gaode_type": "\u4f53\u80b2\u4f11\u95f2\u670d\u52a1;\u8fd0\u52a8\u573a\u9986;\u8db3\u7403\u573a",
                    "rating": 4.8,
                },
                {
                    "poi_id": "svc_training",
                    "name": "\u5c0f\u8d5b\u864e\u5c11\u513f\u8db3\u7403\u57f9\u8bad(\u957f\u98ce\u6821\u533a)",
                    "type": "local_service",
                    "supply_domain": "local_service",
                    "gaode_type": "\u79d1\u6559\u6587\u5316\u670d\u52a1;\u57f9\u8bad\u673a\u6784;\u57f9\u8bad\u673a\u6784",
                    "rating": 4.6,
                },
            ]
        },
    }

    pool, meta = _fast_role_prefilter_pool(
        bundle=bundle,
        domain="local_service",
        role="sports_training",
        node_terms=["\u8db3\u7403", "\u8db3\u7403\u57f9\u8bad", "\u57f9\u8bad\u73ed"],
        global_terms=[],
        constraints={},
        limit=10,
    )

    assert meta["retriever"] == "local_poi_fast_role_prefilter_v1"
    assert [item["poi_id"] for item in pool] == ["svc_training"]


def test_rag_inferrs_yuyuan_anchor_for_shanghai_hanfu_teahouse_request():
    anchors = _inferred_location_anchor_terms(
        {
            "user_input": (
                "\u6211\u60f3\u4f53\u9a8c\u4e00\u6574\u5929\u53e4\u98ce\u4e3b\u9898\uff0c"
                "\u4e0a\u5348\u6c49\u670d\u62cd\u7167\uff0c\u4e0b\u5348\u8336\u9986\u559d\u8336"
            )
        },
        {"city": "\u4e0a\u6d77"},
    )

    assert "\u8c6b\u56ed" in anchors
    assert "\u57ce\u968d\u5e99" in anchors


def test_rag_inferrs_event_and_district_anchors_for_benchmark_like_requests():
    tea_anchors = _inferred_location_anchor_terms(
        {
            "user_input": (
                "\u6211\u60f3\u53c2\u52a02025\u5e74\u7b2c30\u5c4a"
                "\u4e0a\u6d77\u56fd\u9645\u8336\u6587\u5316\u65c5\u6e38\u8282\uff0c"
                "\u5fd9\u5b8c\u60f3\u5728\u6d3b\u52a8\u4e3e\u529e\u533a\u4e70\u676f\u5496\u5561"
            )
        },
        {"city": "\u4e0a\u6d77"},
    )
    child_theatre_anchors = _inferred_location_anchor_terms(
        {
            "user_input": (
                "\u6211\u60f3\u53c2\u52a02025\u4e0a\u6d77\u56fd\u9645\u513f\u7ae5"
                "\u620f\u5267\u827a\u672f\u8282\uff0c\u53c2\u52a0\u5b8c\u60f3\u5728\u6d3b\u52a8\u4e3e\u529e\u533a\u8fc7\u591c"
            )
        },
        {"city": "\u4e0a\u6d77"},
    )
    fudan_anchors = _inferred_location_anchor_terms(
        {"user_input": "\u6211\u5728\u590d\u65e6\u9644\u8fd1\u60f3\u627e\u65e9\u9910\u548c\u4fbf\u5229\u5e97"},
        {"city": "\u4e0a\u6d77"},
    )
    egypt_anchors = _inferred_location_anchor_terms(
        {"user_input": "\u6211\u60f3\u53c2\u52a0\u4e0a\u6d77\u201c\u53e4\u57c3\u53ca\u6587\u660e\u5927\u5c55\u201d"},
        {"city": "\u4e0a\u6d77"},
    )
    zikawei_anchors = _inferred_location_anchor_terms(
        {
            "user_input": (
                "\u6211\u60f3\u53c2\u52a0\u201c\u6d77\u6d3eZIKAWEI\u00b7\u590f\u65e5\u5706\u821e\u66f2\u201d"
                "\u6d3b\u52a8\uff0c\u7136\u540e\u5728\u6d3b\u52a8\u4e3e\u529e\u533a\u627e\u9910\u5385"
            )
        },
        {"city": "\u4e0a\u6d77"},
    )

    assert {"\u9759\u5b89", "\u5927\u7530\u8def"}.issubset(set(tea_anchors))
    assert {"\u9759\u5b89", "\u5357\u4eac\u897f\u8def"}.issubset(set(child_theatre_anchors))
    assert {"\u590d\u65e6", "\u4e94\u89d2\u573a", "\u6768\u6d66"}.issubset(set(fudan_anchors))
    assert {"\u4e0a\u6d77\u535a\u7269\u9986", "\u4eba\u6c11\u5e7f\u573a"}.issubset(set(egypt_anchors))
    assert {"\u5f90\u5bb6\u6c47", "\u5f90\u6c47\u6ee8\u6c5f"}.issubset(set(zikawei_anchors))


def test_rag_strictly_retrieves_cultural_photo_and_teahouse(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "activities.json",
        [
            {
                "poi_id": "act_spa",
                "name": "\u6cf0\u5b81\u517b\u751f",
                "type": "activity",
                "category": "SPA\u6309\u6469",
                "coordinates": "121.470,31.220",
                "price": 168,
                "rating": 4.8,
                "available": True,
            },
            {
                "poi_id": "act_hanfu",
                "name": "\u8001\u4e0a\u6d77\u65d7\u888d\u6c49\u670d",
                "type": "activity",
                "category": "\u6c49\u670d\u62cd\u7167",
                "gaode_type": "\u8d2d\u7269\u670d\u52a1;\u670d\u88c5\u978b\u5e3d\u76ae\u5177\u5e97",
                "coordinates": "121.471,31.221",
                "price": 80,
                "rating": 4.5,
                "available": True,
            },
        ],
    )
    _write_json(
        tmp_path / "restaurants.json",
        [
            {
                "poi_id": "res_hotpot",
                "name": "\u6d77\u5e95\u635e\u706b\u9505",
                "type": "restaurant",
                "category": "\u706b\u9505",
                "coordinates": "121.472,31.222",
                "price": 120,
                "rating": 4.8,
                "available": True,
            },
            {
                "poi_id": "res_tea",
                "name": "\u8ff9\u53d9\u8336\u7a7a\u95f4",
                "type": "restaurant",
                "category": "\u8336\u9986",
                "coordinates": "121.473,31.223",
                "price": 88,
                "rating": 4.7,
                "available": True,
            },
            {
                "poi_id": "res_dinner",
                "name": "\u672c\u5e2e\u83dc\u665a\u9910\u9986",
                "type": "restaurant",
                "category": "\u672c\u5e2e\u83dc",
                "coordinates": "121.474,31.224",
                "price": 128,
                "rating": 4.6,
                "available": True,
            },
        ],
    )

    monkeypatch.setenv("WF_B_RAG_ENABLED", "1")
    monkeypatch.setenv("WF_B_RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(tmp_path))

    state = {
        "user_input": (
            "\u4f53\u9a8c\u6c49\u670d\u62cd\u7167\uff0c"
            "\u7136\u540e\u53bb\u9644\u8fd1\u4f53\u9a8c\u8336\u827a\u4f11\u606f\u4e00\u4e0b\uff0c"
            "\u518d\u627e\u4e2a\u5730\u65b9\u5403\u665a\u996d"
        ),
        "scene_type": "solo",
        "constraints": {
            "raw_text": "\u6c49\u670d\u62cd\u7167 \u8336\u827a \u665a\u996d",
            "city": "\u4e0a\u6d77",
            "budget": 500,
            "max_distance_km": 10,
            "origin_coordinates": "121.470,31.220",
        },
        "scenario_activities": [],
        "execution_log": [],
    }

    state.update(b_poi_rag_node(state))

    evidence_by_role = {
        block["role"]: block["candidates"]
        for block in state["b_rag_candidate_evidence"]["node_evidence"]
    }
    assert evidence_by_role["cultural_photo"][0]["poi_id"] == "act_hanfu"
    assert evidence_by_role["tea_house"][0]["poi_id"] == "res_tea"


def test_rag_keeps_role_match_when_location_anchor_pool_is_too_generic(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "restaurants.json",
        [
            {
                "poi_id": "res_fudan_hotpot",
                "name": "复旦附近火锅",
                "type": "restaurant",
                "restaurant_category": "火锅",
                "tags": ["复旦", "五角场"],
                "raw": {
                    "name": "复旦附近火锅",
                    "address": "复旦大学旁",
                    "biz_ext": {"tag": "毛肚,牛肉,火锅", "cost": "120.00"},
                },
                "coordinates": "121.500,31.300",
                "price": 120,
                "rating": 4.7,
                "available": True,
            },
            {
                "poi_id": "res_baozi",
                "name": "城市包子早餐",
                "type": "restaurant",
                "restaurant_category": "早餐",
                "tags": ["早餐", "包子"],
                "raw": {
                    "name": "城市包子早餐",
                    "biz_ext": {"tag": "鲜肉包子,小馄饨,豆浆", "cost": "9.00"},
                },
                "coordinates": "121.470,31.220",
                "price": 90,
                "rating": 4.3,
                "available": True,
            },
        ],
    )
    monkeypatch.setenv("WF_B_RAG_ENABLED", "1")
    monkeypatch.setenv("WF_B_RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(tmp_path))

    result = b_poi_rag_node(
        {
            "user_input": "我在复旦附近找10元以内能吃饱的包子早餐",
            "scene_type": "low_budget",
            "constraints": {
                "raw_text": "我在复旦附近找10元以内能吃饱的包子早餐",
                "city": "上海",
                "scene": "low_budget",
                "b_itinerary_blueprint": {
                    "version": "b_itinerary_blueprint_v1",
                    "template_mode": "multi_node",
                    "requires_rag": True,
                    "node_intents": [
                        {
                            "node_id": "intent_01",
                            "role": "restaurant_breakfast",
                            "label": "早餐",
                            "supply_domain": "restaurant",
                            "search_terms": ["早餐", "包子"],
                        }
                    ],
                    "unsupported_roles": [],
                },
            },
            "user_profile": {},
            "execution_log": [],
        }
    )

    block = result["b_rag_candidate_evidence"]["node_evidence"][0]
    assert block["candidates"][0]["poi_id"] == "res_baozi"
    assert block["candidates"][0]["price"] == 9.0
    assert block["retrieval_meta"].get("location_anchor_pool_skipped") in {
        "role_terms_too_sparse",
        "too_small",
    }


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


def test_rag_retrieves_park_scenic_walk_as_activity(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "activities.json",
        [
            {
                "poi_id": "act_riverside_park",
                "name": "\u5f90\u6c47\u6ee8\u6c5f\u7eff\u5730\u516c\u56ed",
                "type": "activity",
                "category": "\u57ce\u5e02\u516c\u56ed",
                "primary_category": "\u516c\u56ed\u591c\u666f",
                "tags": ["\u516c\u56ed", "\u6ee8\u6c5f", "\u591c\u666f", "\u6563\u6b65"],
                "coordinates": "121.459,31.185",
                "rating": 4.8,
                "available": True,
            },
            {
                "poi_id": "act_spa",
                "name": "\u5f90\u6c47\u8db3\u7597\u6309\u6469",
                "type": "activity",
                "category": "\u8db3\u7597",
                "tags": ["\u6309\u6469", "\u4f11\u606f"],
                "coordinates": "121.460,31.186",
                "rating": 4.7,
                "available": True,
            },
        ],
    )
    _write_json(
        tmp_path / "restaurants.json",
        [
            {
                "poi_id": "res_cafe",
                "name": "\u6ee8\u6c5f\u5496\u5561\u9986",
                "type": "restaurant",
                "restaurant_category": "\u5496\u5561\u751c\u54c1",
                "tags": ["\u5496\u5561", "\u4e0b\u5348\u8336"],
                "coordinates": "121.458,31.186",
                "rating": 4.6,
                "available": True,
            }
        ],
    )

    monkeypatch.setenv("WF_B_RAG_ENABLED", "1")
    monkeypatch.setenv("WF_B_RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WF_MOCK_DATA_DIR", str(tmp_path))

    state = {
        "user_input": (
            "\u665a\u4e0a\u60f3\u5728\u5f90\u6c47\u6ee8\u6c5f\u627e\u4e2a\u516c\u56ed\u6563\u6b65\u770b\u591c\u666f\uff0c"
            "\u518d\u627e\u4e2a\u5496\u5561\u9986\u5750\u5750\u3002"
        ),
        "scene_type": "solo",
        "constraints": {
            "raw_text": "\u5f90\u6c47\u6ee8\u6c5f \u516c\u56ed \u6563\u6b65 \u591c\u666f \u5496\u5561\u9986",
            "city": "\u4e0a\u6d77",
            "origin_coordinates": "121.455,31.184",
        },
        "user_profile": {},
        "scenario_activities": [],
        "execution_log": [],
    }

    evidence = b_poi_rag_node(state)["b_rag_candidate_evidence"]
    park_block = next(block for block in evidence["node_evidence"] if block["role"] == "park_scenic_walk")

    assert park_block["coverage_status"] == "covered"
    assert park_block["candidates"][0]["poi_id"] == "act_riverside_park"


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
    assert evidence["retrieval_trace"][0]["retriever"] == "local_poi_memory_bm25_v1+structured_rerank"
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
