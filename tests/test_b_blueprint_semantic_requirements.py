from src.nodes.b_itinerary_blueprint import build_b_itinerary_blueprint
from src.nodes.candidate_generator import (
    _explicit_activity_requirements,
    _matches_lodging_identity,
    _matches_strict_node_role,
    _rank_pool_for_node_intent,
)


def test_activity_requirements_include_blueprint_search_terms():
    constraints = {
        "planning_preferences": {"activity_type": ["多人活动"]},
        "b_itinerary_blueprint": {
            "node_intents": [
                {
                    "role": "board_game_escape",
                    "supply_domain": "activity",
                    "search_terms": ["桌游", "密室"],
                },
                {
                    "role": "restaurant_specific",
                    "supply_domain": "restaurant",
                    "search_terms": ["便宜好吃"],
                },
            ]
        },
    }

    requirements = _explicit_activity_requirements(constraints)

    assert {"桌游", "密室"}.intersection(requirements)


def test_lodging_identity_rejects_restaurant_named_mixed_gaode_poi():
    mixed_restaurant = {
        "type": "hotel",
        "name": "新广开·海鲜鲁菜·非遗渔家宴",
        "gaode_type": "餐饮服务;中餐厅;综合酒楼|住宿服务;宾馆酒店;宾馆酒店",
    }
    hotel = {
        "type": "hotel",
        "name": "青岛广业锦江大酒店",
        "gaode_type": "住宿服务;宾馆酒店;四星级宾馆",
    }

    assert _matches_lodging_identity(mixed_restaurant) is False
    assert _matches_lodging_identity(hotel) is True


def test_karaoke_role_rejects_food_heavy_misclassified_gaode_poi():
    mixed_food_ktv = {
        "type": "activity",
        "name": "皇朝尊会(粤菜)",
        "gaode_type": "体育休闲服务;娱乐场所;KTV",
        "raw": {
            "biz_ext": {
                "tag": "皇朝虾饺,雪山包,凤凰流沙包,榴莲酥,烧鹅拼叉烧,双人餐",
                "rectag": "广式烧腊",
                "keytag": "粤菜",
            }
        },
    }
    real_ktv = {
        "type": "activity",
        "name": "星聚会KTV",
        "gaode_type": "体育休闲服务;娱乐场所;KTV",
    }

    assert _matches_strict_node_role(mixed_food_ktv, "karaoke") is False
    assert _matches_strict_node_role(real_ktv, "karaoke") is True


def test_blueprint_preserves_family_handcraft_terms():
    message = "周六想带孩子玩一整天，上午找个亲子手作，中午吃清淡一点，下午去室内乐园，晚上吃本帮菜。"
    blueprint = build_b_itinerary_blueprint(
        {
            "user_input": message,
            "raw_user_input": message,
            "constraints": {
                "raw_text": "周六 一整天 孩子 亲子手作 中午 清淡 下午 室内乐园 晚上 本帮菜"
            },
        }
    )

    first_terms = set(blueprint["node_intents"][0]["search_terms"])

    assert "亲子手作" in first_terms or "手作" in first_terms or "陶艺" in first_terms


def test_blueprint_keeps_meal_specific_terms_in_the_right_daypart():
    message = (
        "\u5468\u516d\u60f3\u5e26\u5b69\u5b50\u73a9\u4e00\u6574\u5929\uff0c"
        "\u4e0a\u5348\u627e\u4e2a\u4eb2\u5b50\u624b\u4f5c\uff0c"
        "\u4e2d\u5348\u5403\u6e05\u6de1\u4e00\u70b9\uff0c"
        "\u4e0b\u5348\u53bb\u5ba4\u5185\u4e50\u56ed\uff0c"
        "\u665a\u4e0a\u5403\u672c\u5e2e\u83dc\u3002"
    )
    blueprint = build_b_itinerary_blueprint(
        {
            "user_input": message,
            "raw_user_input": message,
            "scene_type": "family",
            "constraints": {"raw_text": message, "city": "\u4e0a\u6d77"},
        },
        constraints={"raw_text": message, "city": "\u4e0a\u6d77"},
    )
    by_role = {
        item["role"]: set(item.get("search_terms", []))
        for item in blueprint["node_intents"]
    }

    assert "\u6e05\u6de1" in by_role["restaurant_lunch"]
    assert "\u672c\u5e2e\u83dc" not in by_role["restaurant_lunch"]
    assert "\u672c\u5e2e\u83dc" in by_role["restaurant_dinner"]


def test_blueprint_does_not_turn_negated_food_into_restaurant_node():
    message = (
        "\u5468\u672b\u4e0b\u5348\u548c\u670b\u53cb\u60f3\u5148\u73a9"
        "\u684c\u6e38\u6216\u5bc6\u5ba4\uff0c\u665a\u4e0a\u518d\u627e"
        "\u4e2aKTV\u5531\u6b4c\uff0c\u4e0d\u8981\u706b\u9505\u70e7\u70e4\u3002"
    )
    blueprint = build_b_itinerary_blueprint(
        {
            "user_input": message,
            "raw_user_input": message,
            "scene_type": "friends",
            "constraints": {"raw_text": message, "city": "\u4e0a\u6d77"},
        },
        constraints={"raw_text": message, "city": "\u4e0a\u6d77"},
    )
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "board_game_escape" in roles
    assert "karaoke" in roles
    assert "restaurant_specific" not in roles


def test_family_handcraft_intent_prefers_handcraft_candidate():
    intent = {
        "node_id": "intent_01",
        "role": "family_activity",
        "supply_domain": "activity",
        "search_terms": ["孩子", "亲子手作", "陶艺"],
    }
    generic_playground = {
        "poi_id": "act_playground",
        "name": "综合游乐场",
        "type": "activity",
        "tags": ["儿童友好", "低强度"],
        "rating": 4.6,
        "available": True,
        "queue_time_min": 8,
    }
    handcraft = {
        "poi_id": "act_handcraft",
        "name": "如一工社亲子陶艺DIY",
        "type": "activity",
        "category": "手作体验",
        "sub_category": "创意手作",
        "experience_type": "手作工作坊",
        "tags": ["室内", "动手体验"],
        "rating": 4.2,
        "available": True,
        "queue_time_min": 8,
    }

    ranked = _rank_pool_for_node_intent(intent, [generic_playground, handcraft], [])

    assert ranked[0]["poi_id"] == "act_handcraft"
