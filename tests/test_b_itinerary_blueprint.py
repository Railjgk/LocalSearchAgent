from src.nodes.b_itinerary_blueprint import build_b_itinerary_blueprint


def test_detects_multinode_exhibition_lunch_souvenir():
    state = {
        "user_input": "我想参加上海“红翠斗芳菲：宋元明漆器珍品展”，然后在附近吃个中饭，还想买点上海特产带回去。",
        "scene_type": "solo",
        "constraints": {"raw_text": "看展 吃中饭 买特产"},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])

    roles = [item["role"] for item in blueprint["node_intents"]]
    assert blueprint["template_mode"] == "multi_node"
    assert blueprint["requires_rag"] is True
    assert "exhibition" in roles
    assert "restaurant_lunch" in roles
    assert "souvenir_shopping" in roles
    assert "souvenir_shopping" in blueprint["unsupported_roles"]
    assert blueprint["named_entities"] == ["红翠斗芳菲：宋元明漆器珍品展"]


def test_detects_overnight_lodging_restaurant_convenience_parking():
    state = {
        "user_input": "我们一家四口想在外滩附近找个有厨房的住处，晚上吃蟹黄面，再找便利店买日用品，最后找停车场。",
        "scene_type": "family",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})

    roles = [item["role"] for item in blueprint["node_intents"]]
    assert blueprint["planning_horizon"] == "overnight"
    assert blueprint["planning_days"] == 2
    assert blueprint["template_mode"] == "multi_node"
    assert roles[:2] == ["lodging", "family_activity"] or "lodging" in roles
    assert "restaurant_dinner" in roles
    assert "convenience_store" in roles
    assert "parking" in roles
    assert set(["lodging", "convenience_store", "parking"]).issubset(
        set(blueprint["unsupported_roles"])
    )
    assert blueprint["time_skeleton"]["days"][0]["slots"]
    assert blueprint["time_skeleton"]["days"][1]["day"] == 2


def test_legacy_pair_for_simple_family_activity_meal():
    state = {
        "user_input": "今天下午带孩子找个亲子活动，再吃个轻食。",
        "scene_type": "family",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})

    assert blueprint["template_mode"] == "legacy_pair"
    assert blueprint["planning_horizon"] == "half_day"
    assert blueprint["planning_days"] == 1
    assert blueprint["unsupported_roles"] == []


def test_two_restaurant_nodes_use_multinode_not_legacy_pair():
    state = {
        "user_input": (
            "\u5728\u9759\u5b89\u5bfa\u8bf7\u516c\u53f8\u524d\u8f88\u7b80\u5355\u5403\u4e2a\u9c81\u83dc\uff0c"
            "\u518d\u627e\u4e2a\u5496\u5561\u5385\u5750\u5750\u3002"
        ),
        "scene_type": "friends",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert blueprint["template_mode"] == "multi_node"
    assert "restaurant_specific" in roles
    assert "cafe" in roles


def test_detects_two_day_horizon():
    state = {
        "user_input": "周末两天想安排亲子活动、晚餐，第二天上午再逛个博物馆。",
        "scene_type": "family",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})

    assert blueprint["planning_horizon"] == "two_day"
    assert blueprint["planning_days"] == 2
    assert len(blueprint["time_skeleton"]["days"]) == 2


def test_full_day_family_keeps_morning_and_afternoon_activity_nodes():
    state = {
        "user_input": "周六想带孩子玩一整天，上午找个亲子手作，中午吃清淡一点，下午去室内乐园，晚上吃本帮菜。",
        "scene_type": "family",
        "constraints": {"raw_text": "周六 一整天 孩子 亲子手作 中午 清淡 下午 室内乐园 晚上 本帮菜"},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert blueprint["template_mode"] == "multi_node"
    assert blueprint["planning_horizon"] == "full_day"
    assert "family_activity" in roles
    assert "family_indoor_play" in roles
    assert "restaurant_lunch" in roles
    assert "restaurant_dinner" in roles


def test_detects_localsearchbench_retail_and_wellness_roles():
    state = {
        "user_input": "我在静安寺附近，想买美妆日化伴手礼，配一束鲜花，找个地方捏脚休息，再去咖啡厅坐坐，最后去便利店买零食。",
        "scene_type": "solo",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})

    roles = [item["role"] for item in blueprint["node_intents"]]
    assert "beauty_cosmetics" in roles
    assert "souvenir_shopping" in roles
    assert "flower_shop" in roles
    assert "wellness_massage" in roles
    assert "cafe" in roles
    assert "convenience_store" in roles
    assert set(["beauty_cosmetics", "flower_shop"]).issubset(
        set(blueprint["unsupported_roles"])
    )
    assert "wellness_massage" not in blueprint["unsupported_roles"]


def test_detects_breakfast_as_restaurant_node():
    state = {
        "user_input": "民宿在南京路附近，早上想吃包子，再找便利店买点东西。",
        "scene_type": "solo",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "lodging" in roles
    assert "restaurant_breakfast" in roles
    assert "convenience_store" in roles


def test_sparse_review_does_not_trigger_spa_role():
    state = {
        "user_input": "Family outing needs trustworthy merchants with enough review evidence; avoid new or sparse-review places.",
        "scene_type": "family",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "wellness_massage" not in roles
    assert blueprint["template_mode"] == "legacy_pair"


def test_detects_friends_entertainment_chain():
    state = {
        "user_input": "今晚几个朋友想先玩桌游或者密室，再吃火锅，饭后如果不累可以唱个KTV。",
        "scene_type": "friends",
        "constraints": {"raw_text": "朋友 桌游 密室 火锅 KTV 唱歌"},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert blueprint["template_mode"] == "multi_node"
    assert "board_game_escape" in roles
    assert "restaurant_specific" in roles
    assert "karaoke" in roles


def test_detects_cultural_photo_tea_and_dinner_chain():
    state = {
        "user_input": (
            "\u60f3\u627e\u4e2a\u8fd1\u4e00\u70b9\u7684\u5730\u65b9\u4f53\u9a8c\u6c49\u670d\u62cd\u7167\uff0c"
            "\u62cd\u5b8c\u540e\u60f3\u53bb\u9644\u8fd1\u4f53\u9a8c\u8336\u827a\u4f11\u606f\u4e00\u4e0b\uff0c"
            "\u7136\u540e\u518d\u627e\u4e2a\u6709\u7279\u8272\u7684\u5730\u65b9\u5403\u665a\u996d\u3002"
        ),
        "scene_type": "solo",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert blueprint["template_mode"] == "multi_node"
    assert "cultural_photo" in roles
    assert "tea_house" in roles
    assert "restaurant_dinner" in roles
