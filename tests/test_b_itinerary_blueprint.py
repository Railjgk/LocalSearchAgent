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


def test_shanghai_xiaolongbao_adds_restaurant_node_after_cultural_photo():
    state = {
        "user_input": (
            "\u6211\u60f3\u5b89\u6392\u4e2a\u4e0b\u5348\u7684\u6c49\u670d\u4f53\u9a8c\u8def\u7ebf\uff0c"
            "\u5148\u5316\u5986\u62cd\u7167\uff0c\u8fd8\u60f3\u627e\u4e2a\u6444\u5f71\u5e08\u5e2e\u5fd9\u62cd\u7167\u7559\u5ff5\uff0c"
            "\u6700\u540e\u53bb\u5403\u4e0a\u6d77\u5c0f\u7b3c\u5305\u3002"
        ),
        "scene_type": "solo",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "cultural_photo" in roles
    assert "restaurant_specific" in roles


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


def test_two_day_lodging_stays_on_day_one_and_following_nodes_move_to_day_two():
    state = {
        "user_input": (
            "\u5468\u672b\u4e24\u5929\u5e26\u5b69\u5b50\u5728\u4e0a\u6d77\u8f7b\u677e\u73a9\u4e00\u4e0b\uff0c"
            "\u7b2c\u4e00\u5929\u4e0b\u5348\u4eb2\u5b50\u6d3b\u52a8\uff0c\u665a\u4e0a\u5403\u996d\uff0c"
            "\u4f4f\u4e00\u665a\uff0c\u7b2c\u4e8c\u5929\u4e0a\u5348\u901b\u535a\u7269\u9986\u518d\u5403\u5348\u996d\u3002"
        ),
        "scene_type": "family",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})

    assert blueprint["planning_horizon"] == "two_day"
    slots = [
        slot
        for day in blueprint["time_skeleton"]["days"]
        for slot in day["slots"]
    ]
    by_role = {slot["role"]: slot for slot in slots}
    assert by_role["lodging"]["day"] == 1
    assert by_role["lodging"]["start_time"] == "20:30"
    assert by_role["exhibition"]["day"] == 2
    assert by_role["restaurant_lunch"]["day"] == 2


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


def test_detects_service_lookup_without_default_dinner():
    state = {
        "user_input": "\u6700\u8fd1\u7259\u75bc\uff0c\u60f3\u627e\u4e2a\u9760\u8c31\u7684\u7259\u79d1\u8bca\u6240\u770b\u7259\uff0c\u6bd4\u8f83\u4e00\u4e0b\u6d17\u7259\u548c\u8865\u7259\u4ef7\u683c",
        "scene_type": "solo",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert roles == ["dental_clinic"]
    assert blueprint["template_mode"] == "multi_node"
    assert blueprint["requires_rag"] is True
    assert "dental_clinic" in blueprint["unsupported_roles"]


def test_detects_pet_service_chain_roles():
    state = {
        "user_input": (
            "\u60f3\u5e26\u91d1\u6bdb\u53bb\u5ba0\u7269\u7f8e\u5bb9\u5e97\u505a\u4e2aspa\uff0c"
            "\u6d17\u5b8c\u540e\u53bb\u5ba0\u7269\u53cb\u597d\u7684\u5496\u5561\u9986\u4f11\u606f\uff0c"
            "\u518d\u53bb\u5ba0\u7269\u533b\u9662\u505a\u4e2a\u4f53\u68c0\uff0c"
            "\u6700\u540e\u53bb\u5ba0\u7269\u5e97\u4e70\u70b9\u8425\u517b\u54c1\u3002"
        ),
        "scene_type": "solo",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert {"pet_grooming", "pet_cafe", "pet_hospital", "pet_store"}.issubset(set(roles))
    assert "wellness_massage" not in roles
    assert "cafe" not in roles
    assert {"pet_grooming", "pet_hospital", "pet_store"}.issubset(set(blueprint["unsupported_roles"]))


def test_detects_nightlife_show_movie_nail_roles():
    state = {
        "user_input": (
            "\u5148\u770b\u8131\u53e3\u79c0\uff0c\u518d\u505a\u7f8e\u7532\uff0c"
            "\u665a\u4e0a\u5403\u706b\u9505\uff0c\u770b\u7535\u5f71\uff0c\u6700\u540e\u53bb\u9152\u5427\u559d\u4e00\u676f\u3002"
        ),
        "scene_type": "friends",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert {"talk_show", "nail_salon", "restaurant_dinner", "cinema", "bar"}.issubset(set(roles))


def test_music_bar_live_does_not_duplicate_show_or_internet_cafe_roles():
    state = {
        "user_input": (
            "\u5468\u65e5\u60f3\u5728\u6768\u6d66\u5b89\u6392\u4e00\u5929\u6587\u827a\u4e4b\u65c5\uff0c"
            "\u4e0a\u5348\u53bb\u7f8e\u672f\u9986\u770b\u5c55\u89c8\uff0c"
            "\u4e2d\u5348\u53bb\u7f51\u7ea2\u9910\u5385\u5403brunch\uff0c"
            "\u4e0b\u5348\u53bb\u5267\u672c\u6740\u5e97\u73a9\u63a8\u7406\u6e38\u620f\uff0c"
            "\u665a\u4e0a\u53bb\u97f3\u4e50\u9152\u5427\u542clive\u6f14\u51fa\u3002"
        ),
        "scene_type": "solo",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert roles == ["exhibition", "restaurant_lunch", "board_game_escape", "bar"]


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


def test_detects_named_civilization_exhibition_as_activity_node():
    state = {
        "user_input": "我想参加上海“古埃及文明大展”，还想在附近找个咖啡厅，另外也想找个便利店买点零食。",
        "constraints": {
            "raw_text": "古埃及文明大展 咖啡厅 便利店",
            "city": "上海",
        },
    }

    blueprint = build_b_itinerary_blueprint(state)
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "exhibition" in roles
    assert "cafe" in roles
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
    assert "wellness_massage" not in roles


def test_blueprint_ignores_false_family_tags_without_child_context():
    state = {
        "user_input": (
            "\u6211\u5468\u672b\u60f3\u4f53\u9a8c\u4e00\u6574\u5929\u7684\u53e4\u98ce\u4e3b\u9898\u6d3b\u52a8\uff0c"
            "\u4e0a\u5348\u53bb\u6c49\u670d\u9986\u62cd\u7167\uff0c\u4e2d\u5348\u5403\u996d\uff0c"
            "\u4e0b\u5348\u53bb\u8336\u9986\u559d\u8336\u3002"
        ),
        "scene_type": "solo",
        "scenario_activities": ["\u4eb2\u5b50", "\u513f\u7ae5\u53cb\u597d"],
        "constraints": {
            "raw_text": (
                "\u6211\u5468\u672b\u60f3\u4f53\u9a8c\u4e00\u6574\u5929\u7684\u53e4\u98ce\u4e3b\u9898\u6d3b\u52a8\uff0c"
                "\u4e0a\u5348\u53bb\u6c49\u670d\u9986\u62cd\u7167\uff0c\u4e2d\u5348\u5403\u996d\uff0c"
                "\u4e0b\u5348\u53bb\u8336\u9986\u559d\u8336\u3002"
            ),
            "planning_preferences": {"activity_type": ["\u4eb2\u5b50"]},
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "cultural_photo" in roles
    assert "tea_house" in roles
    assert "family_activity" not in roles
    assert "family_indoor_play" not in roles


def test_blueprint_ignores_false_citywalk_tags_without_raw_citywalk_context():
    state = {
        "user_input": (
            "\u6211\u5468\u672b\u60f3\u4f53\u9a8c\u4e00\u6574\u5929\u7684\u53e4\u98ce\u4e3b\u9898\u6d3b\u52a8\uff0c"
            "\u4e0a\u5348\u53bb\u6c49\u670d\u9986\u62cd\u7167\uff0c\u4e2d\u5348\u5403\u996d\uff0c"
            "\u4e0b\u5348\u53bb\u8336\u9986\u559d\u8336\u3002"
        ),
        "scene_type": "solo",
        "scenario_activities": ["Citywalk", "\u672c\u5730\u63a2\u7d22"],
        "constraints": {
            "raw_text": (
                "\u6211\u5468\u672b\u60f3\u4f53\u9a8c\u4e00\u6574\u5929\u7684\u53e4\u98ce\u4e3b\u9898\u6d3b\u52a8\uff0c"
                "\u4e0a\u5348\u53bb\u6c49\u670d\u9986\u62cd\u7167\uff0c\u4e2d\u5348\u5403\u996d\uff0c"
                "\u4e0b\u5348\u53bb\u8336\u9986\u559d\u8336\u3002"
            ),
            "planning_preferences": {"activity_type": ["Citywalk", "\u672c\u5730\u63a2\u7d22"]},
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "cultural_photo" in roles
    assert "tea_house" in roles
    assert "citywalk_market" not in roles


def test_blueprint_keeps_park_night_walk_separate_from_massage_rest():
    state = {
        "user_input": (
            "\u6211\u60f3\u5728\u5f90\u6c47\u533a\u5b89\u6392\u4e00\u6761\u5468\u672b\u4e0b\u5348\u7684citywalk\u8def\u7ebf\uff0c"
            "\u5148\u53bb\u770b\u770b\u5386\u53f2\u5efa\u7b51\uff0c"
            "\u7136\u540e\u627e\u4e2a\u5496\u5561\u9986\u559d\u4e0b\u5348\u8336\u4f11\u606f\u4e00\u4e0b\uff0c"
            "\u6700\u540e\u665a\u4e0a\u53bb\u516c\u56ed\u6563\u6b65\u770b\u591c\u666f\u3002"
        ),
        "scene_type": "solo",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "citywalk_market" in roles
    assert "cafe" in roles
    assert "park_scenic_walk" in roles
    assert "wellness_massage" not in roles


def test_blueprint_treats_child_history_culture_as_culture_not_generic_play():
    state = {
        "user_input": (
            "\u5468\u672b\u5e26\u5b69\u5b50\u53bb\u770b\u770b\u5f90\u6c47\u533a\u5386\u53f2\u6587\u5316\u7684\u5730\u65b9\uff0c"
            "\u6700\u597d\u6709\u4e13\u4e1a\u8bb2\u89e3\uff0c\u518d\u627e\u4e2a\u9910\u5385\u5403\u996d\uff0c"
            "\u53e6\u5916\u627e\u4e2a\u4fbf\u5229\u5e97\u4e70\u96f6\u98df\u996e\u6599\u3002"
        ),
        "scene_type": "family",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "citywalk_market" in roles or "exhibition" in roles
    assert "restaurant_specific" in roles
    assert "convenience_store" in roles
    assert "family_activity" not in roles


def test_blueprint_does_not_treat_business_area_park_name_as_park_walk():
    state = {
        "user_input": (
            "\u6211\u5728\u9f99\u4e4b\u68a6\u9644\u8fd1\u5de5\u4f5c\uff0c"
            "\u4e2d\u5348\u60f3\u5403\u4e2a\u4fbf\u5b9c\u7684\u65e5\u5f0f\u732a\u6392\u5957\u9910\uff0c"
            "\u73af\u7403\u6e2f\u3001\u4e2d\u5c71\u516c\u56ed\u3001\u9759\u5b89\u5bfa\u8fd9\u4e09\u4e2a\u5546\u5708\u54ea\u5bb6\u5e97\u7684\u5de5\u4f5c\u65e5\u7279\u60e0\u6700\u5b9e\u60e0\uff1f"
        ),
        "scene_type": "solo",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "restaurant_lunch" in roles
    assert "park_scenic_walk" not in roles


def test_blueprint_still_detects_explicit_foot_massage_rest():
    state = {
        "user_input": (
            "\u9759\u5b89\u5bfa\u9644\u8fd1\u60f3\u4e70\u4e2a\u4f34\u624b\u793c\uff0c"
            "\u627e\u4e2a\u5730\u65b9\u634f\u811a\u4f11\u606f\u4e00\u4e0b\uff0c"
            "\u518d\u53bb\u5496\u5561\u5385\u5750\u5750\u3002"
        ),
        "scene_type": "solo",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "wellness_massage" in roles
    assert "cafe" in roles


def test_blueprint_drops_generic_restaurant_from_non_meal_recommendation_context():
    state = {
        "user_input": (
            "\u53c2\u52a0\u5b8c\u8336\u6587\u5316\u8282\uff0c"
            "\u60f3\u5728\u4e3e\u529e\u533a\u4e70\u676f\u5496\u5561\u6b47\u4f1a\u513f\uff0c"
            "\u518d\u627e\u4e2a\u4fbf\u5229\u5e97\u4e70\u96f6\u98df\u996e\u6599\uff0c\u6709\u6ca1\u6709\u5408\u9002\u63a8\u8350"
        ),
        "scene_type": "solo",
        "scenario_activities": ["\u9910\u996e", "\u63a8\u8350"],
        "constraints": {
            "raw_text": (
                "\u53c2\u52a0\u5b8c\u8336\u6587\u5316\u8282\uff0c"
                "\u60f3\u5728\u4e3e\u529e\u533a\u4e70\u676f\u5496\u5561\u6b47\u4f1a\u513f\uff0c"
                "\u518d\u627e\u4e2a\u4fbf\u5229\u5e97\u4e70\u96f6\u98df\u996e\u6599"
            ),
            "planning_preferences": {"restaurant_type": ["\u9910\u996e"]},
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "cafe" in roles
    assert "convenience_store" in roles
    assert "restaurant_specific" not in roles


def test_blueprint_extracts_spa_and_fitness_as_separate_supported_nodes():
    blueprint = build_b_itinerary_blueprint(
        {
            "user_input": (
                "\u9759\u5b89\u5bfa\u9644\u8fd1\u627e\u4e2aSPA\u548c\u5065\u8eab\u623f\uff0c"
                "\u7136\u540e\u7ed9\u9886\u5bfc\u627e\u4e2a\u9910\u5385"
            ),
            "constraints": {
                "raw_text": (
                    "\u9759\u5b89\u5bfa SPA \u5065\u8eab\u623f \u9910\u5385"
                )
            },
        }
    )
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "wellness_massage" in roles
    assert "fitness" in roles
    assert "restaurant_specific" in roles


def test_blueprint_merges_generic_restaurant_with_later_dinner_role():
    blueprint = build_b_itinerary_blueprint(
        {
            "user_input": (
                "\u5148\u627e\u4e2a\u9910\u5385\uff0c\u665a\u4e0a\u5403\u987f\u6e05\u6de1\u665a\u9910"
            ),
            "constraints": {
                "raw_text": "\u9910\u5385 \u665a\u4e0a \u665a\u9910",
            },
        }
    )
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert roles.count("restaurant_dinner") == 1
    assert "restaurant_specific" not in roles


def test_blueprint_keeps_specific_food_as_evening_dinner_node():
    state = {
        "user_input": (
            "\u6211\u4eec\u4e00\u5bb6\u56db\u53e3\u60f3\u5728\u5916\u6ee9\u9644\u8fd1"
            "\u627e\u4e2a\u6709\u53a8\u623f\u7684\u4f4f\u5904\uff0c"
            "\u665a\u4e0a\u80fd\u53bb\u54ea\u5403\u87f9\u9ec4\u9762\uff1f"
            "\u8fd8\u60f3\u627e\u4e2a\u4fbf\u5229\u5e97\u4e70\u70b9\u65e5\u7528\u54c1\uff0c"
            "\u6700\u540e\u8fd8\u9700\u8981\u627e\u4e2a\u505c\u8f66\u573a\u505c\u8f66\u3002"
        ),
        "constraints": {
            "raw_text": (
                "\u5916\u6ee9 \u6709\u53a8\u623f \u4f4f\u5904 "
                "\u665a\u4e0a\u80fd\u53bb\u54ea\u5403\u87f9\u9ec4\u9762 "
                "\u4fbf\u5229\u5e97 \u65e5\u7528\u54c1 \u505c\u8f66\u573a"
            )
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "lodging" in roles
    assert "restaurant_dinner" in roles
    assert "convenience_store" in roles
    assert "parking" in roles
    assert "restaurant_specific" not in roles


def test_blueprint_models_huangpu_river_cruise_as_cruise_not_walk():
    state = {
        "user_input": (
            "\u8bf7\u670b\u53cb\u6765\u591c\u6e38\u9ec4\u6d66\u6c5f\uff0c"
            "\u6709\u6ca1\u6709\u63d0\u4f9b\u5305\u53a2\u548c\u81ea\u52a9\u9910\u7684\u6e38\u8239\uff1f"
            "\u8fd8\u60f3\u627e\u4e2a\u505c\u8f66\u573a\u505c\u8f66\u3002"
        ),
        "constraints": {
            "raw_text": "\u591c\u6e38\u9ec4\u6d66\u6c5f \u5305\u53a2 \u81ea\u52a9\u9910 \u6e38\u8239 \u505c\u8f66\u573a",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "river_cruise" in roles
    assert "parking" in roles
    assert "park_scenic_walk" not in roles


def test_blueprint_treats_business_banquet_as_restaurant_need():
    state = {
        "user_input": (
            "\u6211\u8981\u5728\u5f90\u5bb6\u6c47\u8fd9\u8fb9\u5bb4\u8bf7\u91cd\u8981\u5ba2\u6237\uff0c"
            "\u5ba2\u6237\u53ef\u80fd\u9700\u8981\u4f4f\u5bbf\uff0c"
            "\u8fd8\u60f3\u5728\u9644\u8fd1\u627e\u4e2a\u505c\u8f66\u573a\uff0c"
            "\u54ea\u91cc\u6bd4\u8f83\u5408\u9002\uff1f"
        ),
        "constraints": {
            "raw_text": (
                "\u5f90\u5bb6\u6c47 \u5bb4\u8bf7 \u91cd\u8981\u5ba2\u6237 "
                "\u4f4f\u5bbf \u505c\u8f66\u573a"
            ),
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "restaurant_specific" in roles
    assert "lodging" in roles
    assert "parking" in roles


def test_blueprint_treats_xiangsheng_as_performance_need():
    state = {
        "user_input": (
            "\u6211\u4f4f\u5728\u9646\u5bb6\u5634\u9644\u8fd1\uff0c"
            "\u5468\u672b\u60f3\u5b89\u6392\u4e00\u4e2a\u6587\u827a\u4e0b\u5348\uff1a"
            "\u5148\u627e\u4e2a\u5730\u65b9\u770b\u76f8\u58f0\uff0c"
            "\u7136\u540e\u53bb\u9644\u8fd1\u5403\u672c\u5e2e\u83dc\u665a\u9910\uff0c"
            "\u996d\u540e\u60f3\u559d\u5496\u5561\u804a\u5929\uff0c"
            "\u6700\u540e\u80fd\u4e70\u70b9\u4e0a\u6d77\u7279\u4ea7\u5e26\u56de\u5bb6\u3002"
        ),
        "constraints": {
            "raw_text": (
                "\u9646\u5bb6\u5634 \u6587\u827a\u4e0b\u5348 \u76f8\u58f0 "
                "\u672c\u5e2e\u83dc \u665a\u9910 \u5496\u5561 \u4e0a\u6d77\u7279\u4ea7"
            ),
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "talk_show" in roles
    assert "restaurant_dinner" in roles
    assert "cafe" in roles
    assert "souvenir_shopping" in roles


def test_blueprint_keeps_weekend_theatre_and_xiangsheng_as_two_day_plan():
    state = {
        "user_input": (
            "想在浦东新区安排文艺周末。周六晚上7点看话剧，"
            "之后附近吃夜宵。周日上午去文艺咖啡馆，"
            "下午2点看相声，然后买些上海特产伴手礼。"
        ),
        "constraints": {
            "raw_text": (
                "浦东新区 周六 晚上 话剧 夜宵 周日 上午 文艺咖啡馆 "
                "下午2点 相声 上海特产 伴手礼"
            ),
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]
    slot_by_role = {
        slot["role"]: slot
        for day in blueprint["time_skeleton"]["days"]
        for slot in day["slots"]
    }

    assert blueprint["planning_horizon"] == "two_day"
    assert blueprint["planning_days"] == 2
    assert roles == [
        "theatre_performance",
        "restaurant_specific",
        "cafe",
        "talk_show",
        "souvenir_shopping",
    ]
    assert slot_by_role["theatre_performance"]["day"] == 1
    assert slot_by_role["restaurant_specific"]["part_of_day"] == "late_evening"
    assert slot_by_role["cafe"]["day"] == 2
    assert slot_by_role["talk_show"]["start_time"] == "14:00"
    assert slot_by_role["souvenir_shopping"]["day"] == 2


def test_full_day_constraint_expands_simple_request_into_default_itinerary():
    state = {
        "user_input": (
            "\u6211\u4eba\u5728\u4e0a\u6d77\uff0c\u4f46\u662f\u8fd9\u4e2a\u5468\u672b"
            "\u60f3\u5728\u9752\u5c9b\u8f7b\u677e\u73a9\u4e00\u5929\uff0c"
            "\u60f3\u5403\u6d77\u9c9c\uff0c\u522b\u592a\u7d2f"
        ),
        "scene_type": "solo",
        "constraints": {
            "city": "\u9752\u5c9b",
            "time_window": "full_day",
            "duration_range": [6, 10],
            "raw_text": "\u9752\u5c9b \u8f7b\u677e \u4e00\u5929 \u6d77\u9c9c",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert blueprint["planning_horizon"] == "full_day"
    assert blueprint["template_mode"] == "multi_node"
    assert roles == ["citywalk_market", "restaurant_lunch", "park_scenic_walk", "restaurant_dinner"]
    assert "\u6d77\u9c9c" in blueprint["node_intents"][1]["search_terms"]


def test_two_day_constraint_expands_simple_request_into_day_split():
    state = {
        "user_input": "\u5468\u672b\u4e24\u5929\u53bb\u9752\u5c9b\u653e\u677e\uff0c\u60f3\u5403\u6d77\u9c9c",
        "scene_type": "solo",
        "constraints": {
            "city": "\u9752\u5c9b",
            "time_window": "two_day",
            "duration_range": [12, 20],
            "raw_text": "\u9752\u5c9b \u4e24\u5929 \u6d77\u9c9c",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]
    day_slots = blueprint["time_skeleton"]["days"]

    assert blueprint["planning_horizon"] == "two_day"
    assert blueprint["planning_days"] == 2
    assert roles == ["citywalk_market", "restaurant_dinner", "park_scenic_walk", "restaurant_lunch"]
    assert {slot["day"] for day in day_slots for slot in day["slots"]} == {1, 2}


def test_two_day_qingdao_seafood_keeps_restaurant_node_with_lodging_and_family_activity():
    state = {
        "user_input": (
            "\u8fd9\u4e2a\u5468\u672b\u5728\u9752\u5c9b\u4f4f\u4e24\u5929\uff0c"
            "\u60f3\u5403\u672c\u5730\u6d77\u9c9c\uff0c\u901b\u6d77\u8fb9\uff0c"
            "\u5b89\u6392\u4f4f\u5bbf\u548c\u4eb2\u5b50\u6d3b\u52a8\u3002"
        ),
        "scene_type": "solo",
        "constraints": {
            "city": "\u9752\u5c9b",
            "raw_text": (
                "\u9752\u5c9b \u4f4f\u4e24\u5929 \u672c\u5730\u6d77\u9c9c "
                "\u6d77\u8fb9 \u4f4f\u5bbf \u4eb2\u5b50\u6d3b\u52a8"
            ),
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]
    restaurant_terms = {
        term
        for item in blueprint["node_intents"]
        if item["supply_domain"] == "restaurant"
        for term in item.get("search_terms", [])
    }

    assert blueprint["planning_horizon"] == "two_day"
    assert "lodging" in roles
    assert "family_activity" in roles
    assert any(role in roles for role in ("restaurant_specific", "restaurant_dinner"))
    assert "\u6d77\u9c9c" in restaurant_terms


def test_specific_japanese_bbq_terms_enter_restaurant_blueprint():
    state = {
        "user_input": "今晚想吃日式烧肉，最好炭火，环境别太吵，再安排一个轻松的活动。",
        "scene_type": "couple",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    restaurant_intents = [
        item for item in blueprint["node_intents"] if item["supply_domain"] == "restaurant"
    ]

    assert restaurant_intents
    assert "restaurant_specific" in [item["role"] for item in restaurant_intents]
    restaurant_terms = {
        term
        for item in restaurant_intents
        for term in item.get("search_terms", [])
    }
    assert {"日式烧肉", "炭火"}.issubset(restaurant_terms)
