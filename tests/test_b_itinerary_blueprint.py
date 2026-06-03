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
