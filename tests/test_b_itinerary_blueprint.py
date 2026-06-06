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


def test_two_day_request_with_home_stay_does_not_add_lodging():
    state = {
        "user_input": (
            "爸妈周五14:30到虹桥，周六17点前还要送回虹桥，"
            "他们住我家不订酒店。想两天带他们轻松看看上海，"
            "最好有一个博物馆或展馆，吃点本帮菜，最后买点伴手礼。"
        ),
        "scene_type": "family",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert blueprint["planning_horizon"] == "two_day"
    assert blueprint["planning_days"] == 2
    assert "lodging" not in roles
    assert "lodging" not in blueprint["unsupported_roles"]
    assert "exhibition" in roles
    assert "souvenir_shopping" in roles


def test_hotel_return_destination_does_not_add_lodging_or_overnight():
    state = {
        "user_input": (
            "今晚临时接待一个外地客户，我18:10从静安寺下班，"
            "客户19:45要到人民广场附近开线上会，21:00前回南京东路酒店。"
            "先找个安静能聊方案的茶室坐半小时，然后吃上海特色晚饭。"
        ),
        "scene_type": "business",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert blueprint["planning_horizon"] != "overnight"
    assert blueprint["planning_days"] == 1
    assert "lodging" not in roles
    assert "lodging" not in blueprint["unsupported_roles"]
    assert "tea_house" in roles
    assert "restaurant_dinner" in roles


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


def test_blueprint_fits_half_day_slots_to_explicit_time_bounds():
    text = (
        "今天16:20以后从杨浦五角场出发，带6岁孩子和外婆找个室内亲子活动，"
        "再吃清淡晚饭，19:30前回到家附近。"
    )
    state = {
        "user_input": text,
        "scene_type": "family",
        "constraints": {
            "raw_text": text,
            "start_time": "16:20",
            "end_time": "19:30",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    slots = blueprint["time_skeleton"]["days"][0]["slots"]

    assert slots
    assert all(slot["start_time"] >= "16:20" for slot in slots)
    assert all(slot["end_time"] <= "19:30" for slot in slots)


def test_blueprint_keeps_bounded_anchor_day_slots_positive_and_sequential():
    text = (
        "周六我陪妈妈过生日，10点在徐家汇洗牙结束后开始，不是要你帮我约牙科，"
        "牙科只是起点。想买一小束花或低糖小蛋糕，再吃个低盐清淡午饭，"
        "下午找个安静茶室歇一会儿看看老建筑，17点前送她到上海南站附近。"
    )
    state = {
        "user_input": text,
        "scene_type": "family",
        "constraints": {
            "raw_text": text,
            "start_time": "10:00",
            "end_time": "17:00",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    slots = blueprint["time_skeleton"]["days"][0]["slots"]
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "dental_clinic" not in roles
    assert slots
    assert all(slot["start_time"] >= "10:00" for slot in slots)
    assert all(slot["end_time"] <= "17:00" for slot in slots)
    assert all(slot["start_time"] < slot["end_time"] for slot in slots)
    assert all(
        left["end_time"] <= right["start_time"]
        for left, right in zip(slots, slots[1:])
    )


def test_blueprint_drops_final_day_slots_that_cannot_fit_before_deadline():
    text = (
        "明天后天陪妈妈在上海待两天，明天上午10点从瑞金医院附近出来，"
        "妈妈刚复查完不能累、也不能吃太油太甜。第一天下午想就近安排一个能坐着休息的文化类活动或茶馆，"
        "晚上早点吃清淡热乎的饭，20点前回酒店休息；第二天上午想顺路买点上海伴手礼带回去，"
        "中午简单吃饭，14:30前到虹桥站。两天总预算1200以内，不用你订医院和高铁；"
        "餐厅或活动能订就订，伴手礼只给可靠购买建议，别说已经买好了。"
    )
    state = {
        "user_input": text,
        "scene_type": "family",
        "constraints": {
            "raw_text": text,
            "start_time": "10:00",
            "end_time": "14:30",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    day_two_slots = blueprint["time_skeleton"]["days"][1]["slots"]

    assert day_two_slots
    assert all(slot["start_time"] < slot["end_time"] for slot in day_two_slots)
    assert all(slot["start_time"] < "14:30" for slot in day_two_slots)
    assert all(slot["end_time"] <= "14:30" for slot in day_two_slots)


def test_blueprint_preserves_saturday_afternoon_to_sunday_noon_window():
    text = (
        "我周五晚上带爸妈到上海，酒店已经订在徐家汇，主要想排周六下午到周日中午的轻松安排。"
        "周六上午妈妈有个已经约好的口腔复诊，不用你帮我订；下午想看个不太累的展，晚上吃清淡点，"
        "爸爸控糖低盐，妈妈膝盖不好不能久走。周日想喝茶休息一下，再买点不占行李的文创伴手礼，"
        "14:30必须到虹桥火车站。预算1200不含酒店，少楼梯少步行，下雨也能改；"
        "牙科和伴手礼办不了就别说订好了。"
    )
    state = {
        "user_input": text,
        "scene_type": "family",
        "constraints": {
            "raw_text": text,
            "time_window": "saturday_afternoon_to_sunday_noon",
            "end_time": "14:30",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])

    assert blueprint["planning_horizon"] == "two_day"
    assert blueprint["planning_days"] == 2
    assert len(blueprint["time_skeleton"]["days"]) == 2
    assert blueprint["time_skeleton"]["days"][1]["slots"]
    assert all(
        slot["end_time"] <= "14:30"
        for slot in blueprint["time_skeleton"]["days"][1]["slots"]
    )


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


def test_two_day_sparse_request_gets_second_day_slots():
    state = {
        "user_input": "我们四个朋友准备在上海玩两天，想多拍点照片，聊聊天放松一下。可能会下雨，有人不能吃辣，预算人均200左右。",
        "scene_type": "friends",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    day_slots = blueprint["time_skeleton"]["days"]

    assert blueprint["planning_horizon"] == "two_day"
    assert blueprint["node_count"] >= 6
    assert len(day_slots[0]["slots"]) >= 3
    assert len(day_slots[1]["slots"]) >= 3
    assert day_slots[1]["slots"]
    assert any(slot["supply_domain"] == "activity" for slot in day_slots[1]["slots"])
    day_two_times = [
        (slot["start_time"], slot["end_time"])
        for slot in day_slots[1]["slots"]
    ]
    assert len(day_two_times) == len(set(day_two_times))
    assert any(slot["supply_domain"] == "restaurant" for slot in day_slots[1]["slots"])


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


def test_one_day_wording_preserves_full_day_horizon_with_few_roles():
    state = {
        "user_input": "我们四个朋友打算在上海玩一天，喜欢拍照、想轻松聊天，有人不能吃辣，预算人均200左右，可能下雨，晚上不想太晚结束。",
        "scene_type": "friends",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})

    assert blueprint["planning_horizon"] == "full_day"
    assert blueprint["planning_days"] == 1
    assert blueprint["template_mode"] == "multi_node"
    assert blueprint["node_count"] >= 4
    assert sum(1 for item in blueprint["node_intents"] if item["supply_domain"] == "activity") >= 2
    assert any(item["supply_domain"] == "restaurant" for item in blueprint["node_intents"])


def test_full_day_family_request_adds_meal_when_food_is_only_a_preference():
    text = "明天想安排一天亲子行程，孩子要玩得安全开心，大人希望吃得健康一些，预算中等，不想来回折腾太多。"
    state = {
        "user_input": text,
        "scene_type": "family",
        "constraints": {"raw_text": text},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert blueprint["planning_horizon"] == "full_day"
    assert blueprint["template_mode"] == "multi_node"
    assert "restaurant_lunch" in roles
    assert "family_activity" in roles
    assert "family_indoor_play" in roles


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


def test_dental_consultation_with_booking_caveat_stays_service_node():
    text = (
        "后天中午我想带妈妈在静安寺附近处理个口腔问题，"
        "最好12:30左右先找个靠谱牙科做洗牙或补牙咨询，"
        "结束后在附近坐一会儿喝点低糖的茶或咖啡，15:30前回公司。"
        "妈妈血糖高，别安排甜品下午茶；两个人总预算500以内。"
        "如果牙科不能直接帮我预约，不要假装订好了，告诉我需要自己确认什么。"
    )
    state = {
        "user_input": text,
        "scene_type": "family",
        "constraints": {
            "raw_text": text,
            "start_time": "12:30",
            "end_time": "15:30",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]
    cafe_node = next(item for item in blueprint["node_intents"] if item["role"] == "cafe")

    assert "dental_clinic" in roles
    assert "cafe" in roles
    assert "dental_clinic" in blueprint["unsupported_roles"]
    assert blueprint["template_mode"] == "multi_node"
    assert blueprint["requires_rag"] is True
    assert "咖啡" in cafe_node["search_terms"]
    assert "甜品" not in cafe_node["search_terms"]
    assert "下午茶" not in cafe_node["search_terms"]


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

    assert {"talk_show", "nail_salon", "restaurant_dinner", "bar"}.issubset(set(roles))


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


def test_blueprint_does_not_turn_dental_aftercare_anchor_into_service_or_bar():
    state = {
        "user_input": (
            "我今晚18:30在静安寺附近洗牙结束，麻药可能还没完全退，"
            "平时我爱吃辣和喝一杯，但今天医生说两小时内别吃太烫太辣太硬，也别喝酒。"
            "帮我找个安静不排队的地方先喝点温的，再吃点软一点清淡的。"
        ),
        "scene_type": "friends",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "dental_clinic" not in roles
    assert "bar" not in roles
    assert any(role in roles for role in ("restaurant_dinner", "restaurant_specific"))


def test_blueprint_does_not_turn_negated_post_filling_anchor_into_dental_service():
    text = (
        "我今天18:10从上海市口腔医院这边出来，刚补完牙，不是要找牙医，"
        "也别按我平时爱火锅KTV夜宵那套来。想就近找个安静能坐一会儿的地方，"
        "再吃点软烂清淡的东西，不能辣、不能太烫、不能硬、不要坚果和酒，"
        "20:30前回家，预算180以内；能订座就订，订不了也别说已经搞定。"
    )
    state = {
        "user_input": text,
        "scene_type": "solo",
        "constraints": {
            "raw_text": text,
            "start_time": "18:10",
            "end_time": "20:30",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "dental_clinic" not in roles
    assert any(role in roles for role in ("cafe", "tea_house", "restaurant_dinner", "restaurant_specific"))


def test_blueprint_keeps_business_dental_aftercare_anchors_out_of_itinerary():
    text = (
        "今天我和同事陪两个外地客户，17:40他们在人民广场附近做完口腔处理出来，"
        "20:45前要送回南京东路酒店。想安排一个轻松的上海味晚饭，"
        "再有一个坐着聊项目的地方，别喝酒、别太辣、别太硬，"
        "也不要需要大声说话的演出或KTV。人均250以内，最好能订位；"
        "如果你不能直接约牙科或酒店接送，就不要把它写成已安排。"
    )
    state = {
        "user_input": text,
        "scene_type": "friends",
        "constraints": {
            "raw_text": text,
            "start_time": "17:40",
            "end_time": "20:45",
            "avoid": ["KTV欢唱", "酒吧"],
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "dental_clinic" not in roles
    assert "lodging" not in roles
    assert "karaoke" not in roles
    assert "talk_show" not in roles
    assert "restaurant_dinner" in roles
    assert "cafe" in roles


def test_blueprint_respects_negated_birthday_escape_and_bar_roles():
    state = {
        "user_input": (
            "周六下午想给朋友过生日，5个人14点后在人民广场集合。"
            "她喜欢安静有仪式感、可以拍照，但这次不想密室或酒吧太吵；"
            "我们想先做个不太久的美甲或买束花，再找地方拍几张照，晚饭要能放小蛋糕。"
        ),
        "scene_type": "friends",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "board_game_escape" not in roles
    assert "bar" not in roles
    assert {"nail_salon", "flower_shop", "cultural_photo", "restaurant_dinner"}.issubset(
        set(roles)
    )


def test_blueprint_uses_completed_football_as_anchor_and_rejects_old_escape_memory():
    state = {
        "user_input": (
            "这个周末想带孩子和外婆轻松安排两天，但两天都回家住，不需要酒店。"
            "周六孩子12点在徐汇足球训练下课，我们从那边开始；刚训练完别再安排高强度，"
            "下午想做点室内手作或看展，晚饭清淡少糖。"
            "周日上午买点伴手礼，中午简单吃，下午再有一个不累的亲子点。"
            "爸爸以前爱密室和重口味夜宵，这次别按那个来。"
        ),
        "scene_type": "family",
        "constraints": {},
    }

    blueprint = build_b_itinerary_blueprint(state, constraints={})
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert blueprint["planning_horizon"] == "two_day"
    assert "lodging" not in roles
    assert "sports_training" not in roles
    assert "board_game_escape" not in roles
    assert "exhibition" in roles
    assert "family_activity" in roles


def test_blueprint_preserves_requested_football_trial_and_dental_followup():
    text = (
        "周六11点后我从浦东世纪公园附近带8岁儿子出发，"
        "想先看看有没有靠谱的少儿足球体验课或试听，"
        "15:00必须到静安寺附近牙科复诊；午饭别辣别太甜，"
        "牙医后如果孩子状态还行就找个安静室内地方坐一会，18:00前回家。"
        "总预算800以内。足球课和牙医如果你不能直接约，就明确告诉我需要自己确认，"
        "别拿普通亲子乐园来替代。"
    )
    state = {
        "user_input": text,
        "scene_type": "family",
        "constraints": {
            "raw_text": text,
            "scene": "family",
            "companions": [{"role": "child", "age": 8}],
            "child_age": 8,
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "sports_training" in roles
    assert "dental_clinic" in roles
    assert "restaurant_lunch" in roles


def test_blueprint_treats_existing_football_trial_and_parent_checkup_as_anchors():
    text = (
        "周五孩子17:00-18:30在浦东源深那边有个少儿足球试听课，已经约好了，"
        "不需要再帮我报名或订培训；我和孩子爸爸下班后去接他，18:45以后附近吃饭。"
        "孩子乳制品过敏，爸爸刚体检完要控糖，别安排奶油蛋糕、披萨这种；"
        "如果顺路能买个给同学生日的小礼物或文具就提醒一下，但买礼物不用假装能下单。"
        "20:30前回家，总预算450以内，餐厅能订就订。"
    )
    state = {
        "user_input": text,
        "scene_type": "family",
        "constraints": {
            "raw_text": text,
            "scene": "family",
            "companions": [{"role": "child"}],
            "child_age": 8,
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "sports_training" not in roles
    assert "pet_hospital" not in roles
    assert "restaurant_specific" in roles or "restaurant_dinner" in roles
    assert "park_scenic_walk" not in roles
    assert "family_activity" not in roles


def test_blueprint_scopes_negation_to_current_phrase_for_two_day_teen_plan():
    text = (
        "这个周末两天帮我排一下：周六10点在虹桥火车站接12岁的侄子和爷爷，"
        "侄子想看看有没有少儿足球试听或青训可以了解，爷爷腿脚一般，最好附近有茶馆能坐着等。"
        "晚上住一晚，靠地铁、别太贵；第二天上午想看个科技馆、博物馆或展，"
        "午饭清淡不辣，15点前送回虹桥。总预算1800以内含住宿，别按幼儿亲子乐园排。"
        "足球课和住宿如果不能直接订，就写成需要电话确认或备选，不要说已经订好了。"
    )
    state = {
        "user_input": text,
        "scene_type": "family",
        "constraints": {
            "raw_text": text,
            "scene": "family",
            "companions": [{"role": "child", "age": 12}],
            "child_age": 12,
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "sports_training" in roles
    assert "lodging" in roles
    assert "exhibition" in roles
    assert "restaurant_lunch" in roles
    assert "tea_house" in roles
    assert "family_activity" not in roles
    assert {"sports_training", "lodging"}.issubset(set(blueprint["unsupported_roles"]))


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


def test_blueprint_respects_current_turn_child_preference_override():
    state = {
        "user_input": (
            "今晚临时和两个同事在静安寺附近聚一下，就我们三个，不带孩子也不带老婆。"
            "想先吃个重庆火锅，辣一点没关系，饭后再找个能唱歌的地方放松，"
            "21:10左右开始，23:50前散。以前那些亲子、轻食偏好这次别套进来。"
        ),
        "scene_type": "friends",
        "constraints": {
            "raw_text": (
                "今晚临时和两个同事在静安寺附近聚一下，就我们三个，不带孩子也不带老婆。"
                "想先吃个重庆火锅，辣一点没关系，饭后再找个能唱歌的地方放松，"
                "21:10左右开始，23:50前散。以前那些亲子、轻食偏好这次别套进来。"
            ),
            "avoid": ["亲子", "儿童友好", "低卡", "轻食"],
        },
        "scenario_activities": ["亲子", "儿童友好"],
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "restaurant_specific" in roles
    assert "karaoke" in roles
    assert "family_activity" not in roles
    assert "family_indoor_play" not in roles


def test_blueprint_respects_no_child_parent_visit_override():
    text = (
        "周日我爸妈来上海，不带孩子，别按亲子路线排。10点在人民广场附近接他们，"
        "想先看一个展或博物馆，中午吃点本帮或上海味道但要少油少糖，"
        "下午想顺路买一束花和小蛋糕，晚上18:30左右给我妈安排一顿安静点的生日晚餐，"
        "20:30前送他们到虹桥火车站。"
    )
    state = {
        "user_input": text,
        "scene_type": "family",
        "constraints": {
            "raw_text": text,
            "start_time": "10:00",
            "end_time": "20:30",
            "avoid": ["亲子", "儿童友好"],
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "exhibition" in roles
    assert "restaurant_lunch" in roles
    assert "restaurant_dinner" in roles
    assert "family_activity" not in roles
    assert "family_indoor_play" not in roles


def test_blueprint_drops_negated_ktv_from_halal_colleague_show_case():
    text = (
        "今晚下班后我和两个同事在静安寺附近碰头，18:10以后开始，22点前散。"
        "先吃饭再看个轻松的脱口秀或小剧场都行，但这次不带孩子，"
        "也别按我以前那种亲子/低卡轻食偏好来；有一位同事是穆斯林，"
        "不吃猪肉也不喝酒。人均220以内，别KTV、别吵酒吧，"
        "地铁方便，能订座和买票就先处理。"
    )
    state = {
        "user_input": text,
        "scene_type": "friends",
        "constraints": {
            "raw_text": text,
            "start_time": "18:10",
            "end_time": "22:00",
            "avoid": ["亲子", "儿童友好", "KTV欢唱", "酒吧"],
            "scenario_activities": ["多人活动", "轻量活动", "KTV欢唱", "清真友好"],
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "restaurant_dinner" in roles
    assert "talk_show" in roles
    assert "karaoke" not in roles
    assert "family_activity" not in roles


def test_blueprint_drops_negated_hotpot_bbq_terms_from_halal_fallback_case():
    text = (
        "今晚临时约了5个朋友，19:10后在人民广场附近碰头，"
        "想先吃顿能照顾清真的晚饭，饭后如果还有精力就看个电影或者唱一小时KTV，"
        "但23:30前一定散，别安排太远。人均260左右，不想现场排很久；"
        "如果清真餐厅订不上，宁可只保留电影/KTV备选，也别随便换成普通火锅烧烤。"
    )
    state = {
        "user_input": text,
        "scene_type": "friends",
        "constraints": {
            "raw_text": text,
            "start_time": "19:10",
            "end_time": "23:30",
            "avoid": ["火锅", "烤肉", "太远", "排队久"],
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    restaurant = next(
        item for item in blueprint["node_intents"] if item["role"] == "restaurant_dinner"
    )

    assert "晚饭" in restaurant["search_terms"]
    assert "餐厅" in restaurant["search_terms"]
    assert "火锅" not in restaurant["search_terms"]
    assert "烧烤" not in restaurant["search_terms"]


def test_blueprint_treats_completed_movie_as_anchor_for_late_hotpot():
    text = (
        "今晚看完电影大概21:20在人民广场附近，我和老婆两个人，不带孩子，"
        "她今天想吃点有仪式感的火锅或暖和的正餐，但不要太辣，"
        "最好能订位，23:30前结束，预算两个人400以内。"
    )
    state = {
        "user_input": text,
        "scene_type": "couple",
        "constraints": {
            "raw_text": text,
            "start_time": "21:20",
            "end_time": "23:30",
            "avoid": ["亲子", "儿童友好", "轻食", "低卡"],
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "cinema" not in roles
    assert "restaurant_specific" in roles
    assert "family_activity" not in roles
    restaurant_slot = next(
        slot
        for slot in blueprint["time_skeleton"]["days"][0]["slots"]
        if slot["supply_domain"] == "restaurant"
    )
    assert restaurant_slot["start_time"] == "21:20"
    assert restaurant_slot["end_time"] <= "23:30"


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


def test_blueprint_keeps_bridal_nodes_and_drops_negated_karaoke():
    text = (
        "周六我们5个女生在徐家汇集合，陪闺蜜婚礼前做个轻松准备日："
        "想先看看能不能做个不刺鼻的美甲/简单造型，再去拍一组好看的汉服或新中式照片，"
        "下午吃一顿不含猪肉和酒精、最好清真友好的饭。"
        "我们以前经常让你排KTV，但这次不要KTV、酒吧或夜店，18:30前散，人均350左右。"
        "美甲造型如果不能下单就写成需要电话确认，拍照或餐厅能订的再帮忙订。"
    )
    state = {
        "user_input": text,
        "scene_type": "friends",
        "constraints": {
            "raw_text": text,
            "avoid": ["KTV欢唱", "酒吧", "夜店"],
            "soft_tags": ["可预约", "需要订座", "堂食"],
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "nail_salon" in roles
    assert "cultural_photo" in roles
    assert "restaurant_dinner" in roles
    assert "karaoke" not in roles
    assert "bar" not in roles
    assert blueprint["template_mode"] == "multi_node"


def test_blueprint_keeps_birthday_dinner_and_karaoke_after_gift_guidance():
    text = (
        "今晚临时给同事过生日，6个人下班后在静安寺附近集合。"
        "想先顺路买一小束花或小蛋糕，吃完饭去唱一小时歌，"
        "22点前散，人均220左右，能订座、订包间就先确认。"
    )
    state = {
        "user_input": text,
        "scene_type": "friends",
        "constraints": {
            "raw_text": text,
            "end_time": "22:00",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]
    slots = blueprint["time_skeleton"]["days"][0]["slots"]

    assert "flower_shop" in roles
    assert "souvenir_shopping" in roles
    assert "restaurant_dinner" in roles
    assert "karaoke" in roles
    assert roles.index("restaurant_dinner") < roles.index("karaoke")
    assert all(slot["end_time"] <= "22:00" for slot in slots)


def test_blueprint_keeps_short_couple_evening_as_half_day_and_sequential():
    text = (
        "今晚我和老婆单独过个小纪念日，不带孩子，18:20以后从陆家嘴下班出发，"
        "想先买一小束花，再找个安静点、有仪式感的晚餐，最好还能散散步聊会儿天；"
        "她海鲜过敏也不太喝酒，21:30前要回到家附近，总预算800以内。"
    )
    state = {
        "user_input": text,
        "scene_type": "couple",
        "constraints": {
            "raw_text": text,
            "start_time": "18:20",
            "end_time": "21:30",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    slots = blueprint["time_skeleton"]["days"][0]["slots"]
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert blueprint["planning_horizon"] == "half_day"
    assert "flower_shop" in roles
    assert all(slot["end_time"] <= "21:30" for slot in slots)
    assert all(
        left["end_time"] <= right["start_time"]
        for left, right in zip(slots, slots[1:])
    )


def test_blueprint_sequences_cross_midnight_late_night_karaoke_window():
    text = (
        "今晚我们3个人22:30以后在人民广场附近碰头，想先吃点不太辣的夜宵，"
        "再找个KTV唱一会儿，最晚1点前散；人均150左右。"
    )
    state = {
        "user_input": text,
        "scene_type": "friends",
        "constraints": {
            "raw_text": text,
            "start_time": "22:30",
            "end_time": "01:00",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    slots = blueprint["time_skeleton"]["days"][0]["slots"]
    restaurant = next(slot for slot in slots if slot["supply_domain"] == "restaurant")
    karaoke = next(slot for slot in slots if slot["role"] == "karaoke")

    assert blueprint["planning_horizon"] == "half_day"
    assert restaurant["start_time"] == "22:30"
    assert restaurant["end_time"] <= karaoke["start_time"]
    assert karaoke["end_time"] == "01:00"


def test_blueprint_keeps_specific_food_as_evening_dinner_node():
    state = {
        "user_input": (
            "我们一家四口想在外滩附近找个有厨房的住处，"
            "晚上能去哪吃蟹黄面？还想找个便利店买点日用品，"
            "最后还需要找个停车场停车。"
        ),
        "constraints": {
            "raw_text": "外滩 有厨房 住处 晚上能去哪吃蟹黄面 便利店 日用品 停车场",
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
        "user_input": "请朋友来夜游黄浦江，有没有提供包厢和自助餐的游船？还想找个停车场停车。",
        "constraints": {
            "raw_text": "夜游黄浦江 包厢 自助餐 游船 停车场",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert "river_cruise" in roles
    assert "parking" in roles
    assert "park_scenic_walk" not in roles


def test_blueprint_treats_business_banquet_as_restaurant_need():
    state = {
        "user_input": "我要在徐家汇这边宴请重要客户，客户可能需要住宿，还想在附近找个停车场，哪里比较合适？",
        "constraints": {
            "raw_text": "徐家汇 宴请 重要客户 住宿 停车场",
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
            "我住在陆家嘴附近，周末想安排一个文艺下午：先找个地方看相声，"
            "然后去附近吃本帮菜晚餐，饭后想喝咖啡聊天，最后能买点上海特产带回家。"
        ),
        "constraints": {
            "raw_text": "陆家嘴 文艺下午 相声 本帮菜 晚餐 咖啡 上海特产",
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
            "想在浦东新区安排文艺周末。周六晚上7点看话剧，之后附近吃夜宵。"
            "周日上午去文艺咖啡馆，下午2点看相声，然后买些上海特产伴手礼。"
        ),
        "constraints": {
            "raw_text": "浦东新区 周六 晚上 话剧 夜宵 周日 上午 文艺咖啡馆 下午2点 相声 上海特产 伴手礼",
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
    assert "theatre_performance" in roles
    assert "restaurant_specific" in roles
    assert "cafe" in roles
    assert "talk_show" in roles
    assert "souvenir_shopping" in roles
    assert slot_by_role["theatre_performance"]["day"] == 1
    assert slot_by_role["restaurant_specific"]["part_of_day"] == "late_evening"
    assert slot_by_role["cafe"]["day"] == 2
    assert slot_by_role["talk_show"]["day"] == 2
    assert slot_by_role["souvenir_shopping"]["day"] == 2


def test_full_day_constraint_expands_simple_request_into_default_itinerary():
    state = {
        "user_input": "我人在上海，但是这个周末想在青岛轻松玩一天，想吃海鲜，别太累",
        "scene_type": "solo",
        "constraints": {
            "city": "青岛",
            "time_window": "full_day",
            "duration_range": [6, 10],
            "raw_text": "青岛 轻松 一天 海鲜",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]

    assert blueprint["planning_horizon"] == "full_day"
    assert blueprint["template_mode"] == "multi_node"
    assert roles == ["citywalk_market", "restaurant_lunch", "park_scenic_walk", "restaurant_dinner"]
    assert "海鲜" in blueprint["node_intents"][1]["search_terms"]


def test_two_day_constraint_expands_simple_request_into_day_split():
    state = {
        "user_input": "周末两天去青岛放松，想吃海鲜",
        "scene_type": "solo",
        "constraints": {
            "city": "青岛",
            "time_window": "two_day",
            "duration_range": [12, 20],
            "raw_text": "青岛 两天 海鲜",
        },
    }

    blueprint = build_b_itinerary_blueprint(state, constraints=state["constraints"])
    roles = [item["role"] for item in blueprint["node_intents"]]
    day_slots = blueprint["time_skeleton"]["days"]

    assert blueprint["planning_horizon"] == "two_day"
    assert blueprint["planning_days"] == 2
    assert "citywalk_market" in roles
    assert "restaurant_dinner" in roles
    assert "restaurant_lunch" in roles
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
