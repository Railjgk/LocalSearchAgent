from src.nodes.intent_parser import (
    _normalize_llm_intent,
    constraints_from_intent,
    parse_intent,
)


def test_intent_parser_counts_explicit_friend_group_phrase_as_total() -> None:
    digit_intent = parse_intent(
        "明天4个朋友想在上海轻松玩一天，10:30左右开始、20:30前散。"
    )
    chinese_intent = parse_intent(
        "我们四个朋友准备在上海玩两天，想多拍点照片，聊聊天放松一下。"
    )

    assert digit_intent["scene"] == "friends"
    assert digit_intent["people_count"] == 4
    assert chinese_intent["scene"] == "friends"
    assert chinese_intent["people_count"] == 4


def test_intent_parser_counts_described_classmate_group_before_singular_exception() -> None:
    intent = parse_intent(
        "我们5个外地大学同学周六12:30到上海虹桥，周日15:00从虹桥走，"
        "想做个毕业前小旅行。一个同学不吃猪肉、乳糖不耐。"
    )

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 5
    assert next(item for item in intent["people"] if item["role"] == "friends")[
        "count"
    ] == 4


def test_intent_parser_counts_we_classmates_before_individual_diet_clause() -> None:
    intent = parse_intent(
        "这个周末我们5个大学同学毕业前想聚两天，但都住上海，不用订酒店。"
        "周六14点以后开始，想拍点照片、吃顿不太贵的晚饭，再唱歌或看场轻松电影，"
        "22点前散；周日10:30再碰头，喝咖啡聊聊天，顺便给老师买一束花或小礼物，"
        "16点前结束。一个人吃素，一个人不吃牛羊肉，人均500以内。"
    )
    constraints = constraints_from_intent(intent)

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 5
    assert next(item for item in intent["people"] if item["role"] == "friends")[
        "count"
    ] == 4
    assert {"素食", "不吃牛羊肉"} <= set(constraints["dietary_constraints"])


def test_intent_parser_counts_explicit_friend_companions_plus_self() -> None:
    intent = parse_intent("我和4个朋友今晚想吃火锅，预算人均200左右。")

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 5


def test_intent_parser_ignores_childcare_location_when_counting_adult_friends() -> None:
    intent = parse_intent(
        "今天孩子在外婆家，这次真的不带娃，也别按亲子偏好来。"
        "16点左右我和两个女生朋友在静安寺附近碰头，给闺蜜补个生日。"
    )

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 3
    assert all(person["role"] != "child" for person in intent["people"])


def test_intent_parser_counts_customer_companions_plus_self() -> None:
    user_input = (
        "今晚临时和两个外地客户在陆家嘴开完会，差不多20点结束，"
        "想找个安静点的地方吃顿上海味道的晚饭。"
    )

    intent = parse_intent(user_input)

    assert intent["people_count"] == 3


def test_intent_parser_prefers_bare_total_count_over_companion_count() -> None:
    intent = parse_intent(
        "今天12:30到14:30在静安寺附近请两个同事吃午饭，三个人，其中一个客户不能吃辣。"
    )

    assert intent["people_count"] == 3


def test_intent_parser_counts_family_household_phrase() -> None:
    intent = parse_intent("想安排一天适合一家三口的行程，孩子要玩得开心，大人饮食要健康。")

    assert intent["scene"] == "family"
    assert intent["people_count"] == 3


def test_intent_parser_counts_child_and_two_elders_plus_self() -> None:
    intent = parse_intent(
        "周日帮我排个一日安排：上午10点从徐汇医院附近出发，"
        "带4岁孩子和两位老人，老人刚检查完不能太累，午饭要低盐清淡。"
    )

    assert intent["scene"] == "family"
    assert intent["people_count"] == 4


def test_intent_parser_treats_adult_only_count_as_total_with_parents() -> None:
    intent = parse_intent(
        "这个周末想给妈妈过生日，周六中午12点从人民广场接爸妈，"
        "周日下午16点前送回杨浦；只有三个大人，不带孩子，别安排儿童乐园。"
    )

    assert intent["people_count"] == 3
    assert all(person["role"] != "child" for person in intent["people"])


def test_intent_parser_counts_adult_people_phrase_with_mother_request() -> None:
    intent = parse_intent(
        "周六下午帮我安排给妈妈过生日，3个成年人，14点从人民广场附近开始。"
        "她血糖偏高，别安排太甜太咸，也不想去人挤人的商场。"
    )

    assert intent["scene"] == "family"
    assert intent["people_count"] == 3
    assert all(person["role"] != "child" for person in intent["people"])


def test_intent_parser_treats_latest_phrase_as_deadline_not_start() -> None:
    intent = parse_intent("晚上吃不辣的餐厅，再去清吧坐一会但别超过22:30。")

    assert intent["time"]["start_time"] is None
    assert intent["time"]["end_time"] == "22:30"


def test_intent_parser_preserves_cross_day_birthday_window_and_group_count() -> None:
    intent = parse_intent(
        "这个周六中午到周日15点，三个女生给闺蜜补过生日，上海市内两天但各自回家不住酒店。"
        "周六晚上吃不辣的餐厅，再去清吧坐一会但别超过22:30；周日想轻松看展或者找茶馆聊天。"
    )

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 3
    assert intent["time"]["window"] == "weekend"
    assert intent["time"]["start_time"] == "12:00"
    assert intent["time"]["end_time"] == "15:00"
    assert intent["time"]["duration_range"] == [27, 27]


def test_intent_parser_treats_negated_hotpot_bbq_as_avoid() -> None:
    intent = parse_intent(
        "今晚6个同事过生日，想先吃饭再去KTV，别安排烤肉火锅那种容易踩雷的。"
    )

    assert "烤肉" not in intent["constraints"]["hard"]
    assert "火锅" not in intent["constraints"]["hard"]
    assert "烤肉" not in intent["constraints"]["soft"]
    assert "火锅" not in intent["constraints"]["soft"]
    assert "烤肉" not in intent["planning_preferences"]["food_type"]
    assert "火锅" not in intent["planning_preferences"]["restaurant_type"]
    assert {"烤肉", "火锅"} <= set(intent["constraints"]["avoid"])


def test_intent_parser_keeps_positive_bbq_when_only_hotpot_is_negated() -> None:
    intent = parse_intent("今晚和朋友想吃烤肉，最好是日式烧肉，别推荐火锅。")

    assert "烤肉" in intent["planning_preferences"]["food_type"]
    assert "烤肉" in intent["planning_preferences"]["restaurant_type"]
    assert "烤肉" in intent["constraints"]["soft"]
    assert "火锅" not in intent["planning_preferences"]["restaurant_type"]
    assert "火锅" in intent["constraints"]["avoid"]


def test_llm_normalization_corrects_customer_companion_count() -> None:
    user_input = (
        "今晚临时和两个外地客户在陆家嘴开完会，差不多20点结束，"
        "想找个安静点的地方吃顿上海味道的晚饭。"
    )
    mock_intent = parse_intent(user_input)
    raw_intent = {
        "scene": "unknown",
        "people_count": 2,
        "people": [
            {"role": "self", "needs": []},
            {"role": "friends", "count": 2, "needs": ["quiet"]},
        ],
    }

    normalized = _normalize_llm_intent(raw_intent, mock_intent, user_input)

    assert normalized["people_count"] == 3


def test_llm_normalization_keeps_explicit_friend_group_total() -> None:
    user_input = (
        "明天4个朋友想在上海轻松玩一天，10:30左右开始、20:30前散。"
        "可能下雨，一个朋友吃素，一个不能吃辣。"
    )
    mock_intent = parse_intent(user_input)
    raw_intent = {
        "scene": "friends",
        "people_count": 5,
        "people": [
            {"role": "self", "needs": []},
            {"role": "friends", "count": 4, "needs": ["group_friendly"]},
        ],
    }

    normalized = _normalize_llm_intent(raw_intent, mock_intent, user_input)

    assert normalized["people_count"] == 4


def test_llm_normalization_moves_risk_tags_out_of_hard_constraints() -> None:
    user_input = "周末两天想安排亲子行程，不想排队太久，也别去人多拥挤的地方。"
    mock_intent = parse_intent(user_input)
    raw_intent = {
        "scene": "family",
        "constraints": {
            "hard": ["儿童友好", "排队久", "人多拥挤"],
            "soft": ["轻食", "高热量"],
            "avoid": ["商场拥挤"],
        },
    }

    normalized = _normalize_llm_intent(raw_intent, mock_intent, user_input)

    assert "儿童友好" in normalized["constraints"]["hard"]
    assert "排队久" not in normalized["constraints"]["hard"]
    assert "人多拥挤" not in normalized["constraints"]["hard"]
    assert "高热量" not in normalized["constraints"]["soft"]
    assert {"排队久", "人多拥挤", "高热量", "商场拥挤"} <= set(
        normalized["constraints"]["avoid"]
    )


def test_llm_normalization_drops_child_role_for_mother_request() -> None:
    user_input = (
        "这个周末想接妈妈来上海放松一下，两天一夜。"
        "妈妈膝盖不好，刚补牙，只能吃软一点、清淡不辣。"
    )
    mock_intent = parse_intent(user_input)
    raw_intent = {
        "scene": "family",
        "people_count": 2,
        "people": [
            {"role": "self", "needs": []},
            {"role": "child", "state": "膝盖不好，刚补牙", "needs": ["low_intensity"]},
        ],
    }

    normalized = _normalize_llm_intent(raw_intent, mock_intent, user_input)

    assert all(person["role"] != "child" for person in normalized["people"])
    assert normalized["people_count"] == 2


def test_intent_parser_does_not_treat_girlfriend_as_friend_group() -> None:
    intent = parse_intent(
        "明天女朋友生日，我15:30以后从人民广场出来，想先买一小束花，"
        "再带她吃安静晚饭，最后看一场轻松脱口秀，21:45前结束。"
        "两个人总预算900以内，她乳糖不耐，别默认奶油蛋糕。"
    )

    assert intent["scene"] == "couple"
    assert all(person["role"] != "friends" for person in intent["people"])
    assert "避开乳糖" in intent["constraints"]["hard"]
    assert "避开乳糖" in intent["planning_preferences"]["food_type"]
    assert intent["people_count"] == 2


def test_intent_parser_preserves_halal_no_pork_and_sequence() -> None:
    text = (
        "今晚临时和4个同事下班聚一下，我18:40在静安寺附近出发，"
        "先吃个能聊天的晚饭，再去唱一会儿KTV，22:15前散。"
        "同事里有一位穆斯林，餐厅要能明确避开猪肉或有清真/素食选择，"
        "大家都不太喝酒，人均220以内。"
    )
    intent = parse_intent(text)
    constraints = constraints_from_intent(intent)

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 5
    assert {"清真友好", "不含猪肉", "素食", "无酒精"} <= set(
        intent["constraints"]["hard"]
    )
    assert constraints["sequence_preference"] == "restaurant_then_activity"


def test_intent_parser_preserves_family_allergy_mobility_and_nap_anchor() -> None:
    intent = parse_intent(
        "周日帮我排个一日安排：上午10点从徐汇医院附近出发，带4岁孩子和两位老人，"
        "老人刚检查完不能太累，午饭要低盐清淡；孩子13:30-15:00基本要午睡，"
        "推婴儿车所以少楼梯。孩子坚果过敏。晚上19点左右想看个亲子剧。"
    )
    constraints = constraints_from_intent(intent)

    assert {"低盐", "避开坚果", "少步行", "婴儿车友好"} <= set(
        intent["constraints"]["hard"]
    )
    assert {"过敏友好", "避开坚果"} <= set(intent["people"][-1]["needs"])
    assert {"低盐", "避开坚果"} <= set(constraints["dietary_constraints"])
    assert {"少步行", "婴儿车友好"} <= set(constraints["accessibility_constraints"])
    assert {"type": "rest", "start_time": "13:30", "end_time": "15:00"} in constraints[
        "time_anchors"
    ]
    assert {"type": "event", "time": "19:00", "label": "亲子剧"} in constraints[
        "time_anchors"
    ]


def test_intent_parser_preserves_parent_visit_weekend_dropoff_anchor() -> None:
    intent = parse_intent(
        "爸妈这周六来上海，我10:20在虹桥站接到他们，"
        "周日14:00还要从虹桥站送他们走。想安排两天轻松一点："
        "周六中午清淡午饭，下午看一个博物馆或展览，最好有讲解；"
        "晚上住得离地铁或虹桥回程方便一点。周日上午吃个上海早点，"
        "再买点不太甜的伴手礼带回去。爸爸控糖，妈妈膝盖不好少楼梯。"
        "总预算2200以内含住宿和吃饭。"
    )
    constraints = constraints_from_intent(intent)

    assert intent["scene"] == "family"
    assert intent["people_count"] == 3
    assert intent["time"]["window"] == "weekend"
    assert intent["time"]["start_time"] == "10:20"
    assert intent["time"]["end_time"] == "14:00"
    assert intent["location"]["origin"] == "虹桥站"
    assert "少糖" in intent["constraints"]["hard"]
    assert "少步行" in constraints["accessibility_constraints"]
