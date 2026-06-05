from src.nodes.intent_parser import _normalize_llm_intent, parse_intent


def test_intent_parser_counts_self_plus_two_colleagues() -> None:
    user_input = (
        "今晚18:30左右我和两个同事在静安寺地铁站附近碰头，"
        "给一个离职同事简单告别。"
    )

    intent = parse_intent(user_input)

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 3
    assert intent["people"] == [
        {"role": "self", "needs": []},
        {"role": "friends", "count": 2, "needs": ["group_friendly", "social"]},
    ]


def test_intent_parser_counts_colleagues_with_later_single_restriction() -> None:
    user_input = (
        "今晚这次不带孩子，也不是亲子安排，我和两个同事在静安寺附近下班后"
        "想先吃个正餐，再找个安静清吧坐一会儿聊项目；一位同事不喝酒"
        "只能点无酒精饮品，别安排火锅烧烤，人均300左右，23点前散。"
    )

    intent = parse_intent(user_input)

    assert intent["scene"] == "friends"
    assert intent["people_count"] == 3
    assert all(person["role"] != "child" for person in intent["people"])


def test_llm_normalization_corrects_self_plus_two_colleagues_count() -> None:
    user_input = (
        "今晚18:30左右我和两个同事在静安寺地铁站附近碰头，"
        "给一个离职同事简单告别。"
    )
    mock_intent = parse_intent(user_input)
    raw_intent = {
        "scene": "friends",
        "people_count": 2,
        "people": [
            {"role": "self", "needs": []},
            {"role": "friends", "count": 2, "needs": ["quiet"]},
        ],
    }

    normalized = _normalize_llm_intent(raw_intent, mock_intent, user_input)

    assert normalized["people_count"] == 3
