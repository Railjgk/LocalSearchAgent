from src.nodes.intent_parser import constraints_from_intent, parse_intent


def _meal_anchors(constraints: dict) -> list[dict]:
    return [
        anchor
        for anchor in constraints["time_anchors"]
        if anchor.get("type") == "meal"
    ]


def test_intent_parser_preserves_lunch_meal_anchor_with_fixed_day_anchors() -> None:
    text = (
        "周日帮我排个一日安排：上午10点从徐汇医院附近出发，带4岁孩子和两位老人，"
        "老人刚检查完不能太累，午饭要低盐清淡；孩子13:30-15:00基本要午睡，"
        "推婴儿车所以少楼梯。晚上19点左右想看个轻松点的演出或亲子剧，"
        "但不要太吵的商场，1000以内。中间如果时间不够宁可少安排，不要硬塞。"
    )
    intent = parse_intent(text)
    constraints = constraints_from_intent(intent)

    lunch_anchor = next(
        anchor
        for anchor in constraints["time_anchors"]
        if anchor.get("type") == "meal" and anchor.get("meal_type") == "午餐"
    )

    assert lunch_anchor["label"] == "午餐"
    assert lunch_anchor["part_of_day"] == "午间"
    assert lunch_anchor["compatibility_role"] == "restaurant_lunch"
    assert {"低盐", "轻食"} <= set(lunch_anchor["dietary"])
    assert "午饭" in lunch_anchor["source_terms"]
    assert lunch_anchor in intent["explicit_constraints"]["time_anchors"]
    assert lunch_anchor in intent["explicit_constraints"]["meal_anchors"]
    assert {"type": "rest", "start_time": "13:30", "end_time": "15:00"} in constraints[
        "time_anchors"
    ]
    assert {"type": "event", "time": "19:00", "label": "亲子剧"} in constraints[
        "time_anchors"
    ]


def test_intent_parser_does_not_infer_lunch_from_midday_departure() -> None:
    intent = parse_intent(
        "这个周末想和对象过纪念日，两天一夜，周六中午出发、周日16点前结束，"
        "上海周边或市内都行，我们自驾，会带一只小狗。"
    )
    constraints = constraints_from_intent(intent)

    assert _meal_anchors(constraints) == []


def test_intent_parser_preserves_single_explicit_dinner_anchor() -> None:
    intent = parse_intent(
        "临时改计划了，今天16:20以后从杨浦五角场出发，带6岁孩子和膝盖不太好的外婆"
        "出去透口气，最好有个不用排很久的室内活动，再吃个清淡晚饭；孩子坚果过敏，"
        "外婆不能走太多，19:30前要回到家附近，总预算600以内。"
    )
    constraints = constraints_from_intent(intent)
    meal_anchors = _meal_anchors(constraints)

    assert len(meal_anchors) == 1
    assert meal_anchors[0]["label"] == "晚餐"
    assert meal_anchors[0]["part_of_day"] == "晚间"
    assert meal_anchors[0]["compatibility_role"] == "restaurant_dinner"
