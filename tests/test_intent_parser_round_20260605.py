from src.nodes.intent_parser import constraints_from_intent, parse_intent


def test_round_20260605_parent_rehab_preserves_trip_and_rest_anchors() -> None:
    text = (
        "这周末想带爸妈在上海市内慢慢住一晚，周六9:30从中山医院附近出发，"
        "周日15:30前回到医院附近。妈妈刚做完小手术，最好能借轮椅或至少电梯方便、少台阶，"
        "周六14:00-16:00一定要休息；爸爸想看看老上海但不要很商业。"
        "吃饭要低盐清淡、不能辣。总预算1800以内含住宿和吃饭。"
    )

    intent = parse_intent(text)
    constraints = constraints_from_intent(intent)

    assert intent["time"]["window"] == "weekend"
    assert intent["time"]["start_time"] == "09:30"
    assert intent["time"]["end_time"] == "15:30"
    assert intent["time"]["duration_range"] == [30, 30]
    assert {"少步行", "低强度", "短距离"} <= set(constraints["accessibility_constraints"])
    assert {"低盐", "轻食", "不辣"} <= set(constraints["dietary_constraints"])
    assert {"type": "rest", "start_time": "14:00", "end_time": "16:00"} in constraints[
        "time_anchors"
    ]


def test_round_20260605_kidfree_anniversary_keeps_no_seafood() -> None:
    intent = parse_intent(
        "我之前经常让你按带娃来排，但今晚孩子在外婆家，别再套亲子方案了。"
        "18:40以后我和老婆从静安寺附近出发，想补一个安静点的小纪念日约会，"
        "再吃一顿不含海鲜的晚饭。22:30前到家，总预算900以内。"
    )

    constraints = constraints_from_intent(intent)

    assert intent["scene"] == "couple"
    assert intent["people_count"] == 2
    assert "避开海鲜" in constraints["dietary_constraints"]
    assert "儿童友好" not in intent["constraints"]["hard"]


def test_round_20260605_business_reception_counts_host_plus_clients() -> None:
    intent = parse_intent(
        "今天临时要接待两个外地客户，14:20在虹桥火车站附近碰头，"
        "晚上20:10前要送到浦东机场T2。晚饭人均260左右，"
        "一个客户不吃猪肉，另一个乳糖不耐。"
    )

    assert intent["people_count"] == 3
    assert intent["time"]["start_time"] == "14:20"
    assert intent["time"]["end_time"] == "20:10"


def test_round_20260605_existing_child_training_uses_post_anchor_dinner_window() -> None:
    text = (
        "周五孩子17:00-18:30在浦东源深那边有个少儿足球试听课，已经约好了，"
        "不需要再帮我报名或订培训；我和孩子爸爸下班后去接他，18:45以后附近吃饭。"
        "孩子乳制品过敏，爸爸刚体检完要控糖，别安排奶油蛋糕、披萨这种；"
        "如果顺路能买个给同学生日的小礼物或文具就提醒一下，但买礼物不用假装能下单。"
        "20:30前回家，总预算450以内，餐厅能订就订。"
    )

    intent = parse_intent(text)
    constraints = constraints_from_intent(intent)

    assert intent["people_count"] == 3
    assert intent["time"]["start_time"] == "18:45"
    assert intent["time"]["end_time"] == "20:30"
    assert intent["time"]["duration_range"] == [1.75, 1.75]
    assert intent["location"]["origin"] == "浦东源深"
    assert {"过敏友好", "避开乳糖", "少糖"} <= set(constraints["dietary_constraints"])
    child = next(person for person in intent["people"] if person["role"] == "child")
    assert "避开乳糖" in child["needs"]
    assert "避开坚果" not in child["needs"]
