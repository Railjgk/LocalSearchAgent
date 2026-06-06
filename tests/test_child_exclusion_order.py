from src.memory.policy import apply_value_memory
from src.memory.storage import load_memory
from src.nodes.intent_parser import constraints_from_intent, parse_intent


def test_child_not_coming_word_order_suppresses_family_defaults() -> None:
    intent = parse_intent(
        "今晚就我和老婆两个人，孩子不带，别再给我排亲子乐园。"
        "18:10从陆家嘴下班，想先吃个热乎但别太辣的晚饭，再去看20:50左右的电影。"
    )

    assert intent["scene"] == "couple"
    assert intent["people_count"] == 2
    assert all(person["role"] != "child" for person in intent["people"])
    assert "亲子" not in intent["planning_preferences"]["activity_type"]
    assert "儿童友好" not in intent["constraints"]["hard"]

    merged = apply_value_memory(constraints_from_intent(intent), load_memory("u001"))

    assert merged["child_age"] is None
    assert "kid_friendly" not in merged["hard_tags"]
    assert "family_care" not in merged["active_value_ids"]


def test_child_away_with_intervening_time_suppresses_family_memory() -> None:
    intent = parse_intent(
        "我老婆刚在九院补完牙出来，孩子今晚放外婆家不带，别再按亲子或减脂轻食那套来。"
        "我们18:10左右从制造局路这边出发，想先吃个软一点、不辣、不喝酒的晚饭，"
        "再去附近安静散散步或看看夜景，21:00前到家。"
    )

    assert intent["scene"] == "couple"
    assert intent["people_count"] == 2
    assert all(person["role"] != "child" for person in intent["people"])
    assert "儿童友好" not in intent["constraints"]["hard"]
    assert "亲子" not in intent["planning_preferences"]["activity_type"]

    merged = apply_value_memory(constraints_from_intent(intent), load_memory("u001"))

    assert merged["child_age"] is None
    assert "kid_friendly" not in merged["hard_tags"]
    assert "family_care" not in merged["active_value_ids"]
