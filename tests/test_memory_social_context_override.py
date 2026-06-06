from src.memory.policy import apply_value_memory, build_user_profile
from src.memory.retrieval import retrieve_relevant_memories
from src.memory.storage import load_memory
from src.nodes.intent_parser import constraints_from_intent, parse_intent


def test_memory_honors_explicit_non_parent_child_non_couple_group_context() -> None:
    text = (
        "这周五晚上两个外地闺蜜到虹桥，我18:30接到人，想安排到周六下午16:20前送回虹桥站。"
        "我们一共3个人，想轻松一点，有上海特色、能拍照聊天，不要按亲子或情侣约会来。"
        "周五晚上先吃饭再看个脱口秀或小剧场，住处希望给个靠近地铁、方便放行李的建议但她们自己订；"
        "周六想早午餐、逛一点有特色的街区，顺路买点伴手礼，时间够的话做个简单美甲。"
        "其中一个乳糖不耐、不喝酒，总预算1800以内不含住宿。"
    )
    memory = load_memory("u001")

    intent = parse_intent(text)
    merged = apply_value_memory(constraints_from_intent(intent), memory)
    retrieved = retrieve_relevant_memories(merged, memory)
    user_profile = build_user_profile(
        merged,
        memory,
        active_value_memory=[],
        retrieved_memories=retrieved,
    )
    retrieved_ids = {item["memory_id"] for item in retrieved}

    assert merged["child_age"] is None
    assert merged["mom_diet"] is None
    assert "kid_friendly" not in merged["hard_tags"]
    assert "儿童友好" not in merged["hard_tags"]
    assert "亲子" not in merged["soft_tags"]
    assert "约会活动" not in merged["soft_tags"]
    assert "浪漫" not in merged["soft_tags"]
    assert "family_care" not in merged["active_value_ids"]
    assert "health" not in merged["active_value_ids"]
    assert "companion_child" not in retrieved_ids
    assert "companion_wife" not in retrieved_ids
    assert "value_family_care" not in retrieved_ids
    assert "value_health" not in retrieved_ids
    assert user_profile["companion_profile"] == {}


def test_memory_does_not_project_child_profile_for_adult_parent_birthday() -> None:
    text = (
        "周六下午帮我安排给妈妈过生日，3个成年人，14点从人民广场附近开始。"
        "她血糖偏高，别安排太甜太咸，也不想去人挤人的商场。"
    )
    memory = load_memory("u001")

    intent = parse_intent(text)
    merged = apply_value_memory(constraints_from_intent(intent), memory)
    retrieved = retrieve_relevant_memories(merged, memory)
    retrieved_ids = {item["memory_id"] for item in retrieved}

    assert intent["people_count"] == 3
    assert merged["child_age"] is None
    assert "kid_friendly" not in merged["hard_tags"]
    assert "family_care" not in merged["active_value_ids"]
    assert "companion_child" not in retrieved_ids
    assert "value_family_care" not in retrieved_ids
