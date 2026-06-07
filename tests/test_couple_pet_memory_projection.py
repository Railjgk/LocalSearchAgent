from src.memory.policy import (
    apply_value_memory,
    build_user_profile,
    select_active_value_memory,
)
from src.memory.retrieval import retrieve_relevant_memories
from src.memory.storage import load_memory
from src.nodes.intent_parser import constraints_from_intent, parse_intent
from src.nodes.scenario_planner import build_scenario_plan


def _project_profile(text: str) -> tuple[dict, dict, set[str], dict]:
    memory = load_memory("u001")
    intent = parse_intent(text)
    constraints = apply_value_memory(constraints_from_intent(intent), memory)
    retrieved = retrieve_relevant_memories(constraints, memory)
    user_profile = build_user_profile(
        constraints,
        memory,
        select_active_value_memory(constraints, memory),
        retrieved,
    )
    scenario_plan = build_scenario_plan(
        {"user_input": text, "intent": intent, "constraints": constraints}
    )
    return (
        constraints,
        user_profile,
        {item["memory_id"] for item in retrieved},
        scenario_plan,
    )


def test_couple_pet_anniversary_filters_stale_child_activity_memory() -> None:
    text = (
        "这个周末想和对象过纪念日，两天一夜，周六中午出发、周日16点前结束，"
        "上海周边或市内都行，我们自驾，会带一只小狗。希望有点仪式感但别太吵，"
        "伴侣海鲜过敏，预算总共1600以内最好含住宿和吃饭。要考虑停车、"
        "宠物友好、下雨备选；如果宠物或住宿订不了，不要硬说搞定，"
        "要告诉我怎么调整。"
    )

    constraints, user_profile, retrieved_ids, scenario_plan = _project_profile(text)

    assert constraints["scene"] == "couple"
    assert constraints["people_count"] == 2
    assert constraints["active_value_ids"] == [
        "convenience",
        "cost_sensitivity",
        "health",
    ]
    assert "companion_wife" in retrieved_ids
    assert "value_health" in retrieved_ids
    assert "companion_child" not in retrieved_ids
    assert "value_family_care" not in retrieved_ids

    activity_preference = set(user_profile["activity_preference"])
    assert not activity_preference.intersection(
        {"parent_child", "kid_friendly", "亲子", "儿童友好"}
    )
    assert {"约会活动", "宠物友好", "停车方便"}.issubset(activity_preference)

    assert {"过敏友好", "避开海鲜"}.issubset(user_profile["food_preference"])
    assert {
        "约会活动",
        "浪漫",
        "仪式感",
        "宠物友好",
        "停车方便",
        "过敏友好",
        "避开海鲜",
    }.issubset(set(scenario_plan["scenario_activities"]))


def test_family_request_keeps_current_child_accessibility_semantics() -> None:
    text = (
        "临时改计划了，今天16:20以后从杨浦五角场出发，带6岁孩子和膝盖不太好的"
        "外婆出去透口气，最好有个不用排很久的室内活动，再吃个清淡晚饭；"
        "孩子坚果过敏，外婆不能走太多，19:30前要回到家附近，总预算600以内，"
        "能预约就先帮我稳住。"
    )

    constraints, user_profile, retrieved_ids, scenario_plan = _project_profile(text)

    assert constraints["scene"] == "family"
    assert "family_care" in constraints["active_value_ids"]
    assert "companion_child" in retrieved_ids
    assert "value_family_care" in retrieved_ids

    activity_preference = set(user_profile["activity_preference"])
    assert {"亲子", "低强度", "室内", "少步行"}.issubset(activity_preference)
    assert "parent_child" in activity_preference
    assert {"儿童友好", "低强度", "少步行"}.issubset(constraints["hard_tags"])
    assert {"亲子", "儿童友好", "过敏友好", "避开坚果"}.issubset(
        set(scenario_plan["scenario_activities"])
    )
