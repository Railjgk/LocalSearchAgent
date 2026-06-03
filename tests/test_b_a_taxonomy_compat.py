from src.nodes.b_utils import (
    collect_preference_sources,
    collect_tag_fields,
    expand_preference_tags,
    get_constraint_config_with_profile,
)


def test_b_accepts_chinese_first_a_handoff_tags() -> None:
    constraints = {
        "people_count": 3,
        "budget": 200,
        "budget_type": "per_person",
        "raw_text": "妻子最近减脂，想吃轻食",
        "hard_tags": ["亲子"],
        "soft_tags": ["室内", "轻食"],
        "avoid": ["排队久", "商场拥挤"],
        "planning_preferences": {
            "activity_type": ["亲子", "室内"],
            "food_type": ["轻食"],
            "emotion_type": ["放松"],
        },
    }

    config = get_constraint_config_with_profile(constraints, {})
    preference_sources = collect_preference_sources(constraints, {}, [])

    assert config["budget"] == 600
    assert config["mom_diet"] == "low_calorie"
    assert "kid_friendly" in preference_sources
    assert "indoor" in preference_sources
    assert "light_food" in preference_sources
    assert "relaxation" in preference_sources
    assert "long_queue" in expand_preference_tags(constraints["avoid"])
    assert "crowded_mall" in expand_preference_tags(constraints["avoid"])


def test_b_accepts_split_cn_tag_fields_from_a() -> None:
    constraints = {
        "hard": ["dine_in"],
        "hard_cn": ["亲子"],
        "soft": ["nearby"],
        "soft_cn": ["低强度", "轻食"],
        "avoid_cn": ["高热量", "外带"],
    }

    assert "dine_in" in collect_tag_fields(constraints, "hard")
    assert "kid_friendly" in collect_tag_fields(constraints, "hard")
    assert "nearby" in collect_tag_fields(constraints, "soft")
    assert "low_intensity" in collect_tag_fields(constraints, "soft")
    assert "light_food" in collect_tag_fields(constraints, "soft")
    assert "high_calorie" in collect_tag_fields(constraints, "avoid")
    assert "takeaway_only" in collect_tag_fields(constraints, "avoid")


def test_light_food_tag_alone_does_not_force_low_calorie_diet() -> None:
    constraints = {
        "soft_tags": ["light_food"],
        "planning_preferences": {"food_type": ["light_food"]},
    }

    config = get_constraint_config_with_profile(constraints, {})

    assert config["mom_diet"] in (None, "")


def test_solo_request_does_not_inherit_spouse_diet_memory() -> None:
    constraints = {"people_count": 1}
    user_profile = {
        "companion_profile": {
            "wife": {
                "state": "dieting",
                "needs": ["low_calorie", "light_food"],
            }
        }
    }

    config = get_constraint_config_with_profile(constraints, user_profile)

    assert config["people_count"] == 1
    assert config["mom_diet"] in (None, "")
