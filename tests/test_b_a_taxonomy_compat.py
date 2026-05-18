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
        "hard_tags": ["????"],
        "soft_tags": ["??", "??"],
        "avoid": ["???", "????"],
        "planning_preferences": {
            "activity_type": ["??", "??"],
            "food_type": ["??"],
            "emotion_type": ["??"],
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
        "hard_cn": ["????"],
        "soft": ["nearby"],
        "soft_cn": ["???", "??"],
        "avoid_cn": ["???", "???"],
    }

    assert "dine_in" in collect_tag_fields(constraints, "hard")
    assert "kid_friendly" in collect_tag_fields(constraints, "hard")
    assert "nearby" in collect_tag_fields(constraints, "soft")
    assert "low_intensity" in collect_tag_fields(constraints, "soft")
    assert "light_food" in collect_tag_fields(constraints, "soft")
    assert "high_calorie" in collect_tag_fields(constraints, "avoid")
    assert "takeaway_only" in collect_tag_fields(constraints, "avoid")
