"""Memory projection policy for planner constraints."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from src.memory.schema import MEMORY_POLICY
from src.memory.utils import to_int
from src.state import ValueMemoryItem


def companion_roles(constraints: dict[str, Any]) -> set[str]:
    companions = constraints.get("companions", []) or []
    return {
        str(item.get("role"))
        for item in companions
        if isinstance(item, dict) and item.get("role")
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _current_request_text(constraints: dict[str, Any]) -> str:
    return str(constraints.get("raw_text") or constraints.get("user_input") or "")


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _has_child_away_clause(text: str) -> bool:
    return bool(
        re.search(
            r"(?:孩子|小孩|小朋友|娃|宝宝)"
            r"[^，。；;,.]{0,12}"
            r"(?:放|去|在|留在|交给|托给)"
            r"[^，。；;,.]{0,12}"
            r"(?:外婆|外公|爷爷|奶奶|姥姥|姥爷|长辈|老人|家里)"
            r"[^，。；;,.]{0,10}"
            r"(?:不带|不同行|不一起|不去|不来)?",
            text,
        )
    )


def _rejects_old_preference_term(text: str, terms: tuple[str, ...]) -> bool:
    old_markers = ("以前", "历史", "旧偏好", "平时", "之前", "那种", "那套")
    reject_markers = (
        "不要按",
        "别按",
        "别把",
        "别套",
        "不要套用",
        "不套用",
        "别来",
    )
    for clause in re.split(r"[，。；;,.]", text):
        if (
            _contains_any(clause, terms)
            and _contains_any(clause, old_markers)
            and _contains_any(clause, reject_markers)
        ):
            return True
    return False


_CHILD_CONTEXT_TERMS = (
    "孩子",
    "小孩",
    "小朋友",
    "带娃",
    "亲子",
    "家庭",
    "一家",
    "宝宝",
    "儿童",
    "儿童友好",
)


def _has_explicit_no_child_request(constraints: dict[str, Any]) -> bool:
    text = _current_request_text(constraints)
    return _contains_any(
        text,
        (
            "不带孩子",
            "孩子不带",
            "孩子去外婆家",
            "孩子在外婆家",
            "孩子去了外婆家",
            "孩子放外婆家",
            "孩子交给外婆",
            "不带小孩",
            "小孩不带",
            "不带小朋友",
            "小朋友不带",
            "不带娃",
            "娃不带",
            "不带宝宝",
            "宝宝不带",
            "没带孩子",
            "没有孩子",
            "这次没孩子",
            "这次没有孩子",
            "这次孩子不去",
            "孩子不去",
            "孩子不同行",
            "孩子不一起",
            "不安排亲子",
            "不要亲子",
            "不要按亲子",
            "别安排亲子",
            "别按亲子",
            "别再给我排亲子",
            "不要儿童友好",
            "别按儿童友好",
        ),
    ) or _has_child_away_clause(text) or bool(re.search(r"别把.*亲子.*套", text))


def _has_explicit_no_spouse_request(constraints: dict[str, Any]) -> bool:
    text = _current_request_text(constraints)
    return _contains_any(
        text,
        (
            "不要按情侣",
            "不要按约会",
            "不要情侣约会",
            "别按情侣",
            "别按约会",
            "别安排情侣",
            "别安排约会",
            "不是情侣",
            "不是约会",
            "不要按情侣约会",
        ),
    ) or bool(
        re.search(
            r"(不要按|别按|别安排|不要安排).{0,12}(情侣|约会|对象|伴侣|夫妻)",
            text,
        )
    )


def _has_explicit_no_diet_request(constraints: dict[str, Any]) -> bool:
    text = _current_request_text(constraints)
    if _contains_any(
        text,
        (
            "不用低卡",
            "不要低卡",
            "别低卡",
            "不用减脂",
            "不要减脂",
            "别减脂",
            "不用轻食",
            "不要轻食",
            "别轻食",
            "不用健康餐",
            "不要健康餐",
            "不想被提醒减肥",
            "不想被低卡限制",
            "不想被减脂限制",
            "不想被轻食限制",
            "不想被低卡减脂限制",
        ),
    ):
        return True
    return bool(
        re.search(
            r"(别按|别再按|不要按|别把|不要套用|不套用).{0,18}"
            r"(低卡|轻食|减脂|减肥)",
            text,
        )
        or re.search(
            r"(低卡|轻食|减脂|减肥).{0,18}(别套|别来|不用|不要|不想被提醒)",
            text,
        )
        or re.search(
            r"(?:不想|不愿|不用|不要)[^，。；;,.]{0,8}"
            r"(?:被|受)?[^，。；;,.]{0,8}"
            r"(?:低卡|轻食|减脂|减肥|健康餐)"
            r"[^，。；;,.]{0,8}(?:限制|约束|绑住|影响)",
            text,
        )
    )


def _has_current_child_context(constraints: dict[str, Any]) -> bool:
    roles = companion_roles(constraints)
    if "child" in roles:
        return True
    text = _current_request_text(constraints)
    if _contains_any(text, _CHILD_CONTEXT_TERMS):
        return True
    return constraints.get("scene") == "parent_child"


def _should_suppress_stale_child_context(constraints: dict[str, Any]) -> bool:
    if _has_explicit_no_child_request(constraints):
        return True
    if _has_current_child_context(constraints):
        return False
    return True


def _rejected_current_turn_terms(constraints: dict[str, Any]) -> set[str]:
    text = _current_request_text(constraints)
    rejected: set[str] = set()

    if _should_suppress_stale_child_context(constraints):
        rejected.update(
            {
                "child",
                "kid_friendly",
                "parent_child",
                "family_care",
                "亲子",
                "儿童友好",
                "孩子",
                "儿童",
                "带娃",
            }
        )

    if _has_explicit_no_spouse_request(constraints):
        rejected.update(
            {
                "wife",
                "partner",
                "spouse",
                "couple",
                "date",
                "romantic",
                "约会活动",
                "情侣",
                "情侣约会",
                "对象",
                "伴侣",
                "夫妻",
                "浪漫",
            }
        )

    rejected_old_preference = _contains_any(
        text,
        ("以前", "历史", "旧偏好", "平时", "之前", "上次", "那种", "那套"),
    ) and (
        _contains_any(
            text,
            ("不要按", "别按", "别再按", "别把", "别套", "不要套用", "不套用"),
        )
        or bool(re.search(r"这次.*别.*来", text))
    )

    if _has_explicit_no_diet_request(constraints) or (
        rejected_old_preference
        and _contains_any(text, ("低卡", "轻食", "减脂", "健康餐", "减肥"))
    ):
        rejected.update(
            {
                "low_calorie",
                "light_food",
                "health",
                "低卡",
                "轻食",
                "减脂",
                "减肥",
                "健康餐",
            }
        )

    if rejected_old_preference or _contains_any(text, ("不要KTV", "别KTV", "不唱歌")):
        if _contains_any(text, ("KTV", "唱歌")):
            rejected.update(
                {
                    "karaoke",
                    "KTV欢唱",
                    "多人活动",
                    "多人友好",
                    "group_friendly",
                    "group_activity",
                    "social",
                    "社交",
                    "热闹",
                }
            )
    if rejected_old_preference and _rejects_old_preference_term(text, ("火锅",)):
        rejected.update({"hotpot", "火锅"})
    if rejected_old_preference and _rejects_old_preference_term(text, ("烤肉", "烧烤")):
        rejected.update({"bbq", "烤肉", "烧烤"})
    if _contains_any(text, ("安静", "不吵", "别太吵", "不要太吵")):
        rejected.update({"热闹", "noisy", "lively"})

    return rejected


def _remove_rejected_terms(values: Any, rejected: set[str], avoid: set[str]) -> list[Any]:
    return [
        value
        for value in _as_list(values)
        if str(value).strip() not in rejected and str(value).strip() not in avoid
    ]


def _sanitize_current_request_exclusions(merged: dict[str, Any]) -> None:
    rejected = _rejected_current_turn_terms(merged)
    if not rejected:
        avoid = {str(item).strip() for item in _as_list(merged.get("avoid")) if item}
        merged["hard_tags"] = _remove_rejected_terms(
            merged.get("hard_tags"),
            set(),
            avoid,
        )
        merged["soft_tags"] = _remove_rejected_terms(
            merged.get("soft_tags"),
            set(),
            avoid,
        )
        return

    merged["avoid"] = _as_list(merged.get("avoid")) + sorted(rejected)
    avoid = {str(item).strip() for item in _as_list(merged.get("avoid")) if item}
    merged["hard_tags"] = _remove_rejected_terms(
        merged.get("hard_tags"),
        rejected,
        avoid,
    )
    merged["soft_tags"] = _remove_rejected_terms(
        merged.get("soft_tags"),
        rejected,
        avoid,
    )

    planning_preferences = deepcopy(merged.get("planning_preferences") or {})
    if isinstance(planning_preferences, dict):
        for key in (
            "activity_type",
            "food_type",
            "restaurant_type",
            "atmosphere_type",
            "emotion_type",
            "facility_type",
        ):
            if key in planning_preferences:
                planning_preferences[key] = _remove_rejected_terms(
                    planning_preferences.get(key),
                    rejected,
                    avoid,
                )
        merged["planning_preferences"] = planning_preferences


def has_child_request(constraints: dict[str, Any]) -> bool:
    if _should_suppress_stale_child_context(constraints):
        return False
    roles = companion_roles(constraints)
    if "child" in roles:
        return True
    if constraints.get("child_age") in (None, ""):
        return False
    return _has_current_child_context(constraints)


def has_spouse_request(constraints: dict[str, Any]) -> bool:
    if _has_explicit_no_spouse_request(constraints):
        return False
    roles = companion_roles(constraints)
    return bool(roles.intersection({"wife", "partner", "spouse"})) or constraints.get(
        "mom_diet"
    ) not in (None, "")


def has_family_memory_context(constraints: dict[str, Any]) -> bool:
    return (
        has_child_request(constraints) or constraints.get("scene") == "family"
    ) and not _should_suppress_stale_child_context(constraints)


def current_child_age(constraints: dict[str, Any], memory: dict[str, Any]) -> Any:
    if constraints.get("child_age") not in (None, ""):
        return constraints.get("child_age")
    return memory.get("companion_profile", {}).get("child", {}).get("age")


def current_mom_diet(constraints: dict[str, Any], memory: dict[str, Any]) -> str | None:
    if _has_explicit_no_diet_request(constraints):
        return None

    if constraints.get("mom_diet") not in (None, ""):
        return constraints.get("mom_diet")

    spouse_profile = memory.get("companion_profile", {}).get("wife", {})
    if spouse_profile.get("state") == "dieting":
        return "low_calorie"

    return None


def _has_explicit_non_family_group(constraints: dict[str, Any]) -> bool:
    """Return true when current input names a social group without family roles."""

    scene = constraints.get("scene")
    roles = companion_roles(constraints)
    return scene in {"friends", "group"} and not roles.intersection(
        {"child", "wife", "partner", "spouse"}
    )


def _profile_companion_projection(
    constraints: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    companion_profile = memory.get("companion_profile", {}) or {}
    projected: dict[str, Any] = {}

    if has_child_request(constraints) or has_family_memory_context(constraints):
        child_profile = companion_profile.get("child")
        if isinstance(child_profile, dict):
            projected["child"] = deepcopy(child_profile)

    if has_spouse_request(constraints):
        for role in ("wife", "partner", "spouse"):
            spouse_profile = companion_profile.get(role)
            if isinstance(spouse_profile, dict):
                projected[role] = deepcopy(spouse_profile)

    return projected


def _profile_preference_projection(
    constraints: dict[str, Any],
    memory: dict[str, Any],
) -> tuple[list[Any], list[Any]]:
    preference_profile = memory.get("preference_profile", {}) or {}
    planning_preferences = constraints.get("planning_preferences", {}) or {}

    if _has_explicit_non_family_group(constraints):
        food_preference: list[Any] = []
        activity_preference: list[Any] = []
    else:
        food_preference = list(preference_profile.get("food", []) or [])
        activity_preference = list(preference_profile.get("activity", []) or [])

    food_preference.extend(planning_preferences.get("food_type", []) or [])
    activity_preference.extend(planning_preferences.get("activity_type", []) or [])
    return food_preference, activity_preference


def select_active_value_memory(
    constraints: dict[str, Any],
    memory: dict[str, Any],
) -> list[ValueMemoryItem]:
    value_memory = memory.get("value_profile", [])
    has_child = has_child_request(constraints)
    has_spouse = has_spouse_request(constraints)
    mom_diet = current_mom_diet(constraints, memory) if has_spouse else None
    active_value_ids = {"convenience", "cost_sensitivity"}

    if has_child or has_family_memory_context(constraints):
        active_value_ids.add("family_care")

    soft_tags = constraints.get("soft_tags", []) or []
    if mom_diet == "low_calorie" or any(
        tag in soft_tags for tag in ("low_calorie", "light_food")
    ):
        active_value_ids.add("health")

    return [item for item in value_memory if item.get("value_id") in active_value_ids]


def build_user_profile(
    constraints: dict[str, Any],
    memory: dict[str, Any],
    active_value_memory: list[ValueMemoryItem],
    retrieved_memories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    food_preference, activity_preference = _profile_preference_projection(
        constraints,
        memory,
    )

    return {
        "user_id": memory.get("user_id"),
        "stable_profile": memory.get("stable_profile", {}),
        "companion_profile": _profile_companion_projection(constraints, memory),
        "companions": constraints.get("companions", []) or [],
        "people_count": constraints.get("people_count"),
        "food_preference": sorted(set(food_preference)),
        "activity_preference": sorted(set(activity_preference)),
        "avoid": constraints.get("avoid", []) or [],
        "value_profile": active_value_memory,
        "retrieved_memory_ids": [
            item.get("memory_id") for item in (retrieved_memories or [])
        ],
    }


def apply_value_memory(
    constraints: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    """Merge memory into calculable planner constraints.

    Current user input remains authoritative. Memory fills defaults, adds
    preferences, and exposes value weights for later ranking.
    """

    merged = deepcopy(constraints)
    merged.setdefault("avoid", [])
    merged.setdefault("hard_tags", [])
    merged.setdefault("soft_tags", [])
    merged.setdefault("companions", [])

    preference_profile = memory.get("preference_profile", {})
    for avoided in preference_profile.get("avoid", []):
        if avoided not in merged["avoid"]:
            merged["avoid"].append(avoided)
    _sanitize_current_request_exclusions(merged)

    child_profile = memory.get("companion_profile", {}).get("child", {})
    if has_child_request(merged):
        child_age = current_child_age(merged, memory)
        merged["child_age"] = child_age
    else:
        child_age = None
        merged["child_age"] = None

    child_age_value = to_int(child_age)
    if child_age_value is not None and child_profile and child_age_value <= 6:
        if "kid_friendly" not in merged["hard_tags"]:
            merged["hard_tags"].append("kid_friendly")

    if has_spouse_request(merged):
        mom_diet = current_mom_diet(merged, memory)
        merged["mom_diet"] = mom_diet
    else:
        mom_diet = None
        merged["mom_diet"] = None

    if mom_diet == "low_calorie":
        for tag in ("low_calorie", "light_food"):
            if tag not in merged["soft_tags"]:
                merged["soft_tags"].append(tag)
        _sanitize_current_request_exclusions(merged)

    defaults = memory.get("derived_defaults", {})
    if merged.get("max_distance_km") in (None, ""):
        merged["max_distance_km"] = defaults.get(
            "max_distance_km",
            15.0,
        )
    if merged.get("transport_mode") in (None, "", "unknown"):
        merged["transport_mode"] = memory.get("stable_profile", {}).get(
            "default_transport",
            "unknown",
        )

    value_memory = select_active_value_memory(merged, memory)
    value_weights = {
        item["value_id"]: round(float(item["score"]), 2) for item in value_memory
    }
    value_confidence = {
        item["value_id"]: round(float(item["confidence"]), 2) for item in value_memory
    }
    merged["value_weights"] = value_weights
    merged["value_confidence"] = value_confidence
    merged["score_weights"] = {
        "group_fit": 0.30 + 0.10 * value_weights.get("family_care", 0.0),
        "availability": 0.20 + 0.05 * value_weights.get("convenience", 0.0),
        "route": 0.20 + 0.05 * value_weights.get("convenience", 0.0),
        "health": 0.10 + 0.10 * value_weights.get("health", 0.0),
        "budget": 0.10 + 0.05 * value_weights.get("cost_sensitivity", 0.0),
        "experience": 0.10,
    }
    current_queue_limit = merged.get("max_queue_time_min")
    if current_queue_limit in (None, ""):
        current_queue_limit = merged.get("max_queue_time")
    if current_queue_limit in (None, ""):
        current_queue_limit = (
            defaults.get("max_queue_time_min", 15)
            if "long_queue" in merged["avoid"]
            else 30
        )
    merged["max_queue_time_min"] = current_queue_limit
    merged["max_queue_time"] = current_queue_limit
    merged["active_value_ids"] = sorted(value_weights)
    merged["avoid"] = sorted(set(merged["avoid"]))
    merged["hard_tags"] = sorted(set(merged["hard_tags"]))
    merged["soft_tags"] = sorted(set(merged["soft_tags"]))
    merged["memory_policy"] = MEMORY_POLICY
    return merged
