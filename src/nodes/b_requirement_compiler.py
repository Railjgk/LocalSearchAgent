"""B-stage requirement compiler for product guardrails.

The compiler turns fuzzy local-life requests into a small, auditable contract
that deterministic B filters can enforce.  LongCat can enrich the contract when
enabled, but the fallback rule compiler always runs and never depends on a
remote model.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from src.nodes.longcat_client import (
    TRUTHY_VALUES,
    chat_completion,
    is_b_ai_enabled,
    load_longcat_config,
    sanitize_longcat_error,
)

try:
    from src.state import PlanState
except ImportError:  # pragma: no cover
    PlanState = dict


B_REQUIREMENT_COMPILER_SYSTEM_PROMPT = (
    "You are WeekendFlow's B-stage requirement compiler. "
    "A has already parsed the user intent; B needs an auditable planning contract. "
    "Return JSON only. Do not invent merchants, prices, distances, ratings, inventory, or memory. "
    "Only set a requirement when the input clearly supports it. "
    "Allowed schema: {"
    "\"hard_requirements\":[\"child_friendly_activity|cafe_non_full_meal|halal_restaurant|"
    "pet_friendly|elder_friendly|parking_needed|late_night_open|restaurant_reservation\"],"
    "\"forbidden_restaurant_groups\":[\"火锅|烤肉|正餐\"],"
    "\"soft_preferences\":[\"...\"],"
    "\"needs_confirmation\":[\"...\"],"
    "\"evidence\":[\"...\"]"
    "}."
)

ALLOWED_HARD_REQUIREMENTS = {
    "child_friendly_activity",
    "cafe_non_full_meal",
    "halal_restaurant",
    "pet_friendly",
    "elder_friendly",
    "parking_needed",
    "late_night_open",
    "restaurant_reservation",
}

ALLOWED_FORBIDDEN_RESTAURANT_GROUPS = {"火锅", "烤肉", "正餐"}

CHILD_COMPANION_TERMS = (
    "孩子",
    "小孩",
    "小朋友",
    "儿童",
    "亲子",
    "宝宝",
    "带娃",
    "家庭",
)


def _env_mapping(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if env is None else env


def is_b_requirement_compiler_enabled(env: Mapping[str, str] | None = None) -> bool:
    env = _env_mapping(env)
    raw_value = env.get("WF_B_AI_REQUIREMENT_COMPILER_ENABLED")
    if raw_value is None:
        return is_b_ai_enabled(env)
    return raw_value.strip().lower() in TRUTHY_VALUES


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


def _dedupe_keep_order(values: list[Any], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _parse_jsonish(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _collect_text_sources(state: PlanState, constraints: dict[str, Any] | None = None) -> list[str]:
    constraints = constraints or state.get("constraints", {}) or {}
    user_profile = state.get("user_profile", {}) or {}
    planning_preferences = constraints.get("planning_preferences", {}) or {}

    values: list[Any] = [
        state.get("user_input"),
        constraints.get("raw_text"),
        state.get("scene_type"),
        constraints.get("scene"),
        constraints.get("mom_diet"),
    ]
    for key in (
        "hard_tags",
        "soft_tags",
        "avoid",
        "companions",
        "scenario_activities",
    ):
        values.extend(_as_list(constraints.get(key)))
    for key in (
        "activity_type",
        "food_type",
        "restaurant_type",
        "atmosphere_type",
        "facility_type",
        "emotion_type",
    ):
        values.extend(_as_list(planning_preferences.get(key)))
    for key in ("food_preference", "activity_preference", "avoid"):
        values.extend(_as_list(user_profile.get(key)))
    values.extend(_as_list(state.get("scenario_activities")))
    return _dedupe_keep_order(values, limit=80)


def _joined_text(state: PlanState, constraints: dict[str, Any] | None = None) -> str:
    return " ".join(_collect_text_sources(state, constraints))


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _has_child_companion_signal(text: str, scene: str) -> bool:
    return (
        scene == "family"
        or "family" in scene.lower()
        or _contains_any(scene, ("家庭", "亲子"))
        or _contains_any(text, CHILD_COMPANION_TERMS)
    )


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_time(value: Any) -> str:
    text = str(value or "").strip()
    if ":" in text:
        parts = text.split(":", 1)
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError:
            return ""
        return f"{hour:02d}:{minute:02d}"
    return ""


def _time_to_minutes(value: str) -> int | None:
    text = _normalize_time(value)
    if not text:
        return None
    hour, minute = text.split(":", 1)
    return int(hour) * 60 + int(minute)


def _deterministic_contract(state: PlanState, constraints: dict[str, Any]) -> dict[str, Any]:
    text = _joined_text(state, constraints)
    child_age = _to_int(constraints.get("child_age"))
    people_count = _to_int(constraints.get("people_count"))
    scene = str(state.get("scene_type") or constraints.get("scene") or "")
    hard_requirements: list[str] = []
    forbidden_groups: list[str] = []
    soft_preferences: list[str] = []
    needs_confirmation: list[str] = []
    evidence: list[str] = []

    if _has_child_companion_signal(text, scene):
        hard_requirements.append("child_friendly_activity")
        soft_preferences.extend(["适龄", "少走路", "安全", "低强度"])
        evidence.append("同行人或场景包含低龄儿童/亲子需求")

    if _contains_any(text, ("咖啡", "甜品", "下午茶", "小坐")) and _contains_any(
        text,
        ("不想吃正餐", "不吃正餐", "不想正餐", "坐一会", "坐一会儿"),
    ):
        hard_requirements.append("cafe_non_full_meal")
        forbidden_groups.append("正餐")
        evidence.append("用户明确表达咖啡小坐且不想吃正餐")

    if _contains_any(text, ("清真", "halal", "穆斯林")):
        hard_requirements.append("halal_restaurant")
        evidence.append("用户提出清真/halal 饮食限制")

    if _contains_any(text, ("带狗", "狗狗", "宠物", "猫狗", "可带宠物", "宠物友好")):
        hard_requirements.append("pet_friendly")
        needs_confirmation.append("宠物友好信息通常依赖商家实时确认")
        evidence.append("用户提出宠物同行需求")

    if _contains_any(text, ("老人", "爸妈", "父母", "走不动", "少走路", "有座位", "无障碍")):
        hard_requirements.append("elder_friendly")
        soft_preferences.extend(["少走路", "有座位", "低强度"])
        evidence.append("用户提出老人/低体力需求")

    if _contains_any(text, ("停车", "免费停车", "好停车")):
        hard_requirements.append("parking_needed")
        needs_confirmation.append("停车信息需要执行前复核")
        evidence.append("用户提出停车需求")

    start_minutes = _time_to_minutes(constraints.get("start_time"))
    if start_minutes is not None and start_minutes >= 21 * 60:
        hard_requirements.append("late_night_open")
        evidence.append("用户开始时间较晚，需要营业时间校验")
    if _contains_any(text, ("夜宵", "十点后", "晚上十点", "凌晨")):
        hard_requirements.append("late_night_open")
        evidence.append("用户提出夜宵/深夜可营业需求")

    if _contains_any(text, ("订座", "预约", "可订", "不要等位")) or (people_count is not None and people_count >= 4):
        hard_requirements.append("restaurant_reservation")

    if _contains_any(
        text,
        (
            "不要火锅",
            "别火锅",
            "别推荐火锅",
            "不要推荐火锅",
            "不推荐火锅",
            "不吃火锅",
            "避开火锅",
        ),
    ):
        forbidden_groups.append("火锅")
    if _contains_any(
        text,
        (
            "不要烤肉",
            "别烤肉",
            "别推荐烤肉",
            "不要推荐烤肉",
            "不推荐烤肉",
            "不吃烤肉",
            "不要烧烤",
            "别烧烤",
            "别推荐烧烤",
            "不要推荐烧烤",
            "不推荐烧烤",
            "不吃烧烤",
            "避开烧烤",
        ),
    ):
        forbidden_groups.append("烤肉")

    return {
        "version": "b_requirement_contract_v1",
        "source": "deterministic",
        "hard_requirements": sorted(set(hard_requirements)),
        "forbidden_restaurant_groups": _dedupe_keep_order(forbidden_groups, limit=6),
        "soft_preferences": _dedupe_keep_order(soft_preferences, limit=12),
        "needs_confirmation": _dedupe_keep_order(needs_confirmation, limit=8),
        "evidence": _dedupe_keep_order(evidence, limit=8),
    }


def _normalize_llm_contract(raw_payload: dict[str, Any]) -> dict[str, Any]:
    hard_requirements = [
        item
        for item in _dedupe_keep_order(_as_list(raw_payload.get("hard_requirements")), limit=12)
        if item in ALLOWED_HARD_REQUIREMENTS
    ]
    forbidden_groups = [
        item
        for item in _dedupe_keep_order(_as_list(raw_payload.get("forbidden_restaurant_groups")), limit=6)
        if item in ALLOWED_FORBIDDEN_RESTAURANT_GROUPS
    ]
    return {
        "hard_requirements": hard_requirements,
        "forbidden_restaurant_groups": forbidden_groups,
        "soft_preferences": _dedupe_keep_order(_as_list(raw_payload.get("soft_preferences")), limit=12),
        "needs_confirmation": _dedupe_keep_order(_as_list(raw_payload.get("needs_confirmation")), limit=8),
        "evidence": _dedupe_keep_order(_as_list(raw_payload.get("evidence")), limit=8),
    }


def _supported_hard_requirement(requirement: str, text: str, constraints: dict[str, Any], state: PlanState) -> bool:
    child_age = _to_int(constraints.get("child_age"))
    people_count = _to_int(constraints.get("people_count"))
    scene = str(state.get("scene_type") or constraints.get("scene") or "")
    if requirement == "child_friendly_activity":
        return _has_child_companion_signal(text, scene)
    if requirement == "cafe_non_full_meal":
        return _contains_any(text, ("咖啡", "甜品", "下午茶", "小坐")) and _contains_any(
            text,
            ("不想吃正餐", "不吃正餐", "不想正餐", "坐一会", "坐一会儿"),
        )
    if requirement == "halal_restaurant":
        return _contains_any(text, ("清真", "halal", "穆斯林"))
    if requirement == "pet_friendly":
        return _contains_any(text, ("带狗", "狗狗", "宠物", "猫狗", "可带宠物", "宠物友好"))
    if requirement == "elder_friendly":
        return _contains_any(text, ("老人", "爸妈", "父母", "走不动", "少走路", "有座位", "无障碍"))
    if requirement == "parking_needed":
        return _contains_any(text, ("停车", "免费停车", "好停车"))
    if requirement == "late_night_open":
        start_minutes = _time_to_minutes(constraints.get("start_time"))
        return (
            (start_minutes is not None and start_minutes >= 21 * 60)
            or _contains_any(text, ("夜宵", "十点后", "晚上十点", "凌晨"))
        )
    if requirement == "restaurant_reservation":
        return (
            _contains_any(text, ("订座", "预约", "可订", "不要等位"))
            or (people_count is not None and people_count >= 4)
        )
    return False


def _supported_forbidden_group(group: str, text: str) -> bool:
    if group == "正餐":
        return _contains_any(text, ("不想吃正餐", "不吃正餐", "不想正餐"))
    if group == "火锅":
        return _contains_any(
            text,
            (
                "不要火锅",
                "别火锅",
                "别推荐火锅",
                "不要推荐火锅",
                "不推荐火锅",
                "不吃火锅",
                "避开火锅",
            ),
        )
    if group == "烤肉":
        return _contains_any(
            text,
            (
                "不要烤肉",
                "别烤肉",
                "别推荐烤肉",
                "不要推荐烤肉",
                "不推荐烤肉",
                "不吃烤肉",
                "不要烧烤",
                "别烧烤",
                "别推荐烧烤",
                "不要推荐烧烤",
                "不推荐烧烤",
                "不吃烧烤",
                "避开烧烤",
            ),
        )
    return False


def _guard_llm_contract(
    llm_contract: dict[str, Any],
    *,
    base_contract: dict[str, Any],
    state: PlanState,
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """Keep LongCat as an enrichment layer, but reject unsupported hard guards."""

    text = _joined_text(state, constraints)
    base_hard = set(base_contract.get("hard_requirements", []))
    base_forbidden = set(base_contract.get("forbidden_restaurant_groups", []))
    hard_requirements = [
        item
        for item in llm_contract.get("hard_requirements", []) or []
        if item in base_hard or _supported_hard_requirement(item, text, constraints, state)
    ]
    forbidden_groups = [
        item
        for item in llm_contract.get("forbidden_restaurant_groups", []) or []
        if item in base_forbidden or _supported_forbidden_group(str(item), text)
    ]
    guarded = dict(llm_contract)
    guarded["hard_requirements"] = hard_requirements
    guarded["forbidden_restaurant_groups"] = forbidden_groups
    return guarded


def generate_b_requirement_contract(
    state: PlanState,
    constraints: dict[str, Any],
    *,
    allow_llm: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    base_contract = _deterministic_contract(state, constraints)
    if not allow_llm:
        return base_contract, {
            "enabled": False,
            "provider": "longcat",
            "success": False,
            "skipped": True,
            "reason": "deterministic_only",
            "deterministic_contract": base_contract,
        }

    env = _env_mapping()
    if not is_b_requirement_compiler_enabled(env):
        return base_contract, None

    config = load_longcat_config(env)
    if config is None:
        return base_contract, {
            "enabled": True,
            "provider": "longcat",
            "success": False,
            "fallback": True,
            "reason": "missing_api_key",
            "deterministic_contract": base_contract,
        }

    payload = {
        "user_input": state.get("user_input"),
        "scene_type": state.get("scene_type"),
        "constraints": constraints,
        "user_profile": state.get("user_profile", {}) or {},
        "scenario_activities": state.get("scenario_activities", []) or [],
        "deterministic_contract": base_contract,
    }
    messages = [
        {"role": "system", "content": B_REQUIREMENT_COMPILER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]

    try:
        response = chat_completion(messages, config=config)
        llm_contract = _normalize_llm_contract(_parse_jsonish(response["content"]))
        llm_contract = _guard_llm_contract(
            llm_contract,
            base_contract=base_contract,
            state=state,
            constraints=constraints,
        )
    except Exception as exc:
        return base_contract, {
            "enabled": True,
            "provider": "longcat",
            "model": config.model,
            "success": False,
            "fallback": True,
            "error_type": type(exc).__name__,
            "error": sanitize_longcat_error(exc)[:300],
            "deterministic_contract": base_contract,
        }

    merged_contract = {
        "version": "b_requirement_contract_v1",
        "source": "deterministic+longcat",
        "hard_requirements": sorted(
            set(base_contract.get("hard_requirements", []))
            | set(llm_contract.get("hard_requirements", []))
        ),
        "forbidden_restaurant_groups": _dedupe_keep_order(
            base_contract.get("forbidden_restaurant_groups", [])
            + llm_contract.get("forbidden_restaurant_groups", []),
            limit=8,
        ),
        "soft_preferences": _dedupe_keep_order(
            base_contract.get("soft_preferences", [])
            + llm_contract.get("soft_preferences", []),
            limit=16,
        ),
        "needs_confirmation": _dedupe_keep_order(
            base_contract.get("needs_confirmation", [])
            + llm_contract.get("needs_confirmation", []),
            limit=10,
        ),
        "evidence": _dedupe_keep_order(
            base_contract.get("evidence", [])
            + llm_contract.get("evidence", []),
            limit=10,
        ),
    }
    return merged_contract, {
        "enabled": True,
        "provider": "longcat",
        "model": response.get("model") or config.model,
        "success": True,
        "fallback": False,
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage", {}),
        "contract": merged_contract,
    }


def apply_b_requirement_contract(
    state: PlanState,
    *,
    constraints: dict[str, Any],
    allow_llm: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Attach a B planning contract to constraints for downstream filters."""

    existing = constraints.get("b_requirement_contract")
    if isinstance(existing, dict) and existing:
        return constraints, existing, None

    contract, metadata = generate_b_requirement_contract(
        state,
        constraints,
        allow_llm=allow_llm,
    )
    enhanced_constraints = dict(constraints)
    enhanced_constraints["b_requirement_contract"] = contract

    planning_preferences = dict(enhanced_constraints.get("planning_preferences") or {})
    if "child_friendly_activity" in contract.get("hard_requirements", []):
        activity_type = _dedupe_keep_order(
            _as_list(planning_preferences.get("activity_type")) + ["亲子", "儿童友好"],
            limit=12,
        )
        planning_preferences["activity_type"] = activity_type
    if "cafe_non_full_meal" in contract.get("hard_requirements", []):
        food_type = _dedupe_keep_order(
            _as_list(planning_preferences.get("food_type")) + ["咖啡", "甜品", "下午茶"],
            limit=12,
        )
        planning_preferences["food_type"] = food_type
    if "halal_restaurant" in contract.get("hard_requirements", []):
        food_type = _dedupe_keep_order(
            _as_list(planning_preferences.get("food_type")) + ["清真"],
            limit=12,
        )
        planning_preferences["food_type"] = food_type
    enhanced_constraints["planning_preferences"] = planning_preferences
    return enhanced_constraints, contract, metadata
