"""B-stage itinerary blueprint extraction.

This module does not retrieve or rank POIs. It only turns a Chinese local-life
request into a small planning skeleton so B can distinguish the current legacy
half-day pair planner from richer multi-node itinerary work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    from src.state import PlanState
except ImportError:  # pragma: no cover
    PlanState = dict


@dataclass(frozen=True)
class RoleDefinition:
    role: str
    label: str
    supply_domain: str
    keywords: tuple[str, ...]
    default_duration_min: int
    current_support: str


ROLE_DEFINITIONS: tuple[RoleDefinition, ...] = (
    RoleDefinition(
        role="lodging",
        label="住宿",
        supply_domain="hotel",
        keywords=("酒店", "住宿", "民宿", "住一晚", "住处", "家庭房", "套房", "有厨房", "过夜", "住酒店"),
        default_duration_min=720,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="exhibition",
        label="展览/博物馆",
        supply_domain="activity",
        keywords=(
            "展览",
            "看展",
            "博物馆",
            "美术馆",
            "画展",
            "展厅",
            "特展",
            "艺术展",
            "文物展",
            "摄影展",
            "珍品展",
            "漆器展",
        ),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="family_activity",
        label="亲子活动",
        supply_domain="activity",
        keywords=("亲子", "孩子", "小朋友", "儿童", "带娃", "游乐园", "儿童乐园"),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="family_indoor_play",
        label="室内亲子乐园",
        supply_domain="activity",
        keywords=("室内乐园", "儿童游乐", "室内游乐", "淘气堡", "蹦床乐园"),
        default_duration_min=150,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="citywalk_market",
        label="城市漫步/市集",
        supply_domain="activity",
        keywords=("citywalk", "城市漫步", "步行街", "市集", "逛街", "本地市集"),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="restaurant_breakfast",
        label="早餐",
        supply_domain="restaurant",
        keywords=("早餐", "早饭", "早上吃", "包子", "馄饨", "豆浆", "粥店", "早点"),
        default_duration_min=45,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="restaurant_lunch",
        label="午餐",
        supply_domain="restaurant",
        keywords=("中午", "中饭", "午饭", "午餐", "吃个中饭", "吃午饭"),
        default_duration_min=70,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="restaurant_dinner",
        label="晚餐",
        supply_domain="restaurant",
        keywords=("晚饭", "晚餐", "晚上吃", "吃晚饭", "浪漫晚餐", "庆祝", "订座", "堂食"),
        default_duration_min=80,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="restaurant_specific",
        label="指定餐饮",
        supply_domain="restaurant",
        keywords=(
            "蟹黄面",
            "本帮菜",
            "火锅",
            "烤肉",
            "烧烤",
            "烧烤店",
            "扒房",
            "牛排",
            "西餐",
            "西餐厅",
            "日料",
            "咖喱",
            "小吃",
            "面馆",
            "轻食",
            "简餐",
            "餐厅",
            "聚餐",
            "夜宵",
            "吃饭",
            "吃个",
            "吃点",
            "吃点东西",
        ),
        default_duration_min=75,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="board_game_escape",
        label="桌游/密室/剧本杀",
        supply_domain="activity",
        keywords=(
            "桌游",
            "棋牌",
            "剧本杀",
            "狼人杀",
            "密室",
            "密室逃脱",
            "推理馆",
            "沉浸式游戏",
        ),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="karaoke",
        label="KTV/唱歌",
        supply_domain="activity",
        keywords=("KTV", "ktv", "唱歌", "卡拉OK", "卡拉ok", "练歌房", "欢唱"),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="internet_cafe",
        label="网吧/电竞",
        supply_domain="activity",
        keywords=("网吧", "网咖", "电竞馆", "电竞", "上网"),
        default_duration_min=120,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="cafe",
        label="咖啡/下午茶",
        supply_domain="restaurant",
        keywords=("咖啡", "咖啡厅", "咖啡馆", "下午茶", "坐坐", "小坐", "甜品"),
        default_duration_min=60,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="souvenir_shopping",
        label="特产/伴手礼",
        supply_domain="shopping",
        keywords=("特产", "伴手礼", "礼品", "礼物", "小礼物", "礼品店", "蛋糕", "生日蛋糕", "带回去", "送朋友", "送老师"),
        default_duration_min=45,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="beauty_cosmetics",
        label="美妆/日化",
        supply_domain="retail",
        keywords=("美妆", "日化", "化妆品", "护肤", "香水", "口红", "彩妆", "美妆店"),
        default_duration_min=35,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="flower_shop",
        label="鲜花/花店",
        supply_domain="retail",
        keywords=("鲜花", "花店", "花束", "买花", "花艺"),
        default_duration_min=25,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="convenience_store",
        label="便利店/日用品",
        supply_domain="shopping",
        keywords=("便利店", "日用品", "超市", "买水", "买点水", "生活用品", "零食"),
        default_duration_min=20,
        current_support="unsupported",
    ),
    RoleDefinition(
        role="cultural_photo",
        label="文化体验/拍照",
        supply_domain="activity",
        keywords=("汉服", "拍照", "写真", "摄影", "古装", "换装", "汉服拍照", "体验汉服"),
        default_duration_min=90,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="tea_house",
        label="茶艺/茶馆休息",
        supply_domain="restaurant",
        keywords=("茶艺", "茶馆", "茶室", "品茶", "喝茶", "茶空间"),
        default_duration_min=60,
        current_support="legacy_restaurant",
    ),
    RoleDefinition(
        role="wellness_massage",
        label="足疗/按摩休息",
        supply_domain="activity",
        keywords=("足疗", "按摩", "捏脚", "修脚", "洗脚", "推拿", "休息一下"),
        default_duration_min=60,
        current_support="legacy_activity",
    ),
    RoleDefinition(
        role="parking",
        label="停车",
        supply_domain="transport_service",
        keywords=("停车", "停车场", "好停车", "免费停车"),
        default_duration_min=15,
        current_support="unsupported",
    ),
)


SEQUENCE_WORDS = ("先", "然后", "再", "顺便", "最后", "接着", "之后")
OVERNIGHT_WORDS = ("住一晚", "住宿", "酒店", "民宿", "住处", "有厨房", "第二天", "过夜")
FULL_DAY_WORDS = ("一整天", "全天", "一天", "上午", "中午", "下午", "晚上")
TWO_DAY_WORDS = ("两天", "2天", "二天", "两日", "2日", "周末两天", "明后天", "第二天", "次日")
NAMED_EVENT_PATTERNS = (
    re.compile(r"“([^”]{4,80})”"),
    re.compile(r"\"([^\"]{4,80})\""),
)


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


def _dedupe_keep_order(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _collect_text(state: PlanState, constraints: dict[str, Any] | None = None) -> str:
    constraints = constraints or state.get("constraints", {}) or {}
    planning_preferences = constraints.get("planning_preferences", {}) or {}
    values: list[Any] = [
        state.get("user_input"),
        constraints.get("raw_text"),
        state.get("scene_type"),
        constraints.get("scene"),
    ]
    for key in ("hard_tags", "soft_tags", "avoid", "scenario_activities"):
        values.extend(_as_list(constraints.get(key)))
    for key in (
        "activity_type",
        "food_type",
        "restaurant_type",
        "experience_type",
        "facility_type",
    ):
        values.extend(_as_list(planning_preferences.get(key)))
    values.extend(_as_list(state.get("scenario_activities")))
    return " ".join(str(value) for value in values if value not in (None, ""))


def _named_entities(text: str) -> list[str]:
    entities: list[str] = []
    for pattern in NAMED_EVENT_PATTERNS:
        entities.extend(match.group(1).strip() for match in pattern.finditer(text))
    return _dedupe_keep_order(entities)


def _role_hits(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for definition in ROLE_DEFINITIONS:
        matched_terms: list[tuple[int, str]] = []
        lowered_text = text.lower()
        for keyword in definition.keywords:
            index = lowered_text.find(keyword.lower())
            if index >= 0:
                matched_terms.append((index, keyword))
        if not matched_terms:
            continue
        matched_terms.sort(key=lambda item: item[0])
        hits.append(
            {
                "role": definition.role,
                "label": definition.label,
                "supply_domain": definition.supply_domain,
                "default_duration_min": definition.default_duration_min,
                "current_support": definition.current_support,
                "matched_terms": _dedupe_keep_order([term for _, term in matched_terms]),
                "first_position": matched_terms[0][0],
            }
        )
    hits.sort(key=lambda item: item["first_position"])
    return hits


def _merge_restaurant_roles(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep meal intent readable without duplicating the same lunch/dinner node."""

    result: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for hit in hits:
        role = hit["role"]
        if role == "restaurant_specific":
            has_specific_meal = any(
                existing["role"] in {"restaurant_breakfast", "restaurant_lunch", "restaurant_dinner"}
                for existing in result
            )
            if has_specific_meal:
                for existing in result:
                    if existing["role"] in {"restaurant_breakfast", "restaurant_lunch", "restaurant_dinner"}:
                        existing["matched_terms"] = _dedupe_keep_order(
                            existing.get("matched_terms", []) + hit.get("matched_terms", [])
                        )
                        existing["label"] = "餐饮"
                continue
        if role in seen_roles:
            continue
        seen_roles.add(role)
        result.append(hit)
    return result


def _planning_horizon(text: str, role_count: int) -> str:
    if any(word in text for word in TWO_DAY_WORDS):
        return "two_day"
    if any(word in text for word in OVERNIGHT_WORDS):
        return "overnight"
    if role_count >= 4 or any(word in text for word in FULL_DAY_WORDS) and "上午" in text and "晚上" in text:
        return "full_day"
    return "half_day"


def _planning_days(horizon: str) -> int:
    return 2 if horizon in {"overnight", "two_day"} else 1


def _time_range_for_role(role: str, sequence_index: int, horizon: str) -> tuple[int, str, str, str]:
    """Return day, start, end, and part-of-day for an intent role."""

    if role == "lodging":
        return 1, "15:00", "次日10:00", "overnight"
    if role == "restaurant_breakfast":
        return 1, "08:30", "09:15", "breakfast"
    if role == "restaurant_lunch":
        return 1, "12:00", "13:10", "lunch"
    if role == "restaurant_dinner":
        return 1, "18:00", "19:20", "dinner"
    if role == "family_indoor_play":
        return 1, "14:30", "17:00", "afternoon"
    if role == "cafe":
        return 1, "15:30", "16:30", "afternoon"
    if role == "board_game_escape":
        return 1, "16:00", "18:00", "late_afternoon"
    if role == "karaoke":
        return 1, "20:00", "22:00", "evening"
    if role == "souvenir_shopping":
        return 1, "16:40", "17:30", "late_afternoon"
    if role == "beauty_cosmetics":
        return 1, "16:20", "16:55", "late_afternoon"
    if role == "flower_shop":
        return 1, "16:55", "17:20", "late_afternoon"
    if role == "wellness_massage":
        return 1, "16:30", "17:30", "late_afternoon"
    if role == "convenience_store":
        return 1, "19:40", "20:00", "evening"
    if role == "parking":
        return 1, "20:00", "20:15", "evening"

    if horizon == "two_day" and sequence_index >= 4:
        return 2, "10:00", "12:00", "morning"
    if sequence_index <= 1 and horizon in {"full_day", "two_day"}:
        return 1, "10:00", "12:00", "morning"
    if sequence_index <= 2:
        return 1, "14:00", "16:00", "afternoon"
    return 1, "16:30", "18:00", "late_afternoon"


def _build_time_skeleton(
    node_intents: list[dict[str, Any]],
    *,
    horizon: str,
) -> dict[str, Any]:
    days: dict[int, list[dict[str, Any]]] = {}
    two_day_split_after = max(2, len(node_intents) // 2) if horizon == "two_day" else None
    for item in node_intents:
        role = str(item.get("role") or "")
        sequence_index = int(item.get("sequence_index") or 1)
        day, start, end, part = _time_range_for_role(
            role,
            sequence_index,
            horizon,
        )
        if two_day_split_after is not None and sequence_index > two_day_split_after:
            day = 2
            if role == "restaurant_breakfast":
                start, end, part = "08:30", "09:15", "breakfast"
            elif role == "restaurant_lunch":
                start, end, part = "12:00", "13:10", "lunch"
            elif role == "restaurant_dinner":
                start, end, part = "18:00", "19:20", "dinner"
            elif role == "cafe":
                start, end, part = "15:30", "16:30", "afternoon"
            elif role not in {"lodging", "convenience_store", "parking"}:
                start, end, part = "10:00", "12:00", "morning"
        entry = {
            "node_id": item.get("node_id"),
            "role": role,
            "label": item.get("label"),
            "supply_domain": item.get("supply_domain"),
            "day": day,
            "start_time": start,
            "end_time": end,
            "part_of_day": part,
            "duration_min": item.get("default_duration_min"),
            "execution_status": "needs_candidate",
        }
        days.setdefault(day, []).append(entry)

    planning_days = _planning_days(horizon)
    day_skeletons = []
    for day_index in range(1, planning_days + 1):
        slots = days.get(day_index, [])
        day_skeletons.append(
            {
                "day": day_index,
                "label": f"Day {day_index}",
                "slots": slots,
                "estimated_active_duration_min": sum(
                    int(slot.get("duration_min") or 0)
                    for slot in slots
                    if slot.get("part_of_day") != "overnight"
                ),
            }
        )

    return {
        "planning_days": planning_days,
        "planning_horizon": horizon,
        "days": day_skeletons,
    }


def build_b_itinerary_blueprint(
    state: PlanState,
    *,
    constraints: dict[str, Any] | None = None,
    scenario_activities: list[str] | None = None,
) -> dict[str, Any]:
    """Build an itinerary skeleton for B framework diagnostics and future planning."""

    del scenario_activities  # Future hook for A-provided node hints.
    constraints = constraints or state.get("constraints", {}) or {}
    text = _collect_text(state, constraints)
    hits = _merge_restaurant_roles(_role_hits(text))

    if not hits:
        hits = [
            {
                "role": "activity",
                "label": "活动",
                "supply_domain": "activity",
                "default_duration_min": 120,
                "current_support": "legacy_activity",
                "matched_terms": [],
                "first_position": 0,
            },
            {
                "role": "restaurant_dinner",
                "label": "餐饮",
                "supply_domain": "restaurant",
                "default_duration_min": 80,
                "current_support": "legacy_restaurant",
                "matched_terms": [],
                "first_position": 1,
            },
        ]

    node_intents: list[dict[str, Any]] = []
    for index, hit in enumerate(hits, start=1):
        node_intents.append(
            {
                "node_id": f"intent_{index:02d}",
                "role": hit["role"],
                "label": hit["label"],
                "supply_domain": hit["supply_domain"],
                "search_terms": hit.get("matched_terms", []),
                "default_duration_min": hit["default_duration_min"],
                "current_support": hit["current_support"],
                "sequence_index": index,
            }
        )

    supported_domains = {"activity", "restaurant"}
    unsupported_roles = [
        item["role"]
        for item in node_intents
        if item["supply_domain"] not in supported_domains
    ]
    domain_counts = {
        "activity": sum(1 for item in node_intents if item["supply_domain"] == "activity"),
        "restaurant": sum(1 for item in node_intents if item["supply_domain"] == "restaurant"),
    }
    is_legacy_pair_shape = (
        len(node_intents) == 2
        and domain_counts["activity"] == 1
        and domain_counts["restaurant"] == 1
        and not unsupported_roles
    )
    has_non_pair_shape = (
        len(node_intents) > 2
        or bool(unsupported_roles)
        or (len(node_intents) == 2 and not is_legacy_pair_shape)
    )
    exact_entities = _named_entities(text)
    requires_rag = bool(
        exact_entities
        or unsupported_roles
        or has_non_pair_shape
        or any(word in text for word in ("附近有哪些", "有什么推荐", "哪吃", "哪里", "哪家"))
    )
    horizon = _planning_horizon(text, len(node_intents))
    time_skeleton = _build_time_skeleton(node_intents, horizon=horizon)

    return {
        "version": "b_itinerary_blueprint_v1",
        "source": "deterministic",
        "template_mode": "multi_node" if has_non_pair_shape else "legacy_pair",
        "planning_horizon": horizon,
        "planning_days": time_skeleton["planning_days"],
        "node_count": len(node_intents),
        "node_intents": node_intents,
        "time_skeleton": time_skeleton,
        "route_pattern": ["start"] + [item["role"] for item in node_intents],
        "unsupported_roles": unsupported_roles,
        "requires_rag": requires_rag,
        "named_entities": exact_entities,
        "sequence_markers": [word for word in SEQUENCE_WORDS if word in text],
        "framework_notes": _framework_notes(has_non_pair_shape, unsupported_roles, requires_rag),
    }


def _framework_notes(
    has_non_pair_shape: bool,
    unsupported_roles: list[str],
    requires_rag: bool,
) -> list[str]:
    notes: list[str] = []
    if has_non_pair_shape:
        notes.append("用户需求不是标准活动+餐厅半天模板，需要多节点 itinerary planner")
    if unsupported_roles:
        notes.append("当前结构化供给缺少部分业态，需要通用 POI/RAG 候选池补足")
    if requires_rag:
        notes.append("需要检索证据确认具体地点、营业时间、价格或用户指定活动")
    if not notes:
        notes.append("可继续使用 legacy activity+restaurant pair planner")
    return notes


def apply_b_itinerary_blueprint(
    state: PlanState,
    *,
    constraints: dict[str, Any],
    scenario_activities: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach the itinerary blueprint to constraints for downstream B nodes."""

    existing = constraints.get("b_itinerary_blueprint")
    if isinstance(existing, dict) and existing:
        return constraints, existing

    blueprint = build_b_itinerary_blueprint(
        state,
        constraints=constraints,
        scenario_activities=scenario_activities,
    )
    enhanced_constraints = dict(constraints)
    enhanced_constraints["b_itinerary_blueprint"] = blueprint
    return enhanced_constraints, blueprint
