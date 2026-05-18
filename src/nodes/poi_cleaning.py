"""Shared POI cleaning rules for real Gaode-backed supply fixtures."""

from __future__ import annotations

from typing import Any


CLOSED_STATUS_TOKENS = (
    "暂停营业",
    "暂不营业",
    "停止营业",
    "已关闭",
    "已停业",
    "歇业",
    "停业",
)

FOOD_SERVICE_TYPE_TOKEN = "餐饮服务"

MARKET_ACTIVITY_TOKENS = (
    "夜市",
    "市集",
    "集市",
    "市场",
    "美食街",
    "小吃街",
)

HANDS_ON_FOOD_ACTIVITY_TOKENS = (
    "DIY",
    "diy",
    "手作",
    "烘焙教室",
    "蛋糕DIY",
    "体验馆",
    "工作室",
)


def _text(value: Any) -> str:
    return "" if value in (None, "") else str(value)


def poi_name_text(item: dict[str, Any]) -> str:
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    return " ".join(part for part in (_text(item.get("name")), _text(raw.get("name"))) if part)


def poi_type_text(item: dict[str, Any]) -> str:
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    return " ".join(
        part
        for part in (
            _text(item.get("gaode_type")),
            _text(raw.get("type")),
            _text(item.get("amap_type")),
        )
        if part
    )


def contains_closed_status(item: dict[str, Any]) -> bool:
    name_text = poi_name_text(item)
    return any(token in name_text for token in CLOSED_STATUS_TOKENS)


def is_food_service_poi(item: dict[str, Any]) -> bool:
    return FOOD_SERVICE_TYPE_TOKEN in poi_type_text(item)


def is_allowed_food_service_activity(item: dict[str, Any]) -> bool:
    category = str(item.get("category") or "")
    name = poi_name_text(item)
    keyword = _text(item.get("gaode_keyword"))
    descriptor = " ".join(
        part
        for part in (
            name,
            keyword,
            _text(item.get("sub_category")),
            _text(item.get("experience_type")),
        )
        if part
    )

    if category == "local_market" and any(token in descriptor for token in MARKET_ACTIVITY_TOKENS):
        return True

    if category == "handcraft" and any(token in descriptor for token in HANDS_ON_FOOD_ACTIVITY_TOKENS):
        return True

    return any(token in name for token in HANDS_ON_FOOD_ACTIVITY_TOKENS)


def is_invalid_activity_poi(item: dict[str, Any]) -> bool:
    return is_food_service_poi(item) and not is_allowed_food_service_activity(item)


def should_exclude_poi(item: dict[str, Any], expected_type: str) -> bool:
    if contains_closed_status(item):
        return True
    if expected_type == "activity" and is_invalid_activity_poi(item):
        return True
    return False
