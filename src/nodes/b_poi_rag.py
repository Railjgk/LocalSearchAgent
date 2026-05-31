"""Local POI RAG retrieval for B-stage itinerary planning.

This node is the replaceable boundary between the future vector/RAG service and
B's deterministic planner.  It retrieves candidate POIs, not final itineraries,
and emits ``b_rag_candidate_evidence`` for ``candidate_generator_node``.
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from src.state import PlanState
except ImportError:  # pragma: no cover
    PlanState = dict

from .b_itinerary_blueprint import build_b_itinerary_blueprint
from .b_poi_memory_index import (
    DEFAULT_DOCUMENT_FIELDS,
    POI_MEMORY_INDEX_VERSION,
    build_poi_memory_index,
    load_poi_memory_index_cache,
    retrieve_poi_memory_candidates,
    save_poi_memory_index_cache,
)
from .b_requirement_compiler import apply_b_requirement_contract
from .b_semantics import (
    B_SEMANTIC_GROUPS,
    b_semantic_terms,
    flatten_semantic_values,
    normalize_semantic_text,
    semantic_groups_in_values,
)
from .b_utils import to_float


DEFAULT_MOCK_DATA_DIR = Path(__file__).resolve().parents[2] / "experiments" / "mock_data"
DEFAULT_INDEX_CACHE_DIR = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "artifacts"
    / "b_poi_memory_index_cache"
)
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on", "auto"}
FALSY_ENV_VALUES = {"0", "false", "no", "off", "disabled"}
SUPPORTED_CONTRACT_VERSION = "b_rag_candidate_evidence_v1"
DEFAULT_TOP_K_PER_NODE = 12
DEFAULT_MAX_SCAN_PER_DOMAIN = 60000
GENERIC_POI_FALLBACK_VERSION = "generic_poi_fallback_v4"
STRICT_ROLE_MATCH_ROLES = {
    "family_activity",
    "family_indoor_play",
    "exhibition",
    "citywalk_market",
    "board_game_escape",
    "internet_cafe",
    "karaoke",
    "cafe",
    "souvenir_shopping",
    "beauty_cosmetics",
    "flower_shop",
    "convenience_store",
    "wellness_massage",
    "parking",
}

DOMAIN_STEMS = {
    "activity": ("activities",),
    "restaurant": ("restaurants",),
    "shopping": ("shopping", "shops", "retail"),
    "retail": ("retail", "shopping", "shops"),
    "hotel": ("hotels", "lodging"),
    "lodging": ("hotels", "lodging"),
    "wellness": ("wellness",),
    "transport_service": ("transport_services", "parking"),
}
GENERIC_POI_FALLBACK_DOMAINS = {"shopping", "retail", "hotel", "transport_service"}
DOMAIN_FALLBACK_HINTS = {
    "shopping": (
        "购物服务",
        "便民商店",
        "便利店",
        "超级市场",
        "超市",
        "土特产",
        "特产",
        "礼品饰品",
        "礼品",
        "专卖店",
        "百货",
    ),
    "retail": (
        "购物服务",
        "个人用品",
        "化妆品",
        "美妆",
        "日化",
        "花鸟鱼虫",
        "花卉",
        "鲜花",
        "礼品饰品",
    ),
    "hotel": (
        "住宿服务",
        "酒店",
        "宾馆",
        "民宿",
        "公寓",
        "旅馆",
        "客栈",
    ),
    "transport_service": (
        "交通设施服务",
        "停车场",
        "停车",
    ),
}
DOMAIN_DEFAULTS = {
    "shopping": {"type": "shopping", "price": 80.0, "duration_min": 25, "queue_time_min": 5},
    "retail": {"type": "shopping", "price": 120.0, "duration_min": 30, "queue_time_min": 5},
    "hotel": {"type": "hotel", "price": 420.0, "duration_min": 720, "queue_time_min": 10},
    "transport_service": {"type": "transport_service", "price": 25.0, "duration_min": 15, "queue_time_min": 5},
}
LOCATION_ANCHOR_TERMS = {
    "外滩",
    "黄浦江",
    "南京路",
    "南京东路",
    "南京西路",
    "人民广场",
    "静安寺",
    "新天地",
    "淮海路",
    "豫园",
    "陆家嘴",
    "徐家汇",
    "田子坊",
    "五角场",
    "虹桥",
    "松江",
    "嘉定",
    "浦东",
}

ROLE_QUERY_TERMS = {
    "family_activity": ("亲子", "儿童", "儿童友好", "室内", "低强度", "游乐园"),
    "family_indoor_play": ("室内乐园", "亲子乐园", "儿童乐园", "儿童游乐", "淘气堡", "蹦床", "游乐园"),
    "exhibition": ("展览", "看展", "博物馆", "美术馆", "艺术展", "文物展", "漆器展"),
    "citywalk_market": ("citywalk", "城市漫步", "市集", "街区", "本地生活"),
    "board_game_escape": ("桌游", "棋牌", "剧本杀", "狼人杀", "密室", "密室逃脱", "推理馆"),
    "internet_cafe": ("网吧", "网咖", "电竞馆", "电竞", "上网"),
    "karaoke": ("KTV", "唱歌", "卡拉OK", "练歌房", "欢唱"),
    "restaurant_breakfast": ("早餐", "早饭", "早点", "包子", "馄饨"),
    "restaurant_lunch": ("午餐", "中饭", "正餐", "简餐"),
    "restaurant_dinner": ("晚餐", "晚饭", "正餐", "堂食"),
    "restaurant_specific": ("餐厅", "吃饭", "正餐", "聚餐", "夜宵", "扒房", "牛排", "西餐", "烧烤", "火锅"),
    "cafe": ("咖啡", "咖啡馆", "下午茶", "甜品", "小坐"),
    "souvenir_shopping": ("特产", "伴手礼", "礼品", "礼物", "小礼物", "蛋糕", "生日蛋糕", "带回去"),
    "beauty_cosmetics": ("美妆", "日化", "化妆品", "护肤"),
    "flower_shop": ("鲜花", "花店", "花束", "花艺"),
    "convenience_store": ("便利店", "日用品", "超市", "买水", "零食"),
    "cultural_photo": ("汉服", "拍照", "写真", "摄影", "古装", "换装", "文化体验"),
    "tea_house": ("茶艺", "茶馆", "茶室", "品茶", "喝茶", "茶空间", "休息"),
    "wellness_massage": ("足疗", "按摩", "洗脚", "推拿", "休息"),
    "parking": ("停车", "停车场", "免费停车", "好停车"),
}

ROLE_REQUIRED_IDENTITY_TERMS = {
    "family_activity": (
        "亲子",
        "儿童",
        "孩子",
        "小朋友",
        "带娃",
        "遛娃",
        "儿童友好",
        "儿童乐园",
        "室内乐园",
        "游乐园",
        "亲子馆",
        "手作",
        "DIY",
        "陶艺",
        "绘画",
        "乐高",
        "积木",
        "早教",
        "研学",
        "儿童剧",
        "绘本",
    ),
    "family_indoor_play": (
        "室内乐园",
        "亲子乐园",
        "儿童乐园",
        "儿童游乐",
        "室内游乐",
        "游乐园",
        "乐园",
        "淘气堡",
        "蹦床",
    ),
}

ROLE_EXCLUDED_IDENTITY_TERMS = {
    "family_activity": (
        "瑜伽",
        "普拉提",
        "健身",
        "健身中心",
        "SPA",
        "足疗",
        "按摩",
        "酒吧",
        "党群服务中心",
        "社区党群",
        "社区服务中心",
        "街道社区",
        "政务服务",
    ),
    "family_indoor_play": (
        "瑜伽",
        "普拉提",
        "健身",
        "SPA",
        "足疗",
        "按摩",
        "党群服务中心",
        "社区党群",
        "社区服务中心",
        "街道社区",
        "政务服务",
    ),
}

MEAL_ROLE_MARKERS = {
    "restaurant_breakfast": ("早上", "早餐", "早饭", "早茶", "早点"),
    "restaurant_lunch": ("中午", "午餐", "午饭", "中饭", "吃个中饭", "吃午饭"),
    "restaurant_dinner": ("晚上", "晚餐", "晚饭", "吃晚饭", "吃晚餐"),
}

MEAL_CONTEXT_STOP_WORDS = (
    "上午",
    "中午",
    "午餐",
    "午饭",
    "下午",
    "晚上",
    "晚餐",
    "晚饭",
    "饭后",
    "然后",
    "再",
    "最后",
)

MEAL_CONTEXT_TERM_EXPANSIONS = {
    "清淡": ("清淡", "轻食", "健康", "少油", "中式轻食", "沙拉"),
    "轻食": ("轻食", "健康", "低卡", "沙拉", "中式轻食"),
    "低卡": ("低卡", "轻食", "健康", "沙拉"),
    "减脂": ("减脂", "低卡", "轻食", "少油"),
    "少油": ("少油", "清淡", "健康"),
    "素食": ("素食", "蔬食", "健康", "清淡"),
    "本帮": ("本帮菜", "上海菜", "家常菜"),
    "上海菜": ("上海菜", "本帮菜", "家常菜"),
    "火锅": ("火锅", "潮汕牛肉火锅", "川渝火锅"),
    "烤肉": ("烤肉", "烧烤", "日式烧肉", "炭火烤肉"),
    "烧烤": ("烧烤", "烤肉", "羊肉串", "炭火"),
    "羊肉串": ("羊肉串", "烧烤", "烤肉"),
    "咖啡": ("咖啡", "咖啡馆", "下午茶"),
    "甜品": ("甜品", "下午茶", "蛋糕"),
}

TEXT_FIELDS = (
    "name",
    "category",
    "sub_category",
    "restaurant_category",
    "primary_category",
    "gaode_keyword",
    "gaode_type",
    "address",
    "location",
    "business_area",
    "tags",
    "source_evidence",
    "review_keywords",
    "signature_dishes",
    "recommended_dishes",
    "dish_tags",
    "package_options",
    "product_options",
    "deal_options",
)
IDENTITY_TEXT_FIELDS = (
    "name",
    "category",
    "sub_category",
    "restaurant_category",
    "primary_category",
    "gaode_keyword",
    "gaode_type",
    "address",
    "location",
    "business_area",
    "tags",
    "review_keywords",
    "signature_dishes",
    "recommended_dishes",
    "dish_tags",
)
_TEXT_BLOB_CACHE_LIMIT = 120000
_TEXT_BLOB_CACHE: dict[tuple[str, tuple[str, ...]], tuple[str, set[str]]] = {}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY_ENV_VALUES


def _falsy(value: Any) -> bool:
    return str(value).strip().lower() in FALSY_ENV_VALUES


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


def _mock_data_dir() -> Path:
    override = os.environ.get("WF_B_RAG_DATA_DIR") or os.environ.get("WF_MOCK_DATA_DIR")
    if override:
        path = Path(override)
        return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path
    return DEFAULT_MOCK_DATA_DIR


def _top_k_per_node() -> int:
    raw = os.environ.get("WF_B_RAG_TOP_K", "").strip()
    try:
        return max(1, min(50, int(raw)))
    except ValueError:
        return DEFAULT_TOP_K_PER_NODE


def _max_scan_per_domain() -> int:
    raw = os.environ.get("WF_B_RAG_MAX_SCAN", "").strip()
    try:
        return max(100, min(200000, int(raw)))
    except ValueError:
        return DEFAULT_MAX_SCAN_PER_DOMAIN


def _index_cache_enabled() -> bool:
    raw = os.environ.get("WF_B_RAG_INDEX_CACHE")
    if raw is None:
        return True
    return not _falsy(raw)


def _index_cache_dir() -> Path:
    override = os.environ.get("WF_B_RAG_INDEX_CACHE_DIR")
    if override:
        path = Path(override)
        return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path
    return DEFAULT_INDEX_CACHE_DIR


def _records_signature(root: Path, stem: str) -> str:
    parts: list[str] = []
    for path in (root / f"{stem}.json", root / f"{stem}.jsonl"):
        try:
            stat = path.stat()
        except OSError:
            continue
        parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    for shard_dir_name in (f"{stem}_shards", f"{stem}_jsonl", f"{stem}.jsonl.d"):
        shard_dir = root / shard_dir_name
        if not shard_dir.exists():
            continue
        for shard in sorted(shard_dir.glob("*.jsonl")):
            try:
                stat = shard.stat()
            except OSError:
                continue
            parts.append(f"{shard_dir.name}/{shard.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts) or f"{stem}:missing"


def _supply_signature(root: Path) -> str:
    stems = {
        "activities",
        "restaurants",
        "shopping",
        "shops",
        "retail",
        "hotels",
        "lodging",
        "wellness",
        "transport_services",
        "parking",
        "availability",
        "products",
        "deals",
        "merchants",
    }
    return "|".join(_records_signature(root, stem) for stem in sorted(stems))


def _domain_signature(root: Path, domain: str) -> str:
    stems = DOMAIN_STEMS.get(domain, (domain,))
    parts = [
        f"index={POI_MEMORY_INDEX_VERSION}",
        f"fields={','.join(DEFAULT_DOCUMENT_FIELDS)}",
        f"max_scan={_max_scan_per_domain()}",
    ]
    parts.extend(_records_signature(root, stem) for stem in stems)
    if _normalize_domain(domain) in GENERIC_POI_FALLBACK_DOMAINS:
        parts.append(f"generic_fallback={GENERIC_POI_FALLBACK_VERSION}")
        parts.append(_records_signature(root, "deduped_pois"))
        parts.append(_records_signature(root, "raw_pois"))
        if _normalize_domain(domain) == "transport_service":
            parts.append(_records_signature(root, "activities"))
            parts.append(_records_signature(root, "restaurants"))
    return "|".join(parts)


def _cache_filename(root: Path, domain: str, signature: str) -> str:
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    root_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", root.name)[:80] or "mock_data"
    domain_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", domain)[:40] or "domain"
    return f"{root_name}_{domain_name}_{digest}.pkl"


def _read_json(path: Path) -> Any:
    if not path.exists():
        return [] if path.suffix == ".json" else None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _items_from_json(raw: Any) -> list[dict[str, Any]]:
    items = raw.get("items", []) if isinstance(raw, dict) and isinstance(raw.get("items"), list) else raw
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
    except Exception:
        return []
    return records


def _read_records(root: Path, stem: str) -> list[dict[str, Any]]:
    records = _items_from_json(_read_json(root / f"{stem}.json"))
    if records:
        return records

    records = _read_jsonl(root / f"{stem}.jsonl")
    if records:
        return records

    for shard_dir_name in (f"{stem}_shards", f"{stem}_jsonl", f"{stem}.jsonl.d"):
        shard_dir = root / shard_dir_name
        if not shard_dir.exists():
            continue
        shard_records: list[dict[str, Any]] = []
        for shard in sorted(shard_dir.glob("*.jsonl")):
            shard_records.extend(_read_jsonl(shard))
        if shard_records:
            return shard_records
    return []


def _group_by_poi(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        poi_id = str(item.get("poi_id") or "").strip()
        if poi_id:
            grouped.setdefault(poi_id, []).append(item)
    return grouped


def _merchant_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in records:
        merchant_id = str(item.get("merchant_id") or "").strip()
        if merchant_id and merchant_id not in result:
            result[merchant_id] = item
    return result


@lru_cache(maxsize=8)
def _load_supply_bundle(root_key: str, signature: str) -> dict[str, Any]:
    del signature
    root = Path(root_key)
    availability_raw = _read_json(root / "availability.json")
    availability = availability_raw if isinstance(availability_raw, dict) else {}
    products_by_poi = _group_by_poi(_read_records(root, "products"))
    deals_by_poi = _group_by_poi(_read_records(root, "deals"))
    merchants = _merchant_by_id(_read_records(root, "merchants"))
    return {
        "root": str(root),
        "domain_items": {},
        "availability": availability,
        "products_by_poi": products_by_poi,
        "deals_by_poi": deals_by_poi,
        "merchants": merchants,
    }


def _supply_bundle(root: Path) -> dict[str, Any]:
    return _load_supply_bundle(str(root.resolve()), _supply_signature(root))


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in records:
        poi_id = str(item.get("poi_id") or item.get("id") or item.get("amap_id") or "").strip()
        name = str(item.get("name") or "").strip()
        key = poi_id or name
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_text(values: list[Any], *, limit: int = 80) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in flatten_semantic_values(values):
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _query_tokens(values: list[Any]) -> list[str]:
    raw: list[Any] = []
    for value in values:
        raw.extend(flatten_semantic_values(value))
        if isinstance(value, str):
            raw.extend(re.split(r"[\s,，。；;、/|+]+", value))
    raw.extend(b_semantic_terms(raw, include_auxiliary=True))
    tokens: list[str] = []
    seen: set[str] = set()
    for value in raw:
        text = normalize_semantic_text(str(value).strip())
        if len(text) < 2 or text in seen:
            continue
        seen.add(text)
        tokens.append(text)
    return tokens


def _location_anchor_terms(values: list[Any]) -> list[str]:
    blob = " ".join(str(value) for value in flatten_semantic_values(values) if value not in (None, ""))
    return [term for term in sorted(LOCATION_ANCHOR_TERMS, key=len, reverse=True) if term in blob]


def _item_matches_location_terms(item: dict[str, Any], location_terms: list[str]) -> bool:
    if not location_terms:
        return False
    blob, values = _text_blob(item)
    return any(term in values or term in blob for term in location_terms)


def _text_blob(item: dict[str, Any]) -> tuple[str, set[str]]:
    return _text_blob_for_fields(item, TEXT_FIELDS)


def _text_blob_for_fields(item: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, set[str]]:
    item_key = str(
        item.get("poi_id")
        or item.get("id")
        or item.get("amap_id")
        or item.get("merchant_id")
        or item.get("name")
        or id(item)
    )
    field_signature = "|".join(str(item.get(field))[:160] for field in fields)
    cache_key = (f"{item_key}:{hash(field_signature)}", fields)
    cached = _TEXT_BLOB_CACHE.get(cache_key)
    if cached:
        return cached

    values: list[Any] = []
    for field in fields:
        values.extend(flatten_semantic_values(item.get(field)))
    normalized = [
        normalize_semantic_text(str(value))
        for value in values
        if len(normalize_semantic_text(str(value))) >= 2
    ]
    result = ("\n".join(normalized), set(normalized))
    if len(_TEXT_BLOB_CACHE) >= _TEXT_BLOB_CACHE_LIMIT:
        _TEXT_BLOB_CACHE.clear()
    _TEXT_BLOB_CACHE[cache_key] = result
    return result


def _matches_any_term(blob: str, values: set[str], terms: list[str]) -> bool:
    return any(term in values or term in blob for term in terms)


BROAD_NEGATIVE_AUXILIARY_TERMS = {
    "meat",
    "social",
    "lively",
    "atmosphere",
    "high_calorie",
    "restaurant",
}


def _negative_group_terms(group: str) -> list[str]:
    payload = B_SEMANTIC_GROUPS.get(group) or {}
    raw_terms = list(payload.get("primary") or [])
    raw_terms.extend(
        term
        for term in (payload.get("auxiliary") or [])
        if normalize_semantic_text(term) not in BROAD_NEGATIVE_AUXILIARY_TERMS
    )
    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        normalized = normalize_semantic_text(term)
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
    return terms


def _item_has_negative_group_evidence(item: dict[str, Any], group: str) -> bool:
    terms = _negative_group_terms(group)
    if not terms:
        return False
    blob, values = _text_blob_for_fields(item, IDENTITY_TEXT_FIELDS)
    return _matches_any_term(blob, values, terms)


def _forbidden_restaurant_groups_for_retrieval(
    state: PlanState,
    constraints: dict[str, Any],
) -> set[str]:
    values: list[Any] = []
    contract = constraints.get("b_requirement_contract") or {}
    if isinstance(contract, dict):
        values.extend(contract.get("forbidden_restaurant_groups") or [])
    values.extend(constraints.get("avoid") or [])
    user_profile = state.get("user_profile", {}) or {}
    if isinstance(user_profile, dict):
        values.extend(user_profile.get("avoid") or [])
    groups = semantic_groups_in_values(values)
    return groups.intersection({"烤肉", "火锅"})


def _coordinates(item: dict[str, Any]) -> tuple[float, float] | None:
    coordinates = item.get("coordinates") or item.get("location")
    if isinstance(coordinates, str) and "," in coordinates:
        left, right = coordinates.split(",", 1)
    elif isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
        left, right = coordinates[0], coordinates[1]
    else:
        left = item.get("longitude") or item.get("lng") or item.get("lon")
        right = item.get("latitude") or item.get("lat")
    try:
        return float(left), float(right)
    except (TypeError, ValueError):
        return None


def _haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    import math

    lng1, lat1 = first
    lng2, lat2 = second
    radius = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lng / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _origin_coordinates(constraints: dict[str, Any]) -> tuple[float, float] | None:
    raw = constraints.get("origin_coordinates") or constraints.get("user_coordinates")
    if isinstance(raw, str) and "," in raw:
        left, right = raw.split(",", 1)
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        left, right = raw[0], raw[1]
    else:
        return None
    try:
        return float(left), float(right)
    except (TypeError, ValueError):
        return None


def _candidate_distance_km(item: dict[str, Any], constraints: dict[str, Any]) -> float:
    origin = _origin_coordinates(constraints)
    coord = _coordinates(item)
    if origin and coord:
        return round(_haversine_km(origin, coord) * 1.35, 2)
    return round(to_float(item.get("distance_km") or item.get("distance"), 0.0), 2)


def _quality_fallback_pool(
    items: list[dict[str, Any]],
    *,
    constraints: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    def rank_key(item: dict[str, Any]) -> tuple[float, float]:
        distance = _candidate_distance_km(item, constraints)
        if distance <= 0:
            distance = 999.0
        return (distance, -to_float(item.get("rating") or item.get("score"), 4.0))

    return sorted(items, key=rank_key)[:limit]


def _merge_auxiliary_fields(item: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    poi_id = str(item.get("poi_id") or item.get("id") or item.get("amap_id") or "").strip()
    merchant_id = str(item.get("merchant_id") or "").strip()
    enriched = dict(item)
    availability = bundle.get("availability", {}).get(poi_id)
    if isinstance(availability, dict):
        for key in (
            "available",
            "available_slots",
            "queue_time_min",
            "inventory_left",
            "reservation_required",
            "holiday_status",
        ):
            if key in availability and enriched.get(key) in (None, "", []):
                enriched[key] = availability[key]
    products = bundle.get("products_by_poi", {}).get(poi_id, [])[:3]
    deals = bundle.get("deals_by_poi", {}).get(poi_id, [])[:3]
    merchant = bundle.get("merchants", {}).get(merchant_id, {})
    if products:
        enriched["product_options"] = products
    if deals:
        enriched["deal_options"] = deals
        enriched["package_options"] = _dedupe_text(
            [
                deal.get("title")
                for deal in deals
            ]
            + [
                component
                for deal in deals
                for component in flatten_semantic_values(deal.get("package_components"))
            ],
            limit=12,
        )
    for key in ("trust_score", "review_count", "operation_stability_score", "business_capabilities"):
        if merchant.get(key) is not None and enriched.get(key) in (None, "", []):
            enriched[key] = merchant.get(key)
    return enriched


def _normalize_domain(domain: str) -> str:
    if domain in {"lodging", "hotel"}:
        return "hotel"
    return str(domain or "")


def _generic_poi_records(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    if "_generic_poi_records" not in bundle:
        root = Path(bundle.get("root") or _mock_data_dir())
        records = _read_records(root, "deduped_pois")
        if not records:
            records = _read_records(root, "raw_pois")
        bundle["_generic_poi_records"] = _dedupe_records(records)
    return bundle.get("_generic_poi_records", [])


def _generic_poi_text(item: dict[str, Any]) -> str:
    values: list[Any] = []
    for key in (
        "name",
        "type",
        "category",
        "address",
        "gaode_type",
        "primary_category",
        "primary_keyword",
        "business_area",
    ):
        values.extend(flatten_semantic_values(item.get(key)))
    for key in ("business", "biz_ext", "raw"):
        nested = item.get(key)
        if isinstance(nested, dict):
            for nested_key in ("tag", "rectag", "keytag", "business_area", "type", "address"):
                values.extend(flatten_semantic_values(nested.get(nested_key)))
    return " ".join(str(value) for value in values if str(value).strip())


def _generic_poi_matches_domain(item: dict[str, Any], domain: str) -> bool:
    text = _generic_poi_text(item)
    hints = DOMAIN_FALLBACK_HINTS.get(domain, ())
    if not hints:
        return False
    if domain == "hotel":
        type_text = str(item.get("type") or item.get("gaode_type") or "")
        raw_type = ""
        raw = item.get("raw")
        if isinstance(raw, dict):
            raw_type = str(raw.get("type") or "")
        type_blob = f"{type_text} {raw_type}"
        if not any(
            term in type_blob
            for term in ("宾馆酒店", "酒店", "民宿", "旅馆", "客栈", "公寓", "度假村")
        ):
            return False
    if domain == "transport_service":
        type_text = str(item.get("type") or item.get("gaode_type") or "")
        raw_type = ""
        raw = item.get("raw")
        if isinstance(raw, dict):
            raw_type = str(raw.get("type") or "")
        name_or_category = " ".join(
            str(item.get(key) or "")
            for key in ("name", "category", "primary_category", "gaode_keyword")
        )
        type_blob = f"{type_text} {raw_type}"
        return "停车场" in type_blob
    return any(hint in text for hint in hints)


def _parse_location(value: Any) -> tuple[float | None, float | None]:
    text = str(value or "").strip()
    if "," not in text:
        return None, None
    left, right = text.split(",", 1)
    try:
        return float(left), float(right)
    except ValueError:
        return None, None


def _normalize_generic_poi(item: dict[str, Any], domain: str) -> dict[str, Any]:
    defaults = DOMAIN_DEFAULTS.get(domain, {})
    poi_id = str(item.get("poi_id") or item.get("id") or item.get("amap_id") or "").strip()
    amap_id = str(item.get("amap_id") or item.get("id") or "").strip()
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    business = item.get("business") if isinstance(item.get("business"), dict) else {}
    biz_ext = item.get("biz_ext") if isinstance(item.get("biz_ext"), dict) else {}
    location = item.get("coordinates") or item.get("location") or raw.get("location")
    lng, lat = _parse_location(location)
    gaode_type = item.get("gaode_type") or item.get("type") or raw.get("type")
    category = (
        item.get("category")
        or item.get("primary_category")
        or business.get("keytag")
        or business.get("rectag")
        or biz_ext.get("keytag")
        or biz_ext.get("rectag")
        or gaode_type
        or domain
    )
    rating = item.get("rating") or biz_ext.get("rating") or business.get("rating")
    cost = item.get("price") or item.get("cost") or biz_ext.get("cost") or business.get("cost")
    normalized = {
        "poi_id": poi_id or (f"gaode_poi_{amap_id}" if amap_id else ""),
        "amap_id": amap_id or None,
        "merchant_id": item.get("merchant_id") or (f"m_gaode_poi_{amap_id}" if amap_id else None),
        "name": item.get("name") or raw.get("name"),
        "type": defaults.get("type") or domain,
        "supply_domain": domain,
        "category": category,
        "primary_category": category,
        "gaode_type": gaode_type,
        "address": item.get("address") or raw.get("address"),
        "location": location,
        "coordinates": location,
        "longitude": lng,
        "latitude": lat,
        "business_area": item.get("business_area") or business.get("business_area") or biz_ext.get("business_area"),
        "tags": [domain, category, gaode_type],
        "price": to_float(cost, defaults.get("price", 80.0)),
        "duration_min": int(defaults.get("duration_min", 30)),
        "queue_time_min": int(defaults.get("queue_time_min", 5)),
        "rating": to_float(rating, 4.2),
        "available": True,
        "source": item.get("source") or "gaode_generic_poi_fallback",
        "source_channel": item.get("source_channel") or "gaode_deduped_pois",
        "source_evidence": [
            f"Gaode generic POI fallback domain={domain}",
            f"amap_id={amap_id}" if amap_id else "",
        ],
        "raw": item,
    }
    return {key: value for key, value in normalized.items() if value not in (None, "", [])}


def _generic_domain_items(bundle: dict[str, Any], domain: str) -> list[dict[str, Any]]:
    items = [
        _normalize_generic_poi(item, domain)
        for item in _generic_poi_records(bundle)
        if _generic_poi_matches_domain(item, domain)
    ]
    if domain == "transport_service" and not items:
        items.extend(_derived_parking_proxy_items(bundle))
    return items


def _parking_proxy_source_records(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    if "_parking_proxy_source_records" not in bundle:
        root = Path(bundle.get("root") or _mock_data_dir())
        records = _read_records(root, "activities") + _read_records(root, "restaurants")
        bundle["_parking_proxy_source_records"] = _dedupe_records(records)
    return bundle.get("_parking_proxy_source_records", [])


def _item_has_parking_signal(item: dict[str, Any]) -> bool:
    if item.get("parking_available") is True:
        return True
    values = [
        item.get("parking_fee_policy"),
        item.get("decision_profile"),
        item.get("fulfillment_actions"),
        item.get("source_evidence"),
        item.get("location"),
        item.get("address"),
    ]
    blob = " ".join(str(value) for value in flatten_semantic_values(values))
    return "停车" in blob or "🅿" in blob


def _normalize_parking_proxy(item: dict[str, Any]) -> dict[str, Any]:
    poi_id = str(item.get("poi_id") or item.get("id") or item.get("amap_id") or "").strip()
    amap_id = str(item.get("amap_id") or item.get("id") or "").strip()
    location = item.get("coordinates") or item.get("location")
    lng, lat = _parse_location(location)
    policy = str(item.get("parking_fee_policy") or "附近可停车，需到店前再次确认").strip()
    source_name = str(item.get("name") or "目标地点").strip()
    source_key = poi_id or amap_id or hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:12]
    return {
        "poi_id": f"parking_proxy_{source_key}",
        "amap_id": amap_id or None,
        "merchant_id": f"m_parking_proxy_{source_key}",
        "name": f"{source_name} 停车指引",
        "type": "transport_service",
        "supply_domain": "transport_service",
        "category": "停车指引",
        "primary_category": "停车指引",
        "gaode_type": "派生停车服务;停车指引",
        "address": item.get("address") or item.get("location"),
        "location": location,
        "coordinates": location,
        "longitude": lng,
        "latitude": lat,
        "business_area": item.get("business_area"),
        "tags": ["停车", "停车场", "停车指引", policy],
        "price": 25.0,
        "duration_min": 15,
        "queue_time_min": 5,
        "rating": to_float(item.get("rating"), 4.0),
        "available": True,
        "parking_proxy": True,
        "parking_proxy_source_poi_id": poi_id or None,
        "parking_fee_policy": policy,
        "source": "b_derived_parking_proxy",
        "source_channel": "local_supply_parking_signal",
        "source_evidence": [
            "Derived parking proxy from local supply parking signal",
            f"source_poi_id={poi_id}" if poi_id else "",
            policy,
        ],
        "raw": item,
    }


def _derived_parking_proxy_items(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _normalize_parking_proxy(item)
        for item in _parking_proxy_source_records(bundle)
        if _item_has_parking_signal(item)
    ]


def _domain_items(bundle: dict[str, Any], domain: str) -> list[dict[str, Any]]:
    domain = _normalize_domain(domain)
    domain_cache = bundle.setdefault("domain_items", {})
    if domain not in domain_cache:
        root = Path(bundle.get("root") or _mock_data_dir())
        items: list[dict[str, Any]] = []
        for stem in DOMAIN_STEMS.get(domain, (domain,)):
            items.extend(_read_records(root, stem))
        if not items and domain in GENERIC_POI_FALLBACK_DOMAINS:
            items.extend(_generic_domain_items(bundle, domain))
        domain_cache[domain] = _dedupe_records(items)
    items = domain_cache.get(domain, [])
    return items[: _max_scan_per_domain()]


def _memory_index_for_domain(
    bundle: dict[str, Any],
    *,
    domain: str,
    items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_domain = _normalize_domain(domain)
    cache = bundle.setdefault("_memory_indexes", {})
    cache_meta = bundle.setdefault("_memory_index_cache_meta", {})
    if normalized_domain in cache:
        cache_meta.setdefault(
            normalized_domain,
            {
                "status": "memory_hit",
                "doc_count": cache[normalized_domain].get("doc_count"),
            },
        )
        return cache[normalized_domain]

    root = Path(bundle.get("root") or _mock_data_dir())
    signature = ""
    cache_path: Path | None = None
    if _index_cache_enabled():
        signature = _domain_signature(root, normalized_domain)
        cache_path = _index_cache_dir() / _cache_filename(root, normalized_domain, signature)
        cached_index = load_poi_memory_index_cache(cache_path, expected_signature=signature)
        if cached_index is not None:
            cache[normalized_domain] = cached_index
            cache_meta[normalized_domain] = {
                "status": "disk_hit",
                "path": str(cache_path),
                "doc_count": cached_index.get("doc_count"),
            }
            return cached_index

    if items is None:
        items = _domain_items(bundle, normalized_domain)
    index = build_poi_memory_index(items)
    cache[normalized_domain] = index
    saved = False
    if cache_path is not None and signature:
        saved = save_poi_memory_index_cache(cache_path, signature=signature, index=index)
    cache_meta[normalized_domain] = {
        "status": "built_saved" if saved else ("built_unsaved" if cache_path else "disabled"),
        "path": str(cache_path) if cache_path else "",
        "doc_count": index.get("doc_count"),
    }
    return index


def _items_from_memory_index(index: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for document in index.get("documents", []) or []:
        item = document.get("item") if isinstance(document, dict) else None
        if isinstance(item, dict):
            items.append(item)
    return items


def _role_terms(intent: dict[str, Any]) -> list[str]:
    role = str(intent.get("role") or "")
    return _dedupe_text(
        [
            intent.get("search_terms"),
            intent.get("label"),
            intent.get("supply_domain"),
            role,
            ROLE_QUERY_TERMS.get(role, ()),
        ],
        limit=40,
    )


def _node_query_terms(state: PlanState, constraints: dict[str, Any], intent: dict[str, Any]) -> tuple[list[str], list[str]]:
    planning_preferences = constraints.get("planning_preferences", {}) or {}
    role = str(intent.get("role") or "")
    domain = str(intent.get("supply_domain") or "")
    node_terms = _role_terms(intent)
    use_local_meal_context = False
    if domain == "activity":
        node_terms.extend(flatten_semantic_values(planning_preferences.get("activity_type")))
        node_terms.extend(flatten_semantic_values(planning_preferences.get("experience_type")))
    elif domain == "restaurant":
        meal_terms = _meal_context_terms(state, constraints, role)
        if meal_terms:
            use_local_meal_context = True
            node_terms.extend(meal_terms)
        else:
            node_terms.extend(flatten_semantic_values(planning_preferences.get("food_type")))
            node_terms.extend(flatten_semantic_values(planning_preferences.get("restaurant_type")))
    elif role in ROLE_QUERY_TERMS:
        node_terms.extend(ROLE_QUERY_TERMS[role])

    global_terms = [
        constraints.get("city"),
        constraints.get("district"),
        constraints.get("business_area"),
        constraints.get("hard_tags"),
        constraints.get("soft_tags"),
        constraints.get("avoid"),
        state.get("scenario_activities"),
    ]
    if not use_local_meal_context:
        global_terms.extend([state.get("user_input"), constraints.get("raw_text")])
    global_terms.extend(
        _location_anchor_terms(
            [
                state.get("user_input"),
                constraints.get("raw_text"),
                constraints.get("district"),
                constraints.get("business_area"),
            ]
        )
    )
    return _query_tokens(node_terms), _query_tokens(global_terms)


def _meal_context_terms(state: PlanState, constraints: dict[str, Any], role: str) -> list[str]:
    markers = MEAL_ROLE_MARKERS.get(role)
    if not markers:
        return []
    text = " ".join(
        str(value)
        for value in (state.get("user_input"), constraints.get("raw_text"))
        if value not in (None, "")
    )
    if not text:
        return []

    marker_positions = [
        (text.find(marker), marker)
        for marker in markers
        if text.find(marker) >= 0
    ]
    if not marker_positions:
        return []
    marker_index, marker = min(marker_positions, key=lambda item: item[0])
    segment_start = marker_index + len(marker)
    segment_end = min(len(text), segment_start + 36)
    for stop_word in MEAL_CONTEXT_STOP_WORDS:
        stop_index = text.find(stop_word, segment_start)
        if stop_index > segment_start and stop_index < segment_end:
            segment_end = stop_index
    segment = text[marker_index:segment_end]

    terms: list[Any] = [marker]
    for keyword, expansions in MEAL_CONTEXT_TERM_EXPANSIONS.items():
        if keyword in segment:
            terms.extend(expansions)
    return _dedupe_text(terms, limit=24)


def _score_item(
    item: dict[str, Any],
    *,
    node_terms: list[str],
    global_terms: list[str],
    constraints: dict[str, Any],
) -> tuple[float, list[str], float, int, int]:
    blob, values = _text_blob(item)
    identity_blob, identity_values = _text_blob_for_fields(item, IDENTITY_TEXT_FIELDS)
    score = 0.0
    evidence: list[str] = []
    matched = 0
    identity_matched = 0
    for term in node_terms:
        if term in values:
            score += 9.0
            matched += 1
            evidence.append(f"字段精确匹配: {term}")
        elif term in blob:
            score += 5.0
            matched += 1
            evidence.append(f"文本匹配: {term}")
        if term in identity_values or term in identity_blob:
            identity_matched += 1
    for term in global_terms:
        if term in values:
            score += 2.0
        elif term in blob:
            score += 0.8
        if term in LOCATION_ANCHOR_TERMS and (term in identity_values or term in identity_blob):
            score += 5.0
            evidence.append(f"空间锚点匹配: {term}")

    rating = to_float(item.get("rating") or item.get("score"), 0.0)
    trust = to_float(item.get("trust_score"), 0.0)
    review_count = to_float(item.get("review_count") or item.get("verified_reviews"), 0.0)
    distance_km = _candidate_distance_km(item, constraints)
    score += min(2.5, max(0.0, rating - 3.5) * 1.2)
    score += min(1.5, trust * 1.2)
    score += min(1.0, review_count / 500)
    if distance_km:
        score -= min(4.0, distance_km * 0.12)
    if item.get("available") is False:
        score -= 5.0
        evidence.append("可用性风险: 当前标记不可用")
    if item.get("available_slots"):
        score += 0.7
        evidence.append("存在可预约/可用时段")
    if item.get("deal_options") or item.get("package_options"):
        score += 0.3
        evidence.append("包含套餐/优惠券线索")
    if matched == 0:
        score -= 8.0
    return score, _dedupe_text(evidence, limit=6), distance_km, matched, identity_matched


def _field_sources(item: dict[str, Any]) -> dict[str, str]:
    existing = item.get("field_sources")
    if isinstance(existing, dict):
        return dict(existing)
    source = str(item.get("source") or item.get("source_channel") or "")
    observed = "observed_gaode" if "gaode" in source else "local_mock_or_observed"
    return {
        "identity": observed,
        "coordinates": observed if item.get("coordinates") or item.get("location") else "missing",
        "rating": "observed_gaode_or_mock",
        "price": "observed_gaode_or_mock",
        "business_hours": "observed_gaode_or_mock",
    }


def _candidate_payload(
    item: dict[str, Any],
    *,
    intent: dict[str, Any],
    score: float,
    evidence: list[str],
    distance_km: float,
) -> dict[str, Any]:
    poi_id = str(item.get("poi_id") or item.get("id") or item.get("amap_id") or "").strip()
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    payload = dict(item)
    payload.update(
        {
            "poi_id": poi_id,
            "amap_id": item.get("amap_id") or item.get("id"),
            "name": item.get("name"),
            "supply_domain": intent.get("supply_domain") or item.get("supply_domain") or item.get("type"),
            "role": intent.get("role"),
            "node_id": intent.get("node_id"),
            "city": item.get("city") or "上海",
            "address": item.get("address") or item.get("location") or raw.get("address"),
            "coordinates": item.get("coordinates") or raw.get("location") or item.get("location"),
            "retrieval_score": round(score, 4),
            "memory_retrieval_score": item.get("_memory_score"),
            "memory_rank": item.get("_memory_rank"),
            "memory_matched_terms": item.get("_memory_matched_terms"),
            "distance_km": distance_km,
            "source": "local_poi_rag",
            "evidence_text": "；".join(evidence[:4]) if evidence else "本地 POI RAG 候选，证据较弱",
            "evidence_fields": {
                "matched_evidence": evidence,
                "rating": item.get("rating"),
                "price": item.get("price"),
                "business_hours": item.get("business_hours"),
                "queue_time_min": item.get("queue_time_min"),
            },
            "field_sources": _field_sources(item),
        }
    )
    return payload


def _retrieve_for_node(
    *,
    state: PlanState,
    constraints: dict[str, Any],
    intent: dict[str, Any],
    bundle: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    domain = str(intent.get("supply_domain") or "")
    normalized_domain = _normalize_domain(domain)
    search_queries = _dedupe_text(
        [
            f"{constraints.get('city') or '上海'} {' '.join(_role_terms(intent)[:4])}",
            constraints.get("raw_text"),
        ],
        limit=5,
    )
    memory_index = _memory_index_for_domain(bundle, domain=normalized_domain)
    domain_size = int(memory_index.get("doc_count") or 0)
    if domain_size <= 0:
        return {
            "node_id": intent.get("node_id"),
            "role": intent.get("role"),
            "label": intent.get("label"),
            "supply_domain": domain,
            "coverage_status": "missing_domain",
            "search_queries": search_queries,
            "candidates": [],
            "missing_reason": f"本地 POI 索引暂未加载 {domain} 供给",
        }

    node_terms, global_terms = _node_query_terms(state, constraints, intent)
    location_terms = [term for term in global_terms if term in LOCATION_ANCHOR_TERMS]
    pool_limit = min(domain_size, max(top_k * 100, 600))
    memory_pool, retrieval_meta = retrieve_poi_memory_candidates(
        memory_index,
        node_terms=node_terms,
        global_terms=location_terms,
        limit=pool_limit,
    )
    retrieval_meta["location_anchor_terms"] = location_terms
    retrieval_meta["domain_size"] = domain_size
    retrieval_meta["index_cache"] = (
        bundle.get("_memory_index_cache_meta", {}).get(normalized_domain, {})
    )
    retrieval_meta["negative_groups"] = sorted(
        _forbidden_restaurant_groups_for_retrieval(state, constraints)
        if normalized_domain == "restaurant"
        else []
    )
    role = str(intent.get("role") or "")
    all_items: list[dict[str, Any]] | None = None
    if memory_pool:
        candidate_pool = memory_pool
        retrieval_meta["fallback_full_scan"] = False
    elif role in STRICT_ROLE_MATCH_ROLES:
        all_items = _items_from_memory_index(memory_index)
        candidate_pool = all_items
        retrieval_meta["fallback_full_scan"] = True
    else:
        all_items = _items_from_memory_index(memory_index)
        candidate_pool = _quality_fallback_pool(all_items, constraints=constraints, limit=pool_limit)
        retrieval_meta["fallback_full_scan"] = "quality_pool"
    if location_terms:
        location_pool = [
            item
            for item in candidate_pool
            if _item_matches_location_terms(item, location_terms)
        ]
        if len(location_pool) < min(top_k, 3):
            if all_items is None:
                all_items = _items_from_memory_index(memory_index)
            full_location_pool = [
                item
                for item in all_items
                if _item_matches_location_terms(item, location_terms)
            ]
            if len(full_location_pool) > len(location_pool):
                location_pool = full_location_pool
        if len(location_pool) >= min(top_k, 3):
            candidate_pool = location_pool
            retrieval_meta["location_anchor_pool_size"] = len(location_pool)
    required_identity_terms = _query_tokens(list(ROLE_REQUIRED_IDENTITY_TERMS.get(role, ())))
    excluded_identity_terms = _query_tokens(list(ROLE_EXCLUDED_IDENTITY_TERMS.get(role, ())))
    forbidden_groups = (
        _forbidden_restaurant_groups_for_retrieval(state, constraints)
        if normalized_domain == "restaurant"
        else set()
    )

    def _score_pool(pool: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any], list[str], float]]:
        pool_scores: list[tuple[float, dict[str, Any], list[str], float]] = []
        for item in pool:
            enriched_item = _merge_auxiliary_fields(item, bundle)
            if forbidden_groups and any(
                _item_has_negative_group_evidence(enriched_item, group)
                for group in forbidden_groups
            ):
                continue
            score, evidence, distance_km, matched_count, identity_match_count = _score_item(
                enriched_item,
                node_terms=node_terms,
                global_terms=global_terms,
                constraints=constraints,
            )
            if role in STRICT_ROLE_MATCH_ROLES and matched_count <= 0:
                continue
            if role in STRICT_ROLE_MATCH_ROLES and identity_match_count <= 0:
                continue
            if required_identity_terms or excluded_identity_terms:
                identity_blob, identity_values = _text_blob_for_fields(enriched_item, IDENTITY_TEXT_FIELDS)
                if required_identity_terms and not _matches_any_term(
                    identity_blob,
                    identity_values,
                    required_identity_terms,
                ):
                    continue
                if excluded_identity_terms and _matches_any_term(
                    identity_blob,
                    identity_values,
                    excluded_identity_terms,
                ):
                    continue
            if score <= -4.0:
                continue
            pool_scores.append((score, enriched_item, evidence, distance_km))
        return pool_scores

    scored = _score_pool(candidate_pool)
    if not scored and memory_pool:
        if all_items is None:
            all_items = _items_from_memory_index(memory_index)
        scored = _score_pool(all_items)
        retrieval_meta["fallback_full_scan"] = "strict_recall_guard"
    scored.sort(key=lambda row: row[0], reverse=True)
    candidates = [
        _candidate_payload(
            item,
            intent=intent,
            score=score,
            evidence=evidence,
            distance_km=distance_km,
        )
        for score, item, evidence, distance_km in scored[:top_k]
    ]
    if candidates:
        weakest_score = min(to_float(item.get("retrieval_score"), 0.0) for item in candidates)
        coverage = "weak" if weakest_score < 2.0 else "covered"
    else:
        coverage = "weak"
    return {
        "node_id": intent.get("node_id"),
        "role": intent.get("role"),
        "label": intent.get("label"),
        "supply_domain": domain,
        "coverage_status": coverage,
        "search_queries": search_queries,
        "retrieval_meta": retrieval_meta,
        "candidates": candidates,
        "missing_reason": "" if candidates else "本地 POI RAG 没有找到足够强的候选",
    }


def _should_run(state: PlanState, blueprint: dict[str, Any]) -> bool:
    constraints = state.get("constraints", {}) or {}
    explicit = state.get("b_poi_rag_enabled", constraints.get("b_poi_rag_enabled"))
    if explicit is not None:
        if isinstance(explicit, str):
            return not _falsy(explicit)
        return bool(explicit)

    env_value = os.environ.get("WF_B_RAG_ENABLED")
    if env_value is not None:
        if _falsy(env_value):
            return False
        return _truthy(env_value)

    if state.get("b_replan_request") or constraints.get("b_replan_request"):
        return True
    if state.get("b_rag_candidate_evidence") or state.get("b_rag_node_candidates"):
        return False
    if constraints.get("b_rag_candidate_evidence") or constraints.get("b_rag_node_candidates"):
        return False
    return bool(
        blueprint.get("requires_rag")
        or blueprint.get("template_mode") == "multi_node"
        or blueprint.get("unsupported_roles")
    )


def _allow_llm_requirement_compiler_for_rag(blueprint: dict[str, Any]) -> bool:
    raw_mode = os.environ.get("WF_B_RAG_REQUIREMENT_COMPILER_MODE", "").strip().lower()
    if raw_mode in {"longcat", "llm", "ai", "remote"}:
        return True
    if raw_mode in {"deterministic", "rule", "rules", "off", "0", "false", "no"}:
        return False
    return False


def b_poi_rag_node(state: PlanState) -> dict[str, Any]:
    """Retrieve local POI evidence for B and emit ``b_rag_candidate_evidence``."""

    execution_log = list(state.get("execution_log", []) or [])
    constraints = dict(state.get("constraints", {}) or {})
    initial_blueprint = (
        state.get("b_itinerary_blueprint")
        or constraints.get("b_itinerary_blueprint")
        or build_b_itinerary_blueprint(state, constraints=constraints)
    )

    if not _should_run(state, initial_blueprint):
        return {}

    constraints, requirement_contract, requirement_metadata = apply_b_requirement_contract(
        state,
        constraints=constraints,
        allow_llm=_allow_llm_requirement_compiler_for_rag(initial_blueprint),
    )
    if requirement_metadata and requirement_metadata.get("success"):
        execution_log.append("[B] b_poi_rag_node applied LongCat requirement compiler before retrieval")
    elif requirement_metadata and requirement_metadata.get("skipped"):
        execution_log.append("[B] b_poi_rag_node used deterministic requirement compiler before retrieval")
    elif requirement_contract.get("hard_requirements"):
        execution_log.append("[B] b_poi_rag_node applied deterministic requirement compiler before retrieval")

    blueprint = (
        constraints.get("b_itinerary_blueprint")
        if isinstance(constraints.get("b_itinerary_blueprint"), dict)
        else None
    ) or build_b_itinerary_blueprint(state, constraints=constraints)
    constraints["b_itinerary_blueprint"] = blueprint

    root = _mock_data_dir()
    bundle = _supply_bundle(root)
    top_k = _top_k_per_node()
    node_evidence = [
        _retrieve_for_node(
            state=state,
            constraints=constraints,
            intent=intent,
            bundle=bundle,
            top_k=top_k,
        )
        for intent in blueprint.get("node_intents", []) or []
    ]
    covered = [
        block.get("node_id")
        for block in node_evidence
        if block.get("coverage_status") == "covered" and block.get("candidates")
    ]
    weak_or_missing = [
        {
            "node_id": block.get("node_id"),
            "role": block.get("role"),
            "coverage_status": block.get("coverage_status"),
            "missing_reason": block.get("missing_reason"),
        }
        for block in node_evidence
        if block.get("coverage_status") != "covered" or not block.get("candidates")
    ]
    evidence = {
        "version": SUPPORTED_CONTRACT_VERSION,
        "query": str(state.get("user_input") or constraints.get("raw_text") or ""),
        "city": constraints.get("city") or "上海",
        "blueprint_version": blueprint.get("version"),
        "blueprint_template_mode": blueprint.get("template_mode"),
        "node_evidence": node_evidence,
        "retrieval_trace": [
            {
                "step": "local_poi_rag",
                "retriever": "local_poi_memory_bm25_v0+structured_rerank",
                "data_dir": str(root),
                "top_k_per_node": top_k,
                "covered_node_count": len(covered),
                "weak_or_missing_node_count": len(weak_or_missing),
            }
        ],
        "coverage_summary": {
            "covered_node_ids": covered,
            "weak_or_missing_nodes": weak_or_missing,
        },
    }
    execution_log.append(
        "[B] b_poi_rag_node retrieved local POI evidence "
        f"(nodes={len(node_evidence)}, covered={len(covered)}, data_dir={root.name})"
    )
    return {
        "constraints": constraints,
        "b_itinerary_blueprint": blueprint,
        "b_rag_candidate_evidence": evidence,
        "b_poi_rag_metadata": evidence["retrieval_trace"][0],
        "execution_log": execution_log,
    }
