"""Fast normalized text matching for B candidate retrieval and ranking."""
from __future__ import annotations

from .b_semantics import b_semantic_terms, flatten_semantic_values, normalize_semantic_text


SEMANTIC_TEXT_CACHE_LIMIT = 60000
_SEMANTIC_TEXT_CACHE: dict[tuple[int, tuple[str, ...]], tuple[tuple, str, set[str]]] = {}


def semantic_cache_signature(item: dict) -> tuple:
    return (
        item.get("poi_id") or item.get("id"),
        item.get("name"),
        item.get("category"),
        item.get("sub_category"),
        item.get("experience_type"),
        item.get("restaurant_category"),
        item.get("primary_category"),
        item.get("primary_keyword"),
        item.get("gaode_keyword"),
    )


def normalized_query_terms(values, *, expand_semantics: bool = False) -> list[str]:
    raw_terms = b_semantic_terms(values, include_auxiliary=True) if expand_semantics else flatten_semantic_values(values)
    result: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        normalized = normalize_semantic_text(term)
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def semantic_text_index(
    item: dict,
    *,
    fields: tuple[str, ...],
) -> tuple[str, set[str]]:
    cache_key = (id(item), fields)
    signature = semantic_cache_signature(item)
    cached = _SEMANTIC_TEXT_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1], cached[2]

    values: list[str] = []
    for field_name in fields:
        values.extend(flatten_semantic_values(item.get(field_name)))

    normalized_values = [
        normalize_semantic_text(value)
        for value in values
        if len(normalize_semantic_text(value)) >= 2
    ]
    value_set = set(normalized_values)
    blob = "\n".join(normalized_values)

    if len(_SEMANTIC_TEXT_CACHE) >= SEMANTIC_TEXT_CACHE_LIMIT:
        _SEMANTIC_TEXT_CACHE.clear()
    _SEMANTIC_TEXT_CACHE[cache_key] = (signature, blob, value_set)
    return blob, value_set


def fast_text_match_score(
    normalized_terms: list[str],
    item: dict,
    *,
    fields: tuple[str, ...],
) -> float:
    if not normalized_terms or not item:
        return 0.0
    blob, value_set = semantic_text_index(item, fields=fields)
    if not blob:
        return 0.0

    best = 0.0
    for term in normalized_terms:
        if term in value_set:
            best = max(best, 3.2)
        elif term in blob:
            best = max(best, 2.3)
    return best


def item_matches_terms(
    item: dict,
    terms: tuple[str, ...],
    *,
    fields: tuple[str, ...],
) -> bool:
    return fast_text_match_score(
        normalized_query_terms(list(terms), expand_semantics=False),
        item,
        fields=fields,
    ) > 0
