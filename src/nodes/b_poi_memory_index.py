"""Local POI memory index used by B-stage RAG retrieval.

The first production-shaped RAG layer should not force B to scan every POI for
every itinerary node. This module builds a small in-process lexical memory over
local POI documents, then returns a narrow candidate pool for B's structured
reranker. It is intentionally dependency-free so it can run in the hackathon
environment and later be swapped for a vector store behind the same contract.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .b_semantics import flatten_semantic_values, normalize_semantic_text


DEFAULT_DOCUMENT_FIELDS = (
    "name",
    "category",
    "sub_category",
    "restaurant_category",
    "primary_category",
    "primary_keyword",
    "gaode_keyword",
    "gaode_type",
    "address",
    "business_area",
    "tags",
    "review_keywords",
    "signature_dishes",
    "recommended_dishes",
    "dish_tags",
    "package_options",
    "_gaode_raw_business",
)

DOMAIN_NOISE_TERMS = {
    "activity",
    "restaurant",
    "餐厅",
    "餐饮",
    "活动",
    "吃饭",
    "正餐",
    "附近",
    "上海",
}

CONTROLLED_SUBSTRING_TERMS = (
    "亲子",
    "儿童",
    "儿童友好",
    "手作",
    "陶艺",
    "绘画",
    "DIY",
    "室内",
    "低强度",
    "室内乐园",
    "亲子乐园",
    "儿童乐园",
    "游乐园",
    "淘气堡",
    "蹦床",
    "展览",
    "看展",
    "博物馆",
    "美术馆",
    "艺术展",
    "文物展",
    "漆器展",
    "城市漫步",
    "citywalk",
    "市集",
    "街区",
    "桌游",
    "棋牌",
    "剧本杀",
    "狼人杀",
    "密室",
    "密室逃脱",
    "KTV",
    "ktv",
    "唱歌",
    "卡拉OK",
    "练歌房",
    "清淡",
    "轻食",
    "低卡",
    "健康",
    "少油",
    "中式轻食",
    "沙拉",
    "本帮菜",
    "上海菜",
    "家常菜",
    "江浙菜",
    "火锅",
    "潮汕牛肉火锅",
    "烤肉",
    "烧烤",
    "日式烧肉",
    "羊肉串",
    "咖啡",
    "咖啡馆",
    "下午茶",
    "甜品",
    "蛋糕",
    "烘焙",
    "伴手礼",
    "特产",
    "礼品",
    "美妆",
    "日化",
    "鲜花",
    "花店",
    "足疗",
    "按摩",
    "便利店",
    "超市",
    "停车",
    "停车场",
    "酒店",
    "住宿",
    "民宿",
    "早餐",
    "早饭",
    "早点",
    "包子",
    "小笼包",
    "小笼",
    "汤包",
    "生煎",
    "生煎包",
    "馄饨",
    "小馄饨",
    "豆浆",
    "粥",
    "油条",
)

MAX_DOCUMENT_TERMS = 520
MAX_POSTING_FRACTION = 0.42
POI_MEMORY_INDEX_VERSION = "local_poi_memory_bm25_v1"
POI_MEMORY_CACHE_VERSION = "local_poi_memory_cache_v3"
_NORMALIZED_CONTROLLED_SUBSTRING_TERMS: tuple[str, ...] | None = None


def _normalize(value: Any) -> str:
    return normalize_semantic_text(str(value or "").strip())


def _controlled_terms() -> tuple[str, ...]:
    global _NORMALIZED_CONTROLLED_SUBSTRING_TERMS
    if _NORMALIZED_CONTROLLED_SUBSTRING_TERMS is None:
        seen: set[str] = set()
        terms: list[str] = []
        for term in CONTROLLED_SUBSTRING_TERMS:
            normalized = _normalize(term)
            if normalized and normalized not in seen:
                seen.add(normalized)
                terms.append(normalized)
        _NORMALIZED_CONTROLLED_SUBSTRING_TERMS = tuple(terms)
    return _NORMALIZED_CONTROLLED_SUBSTRING_TERMS


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _cjk_runs(text: str) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    for char in text:
        if _is_cjk(char):
            current.append(char)
        elif current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    return runs


def _char_ngrams(text: str, *, min_n: int = 2, max_n: int = 4) -> list[str]:
    grams: list[str] = []
    for run in _cjk_runs(text):
        if len(run) < min_n:
            continue
        for n in range(min_n, min(max_n, len(run)) + 1):
            grams.extend(run[index : index + n] for index in range(0, len(run) - n + 1))
    return grams


def _term_stream(values: list[Any]) -> list[str]:
    terms: list[str] = []
    for raw_value in flatten_semantic_values(values):
        text = _normalize(raw_value)
        if len(text) < 2:
            continue
        terms.append(text)
        for part in re.split(r"[\s,，。；;、|/+()（）·\-_:：]+", text):
            part = part.strip()
            if len(part) >= 2:
                terms.append(part)
        terms.extend(re.findall(r"[a-z0-9]{2,}", text.lower()))
    return terms


def _document_values(item: dict[str, Any], fields: tuple[str, ...]) -> list[Any]:
    values: list[Any] = []
    for field in fields:
        if field == "_gaode_raw_business":
            values.extend(_raw_business_values(item))
        else:
            values.extend(flatten_semantic_values(item.get(field)))
    return values


def _raw_business_values(item: dict[str, Any]) -> list[str]:
    """Extract useful Gaode raw business text without noisy source-query traces."""

    raw = item.get("raw")
    if not isinstance(raw, dict):
        return []

    nested_raw = raw.get("raw") if isinstance(raw.get("raw"), dict) else {}
    candidates = [raw, nested_raw]
    business_blocks = []
    for payload in candidates:
        for key in ("biz_ext", "business"):
            value = payload.get(key)
            if isinstance(value, dict):
                business_blocks.append(value)

    values: list[Any] = []
    for payload in candidates:
        for key in ("name", "type", "address", "adname", "cityname", "pname"):
            values.append(payload.get(key))
    for block in business_blocks:
        for key in (
            "tag",
            "rectag",
            "keytag",
            "alias",
            "business_area",
            "opentime_today",
            "opentime_week",
        ):
            values.append(block.get(key))
    return [
        str(value).strip()
        for value in flatten_semantic_values(values)
        if str(value or "").strip()
    ]


def _document_terms(item: dict[str, Any], fields: tuple[str, ...]) -> Counter[str]:
    counter: Counter[str] = Counter()
    values = _document_values(item, fields)
    for term in _term_stream(values):
        if len(counter) >= MAX_DOCUMENT_TERMS and term not in counter:
            continue
        counter[term] += 1
    document_blob = "\n".join(
        text
        for text in (_normalize(value) for value in flatten_semantic_values(values))
        if len(text) >= 2
    )
    for term in _controlled_terms():
        if term in document_blob:
            counter[term] += 2
    return counter


def build_poi_memory_index(
    items: list[dict[str, Any]],
    *,
    fields: tuple[str, ...] = DEFAULT_DOCUMENT_FIELDS,
) -> dict[str, Any]:
    """Build a BM25-like memory index over POI documents."""

    documents: list[dict[str, Any]] = []
    postings: dict[str, list[tuple[int, int]]] = {}
    total_len = 0
    for doc_id, item in enumerate(items):
        terms = _document_terms(item, fields)
        doc_len = max(1, sum(terms.values()))
        total_len += doc_len
        documents.append({"item": item, "length": doc_len})
        for term, tf in terms.items():
            postings.setdefault(term, []).append((doc_id, tf))

    total_docs = len(documents)
    return {
        "version": POI_MEMORY_INDEX_VERSION,
        "documents": documents,
        "postings": postings,
        "doc_count": total_docs,
        "avg_doc_len": (total_len / total_docs) if total_docs else 1.0,
    }


def load_poi_memory_index_cache(
    cache_path: Path,
    *,
    expected_signature: str,
) -> dict[str, Any] | None:
    """Load a local POI memory index cache when its data signature matches."""

    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    index = payload.get("index")
    if not isinstance(metadata, dict) or not isinstance(index, dict):
        return None
    if metadata.get("cache_version") != POI_MEMORY_CACHE_VERSION:
        return None
    if metadata.get("index_version") != POI_MEMORY_INDEX_VERSION:
        return None
    if metadata.get("signature") != expected_signature:
        return None
    return index


def save_poi_memory_index_cache(
    cache_path: Path,
    *,
    signature: str,
    index: dict[str, Any],
) -> bool:
    """Persist a local POI memory index cache with an atomic replace."""

    payload = {
        "metadata": {
            "cache_version": POI_MEMORY_CACHE_VERSION,
            "index_version": POI_MEMORY_INDEX_VERSION,
            "signature": signature,
            "doc_count": index.get("doc_count"),
        },
        "index": index,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        return True
    except Exception:
        return False


def _query_terms(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    raw_terms = _term_stream(values)
    query_blob = "\n".join(
        text
        for text in (_normalize(value) for value in flatten_semantic_values(values))
        if len(text) >= 2
    )
    raw_terms.extend(term for term in _controlled_terms() if term in query_blob)
    for term in raw_terms:
        if term in DOMAIN_NOISE_TERMS or len(term) < 2 or term in seen:
            continue
        seen.add(term)
        result.append(term)
    return result


def retrieve_poi_memory_candidates(
    index: dict[str, Any],
    *,
    node_terms: list[str],
    global_terms: list[str] | None = None,
    limit: int = 800,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve a narrow POI candidate pool plus traceable match evidence."""

    doc_count = int(index.get("doc_count") or 0)
    if doc_count <= 0:
        return [], {
            "retriever": index.get("version") or "local_poi_memory_bm25_v0",
            "query_terms": [],
            "matched_terms": [],
            "candidate_pool_size": 0,
        }

    query_terms = _query_terms(list(node_terms or []) + list(global_terms or [])[:12])
    if not query_terms:
        return [], {
            "retriever": index.get("version") or "local_poi_memory_bm25_v0",
            "query_terms": [],
            "matched_terms": [],
            "candidate_pool_size": 0,
        }

    postings: dict[str, list[tuple[int, int]]] = index.get("postings", {})
    documents: list[dict[str, Any]] = index.get("documents", [])
    avg_doc_len = float(index.get("avg_doc_len") or 1.0)
    scores: dict[int, float] = {}
    matched_by_doc: dict[int, list[str]] = {}
    matched_terms: list[str] = []
    k1 = 1.2
    b = 0.75

    for term in query_terms:
        term_postings = postings.get(term)
        if not term_postings:
            continue
        if len(term_postings) > max(60, int(doc_count * MAX_POSTING_FRACTION)):
            continue
        matched_terms.append(term)
        df = len(term_postings)
        idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
        for doc_id, tf in term_postings:
            doc_len = float(documents[doc_id].get("length") or avg_doc_len)
            denom = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (k1 + 1) / denom)
            matched_by_doc.setdefault(doc_id, []).append(term)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    candidates: list[dict[str, Any]] = []
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        record = dict(documents[doc_id]["item"])
        record["_memory_rank"] = rank
        record["_memory_score"] = round(score, 4)
        record["_memory_matched_terms"] = matched_by_doc.get(doc_id, [])[:12]
        candidates.append(record)

    return candidates, {
        "retriever": index.get("version") or "local_poi_memory_bm25_v0",
        "query_terms": query_terms[:32],
        "matched_terms": matched_terms[:32],
        "candidate_pool_size": len(candidates),
    }
