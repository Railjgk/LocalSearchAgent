#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LocalSearchBench Shanghai RAG probe for WeekendFlow B.

This script is intentionally experimental. It does not modify the production B
planner. It asks whether a light local retrieval layer can expose multi-node
local-life intents and evidence before the legacy activity+restaurant planner
collapses the request into a fixed pair.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/localsearchbench/localsearchbench/"
    "resolve/main/data/train-00000-of-00001.parquet"
)
DEFAULT_DATASET_PATH = (
    REPO_ROOT
    / "experiments"
    / "artifacts"
    / "localsearchbench"
    / "train-00000-of-00001.parquet"
)
DEFAULT_SUPPLY_DIR = (
    REPO_ROOT
    / "experiments"
    / "mock_data"
    / "gaode_supply_shanghai_v2_20260527_full"
)
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "experiments" / "artifacts"


ROLE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "exhibition": {
        "label": "展览/博物馆",
        "domain": "activity",
        "keywords": ["展览", "看展", "博物馆", "美术馆", "展厅", "珍品展"],
        "query_terms": ["展览", "博物馆", "美术馆"],
    },
    "entertainment": {
        "label": "娱乐/演出",
        "domain": "activity",
        "keywords": ["演出", "剧场", "剧院", "脱口秀", "音乐会", "密室", "桌游", "KTV"],
        "query_terms": ["演出", "剧场", "密室", "桌游"],
    },
    "restaurant_lunch": {
        "label": "午餐/中饭",
        "domain": "restaurant",
        "keywords": ["中饭", "午饭", "午餐", "吃个中饭", "吃午饭"],
        "query_terms": ["餐厅", "中餐", "小吃快餐", "简餐"],
    },
    "restaurant_specific": {
        "label": "指定餐饮",
        "domain": "restaurant",
        "keywords": [
            "吃",
            "餐厅",
            "面馆",
            "蟹黄面",
            "蟹粉面",
            "本帮菜",
            "小吃",
            "火锅",
            "烤肉",
            "日料",
            "咖啡",
        ],
        "query_terms": ["餐厅", "小吃", "本帮菜", "面馆"],
    },
    "cafe": {
        "label": "咖啡/甜品",
        "domain": "restaurant",
        "keywords": ["咖啡", "咖啡店", "咖啡馆", "下午茶", "甜品"],
        "query_terms": ["咖啡", "甜品", "下午茶"],
    },
    "souvenir_shopping": {
        "label": "特产/伴手礼",
        "domain": "shopping",
        "keywords": ["特产", "伴手礼", "礼品", "礼物", "带回去", "送朋友"],
        "query_terms": ["特产", "伴手礼", "礼品", "购物"],
    },
    "shopping": {
        "label": "购物",
        "domain": "shopping",
        "keywords": ["购物", "买", "商场", "百货", "南京路", "步行街"],
        "query_terms": ["购物", "商场", "百货"],
    },
    "lodging": {
        "label": "酒店/住宿",
        "domain": "hotel",
        "keywords": ["住宿", "住处", "酒店", "民宿", "套房", "家庭房", "有厨房", "住一晚"],
        "query_terms": ["酒店", "民宿", "住宿", "家庭房", "厨房"],
    },
    "convenience_store": {
        "label": "便利店/日用品",
        "domain": "shopping",
        "keywords": ["便利店", "日用品", "超市", "买水", "生活用品"],
        "query_terms": ["便利店", "超市", "日用品"],
    },
    "parking": {
        "label": "停车",
        "domain": "transport_service",
        "keywords": ["停车", "停车场", "好停车"],
        "query_terms": ["停车场", "停车"],
    },
    "life_service": {
        "label": "生活服务",
        "domain": "life_service",
        "keywords": ["理发", "洗衣", "药店", "医院", "银行", "厕所", "卫生间"],
        "query_terms": ["生活服务"],
    },
}

DOMAIN_TYPE_HINTS = {
    "activity": ["体育休闲服务", "科教文化服务", "风景名胜", "博物馆", "美术馆", "展览"],
    "restaurant": ["餐饮服务", "餐厅", "咖啡", "小吃", "面馆"],
    "shopping": ["购物服务", "商场", "百货", "超市", "便利店", "特产", "礼品"],
    "hotel": ["住宿服务", "酒店", "宾馆", "民宿", "公寓"],
    "transport_service": ["交通设施服务", "停车场", "停车"],
    "life_service": ["生活服务", "医疗保健服务", "公共设施"],
}

LOCATION_HINTS = [
    "人民广场",
    "外滩",
    "南京路",
    "豫园",
    "陆家嘴",
    "静安寺",
    "徐家汇",
    "淮海路",
    "新天地",
    "田子坊",
    "五角场",
    "虹桥",
    "浦东",
    "黄浦",
    "静安",
    "徐汇",
    "长宁",
]


@dataclass
class EvidenceDoc:
    poi_id: str
    name: str
    domain: str
    primary_category: str
    gaode_type: str
    address: str
    district: str
    business_area: str
    rating: str | float | None
    price: str | float | None
    business_hours: str | dict[str, Any] | None
    source: str
    source_queries: list[str]
    mock_field_source: dict[str, list[str]]
    text: str
    raw: dict[str, Any]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _flatten(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            values.extend(_flatten(nested))
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            values.extend(_flatten(nested))
    elif value not in (None, ""):
        values.append(str(value))
    return values


def _first_dict_value(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        first = next(iter(data.values()), {})
        return first if isinstance(first, dict) else {}
    return {}


def ensure_dataset(path: Path, url: str = DEFAULT_DATASET_URL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def _rating_cost_from_raw(poi: dict[str, Any]) -> tuple[Any, Any, Any]:
    biz_ext = poi.get("biz_ext") if isinstance(poi.get("biz_ext"), dict) else {}
    raw = poi.get("raw") if isinstance(poi.get("raw"), dict) else {}
    business = raw.get("business") if isinstance(raw.get("business"), dict) else {}
    rating = biz_ext.get("rating") or business.get("rating")
    cost = biz_ext.get("cost") or business.get("cost")
    hours = biz_ext.get("opentime_week") or business.get("opentime_week")
    return rating, cost, hours


def infer_domain(poi: dict[str, Any], enriched: dict[str, Any] | None = None) -> str:
    if enriched and enriched.get("type") in {"activity", "restaurant"}:
        return str(enriched["type"])
    expected = str(poi.get("expected_type") or "")
    if expected in {"activity", "restaurant"}:
        return expected
    text = " ".join(
        [
            str(poi.get("type") or ""),
            str(poi.get("primary_category") or ""),
            str(poi.get("primary_keyword") or ""),
            str(poi.get("name") or ""),
        ]
    )
    for domain, hints in DOMAIN_TYPE_HINTS.items():
        if any(hint in text for hint in hints):
            return domain
    return "poi"


def _compact_source_queries(poi: dict[str, Any], limit: int = 8) -> list[str]:
    queries: list[str] = []
    for item in _as_list(poi.get("source_queries")):
        if isinstance(item, dict) and item.get("keyword"):
            queries.append(str(item["keyword"]))
    result: list[str] = []
    seen: set[str] = set()
    for query in queries:
        if query not in seen:
            seen.add(query)
            result.append(query)
        if len(result) >= limit:
            break
    return result


def _fields_present(item: dict[str, Any] | None, fields: list[str]) -> list[str]:
    if not isinstance(item, dict):
        return []
    return [field for field in fields if item.get(field) not in (None, "", [], {})]


def build_evidence_docs(supply_dir: Path) -> list[EvidenceDoc]:
    deduped = _read_json(supply_dir / "deduped_pois.json")
    activities = {item.get("amap_id") or item.get("poi_id", "").replace("gaode_act_", ""): item for item in _read_json(supply_dir / "activities.json")}
    restaurants = {item.get("amap_id") or item.get("poi_id", "").replace("gaode_res_", ""): item for item in _read_json(supply_dir / "restaurants.json")}
    products = {item.get("poi_id"): item for item in _read_json(supply_dir / "products.json")}
    deals = {item.get("poi_id"): item for item in _read_json(supply_dir / "deals.json")}
    availability = _read_json(supply_dir / "availability.json")

    docs: list[EvidenceDoc] = []
    for poi in deduped:
        amap_id = str(poi.get("id") or "")
        enriched = activities.get(amap_id) or restaurants.get(amap_id) or {}
        poi_id = str(enriched.get("poi_id") or f"gaode_{amap_id}")
        rating, cost, raw_hours = _rating_cost_from_raw(poi)
        rating = enriched.get("rating", rating)
        price = enriched.get("price", cost)
        business_hours = enriched.get("business_hours") or raw_hours
        business_area = ""
        biz_ext = poi.get("biz_ext") if isinstance(poi.get("biz_ext"), dict) else {}
        raw_business = ((poi.get("raw") or {}).get("business") or {}) if isinstance(poi.get("raw"), dict) else {}
        business_area = str(biz_ext.get("business_area") or raw_business.get("business_area") or "")
        domain = infer_domain(poi, enriched)
        source_queries = _compact_source_queries(poi)
        product = products.get(poi_id)
        deal = deals.get(poi_id)
        avail = availability.get(poi_id) if isinstance(availability, dict) else None

        text_parts = [
            poi.get("name"),
            poi.get("type"),
            poi.get("address"),
            poi.get("adname"),
            business_area,
            poi.get("primary_category"),
            poi.get("primary_keyword"),
            enriched.get("category"),
            enriched.get("sub_category"),
            enriched.get("experience_type"),
            enriched.get("restaurant_category"),
            enriched.get("gaode_keyword"),
            enriched.get("gaode_type"),
            enriched.get("source_evidence"),
            enriched.get("tags"),
            enriched.get("review_keywords"),
            enriched.get("signature_dishes"),
            enriched.get("recommended_dishes"),
            product.get("name") if isinstance(product, dict) else None,
            product.get("package_components") if isinstance(product, dict) else None,
            deal.get("title") if isinstance(deal, dict) else None,
            deal.get("coupon_type") if isinstance(deal, dict) else None,
            source_queries,
        ]
        docs.append(
            EvidenceDoc(
                poi_id=poi_id,
                name=str(enriched.get("name") or poi.get("name") or ""),
                domain=domain,
                primary_category=str(enriched.get("category") or poi.get("primary_category") or ""),
                gaode_type=str(enriched.get("gaode_type") or poi.get("type") or ""),
                address=str(enriched.get("location") or poi.get("address") or ""),
                district=str(poi.get("adname") or ""),
                business_area=business_area,
                rating=rating,
                price=price,
                business_hours=business_hours,
                source=str(enriched.get("source_channel") or "gaode_poi_search"),
                source_queries=source_queries,
                mock_field_source={
                    "merchant_evidence": _fields_present(enriched, ["merchant_id", "trust_score", "review_count", "operation_stability_score"]),
                    "product_evidence": _fields_present(product, ["product_id", "product_type", "name", "price", "package_components"]),
                    "deal_evidence": _fields_present(deal, ["deal_id", "title", "sale_price", "valid_time", "refund_policy"]),
                    "availability_evidence": _fields_present(avail, ["available", "available_slots", "queue_time_min", "inventory_left"]),
                    "route_location_evidence": _fields_present(enriched, ["coordinates", "latitude", "longitude", "distance_km"]),
                    "business_hours_price_rating_evidence": [
                        field
                        for field, value in {"business_hours": business_hours, "price": price, "rating": rating}.items()
                        if value not in (None, "", [], {})
                    ],
                },
                text=" ".join(_flatten(text_parts)),
                raw=poi,
            )
        )
    return docs


class LocalRetriever:
    def __init__(self, docs: list[EvidenceDoc]) -> None:
        self.docs = docs
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, max_features=220000)
        self.matrix = self.vectorizer.fit_transform([doc.text for doc in docs])

    def search(self, query: str, *, role: str, top_k: int = 5) -> list[dict[str, Any]]:
        definition = ROLE_DEFINITIONS.get(role, {})
        domain = definition.get("domain")
        expanded_query = " ".join([query] + definition.get("query_terms", []))
        query_vec = self.vectorizer.transform([expanded_query])
        scores = linear_kernel(query_vec, self.matrix).ravel()
        candidate_count = min(len(scores), max(300, top_k * 80))
        top_indexes = scores.argsort()[-candidate_count:][::-1]

        ranked: list[tuple[float, int]] = []
        for index in top_indexes:
            doc = self.docs[int(index)]
            score = float(scores[int(index)])
            boost = self._heuristic_boost(doc, query, role, domain)
            ranked.append((score + boost, int(index)))
        ranked.sort(reverse=True, key=lambda item: item[0])
        return [self._public_doc(self.docs[index], score) for score, index in ranked[:top_k]]

    def _heuristic_boost(self, doc: EvidenceDoc, query: str, role: str, domain: str | None) -> float:
        boost = 0.0
        if domain and doc.domain == domain:
            boost += 0.16
        elif domain == "shopping" and doc.domain in {"shopping", "poi"}:
            boost += 0.06
        elif domain == "hotel" and any(hint in doc.text for hint in DOMAIN_TYPE_HINTS["hotel"]):
            boost += 0.14
        elif domain == "transport_service" and "停车" in doc.text:
            boost += 0.14

        role_terms = ROLE_DEFINITIONS.get(role, {}).get("query_terms", [])
        boost += min(0.18, 0.04 * sum(1 for term in role_terms if term in doc.text))
        for location in extract_location_hints(query):
            if location in doc.text:
                boost += 0.12
        exact_entities = extract_quoted_entities(query)
        if any(entity and entity in doc.text for entity in exact_entities):
            boost += 0.28
        return boost

    def _public_doc(self, doc: EvidenceDoc, score: float) -> dict[str, Any]:
        return {
            "score": round(score, 4),
            "poi_id": doc.poi_id,
            "name": doc.name,
            "domain": doc.domain,
            "primary_category": doc.primary_category,
            "gaode_type": doc.gaode_type,
            "address": doc.address,
            "district": doc.district,
            "business_area": doc.business_area,
            "rating": doc.rating,
            "price": doc.price,
            "business_hours": doc.business_hours,
            "source": doc.source,
            "source_queries": doc.source_queries[:5],
            "mock_field_source": doc.mock_field_source,
        }


def extract_quoted_entities(text: str) -> list[str]:
    entities = re.findall(r"[“\"]([^”\"]{2,80})[”\"]", text or "")
    return _dedupe(entities)


def extract_location_hints(text: str) -> list[str]:
    return [hint for hint in LOCATION_HINTS if hint in (text or "")]


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _nearby_query_terms(question: str, role: str) -> list[str]:
    terms = list(ROLE_DEFINITIONS.get(role, {}).get("query_terms", []))
    terms.extend(extract_location_hints(question))
    for entity in extract_quoted_entities(question):
        if role == "exhibition":
            terms.append(entity)
    food_terms = ["蟹黄面", "蟹粉面", "本帮菜", "咖啡", "小吃", "火锅", "烤肉", "日料"]
    terms.extend(term for term in food_terms if term in question)
    if "有厨房" in question and role == "lodging":
        terms.append("有厨房")
    return _dedupe(terms)


def extract_node_intents(question: str) -> list[dict[str, Any]]:
    hits: list[tuple[int, str, list[str]]] = []
    lowered = question.lower()
    for role, definition in ROLE_DEFINITIONS.items():
        matched: list[tuple[int, str]] = []
        for keyword in definition["keywords"]:
            index = lowered.find(keyword.lower())
            if index >= 0:
                matched.append((index, keyword))
        if matched:
            matched.sort(key=lambda item: item[0])
            hits.append((matched[0][0], role, _dedupe([term for _, term in matched])))

    hits.sort(key=lambda item: item[0])
    roles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, role, matched_terms in hits:
        if role == "restaurant_specific" and "cafe" in seen and matched_terms == ["咖啡"]:
            continue
        if role == "shopping" and ("souvenir_shopping" in seen or "convenience_store" in seen):
            continue
        if role in seen:
            continue
        seen.add(role)
        definition = ROLE_DEFINITIONS[role]
        roles.append(
            {
                "node_id": f"intent_{len(roles) + 1:02d}",
                "role": role,
                "label": definition["label"],
                "supply_domain": definition["domain"],
                "matched_terms": matched_terms,
                "search_terms": _nearby_query_terms(question, role),
            }
        )

    if not roles:
        roles.append(
            {
                "node_id": "intent_01",
                "role": "general_local_search",
                "label": "通用本地搜索",
                "supply_domain": "poi",
                "matched_terms": [],
                "search_terms": extract_location_hints(question) or [question[:24]],
            }
        )
    return roles


def parse_search_hops(search_path: str) -> list[dict[str, Any]]:
    hops: list[dict[str, Any]] = []
    pattern = re.compile(r"第([一二三四五六七八九十\d]+)跳:\s*搜索\"([^\"]+)\"")
    for match in pattern.finditer(search_path or ""):
        query = match.group(2)
        roles = [intent["role"] for intent in extract_node_intents(query)]
        hops.append({"hop": match.group(1), "query": query, "roles": roles})
    return hops


def parse_answer_sequence(answer: str) -> list[str]:
    text = answer or ""
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        text = boxed.group(1)
    text = re.sub(r"参考答案[:：]|最终参考答案[:：]", "", text)
    parts = re.split(r"→|->|；|;|\n", text)
    cleaned: list[str] = []
    for part in parts:
        item = re.sub(r"（.*?）|\(.*?\)", "", part).strip(" ：:，,。")
        if len(item) >= 2:
            cleaned.append(item)
    return _dedupe(cleaned)[:10]


def task_types_for_roles(roles: list[str]) -> list[str]:
    mapping = {
        "restaurant_lunch": "餐饮",
        "restaurant_specific": "餐饮",
        "cafe": "餐饮",
        "exhibition": "娱乐",
        "entertainment": "娱乐",
        "shopping": "购物",
        "souvenir_shopping": "购物",
        "lodging": "酒店/住宿",
        "parking": "停车/便利店/生活服务",
        "convenience_store": "停车/便利店/生活服务",
        "life_service": "停车/便利店/生活服务",
    }
    result = _dedupe([mapping.get(role, "其他") for role in roles])
    if len(result) >= 2:
        result.append("多节点混合 itinerary")
    return result


def answer_coverage(answer_names: list[str], node_results: list[dict[str, Any]]) -> dict[str, Any]:
    retrieved = []
    for node in node_results:
        for item in node.get("top_evidence", []):
            retrieved.append(str(item.get("name") or ""))
    matched: list[str] = []
    for answer in answer_names:
        for name in retrieved:
            if answer and name and (answer in name or name in answer):
                matched.append(answer)
                break
    return {
        "answer_items": answer_names,
        "matched_answer_items": _dedupe(matched),
        "matched_count": len(_dedupe(matched)),
        "total_answer_items": len(answer_names),
        "coverage": round(len(_dedupe(matched)) / len(answer_names), 3) if answer_names else None,
    }


def hop_role_coverage(benchmark_hops: list[dict[str, Any]], predicted_roles: list[str]) -> dict[str, Any]:
    expected_roles = _dedupe([role for hop in benchmark_hops for role in hop.get("roles", [])])
    matched = [role for role in expected_roles if role in predicted_roles]
    return {
        "expected_roles_from_search_path": expected_roles,
        "predicted_roles": predicted_roles,
        "matched_roles": matched,
        "coverage": round(len(matched) / len(expected_roles), 3) if expected_roles else None,
    }


def load_shanghai_rows(dataset_path: Path, limit: int) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    shanghai = df[df["City"].eq("上海")].copy()
    if limit:
        shanghai = shanghai.head(limit)
    return shanghai


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = ensure_dataset(Path(args.dataset_path))
    rows = load_shanghai_rows(dataset_path, args.limit)
    docs = build_evidence_docs(Path(args.supply_dir))
    retriever = LocalRetriever(docs)

    results: list[dict[str, Any]] = []
    task_type_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    hop_coverages: list[float] = []
    answer_coverages: list[float] = []

    for row_index, row in rows.iterrows():
        question = str(row["Question"])
        intents = extract_node_intents(question)
        node_results = []
        for intent in intents:
            role = intent["role"]
            query = " ".join([question] + intent.get("search_terms", []))
            top_evidence = retriever.search(query, role=role, top_k=args.top_k)
            node_results.append({**intent, "top_evidence": top_evidence})

        predicted_roles = [intent["role"] for intent in intents]
        benchmark_hops = parse_search_hops(str(row["Multi-hop search path"]))
        answer_names = parse_answer_sequence(str(row["Answer"]))
        hop_cov = hop_role_coverage(benchmark_hops, predicted_roles)
        ans_cov = answer_coverage(answer_names, node_results)
        types = task_types_for_roles(predicted_roles)
        task_type_counts.update(types)
        role_counts.update(predicted_roles)
        if isinstance(hop_cov["coverage"], float):
            hop_coverages.append(hop_cov["coverage"])
        if isinstance(ans_cov["coverage"], float):
            answer_coverages.append(ans_cov["coverage"])

        results.append(
            {
                "dataset_index": int(row_index),
                "city": row["City"],
                "difficulty": row["Difficulty"],
                "hop_count": int(row["Hop Count"]),
                "question": question,
                "task_types": types,
                "benchmark_hops": benchmark_hops,
                "benchmark_answer_sequence": answer_names,
                "node_intents": node_results,
                "hop_role_coverage": hop_cov,
                "answer_name_coverage": ans_cov,
            }
        )

    domain_counts = Counter(doc.domain for doc in docs)
    category_counts = Counter(doc.primary_category or "unknown" for doc in docs)
    summary = {
        "total_cases": len(results),
        "city": "上海",
        "avg_hop_role_coverage": round(sum(hop_coverages) / len(hop_coverages), 3) if hop_coverages else None,
        "avg_answer_name_coverage": round(sum(answer_coverages) / len(answer_coverages), 3) if answer_coverages else None,
        "task_type_counts": dict(task_type_counts),
        "node_role_counts": dict(role_counts),
        "supply_domain_counts": dict(domain_counts),
        "top_supply_categories": dict(category_counts.most_common(20)),
        "sample_10": [
            {
                "dataset_index": item["dataset_index"],
                "question": item["question"],
                "hop_count": item["hop_count"],
                "multi_hop_queries": [hop["query"] for hop in item["benchmark_hops"]],
                "answer_sequence": item["benchmark_answer_sequence"],
                "task_types": item["task_types"],
                "node_roles": [node["role"] for node in item["node_intents"]],
            }
            for item in results[:10]
        ],
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": "localsearchbench/localsearchbench",
        "dataset_path": str(dataset_path),
        "supply_dir": str(Path(args.supply_dir).resolve()),
        "method": {
            "intent_extractor": "deterministic Chinese keyword roles with location/entity hints",
            "retriever": "scikit-learn char_wb TF-IDF ngram(2,4) over local Gaode/mock evidence plus role/location boosts",
            "top_k_per_node": args.top_k,
        },
        "rag_evidence_schema": rag_evidence_schema(),
        "summary": summary,
        "results": results,
    }


def rag_evidence_schema() -> dict[str, Any]:
    return {
        "merchant_evidence": [
            "poi_id/amap_id/name/address/coordinates",
            "merchant_id/trust_score/review_count/operation_stability_score",
            "gaode_type/primary_category/business_area/source_queries",
        ],
        "product_deal_evidence": [
            "product_id/product_type/package_components/price",
            "deal_id/title/sale_price/valid_time/refund_policy",
        ],
        "review_ugc_evidence": [
            "review_keywords/signature_dishes/recommended_dishes when present",
            "Gaode rating/review_count as structured quality signals",
        ],
        "route_location_evidence": [
            "longitude/latitude/address/district/business_area",
            "future: route graph and travel-time overlays",
        ],
        "business_hours_price_rating_evidence": [
            "business_hours/open time/holiday_status",
            "price/cost/rating/queue_time/inventory",
        ],
        "mock_field_source": [
            "Gaode facts remain source_channel=gaode_poi_search",
            "WeekendFlow mock owns inventory/deal/refund_policy/queue/reservation_slots",
        ],
    }


def _format_percent(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# LocalSearchBench RAG Probe - Shanghai",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Dataset: {report['source_dataset']}",
        f"- Cases: {summary['total_cases']} Shanghai samples",
        f"- Avg hop-role coverage: {_format_percent(summary['avg_hop_role_coverage'])}",
        f"- Avg exact answer-name coverage in local supply: {_format_percent(summary['avg_answer_name_coverage'])}",
        "",
        "## What The 100 Shanghai Cases Look Like",
        "",
    ]
    for key, value in summary["task_type_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Node role counts:"])
    for key, value in summary["node_role_counts"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Why Current B Fails",
            "",
            "Current B still generates and scores a legacy pair: one activity node plus one restaurant node. "
            "The LocalSearchBench Shanghai questions often require 3-4 evidence nodes, including lodging, shopping, convenience stores, and parking. "
            "When the request contains unsupported roles, the pair planner must either drop them or reinterpret them as the nearest activity/restaurant category, which explains failures such as exhibition+meal+souvenir becoming SPA+yakitori.",
            "",
            "The existing structured supply is strong for restaurants and activity-like POIs, but weaker for hotels, convenience stores, parking, and souvenir shopping as first-class planning nodes. "
            "A RAG layer can reveal those missing node intents before B commits to a template.",
            "",
            "## What RAG Should Add",
            "",
            "- Decompose a question into node intents such as exhibition, lunch, souvenir shop, lodging, convenience store, and parking.",
            "- Retrieve semantic evidence for each node from Gaode facts plus WeekendFlow mock fields.",
            "- Surface evidence gaps when local supply lacks a needed domain instead of forcing an activity+restaurant pair.",
            "- Give LLM components grounded snippets for explanation, candidate justification, and clarification.",
            "",
            "## What RAG Must Not Replace",
            "",
            "- Hard constraint filtering for time, distance, budget, inventory, reservation, and business hours.",
            "- Structured ranking features such as coordinates, queue time, rating, price, availability, and execution readiness.",
            "- C-stage execution checks. RAG evidence can suggest candidates; it cannot verify stock, reserve seats, buy coupons, or guarantee route feasibility.",
            "",
            "## Proposed Evidence Shape",
            "",
        ]
    )
    for key, values in report["rag_evidence_schema"].items():
        lines.append(f"- {key}: {', '.join(values)}")

    lines.extend(["", "## Sample 10 Shanghai Cases", ""])
    for item in summary["sample_10"]:
        lines.extend(
            [
                f"### #{item['dataset_index']} / {item['hop_count']} hops",
                "",
                f"- Question: {item['question']}",
                f"- Multi-hop queries: {' | '.join(item['multi_hop_queries'])}",
                f"- Answer sequence: {' -> '.join(item['answer_sequence'])}",
                f"- Task types: {', '.join(item['task_types'])}",
                f"- Extracted node roles: {', '.join(item['node_roles'])}",
                "",
            ]
        )

    lines.extend(["## Probe Details", ""])
    for item in report["results"][:20]:
        lines.extend(
            [
                f"### #{item['dataset_index']} {item['difficulty']} / {item['hop_count']} hops",
                "",
                f"- Question: {item['question']}",
                f"- Hop-role coverage: {_format_percent(item['hop_role_coverage']['coverage'])}",
                f"- Answer-name coverage: {_format_percent(item['answer_name_coverage']['coverage'])}",
            ]
        )
        for node in item["node_intents"]:
            evidence_names = [
                f"{e['name']} ({e['domain']}, {e['primary_category'] or e['gaode_type']})"
                for e in node.get("top_evidence", [])[:3]
            ]
            lines.append(f"- {node['role']}: {'; '.join(evidence_names)}")
        lines.append("")

    lines.extend(
        [
            "## Next Integration Step",
            "",
            "Add a B-side retrieval adapter before candidate generation: `question -> node_intents -> evidence bundles -> planner contract`. "
            "Candidate generation should still call structured loaders for supported domains, while unsupported or sparse domains produce explicit evidence gaps and relaxation/clarification options. "
            "The optimizer can then score multi-node itineraries using structured fields, with RAG evidence only as grounded semantic context.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LocalSearchBench Shanghai RAG probe.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--supply-dir", default=str(DEFAULT_SUPPLY_DIR))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Write JSON/Markdown probe artifacts. By default the script only prints a compact summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    report = run_probe(args)
    output: dict[str, Any] = {"summary": report["summary"]}
    if args.write_artifacts:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifact_dir / f"localsearchbench_rag_probe_{args.date}.json"
        md_path = artifact_dir / f"localsearchbench_rag_probe_{args.date}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(report, md_path)
        output.update(
            {
                "json_report": str(json_path),
                "md_report": str(md_path),
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
