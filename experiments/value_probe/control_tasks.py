"""Control-task records for value probe selectivity checks."""

from __future__ import annotations

import random
import re
from copy import deepcopy
from math import exp, log
from typing import Any

from experiments.value_probe.constants import VALUE_IDS


VALUE_KEYWORDS = {
    "家庭照护": (
        "孩子",
        "小孩",
        "老人",
        "爸妈",
        "父母",
        "家人",
        "家庭",
        "带娃",
        "照顾",
        "亲子",
    ),
    "健康克制": (
        "健康",
        "清淡",
        "少油",
        "低脂",
        "低糖",
        "养生",
        "新鲜",
        "营养",
        "不辣",
        "舒服",
    ),
    "省心便利": (
        "方便",
        "近",
        "附近",
        "少排队",
        "不用等",
        "预约",
        "停车",
        "地铁",
        "省事",
        "省心",
    ),
    "价格敏感": (
        "预算",
        "便宜",
        "划算",
        "优惠",
        "团购",
        "人均",
        "省钱",
        "性价比",
        "别太贵",
        "贵",
    ),
    "品质可靠": (
        "评分",
        "评价",
        "口碑",
        "靠谱",
        "稳定",
        "老店",
        "卫生",
        "少踩雷",
        "放心",
        "服务",
    ),
    "体验享受": (
        "好玩",
        "沉浸",
        "体验感",
        "有意思",
        "尽兴",
        "丰富",
        "开心",
        "满足",
        "玩得",
        "互动体验",
    ),
    "新奇探索": (
        "新店",
        "小众",
        "没去过",
        "探索",
        "新鲜",
        "宝藏",
        "隐藏",
        "尝试",
        "冷门",
        "开业",
    ),
    "社交连接": (
        "朋友",
        "聚会",
        "聊天",
        "热闹",
        "互动",
        "多人",
        "一起玩",
        "组局",
        "联络",
        "团建",
    ),
    "氛围仪式": (
        "氛围",
        "仪式感",
        "纪念日",
        "浪漫",
        "出片",
        "约会",
        "景观",
        "布置",
        "精致",
        "生日",
    ),
    "舒适安全": (
        "安全",
        "不累",
        "不挤",
        "低强度",
        "舒服",
        "无烟",
        "安静",
        "风险",
        "老人友好",
        "不赶",
    ),
}

OPPOSITE_KEYWORDS = {
    "家庭照护": (
        "不用管孩子",
        "不带孩子",
        "大人自己",
        "孩子不去",
        "不考虑老人",
    ),
    "健康克制": (
        "重口味",
        "辣",
        "火锅",
        "烧烤",
        "炸鸡",
        "薯条",
        "酒吧",
        "喝酒",
        "不健康",
    ),
    "省心便利": (
        "绕路",
        "远一点",
        "排队",
        "换乘",
        "慢慢逛",
        "不好找",
        "藏在",
        "巷子",
        "不赶",
    ),
    "价格敏感": (
        "贵点",
        "预算放宽",
        "不差钱",
        "不用省",
        "高端",
        "奢侈",
        "人均两百",
        "人均三百",
    ),
    "品质可靠": (
        "评价少",
        "新店",
        "试错",
        "不看评分",
        "随便试",
        "踩雷也行",
    ),
    "体验享受": (
        "普通就行",
        "不好玩也行",
        "完成任务",
        "实用即可",
        "不求体验",
    ),
    "新奇探索": (
        "熟悉",
        "常去",
        "别试新",
        "稳妥",
        "不冒险",
        "老地方",
    ),
    "社交连接": (
        "一个人",
        "独处",
        "少社交",
        "不聊天",
        "安静待着",
        "自己吃",
    ),
    "氛围仪式": (
        "不在意氛围",
        "不用仪式感",
        "朴素",
        "实用",
        "普通就好",
    ),
    "舒适安全": (
        "累一点",
        "刺激",
        "拥挤也行",
        "高强度",
        "冒险",
        "不怕累",
    ),
}


def shuffled_relation_records(records: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Return copies with relation labels randomly permuted."""

    rng = random.Random(seed)
    relations = [str(record.get("relation", "")) for record in records]
    rng.shuffle(relations)
    shuffled = deepcopy(records)
    for record, relation in zip(shuffled, relations, strict=True):
        record["relation"] = relation
    return shuffled


def shuffled_value_records(records: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Return copies with target values permuted and target scores refreshed."""

    rng = random.Random(seed)
    values = [str(record.get("target_value", "")) for record in records]
    rng.shuffle(values)
    shuffled = deepcopy(records)
    for record, value_id in zip(shuffled, values, strict=True):
        record["target_value"] = value_id
        positive_scores = record.get("all_positive_scores", {})
        negative_scores = record.get("all_negative_scores", {})
        if isinstance(positive_scores, dict):
            record["positive_score"] = float(positive_scores.get(value_id, 0.0))
        if isinstance(negative_scores, dict):
            record["negative_score"] = float(negative_scores.get(value_id, 0.0))
    return shuffled


def keyword_baseline_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a simple keyword-rule baseline with the same scored-record schema."""

    scored = []
    for row in rows:
        text = str(row.get("text", ""))
        positive_scores = {
            value_id: keyword_score(text, VALUE_KEYWORDS[value_id])
            for value_id in VALUE_IDS
        }
        negative_scores = {
            value_id: keyword_score(text, OPPOSITE_KEYWORDS[value_id])
            for value_id in VALUE_IDS
        }
        target = str(row.get("target_value", ""))
        scored.append(
            {
                "example_id": row.get("example_id", ""),
                "split": row.get("split", ""),
                "target_value": target,
                "relation": row.get("relation", ""),
                "positive_score": positive_scores.get(target, 0.0),
                "negative_score": negative_scores.get(target, 0.0),
                "all_positive_scores": positive_scores,
                "all_negative_scores": negative_scores,
            }
        )
    return scored


def length_baseline_records(
    rows: list[dict[str, Any]],
    *,
    max_chars: int = 120,
) -> list[dict[str, Any]]:
    """Build a value-agnostic baseline from text length only."""

    scored = []
    for row in rows:
        text = str(row.get("text", ""))
        length = len(re.sub(r"\s+", "", text))
        score = min(1.0, length / max_chars)
        positive_scores = {value_id: score for value_id in VALUE_IDS}
        negative_scores = {value_id: score for value_id in VALUE_IDS}
        target = str(row.get("target_value", ""))
        scored.append(
            {
                "example_id": row.get("example_id", ""),
                "split": row.get("split", ""),
                "target_value": target,
                "relation": row.get("relation", ""),
                "positive_score": positive_scores.get(target, 0.0),
                "negative_score": negative_scores.get(target, 0.0),
                "all_positive_scores": positive_scores,
                "all_negative_scores": negative_scores,
            }
        )
    return scored


def lexical_nb_baseline_records(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    ngram_range: tuple[int, int] = (1, 2),
    alpha: float = 1.0,
) -> list[dict[str, Any]]:
    """Train a tiny character n-gram Naive Bayes baseline and score eval rows."""

    scored = []
    positive_models = {
        value_id: fit_binary_nb(
            train_rows,
            value_id=value_id,
            positive_relation="related",
            ngram_range=ngram_range,
            alpha=alpha,
        )
        for value_id in VALUE_IDS
    }
    negative_models = {
        value_id: fit_binary_nb(
            train_rows,
            value_id=value_id,
            positive_relation="opposite",
            ngram_range=ngram_range,
            alpha=alpha,
        )
        for value_id in VALUE_IDS
    }
    for row in eval_rows:
        text = str(row.get("text", ""))
        positive_scores = {
            value_id: score_binary_nb(positive_models[value_id], text)
            for value_id in VALUE_IDS
        }
        negative_scores = {
            value_id: score_binary_nb(negative_models[value_id], text)
            for value_id in VALUE_IDS
        }
        target = str(row.get("target_value", ""))
        scored.append(
            {
                "example_id": row.get("example_id", ""),
                "split": row.get("split", ""),
                "target_value": target,
                "relation": row.get("relation", ""),
                "positive_score": positive_scores.get(target, 0.0),
                "negative_score": negative_scores.get(target, 0.0),
                "all_positive_scores": positive_scores,
                "all_negative_scores": negative_scores,
            }
        )
    return scored


def keyword_score(text: str, keywords: tuple[str, ...]) -> float:
    matches = sum(1 for keyword in keywords if keyword and keyword in text)
    if matches <= 0:
        return 0.0
    return min(1.0, 0.35 + 0.2 * matches)


def fit_binary_nb(
    rows: list[dict[str, Any]],
    *,
    value_id: str,
    positive_relation: str,
    ngram_range: tuple[int, int],
    alpha: float,
) -> dict[str, Any]:
    class_feature_counts = {0: {}, 1: {}}
    class_totals = {0: 0, 1: 0}
    class_doc_counts = {0: 0, 1: 0}
    vocabulary: set[str] = set()
    for row in rows:
        if row.get("target_value") != value_id:
            continue
        label = 1 if row.get("relation") == positive_relation else 0
        class_doc_counts[label] += 1
        for feature in text_ngrams(str(row.get("text", "")), ngram_range=ngram_range):
            vocabulary.add(feature)
            class_feature_counts[label][feature] = class_feature_counts[label].get(feature, 0) + 1
            class_totals[label] += 1

    total_docs = class_doc_counts[0] + class_doc_counts[1]
    if total_docs == 0:
        prior_positive = 0.5
    else:
        prior_positive = (class_doc_counts[1] + alpha) / (total_docs + 2 * alpha)
    return {
        "class_feature_counts": class_feature_counts,
        "class_totals": class_totals,
        "prior_positive": prior_positive,
        "vocabulary": vocabulary,
        "ngram_range": ngram_range,
        "alpha": alpha,
    }


def score_binary_nb(model: dict[str, Any], text: str) -> float:
    features = text_ngrams(text, ngram_range=model["ngram_range"])
    vocabulary = model["vocabulary"]
    vocab_size = max(1, len(vocabulary))
    alpha = float(model["alpha"])
    prior_positive = float(model["prior_positive"])
    log_odds = log(prior_positive) - log(1 - prior_positive)
    for feature in features:
        pos_count = model["class_feature_counts"][1].get(feature, 0)
        neg_count = model["class_feature_counts"][0].get(feature, 0)
        pos_total = model["class_totals"][1]
        neg_total = model["class_totals"][0]
        pos_prob = (pos_count + alpha) / (pos_total + alpha * vocab_size)
        neg_prob = (neg_count + alpha) / (neg_total + alpha * vocab_size)
        log_odds += log(pos_prob) - log(neg_prob)
    return sigmoid(log_odds)


def text_ngrams(text: str, *, ngram_range: tuple[int, int]) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return []
    features = []
    min_n, max_n = ngram_range
    for ngram_size in range(min_n, max_n + 1):
        if ngram_size <= 0:
            continue
        features.extend(
            compact[index : index + ngram_size]
            for index in range(0, max(0, len(compact) - ngram_size + 1))
        )
    return features


def sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1 / (1 + z)
    z = exp(value)
    return z / (1 + z)
