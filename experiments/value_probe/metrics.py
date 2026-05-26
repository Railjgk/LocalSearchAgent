"""Small metric helpers without a scikit-learn dependency."""

from __future__ import annotations

from collections import defaultdict
from math import isnan

from experiments.value_probe.constants import VALUE_IDS


def binary_f1(labels: list[int], scores: list[float], threshold: float = 0.5) -> dict[str, float]:
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, y_hat in zip(labels, preds, strict=True) if y == 1 and y_hat == 1)
    fp = sum(1 for y, y_hat in zip(labels, preds, strict=True) if y == 0 and y_hat == 1)
    fn = sum(1 for y, y_hat in zip(labels, preds, strict=True) if y == 1 and y_hat == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def binary_auc(labels: list[int], scores: list[float]) -> float:
    """Compute ROC AUC from ranks, returning NaN for degenerate labels."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2
        rank_sum += sum(label for _, label in pairs[index:end]) * avg_rank
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def macro_metrics(records: list[dict[str, object]], threshold: float = 0.5) -> dict[str, float]:
    """Evaluate target-value `related` detection from scored records."""

    by_value: dict[str, tuple[list[int], list[float]]] = {
        value_id: ([], []) for value_id in VALUE_IDS
    }
    for record in records:
        value_id = str(record["target_value"])
        label = 1 if record["relation"] == "related" else 0
        score = float(record["positive_score"])
        by_value[value_id][0].append(label)
        by_value[value_id][1].append(score)

    result: dict[str, float] = {}
    f1s: list[float] = []
    aucs: list[float] = []
    for value_id in VALUE_IDS:
        labels, scores = by_value[value_id]
        f1_result = binary_f1(labels, scores, threshold=threshold)
        auc = binary_auc(labels, scores)
        result[f"{value_id}_precision"] = f1_result["precision"]
        result[f"{value_id}_recall"] = f1_result["recall"]
        result[f"{value_id}_f1"] = f1_result["f1"]
        result[f"{value_id}_auc"] = auc
        f1s.append(f1_result["f1"])
        if not isnan(auc):
            aucs.append(auc)
    result["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    result["macro_auc"] = sum(aucs) / len(aucs) if aucs else float("nan")
    return result


def diagonal_dominance(records: list[dict[str, object]]) -> dict[str, float]:
    """Compare matched related corpus scores against mismatched corpus scores."""

    related_scores: dict[str, list[float]] = defaultdict(list)
    mismatched_scores: dict[str, list[float]] = defaultdict(list)
    for record in records:
        target = str(record["target_value"])
        all_scores = record.get("all_positive_scores", {})
        if not isinstance(all_scores, dict):
            continue
        if record["relation"] == "related":
            related_scores[target].append(float(all_scores[target]))
            for value_id in VALUE_IDS:
                if value_id != target:
                    mismatched_scores[target].append(float(all_scores.get(value_id, 0.0)))

    result: dict[str, float] = {}
    margins: list[float] = []
    for value_id in VALUE_IDS:
        matched = related_scores[value_id]
        mismatched = mismatched_scores[value_id]
        matched_mean = sum(matched) / len(matched) if matched else 0.0
        mismatched_mean = sum(mismatched) / len(mismatched) if mismatched else 0.0
        margin = matched_mean - mismatched_mean
        result[f"{value_id}_diagonal_margin"] = margin
        margins.append(margin)
    result["mean_diagonal_margin"] = sum(margins) / len(margins) if margins else 0.0
    return result

