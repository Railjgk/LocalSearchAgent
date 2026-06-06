"""Compute extended value-probe evaluation metrics from saved score files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.value_probe.constants import VALUE_IDS
from experiments.value_probe.control_tasks import (
    keyword_baseline_records,
    length_baseline_records,
    lexical_nb_baseline_records,
    shuffled_relation_records,
    shuffled_value_records,
)
from experiments.value_probe.data import read_jsonl
from experiments.value_probe.metrics import binary_auc, binary_f1, diagonal_dominance, macro_metrics


RELATION_CLASSES = ("related", "opposite", "unrelated")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_or_empty(path: Path) -> Any:
    if not path.exists():
        return []
    return read_json(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / len(values) if values else float("nan")


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return float("nan")
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    numerator = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(x_values, y_values, strict=True)
    )
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_values))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_values))
    return numerator / (denom_x * denom_y) if denom_x and denom_y else float("nan")


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        avg_rank = (index + 1 + end) / 2
        for original_index, _value in indexed[index:end]:
            ranks[original_index] = avg_rank
        index = end
    return ranks


def spearman(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return float("nan")
    return pearson(average_ranks(x_values), average_ranks(y_values))


def binary_pr_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    pairs = sorted(
        zip(scores, labels, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    true_positives = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(pairs, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def ece(labels: list[int], scores: list[float], *, bins: int = 10) -> float:
    if not labels:
        return 0.0
    total = len(labels)
    error = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        in_bin = [
            index
            for index, score in enumerate(scores)
            if (low <= score < high) or (bin_index == bins - 1 and score == high)
        ]
        if not in_bin:
            continue
        avg_confidence = sum(scores[index] for index in in_bin) / len(in_bin)
        avg_accuracy = sum(labels[index] for index in in_bin) / len(in_bin)
        error += len(in_bin) / total * abs(avg_confidence - avg_accuracy)
    return error


def brier(labels: list[int], scores: list[float]) -> float:
    if not labels:
        return 0.0
    squared_errors = (
        (score - label) ** 2
        for label, score in zip(labels, scores, strict=True)
    )
    return sum(squared_errors) / len(labels)


def threshold_sweep(
    labels: list[int],
    scores: list[float],
    *,
    step: float = 0.05,
) -> list[dict[str, float]]:
    rows = []
    steps = int(1.0 / step)
    for index in range(steps + 1):
        threshold = round(index * step, 10)
        metrics = binary_f1(labels, scores, threshold=threshold)
        rows.append(
            {
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    return rows


def predict_relation(record: dict[str, Any], *, threshold: float) -> str:
    positive = float(record.get("positive_score", 0.0))
    negative = float(record.get("negative_score", 0.0))
    if positive < threshold and negative < threshold:
        return "unrelated"
    return "related" if positive >= negative else "opposite"


def multiclass_f1(
    labels: list[str],
    predictions: list[str],
    classes: tuple[str, ...],
) -> dict[str, float]:
    result: dict[str, float] = {}
    f1s = []
    for class_name in classes:
        tp = sum(
            1
            for y, y_hat in zip(labels, predictions, strict=True)
            if y == class_name and y_hat == class_name
        )
        fp = sum(
            1
            for y, y_hat in zip(labels, predictions, strict=True)
            if y != class_name and y_hat == class_name
        )
        fn = sum(
            1
            for y, y_hat in zip(labels, predictions, strict=True)
            if y == class_name and y_hat != class_name
        )
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        result[f"{class_name}_precision"] = precision
        result[f"{class_name}_recall"] = recall
        result[f"{class_name}_f1"] = f1
        f1s.append(f1)
    result["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    return result


def relation_confusion_rows(
    records: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    counts = Counter(
        (
            str(record.get("target_value", "")),
            str(record.get("relation", "")),
            predict_relation(record, threshold=threshold),
        )
        for record in records
    )
    rows = []
    for value_id in ("all", *VALUE_IDS):
        for actual in RELATION_CLASSES:
            row = {"target_value": value_id, "actual_relation": actual}
            total = 0
            for predicted in RELATION_CLASSES:
                if value_id == "all":
                    count = sum(
                        counts[(inner_value, actual, predicted)]
                        for inner_value in VALUE_IDS
                    )
                else:
                    count = counts[(value_id, actual, predicted)]
                row[f"predicted_{predicted}"] = count
                total += count
            row["total"] = total
            rows.append(row)
    return rows


def sentence_relation_metrics(records: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    labels_all = [str(record.get("relation", "")) for record in records]
    predictions_all = [predict_relation(record, threshold=threshold) for record in records]
    overall = multiclass_f1(labels_all, predictions_all, RELATION_CLASSES)
    result.update({f"overall_3way_{key}": value for key, value in overall.items()})

    per_value_3way = []
    positive_f1s = []
    negative_f1s = []
    positive_aucs = []
    negative_aucs = []
    positive_pr_aucs = []
    negative_pr_aucs = []
    opposite_aucs = []
    top1_hits_signed = []
    target_ranks = []
    reciprocal_ranks = []
    for value_id in VALUE_IDS:
        value_records = [record for record in records if record.get("target_value") == value_id]
        labels = [str(record.get("relation", "")) for record in value_records]
        predictions = [predict_relation(record, threshold=threshold) for record in value_records]
        value_3way = multiclass_f1(labels, predictions, RELATION_CLASSES)
        per_value_3way.append(value_3way["macro_f1"])
        result[f"{value_id}_3way_macro_f1"] = value_3way["macro_f1"]

        positive_labels = [
            1 if record.get("relation") == "related" else 0
            for record in value_records
        ]
        positive_scores = [float(record.get("positive_score", 0.0)) for record in value_records]
        positive_f1 = binary_f1(positive_labels, positive_scores, threshold=threshold)
        positive_auc = binary_auc(positive_labels, positive_scores)
        positive_pr = binary_pr_auc(positive_labels, positive_scores)
        positive_f1s.append(positive_f1["f1"])
        positive_aucs.append(positive_auc)
        positive_pr_aucs.append(positive_pr)
        result[f"{value_id}_positive_vs_rest_f1"] = positive_f1["f1"]
        result[f"{value_id}_positive_vs_rest_auc"] = positive_auc
        result[f"{value_id}_positive_vs_rest_pr_auc"] = positive_pr

        negative_labels = [
            1 if record.get("relation") == "opposite" else 0
            for record in value_records
        ]
        negative_scores = [float(record.get("negative_score", 0.0)) for record in value_records]
        negative_f1 = binary_f1(negative_labels, negative_scores, threshold=threshold)
        negative_auc = binary_auc(negative_labels, negative_scores)
        negative_pr = binary_pr_auc(negative_labels, negative_scores)
        negative_f1s.append(negative_f1["f1"])
        negative_aucs.append(negative_auc)
        negative_pr_aucs.append(negative_pr)
        result[f"{value_id}_negative_vs_rest_f1"] = negative_f1["f1"]
        result[f"{value_id}_negative_vs_rest_auc"] = negative_auc
        result[f"{value_id}_negative_vs_rest_pr_auc"] = negative_pr

        negative_records = [
            record
            for record in value_records
            if record.get("relation") in {"opposite", "unrelated"}
        ]
        opposite_labels = [
            1 if record.get("relation") == "opposite" else 0
            for record in negative_records
        ]
        opposite_scores = [float(record.get("negative_score", 0.0)) for record in negative_records]
        opposite_auc = binary_auc(opposite_labels, opposite_scores)
        result[f"{value_id}_opposite_vs_unrelated_auc"] = opposite_auc
        opposite_aucs.append(opposite_auc)

        for record in value_records:
            relation = str(record.get("relation", ""))
            if relation == "related":
                score_map = record.get("all_positive_scores", {})
            elif relation == "opposite":
                score_map = record.get("all_negative_scores", {})
            else:
                continue
            if not isinstance(score_map, dict) or value_id not in score_map:
                continue
            ranked_values = sorted(
                VALUE_IDS,
                key=lambda candidate: float(score_map.get(candidate, 0.0)),
                reverse=True,
            )
            rank = ranked_values.index(value_id) + 1
            top1_hits_signed.append(1.0 if rank == 1 else 0.0)
            target_ranks.append(float(rank))
            reciprocal_ranks.append(1.0 / rank)

    result["macro_3way_f1"] = sum(per_value_3way) / len(per_value_3way) if per_value_3way else 0.0
    result["macro_positive_vs_rest_f1"] = finite_mean(positive_f1s)
    result["macro_positive_vs_rest_auc"] = finite_mean(positive_aucs)
    result["macro_positive_vs_rest_pr_auc"] = finite_mean(positive_pr_aucs)
    result["macro_negative_vs_rest_f1"] = finite_mean(negative_f1s)
    result["macro_negative_vs_rest_auc"] = finite_mean(negative_aucs)
    result["macro_negative_vs_rest_pr_auc"] = finite_mean(negative_pr_aucs)
    result["macro_opposite_vs_unrelated_auc"] = finite_mean(opposite_aucs)
    result["top1_matched_value_accuracy"] = (
        sum(top1_hits_signed) / len(top1_hits_signed) if top1_hits_signed else 0.0
    )
    result["matched_rank_mean"] = sum(target_ranks) / len(target_ranks) if target_ranks else 0.0
    result["matched_mrr"] = (
        sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    )
    return result


def sentence_continuous_metrics(
    records: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    signed_labels = []
    signed_scores = []
    positive_labels = []
    positive_scores = []
    negative_labels = []
    negative_scores = []
    for record in records:
        relation = str(record.get("relation", ""))
        signed_labels.append({"related": 1.0, "opposite": -1.0}.get(relation, 0.0))
        positive = float(record.get("positive_score", 0.0))
        negative = float(record.get("negative_score", 0.0))
        signed_scores.append(positive - negative)
        positive_labels.append(1 if relation == "related" else 0)
        positive_scores.append(positive)
        negative_labels.append(1 if relation == "opposite" else 0)
        negative_scores.append(negative)

    return {
        "signed_pearson_r": pearson(signed_labels, signed_scores),
        "signed_spearman_rho": spearman(signed_labels, signed_scores),
        "positive_brier_score": brier(positive_labels, positive_scores),
        "positive_ece": ece(positive_labels, positive_scores),
        "positive_pr_auc": binary_pr_auc(positive_labels, positive_scores),
        "negative_brier_score": brier(negative_labels, negative_scores),
        "negative_ece": ece(negative_labels, negative_scores),
        "negative_pr_auc": binary_pr_auc(negative_labels, negative_scores),
        "positive_threshold_sweep": threshold_sweep(positive_labels, positive_scores),
        "negative_threshold_sweep": threshold_sweep(negative_labels, negative_scores),
        "relation_threshold": threshold,
    }


def diagonal_dominance_for_records(records: list[dict[str, Any]]) -> float:
    metrics = diagonal_dominance(records)
    return float(metrics.get("mean_diagonal_margin", 0.0))


def bootstrap_ci(
    records: list[dict[str, Any]],
    metric_fn: Callable[[list[dict[str, Any]]], float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if not records or samples <= 0:
        return {
            "mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    rng = random.Random(seed)
    values = []
    for _sample_index in range(samples):
        sample = [records[rng.randrange(len(records))] for _ in records]
        values.append(metric_fn(sample))
    return {
        "mean": sum(values) / len(values),
        "ci95_low": percentile(values, 0.025),
        "ci95_high": percentile(values, 0.975),
    }


def specificity_metrics(
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagonal = diagonal_dominance(records)
    ci = bootstrap_ci(
        [record for record in records if record.get("relation") == "related"],
        diagonal_dominance_for_records,
        samples=bootstrap_samples,
        seed=seed,
    )
    result: dict[str, Any] = {
        "mean_diagonal_margin": diagonal.get("mean_diagonal_margin", 0.0),
        "mean_diagonal_margin_ci95_low": ci["ci95_low"],
        "mean_diagonal_margin_ci95_high": ci["ci95_high"],
        "bootstrap_samples": bootstrap_samples,
        "per_value_diagonal_margin": {
            value_id: diagonal.get(f"{value_id}_diagonal_margin", 0.0)
            for value_id in VALUE_IDS
        },
    }

    effect_sizes = []
    leakage_rows = []
    related_by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("relation") == "related":
            related_by_value[str(record.get("target_value", ""))].append(record)
    for target in VALUE_IDS:
        matched = []
        mismatched = []
        row = {"target_value": target}
        target_records = related_by_value[target]
        for value_id in VALUE_IDS:
            scores = [
                float(record.get("all_positive_scores", {}).get(value_id, 0.0))
                for record in target_records
                if isinstance(record.get("all_positive_scores", {}), dict)
            ]
            row[value_id] = sum(scores) / len(scores) if scores else 0.0
            if value_id == target:
                matched.extend(scores)
            else:
                mismatched.extend(scores)
        leakage_rows.append(row)
        effect = cohen_d(matched, mismatched)
        result[f"{target}_matched_vs_mismatched_effect_size"] = effect
        effect_sizes.append(effect)
    result["matched_vs_mismatched_effect_size"] = finite_mean(effect_sizes)
    result["cross_value_leakage_matrix"] = leakage_rows
    return result, leakage_rows


def cohen_d(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    var_left = sum((value - mean_left) ** 2 for value in left) / (len(left) - 1)
    var_right = sum((value - mean_right) ** 2 for value in right) / (len(right) - 1)
    pooled = math.sqrt(
        ((len(left) - 1) * var_left + (len(right) - 1) * var_right)
        / (len(left) + len(right) - 2)
    )
    return (mean_left - mean_right) / pooled if pooled else float("nan")


def flatten_fluent_tokens(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens = []
    for record in records:
        for token in record.get("word_scores", []) or []:
            tokens.append(
                {
                    "example_id": record.get("example_id", ""),
                    "target_value": record.get("target_value", ""),
                    "relation": record.get("relation", ""),
                    "gold_score": float(token.get("gold_score", 0.0)),
                    "predicted_score": float(token.get("predicted_score", 0.0)),
                }
            )
    return tokens


def fluent_metrics(
    records: list[dict[str, Any]],
    *,
    score_scale: float,
    precision_k: list[int],
) -> dict[str, Any]:
    tokens = flatten_fluent_tokens(records)
    gold = [float(token["gold_score"]) for token in tokens]
    pred = [float(token["predicted_score"]) for token in tokens]
    nonzero_indices = [index for index, value in enumerate(gold) if value != 0.0]
    evidence_labels = [1 if value != 0.0 else 0 for value in gold]
    evidence_scores = [min(1.0, abs(value) / score_scale) for value in pred]
    result: dict[str, Any] = {
        "num_tokens": len(tokens),
        "num_nonzero_tokens": len(nonzero_indices),
        "all_token_pearson_r": pearson(gold, pred),
        "all_token_spearman_rho": spearman(gold, pred),
        "nonzero_token_pearson_r": pearson(
            [gold[index] for index in nonzero_indices],
            [pred[index] for index in nonzero_indices],
        ),
        "nonzero_token_spearman_rho": spearman(
            [gold[index] for index in nonzero_indices],
            [pred[index] for index in nonzero_indices],
        ),
        "evidence_pr_auc": binary_pr_auc(evidence_labels, evidence_scores),
        "evidence_brier_score": brier(evidence_labels, evidence_scores),
        "evidence_ece": ece(evidence_labels, evidence_scores),
        "evidence_threshold_sweep": threshold_sweep(evidence_labels, evidence_scores),
    }
    ranked = sorted(range(len(tokens)), key=lambda index: evidence_scores[index], reverse=True)
    total_positive = sum(evidence_labels)
    for k in precision_k:
        k = min(k, len(ranked))
        selected = ranked[:k]
        hits = sum(evidence_labels[index] for index in selected)
        result[f"precision_at_{k}"] = safe_div(hits, k)
        result[f"recall_at_{k}"] = safe_div(hits, total_positive)
    return result


def trapezoid_auc(rows: list[dict[str, Any]], metric_key: str) -> float:
    points = sorted(
        (float(row["layer"]), float(row[metric_key]))
        for row in rows
        if "layer" in row and metric_key in row and row[metric_key] is not None
    )
    if len(points) < 2:
        return float("nan")
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        area += (x1 - x0) * (y0 + y1) / 2
    span = points[-1][0] - points[0][0]
    return area / span if span else float("nan")


def layer_stability_metrics(
    sentence_layer_metrics: list[dict[str, Any]],
    fluent_layer_metrics: list[dict[str, Any]],
    *,
    seed_run_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if sentence_layer_metrics:
        result["sentence_best_layer_by_val_macro_auc"] = max(
            sentence_layer_metrics,
            key=lambda row: float(row.get("val_macro_auc", float("-inf"))),
        ).get("layer")
        result["sentence_best_layer_by_test_macro_auc"] = max(
            sentence_layer_metrics,
            key=lambda row: float(row.get("test_macro_auc", float("-inf"))),
        ).get("layer")
        for key in (
            "val_macro_auc",
            "test_macro_auc",
            "val_macro_f1",
            "test_macro_f1",
            "val_mean_diagonal_margin",
            "test_mean_diagonal_margin",
        ):
            result[f"sentence_layer_curve_auc_{key}"] = trapezoid_auc(
                sentence_layer_metrics,
                key,
            )
    if fluent_layer_metrics:
        result["fluent_best_layer_by_val_rmse"] = min(
            fluent_layer_metrics,
            key=lambda row: float(row.get("val_rmse", float("inf"))),
        ).get("layer")
        result["fluent_best_layer_by_test_rmse"] = min(
            fluent_layer_metrics,
            key=lambda row: float(row.get("test_rmse", float("inf"))),
        ).get("layer")
        for key in (
            "val_rmse",
            "test_rmse",
            "val_nonzero_direction_accuracy",
            "test_nonzero_direction_accuracy",
        ):
            result[f"fluent_layer_curve_auc_{key}"] = trapezoid_auc(
                fluent_layer_metrics,
                key,
            )
    seed_run_dirs = seed_run_dirs or []
    if seed_run_dirs:
        result.update(seed_run_stability_metrics(seed_run_dirs))
    unavailable = []
    if not seed_run_dirs:
        unavailable.extend(
            [
                "best_layer_across_seeds requires --seed-run-dirs",
                "probe_direction_cosine_similarity requires --seed-run-dirs",
            ]
        )
    result["unavailable_stability_checks"] = unavailable
    return result


def seed_run_stability_metrics(seed_run_dirs: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {"seed_run_dirs": [str(path) for path in seed_run_dirs]}
    sentence_best_layers = []
    fluent_best_layers = []
    for run_dir in seed_run_dirs:
        sentence_metrics_path = run_dir / "probes" / "layer_metrics.json"
        fluent_metrics_path = run_dir / "fluent_level_probes" / "layer_metrics.json"
        if sentence_metrics_path.exists():
            rows = read_json(sentence_metrics_path)
            if rows:
                best = max(rows, key=lambda row: float(row.get("val_macro_auc", float("-inf"))))
                sentence_best_layers.append(int(best["layer"]))
        if fluent_metrics_path.exists():
            rows = read_json(fluent_metrics_path)
            if rows:
                best = min(rows, key=lambda row: float(row.get("val_rmse", float("inf"))))
                fluent_best_layers.append(int(best["layer"]))

    result["sentence_best_layers_across_seeds"] = sentence_best_layers
    result["sentence_best_layer_mode_across_seeds"] = mode_or_none(sentence_best_layers)
    result["sentence_best_layer_range_across_seeds"] = range_or_none(sentence_best_layers)
    result["fluent_best_layers_across_seeds"] = fluent_best_layers
    result["fluent_best_layer_mode_across_seeds"] = mode_or_none(fluent_best_layers)
    result["fluent_best_layer_range_across_seeds"] = range_or_none(fluent_best_layers)
    result.update(probe_direction_cosine_metrics(seed_run_dirs, sentence_best_layers, "sentence"))
    result.update(probe_direction_cosine_metrics(seed_run_dirs, fluent_best_layers, "fluent"))
    return result


def mode_or_none(values: list[int]) -> int | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def range_or_none(values: list[int]) -> int | None:
    if not values:
        return None
    return max(values) - min(values)


def probe_direction_cosine_metrics(
    seed_run_dirs: list[Path],
    best_layers: list[int],
    probe_kind: str,
) -> dict[str, Any]:
    if len(seed_run_dirs) < 2 or len(best_layers) < 2:
        return {f"{probe_kind}_probe_direction_cosine_mean": float("nan")}
    vectors = []
    for run_dir, layer in zip(seed_run_dirs, best_layers, strict=False):
        if probe_kind == "sentence":
            checkpoint_path = run_dir / "probes" / f"layer_{layer:02d}" / "probe.pt"
            keys = ("positive_head.weight", "negative_head.weight")
        else:
            checkpoint_path = run_dir / "fluent_level_probes" / f"layer_{layer:02d}" / "probe.pt"
            keys = ("score_head.weight",)
        vector = load_probe_vector(checkpoint_path, keys)
        if vector:
            vectors.append(vector)
    cosines = [
        cosine(left, right)
        for index, left in enumerate(vectors)
        for right in vectors[index + 1 :]
    ]
    return {
        f"{probe_kind}_probe_direction_cosine_mean": finite_mean(cosines),
        f"{probe_kind}_probe_direction_cosine_pair_count": len(cosines),
    }


def load_probe_vector(checkpoint_path: Path, keys: tuple[str, ...]) -> list[float]:
    if not checkpoint_path.exists():
        return []
    try:
        import torch
    except ImportError:
        return []
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    values = []
    for key in keys:
        tensor = state_dict.get(key)
        if tensor is not None:
            values.extend(float(value) for value in tensor.detach().flatten().tolist())
    return values


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return float("nan")
    numerator = sum(x * y for x, y in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(x * x for x in left))
    right_norm = math.sqrt(sum(y * y for y in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else float("nan")


def control_and_selectivity_metrics(
    sentence_records: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    *,
    threshold: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    real_metrics = compact_selectivity_source(sentence_records, threshold=threshold)
    controls = {
        "random_label": shuffled_relation_records(sentence_records, seed=seed),
        "shuffled_value": shuffled_value_records(sentence_records, seed=seed + 1),
        "keyword_baseline": keyword_baseline_records(eval_rows),
        "lexical_nb_baseline": lexical_nb_baseline_records(train_rows, eval_rows),
        "length_baseline": length_baseline_records(eval_rows),
    }
    control_metrics = {}
    selectivity = {}
    for name, records in controls.items():
        metrics = compact_selectivity_source(records, threshold=threshold)
        control_metrics[name] = metrics
        selectivity[name] = {
            "selectivity_macro_auc": real_metrics["macro_auc"] - metrics["macro_auc"],
            "selectivity_macro_f1": real_metrics["macro_f1"] - metrics["macro_f1"],
            "selectivity_diagonal_margin": (
                real_metrics["mean_diagonal_margin"] - metrics["mean_diagonal_margin"]
            ),
            "selectivity_pr_auc": real_metrics["positive_pr_auc"] - metrics["positive_pr_auc"],
        }
    control_metrics["real_task"] = real_metrics
    control_metrics["template_split"] = template_split_metrics(
        sentence_records,
        eval_rows,
        threshold=threshold,
    )
    return control_metrics, selectivity


def compact_selectivity_source(
    records: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, float]:
    base = macro_metrics(records, threshold=threshold)
    diagonal = diagonal_dominance(records)
    continuous = sentence_continuous_metrics(records, threshold=threshold)
    return {
        "macro_auc": float(base.get("macro_auc", float("nan"))),
        "macro_f1": float(base.get("macro_f1", 0.0)),
        "mean_diagonal_margin": float(diagonal.get("mean_diagonal_margin", 0.0)),
        "positive_pr_auc": float(continuous.get("positive_pr_auc", float("nan"))),
    }


def template_split_metrics(
    sentence_records: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    row_by_id = {str(row.get("example_id", "")): row for row in eval_rows}
    candidate_fields = ("template_id", "template", "semantic_pattern", "text_type")
    template_field = next(
        (
            field
            for field in candidate_fields
            if any(field in row and row.get(field) for row in eval_rows)
        ),
        None,
    )
    if template_field is None:
        return {
            "status": "unavailable",
            "reason": "dataset rows do not include template_id/template/semantic_pattern/text_type",
        }
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sentence_records:
        row = row_by_id.get(str(record.get("example_id", "")), {})
        template = str(row.get(template_field, ""))
        if template:
            by_template[template].append(record)
    return {
        "status": "computed",
        "field": template_field,
        "groups": {
            template: compact_selectivity_source(records, threshold=threshold)
            for template, records in sorted(by_template.items())
        },
    }


def length_bias_metrics(
    sentence_records: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row_by_id = {str(row.get("example_id", "")): row for row in eval_rows}
    lengths = []
    positive_scores = []
    negative_scores = []
    signed_scores = []
    for record in sentence_records:
        row = row_by_id.get(str(record.get("example_id", "")))
        if row is None:
            continue
        text = re.sub(r"\s+", "", str(row.get("text", "")))
        lengths.append(float(len(text)))
        positive = float(record.get("positive_score", 0.0))
        negative = float(record.get("negative_score", 0.0))
        positive_scores.append(positive)
        negative_scores.append(negative)
        signed_scores.append(positive - negative)
    return {
        "num_examples": len(lengths),
        "positive_score_length_pearson_r": pearson(lengths, positive_scores),
        "negative_score_length_pearson_r": pearson(lengths, negative_scores),
        "signed_score_length_pearson_r": pearson(lengths, signed_scores),
    }


def paired_consistency_metrics(
    sentence_records: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row_by_id = {str(row.get("example_id", "")): row for row in eval_rows}
    record_by_id = {str(record.get("example_id", "")): record for record in sentence_records}
    group_field = next(
        (
            field
            for field in ("paraphrase_group_id", "noise_group_id", "pair_id")
            if any(field in row and row.get(field) for row in eval_rows)
        ),
        None,
    )
    if group_field is None:
        return {
            "status": "unavailable",
            "reason": "dataset rows do not include paraphrase_group_id/noise_group_id/pair_id",
        }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example_id, row in row_by_id.items():
        record = record_by_id.get(example_id)
        group = str(row.get(group_field, ""))
        if record is not None and group:
            groups[group].append(record)

    top1_matches = []
    rank_ranges = []
    for group_records in groups.values():
        if len(group_records) < 2:
            continue
        top_values = []
        ranks = []
        for record in group_records:
            score_map = record.get("all_positive_scores", {})
            if not isinstance(score_map, dict):
                continue
            ranked_values = sorted(
                VALUE_IDS,
                key=lambda value_id: float(score_map.get(value_id, 0.0)),
                reverse=True,
            )
            target = str(record.get("target_value", ""))
            top_values.append(ranked_values[0])
            if target in ranked_values:
                ranks.append(ranked_values.index(target) + 1)
        if len(top_values) >= 2:
            top1_matches.append(1.0 if len(set(top_values)) == 1 else 0.0)
        if len(ranks) >= 2:
            rank_ranges.append(float(max(ranks) - min(ranks)))

    return {
        "status": "computed",
        "field": group_field,
        "num_groups": len(groups),
        "paraphrase_or_noise_top1_consistency": finite_mean(top1_matches),
        "paraphrase_or_noise_target_rank_range_mean": finite_mean(rank_ranges),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen3_4b_token_probe"),
    )
    parser.add_argument("--sentence-layer", type=int, default=11)
    parser.add_argument("--fluent-layer", type=int, default=29)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--score-scale", type=float, default=6.0)
    parser.add_argument("--precision-k", default="10,50,100")
    parser.add_argument(
        "--seed-run-dirs",
        default="",
        help="Comma-separated run dirs for across-seed best-layer and direction-cosine checks.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "eval_extended"
    precision_k = [int(part) for part in args.precision_k.split(",") if part.strip()]

    sentence_scores_path = (
        args.run_dir
        / "probes"
        / f"layer_{args.sentence_layer:02d}"
        / f"{args.split}_scores.json"
    )
    fluent_scores_path = (
        args.run_dir
        / "fluent_level_probes"
        / f"layer_{args.fluent_layer:02d}"
        / f"{args.split}_scores.json"
    )
    dataset_path = args.run_dir / "dataset" / f"{args.split}.jsonl"
    train_dataset_path = args.run_dir / "dataset" / "train.jsonl"
    sentence_layer_metrics_path = args.run_dir / "probes" / "layer_metrics.json"
    fluent_layer_metrics_path = args.run_dir / "fluent_level_probes" / "layer_metrics.json"

    sentence_records = read_json(sentence_scores_path)
    fluent_records = read_json_or_empty(fluent_scores_path)
    dataset_rows = read_jsonl(dataset_path)
    train_dataset_rows = read_jsonl(train_dataset_path)
    seed_run_dirs = [
        Path(part.strip())
        for part in args.seed_run_dirs.split(",")
        if part.strip()
    ]

    relation_metrics = sentence_relation_metrics(sentence_records, threshold=args.threshold)
    relation_confusion = relation_confusion_rows(sentence_records, threshold=args.threshold)
    continuous_metrics = sentence_continuous_metrics(sentence_records, threshold=args.threshold)
    specificity, leakage_rows = specificity_metrics(
        sentence_records,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    fluent = fluent_metrics(fluent_records, score_scale=args.score_scale, precision_k=precision_k)
    control_metrics, selectivity = control_and_selectivity_metrics(
        sentence_records,
        dataset_rows,
        train_dataset_rows,
        threshold=args.threshold,
        seed=args.seed,
    )
    layer_stability = layer_stability_metrics(
        read_json_or_empty(sentence_layer_metrics_path),
        read_json_or_empty(fluent_layer_metrics_path),
        seed_run_dirs=seed_run_dirs,
    )
    robustness = {
        "length_bias": length_bias_metrics(sentence_records, dataset_rows),
        "paired_consistency": paired_consistency_metrics(sentence_records, dataset_rows),
    }

    write_json(output_dir / "sentence_relation_metrics.json", relation_metrics)
    write_csv(
        output_dir / "relation_confusion_matrix.csv",
        relation_confusion,
        fieldnames=[
            "target_value",
            "actual_relation",
            "predicted_related",
            "predicted_opposite",
            "predicted_unrelated",
            "total",
        ],
    )
    write_json(output_dir / "sentence_continuous_calibration.json", continuous_metrics)
    write_json(output_dir / "specificity_metrics.json", specificity)
    write_csv(
        output_dir / "cross_value_leakage_matrix.csv",
        leakage_rows,
        fieldnames=["target_value", *VALUE_IDS],
    )
    write_json(output_dir / "fluent_continuous_metrics.json", fluent)
    write_json(output_dir / "control_task_metrics.json", control_metrics)
    write_json(output_dir / "selectivity_metrics.json", selectivity)
    write_json(output_dir / "layer_stability_metrics.json", layer_stability)
    write_json(output_dir / "robustness_metrics.json", robustness)
    write_json(
        output_dir / "eval_summary.json",
        {
            "split": args.split,
            "sentence_layer": args.sentence_layer,
            "fluent_layer": args.fluent_layer,
            "relation_threshold": args.threshold,
            "macro_3way_f1": relation_metrics["macro_3way_f1"],
            "macro_opposite_vs_unrelated_auc": relation_metrics[
                "macro_opposite_vs_unrelated_auc"
            ],
            "mean_diagonal_margin": specificity["mean_diagonal_margin"],
            "mean_diagonal_margin_ci95_low": specificity["mean_diagonal_margin_ci95_low"],
            "mean_diagonal_margin_ci95_high": specificity["mean_diagonal_margin_ci95_high"],
            "fluent_evidence_pr_auc": fluent.get("evidence_pr_auc"),
            "fluent_available": bool(fluent_records),
            "output_dir": str(output_dir),
        },
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
