"""Train fluent-level value probes from word-level signed scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.value_probe.constants import VALUE_IDS, VALUE_TO_INDEX
from experiments.value_probe.data import read_jsonl
from experiments.value_probe.fluent_alignment import (
    aggregate_token_scores_to_words,
    mean_word_score_for_span,
    tokenizer_token_spans,
    word_char_spans,
)
from experiments.value_probe.modeling import FluentValueProbe, require_torch


torch, nn = require_torch()


def require_transformers():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("train_fluent_level_probe.py requires transformers.") from exc
    return AutoTokenizer


def parse_layers(raw_layers: str, available_layers: list[int]) -> list[int]:
    if raw_layers == "all":
        return available_layers
    requested: set[int] = set()
    for part in raw_layers.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            requested.update(range(int(start_raw), int(end_raw) + 1))
        else:
            requested.add(int(part))
    missing = sorted(requested - set(available_layers))
    if missing:
        raise ValueError(f"Requested uncached layers: {missing}")
    return sorted(requested)


def load_cache_index(cache_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[int, list[int]]]]:
    config = json.loads((cache_dir / "cache_config.json").read_text(encoding="utf-8"))
    metadata_rows = read_jsonl(cache_dir / "metadata.jsonl")
    index: dict[str, dict[int, list[int]]] = {"train": {}, "val": {}, "test": {}, "all": {}}
    for row in metadata_rows:
        shard = int(row["shard"])
        offset = int(row["offset"])
        split = str(row["split"])
        index.setdefault(split, {}).setdefault(shard, []).append(offset)
        index["all"].setdefault(shard, []).append(offset)
    return config, metadata_rows, index


def load_fluent_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        example_id = str(row.get("example_id") or "")
        if not example_id:
            raise ValueError(f"Missing example_id in {path}")
        rows[example_id] = row
    return rows


def build_label_cache(
    *,
    cache_dir: Path,
    fluent_rows: dict[str, dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    tokenizer: Any,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    metadata_by_shard: dict[int, list[dict[str, Any]]] = {}
    for row in metadata_rows:
        metadata_by_shard.setdefault(int(row["shard"]), []).append(row)

    label_cache: dict[int, dict[str, Any]] = {}
    summary = {
        "aligned_examples": 0,
        "supervised_model_tokens": 0,
        "truncated_examples": 0,
        "alignment_errors": [],
    }
    for shard, shard_rows in sorted(metadata_by_shard.items()):
        shard_dir = cache_dir / f"shard_{shard:05d}"
        meta = torch.load(shard_dir / "meta.pt", map_location="cpu", weights_only=False)
        input_ids = meta["input_ids"]
        attention_mask = meta["attention_mask"]
        labels = torch.zeros(input_ids.shape, dtype=torch.float32)
        label_mask = torch.zeros(input_ids.shape, dtype=torch.float32)
        target_indices = torch.zeros((input_ids.shape[0],), dtype=torch.long)
        token_spans_by_offset: dict[int, list[dict[str, Any]]] = {}
        word_spans_by_offset: dict[int, list[dict[str, Any]]] = {}

        for row in shard_rows:
            offset = int(row["offset"])
            example_id = str(row["example_id"])
            fluent_row = fluent_rows.get(example_id)
            if fluent_row is None:
                raise ValueError(f"Missing fluent annotation for {example_id}")
            target_value = str(row["target_value"])
            target_indices[offset] = VALUE_TO_INDEX[target_value]
            try:
                source, word_spans = word_char_spans(fluent_row)
                token_spans = tokenizer_token_spans(
                    tokenizer=tokenizer,
                    input_ids=input_ids[offset].tolist(),
                    attention_mask=attention_mask[offset].tolist(),
                    source=source,
                )
                covered_end = max((int(span["end"]) for span in token_spans), default=0)
                if covered_end < len(source):
                    summary["truncated_examples"] += 1
                for token_span in token_spans:
                    token_index = int(token_span["token_index"])
                    if not token_span["is_supervised"]:
                        continue
                    score = mean_word_score_for_span(
                        word_spans,
                        int(token_span["start"]),
                        int(token_span["end"]),
                    )
                    labels[offset, token_index] = float(score)
                    label_mask[offset, token_index] = 1.0
                token_spans_by_offset[offset] = token_spans
                word_spans_by_offset[offset] = word_spans
                summary["aligned_examples"] += 1
                summary["supervised_model_tokens"] += int(label_mask[offset].sum().item())
            except Exception as exc:  # noqa: BLE001
                summary["alignment_errors"].append({"example_id": example_id, "error": str(exc)})
                raise

        label_cache[shard] = {
            "labels": labels,
            "label_mask": label_mask,
            "target_indices": target_indices,
            "token_spans_by_offset": token_spans_by_offset,
            "word_spans_by_offset": word_spans_by_offset,
        }
    return label_cache, summary


def load_shard(
    *,
    cache_dir: Path,
    label_cache: dict[int, dict[str, Any]],
    shard: int,
    layer: int,
    offsets: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    shard_dir = cache_dir / f"shard_{shard:05d}"
    hidden = torch.load(shard_dir / f"layer_{layer:02d}.pt", map_location="cpu", weights_only=False)
    index = torch.tensor(offsets, dtype=torch.long)
    labels = label_cache[shard]["labels"]
    label_mask = label_cache[shard]["label_mask"]
    target_indices = label_cache[shard]["target_indices"]
    return {
        "hidden": hidden.index_select(0, index).to(device),
        "labels": labels.index_select(0, index).to(device),
        "label_mask": label_mask.index_select(0, index).to(device),
        "target_indices": target_indices.index_select(0, index).to(device),
    }


def gather_target_scores(scores: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    gather_index = target_indices.view(-1, 1, 1).expand(-1, scores.shape[1], 1)
    return scores.gather(dim=2, index=gather_index).squeeze(-1)


def masked_regression_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    nonzero_weight: float,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "mse":
        raw_loss = (predictions - labels).pow(2)
    elif loss_type == "smooth_l1":
        raw_loss = nn.functional.smooth_l1_loss(predictions, labels, reduction="none")
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    weights = torch.ones_like(labels) + (labels.abs() > 0).to(labels.dtype) * nonzero_weight
    weighted_mask = mask * weights
    denom = weighted_mask.sum().clamp_min(1.0)
    return (raw_loss * weighted_mask).sum() / denom


def train_epoch(
    *,
    model: FluentValueProbe,
    optimizer: torch.optim.Optimizer,
    cache_dir: Path,
    label_cache: dict[int, dict[str, Any]],
    layer: int,
    split_index: dict[int, list[int]],
    device: torch.device,
    l1_weight: float,
    nonzero_weight: float,
    loss_type: str,
    seed: int,
) -> float:
    model.train()
    shard_items = list(split_index.items())
    random.Random(seed).shuffle(shard_items)
    losses: list[float] = []
    for shard, offsets in shard_items:
        batch = load_shard(
            cache_dir=cache_dir,
            label_cache=label_cache,
            shard=shard,
            layer=layer,
            offsets=offsets,
            device=device,
        )
        scores = model(batch["hidden"])
        target_scores = gather_target_scores(scores, batch["target_indices"])
        loss = masked_regression_loss(
            target_scores,
            batch["labels"],
            batch["label_mask"],
            nonzero_weight=nonzero_weight,
            loss_type=loss_type,
        )
        loss = loss + l1_weight * model.score_head.weight.abs().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else 0.0


@torch.no_grad()
def score_split(
    *,
    model: FluentValueProbe,
    cache_dir: Path,
    label_cache: dict[int, dict[str, Any]],
    layer: int,
    split_index: dict[int, list[int]],
    metadata_rows: list[dict[str, Any]],
    device: torch.device,
    include_word_scores: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    metadata_by_shard_offset = {
        (int(row["shard"]), int(row["offset"])): row for row in metadata_rows
    }
    predictions_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    scored_rows: list[dict[str, Any]] = []
    for shard, offsets in sorted(split_index.items()):
        batch = load_shard(
            cache_dir=cache_dir,
            label_cache=label_cache,
            shard=shard,
            layer=layer,
            offsets=offsets,
            device=device,
        )
        scores = model(batch["hidden"])
        target_scores = gather_target_scores(scores, batch["target_indices"])
        for local_index, offset in enumerate(offsets):
            mask = batch["label_mask"][local_index].bool().detach().cpu()
            predictions = target_scores[local_index].detach().cpu()[mask]
            labels = batch["labels"][local_index].detach().cpu()[mask]
            predictions_all.append(predictions)
            labels_all.append(labels)
            row = metadata_by_shard_offset[(shard, offset)]
            scored: dict[str, Any] = {
                "example_id": row["example_id"],
                "split": row["split"],
                "target_value": row["target_value"],
                "relation": row["relation"],
            }
            if include_word_scores:
                raw_token_scores = target_scores[local_index].detach().cpu().tolist()
                scored["word_scores"] = aggregate_token_scores_to_words(
                    word_spans=label_cache[shard]["word_spans_by_offset"][offset],
                    token_spans=label_cache[shard]["token_spans_by_offset"][offset],
                    token_scores=raw_token_scores,
                )
            scored_rows.append(scored)

    predictions_tensor = torch.cat(predictions_all) if predictions_all else torch.empty(0)
    labels_tensor = torch.cat(labels_all) if labels_all else torch.empty(0)
    metrics = regression_metrics(predictions_tensor, labels_tensor)
    return metrics, scored_rows


def direction_with_deadzone(values: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return torch.where(
        values > threshold,
        torch.ones_like(values),
        torch.where(values < -threshold, -torch.ones_like(values), torch.zeros_like(values)),
    )


def regression_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    if predictions.numel() == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "direction_accuracy": 0.0,
            "nonzero_direction_accuracy": 0.0,
            "num_tokens": 0.0,
            "num_nonzero_tokens": 0.0,
        }
    errors = predictions - labels
    mae = float(errors.abs().mean().item())
    rmse = float(torch.sqrt(errors.pow(2).mean()).item())
    predicted_direction = torch.sign(predictions)
    gold_direction = torch.sign(labels)
    threshold_predicted_direction = direction_with_deadzone(predictions)
    direction_accuracy = float((predicted_direction == gold_direction).float().mean().item())
    threshold_direction_accuracy = float(
        (threshold_predicted_direction == gold_direction).float().mean().item()
    )
    nonzero = labels != 0
    if bool(nonzero.any()):
        nonzero_direction_accuracy = float(
            (predicted_direction[nonzero] == gold_direction[nonzero]).float().mean().item()
        )
        threshold_nonzero_direction_accuracy = float(
            (threshold_predicted_direction[nonzero] == gold_direction[nonzero]).float().mean().item()
        )
    else:
        nonzero_direction_accuracy = 0.0
        threshold_nonzero_direction_accuracy = 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "direction_accuracy": direction_accuracy,
        "threshold_direction_accuracy": threshold_direction_accuracy,
        "nonzero_direction_accuracy": nonzero_direction_accuracy,
        "threshold_nonzero_direction_accuracy": threshold_nonzero_direction_accuracy,
        "num_tokens": float(labels.numel()),
        "num_nonzero_tokens": float(nonzero.sum().item()),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def train_layer(
    args: argparse.Namespace,
    layer: int,
    cache_config: dict[str, Any],
    label_cache: dict[int, dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    index: dict[str, dict[int, list[int]]],
) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    sample_shard = next(iter(index["train"]))
    sample_offsets = index["train"][sample_shard][:1]
    sample_batch = load_shard(
        cache_dir=args.cache_dir,
        label_cache=label_cache,
        shard=sample_shard,
        layer=layer,
        offsets=sample_offsets,
        device=device,
    )
    hidden_size = int(sample_batch["hidden"].shape[-1])
    del sample_batch

    model = FluentValueProbe(
        hidden_size=hidden_size,
        num_values=len(VALUE_IDS),
        dropout=args.dropout,
        score_scale=args.score_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    layer_dir = args.output_dir / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    best_metric = math.inf
    best_epoch_payload: dict[str, Any] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            cache_dir=args.cache_dir,
            label_cache=label_cache,
            layer=layer,
            split_index=index["train"],
            device=device,
            l1_weight=args.l1_weight,
            nonzero_weight=args.nonzero_weight,
            loss_type=args.loss,
            seed=args.seed + epoch + layer * 1000,
        )
        val_metrics, val_scores = score_split(
            model=model,
            cache_dir=args.cache_dir,
            label_cache=label_cache,
            layer=layer,
            split_index=index["val"],
            metadata_rows=metadata_rows,
            device=device,
            include_word_scores=args.write_word_scores,
        )
        metric = float(val_metrics.get(args.selection_metric, math.inf))
        epoch_payload = {
            "layer": layer,
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_payload)
        print(json.dumps(epoch_payload, ensure_ascii=False, sort_keys=True))
        if metric < best_metric:
            best_metric = metric
            best_epoch_payload = epoch_payload
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hidden_size": hidden_size,
                    "value_ids": list(VALUE_IDS),
                    "layer": layer,
                    "score_scale": args.score_scale,
                    "cache_config": cache_config,
                    "training_args": vars(args),
                    "best_epoch": epoch,
                    "best_metrics": val_metrics,
                },
                layer_dir / "probe.pt",
            )
            write_json(layer_dir / "val_scores.json", val_scores)

    write_json(layer_dir / "history.json", history)
    checkpoint = torch.load(layer_dir / "probe.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, test_scores = score_split(
        model=model,
        cache_dir=args.cache_dir,
        label_cache=label_cache,
        layer=layer,
        split_index=index["test"],
        metadata_rows=metadata_rows,
        device=device,
        include_word_scores=args.write_word_scores,
    )
    write_json(layer_dir / "test_scores.json", test_scores)
    write_json(layer_dir / "test_metrics.json", test_metrics)
    return {
        "layer": layer,
        "best_epoch": best_epoch_payload["epoch"] if best_epoch_payload else None,
        "best_val_metric": best_metric,
        **(best_epoch_payload or {}),
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen3_4b_token_probe/activations"),
    )
    parser.add_argument(
        "--fluent-data",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen3_4b_token_probe/fluent_level/all.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen3_4b_token_probe/fluent_level_probes"),
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--layers", default="11")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--l1-weight", type=float, default=1e-4)
    parser.add_argument("--nonzero-weight", type=float, default=3.0)
    parser.add_argument("--score-scale", type=float, default=6.0)
    parser.add_argument("--loss", choices=("smooth_l1", "mse"), default="smooth_l1")
    parser.add_argument("--selection-metric", default="rmse")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--write-word-scores", action="store_true")
    parser.add_argument("--align-only", action="store_true", help="Build and validate label alignment, then exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_config, metadata_rows, index = load_cache_index(args.cache_dir)
    model_name = args.model or str(cache_config["model"])
    AutoTokenizer = require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    fluent_rows = load_fluent_rows(args.fluent_data)
    label_cache, label_summary = build_label_cache(
        cache_dir=args.cache_dir,
        fluent_rows=fluent_rows,
        metadata_rows=metadata_rows,
        tokenizer=tokenizer,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "label_alignment_summary.json", label_summary)
    if label_summary["alignment_errors"]:
        raise RuntimeError("Fluent label alignment failed")
    if args.align_only:
        print(json.dumps(label_summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    layers = parse_layers(args.layers, list(map(int, cache_config["cached_layers"])))
    results = []
    for layer in layers:
        results.append(train_layer(args, layer, cache_config, label_cache, metadata_rows, index))

    metrics_path = args.output_dir / "layer_metrics.csv"
    if results:
        fieldnames = sorted({key for row in results for key in row})
        with metrics_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    write_json(args.output_dir / "layer_metrics.json", results)
    best = min(results, key=lambda row: float(row.get("best_val_metric", math.inf))) if results else {}
    write_json(args.output_dir / "best_layer.json", best)
    print(json.dumps({"metrics_path": str(metrics_path), "best": best}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
