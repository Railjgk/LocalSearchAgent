"""Train one token-level value probe per cached Qwen layer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.value_probe.constants import VALUE_IDS
from experiments.value_probe.metrics import diagonal_dominance, macro_metrics
from experiments.value_probe.modeling import TokenValueProbe, aggregate, require_torch


torch, nn = require_torch()


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


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_cache_index(cache_dir: Path) -> tuple[dict[str, object], dict[str, dict[int, list[int]]]]:
    config = json.loads((cache_dir / "cache_config.json").read_text(encoding="utf-8"))
    index: dict[str, dict[int, list[int]]] = {"train": {}, "val": {}, "test": {}, "all": {}}
    for row in read_jsonl(cache_dir / "metadata.jsonl"):
        shard = int(row["shard"])
        offset = int(row["offset"])
        split = str(row["split"])
        index.setdefault(split, {}).setdefault(shard, []).append(offset)
        index["all"].setdefault(shard, []).append(offset)
    return config, index


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (raw_loss * mask).sum() / denom


def load_shard(
    cache_dir: Path,
    shard: int,
    layer: int,
    offsets: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    shard_dir = cache_dir / f"shard_{shard:05d}"
    meta = torch.load(shard_dir / "meta.pt", map_location="cpu")
    hidden = torch.load(shard_dir / f"layer_{layer:02d}.pt", map_location="cpu")
    index = torch.tensor(offsets, dtype=torch.long)
    return {
        "hidden": hidden.index_select(0, index).to(device),
        "attention_mask": meta["attention_mask"].index_select(0, index).to(device),
        "positive_labels": meta["positive_labels"].index_select(0, index).to(device),
        "negative_labels": meta["negative_labels"].index_select(0, index).to(device),
        "supervision_mask": meta["supervision_mask"].index_select(0, index).to(device),
        "target_indices": meta["target_indices"].index_select(0, index).to(device),
    }


def train_epoch(
    *,
    model: TokenValueProbe,
    optimizer: torch.optim.Optimizer,
    cache_dir: Path,
    layer: int,
    split_index: dict[int, list[int]],
    device: torch.device,
    aggregation: str,
    temperature: float,
    topk: int,
    l1_weight: float,
    seed: int,
) -> float:
    model.train()
    shard_items = list(split_index.items())
    random.Random(seed).shuffle(shard_items)
    losses: list[float] = []
    for shard, offsets in shard_items:
        batch = load_shard(cache_dir, shard, layer, offsets, device)
        positive_token_logits, negative_token_logits = model(batch["hidden"])
        positive_doc_logits = aggregate(positive_token_logits, batch["attention_mask"], aggregation, temperature, topk)
        negative_doc_logits = aggregate(negative_token_logits, batch["attention_mask"], aggregation, temperature, topk)
        positive_loss = masked_bce_loss(
            positive_doc_logits,
            batch["positive_labels"],
            batch["supervision_mask"],
        )
        negative_loss = masked_bce_loss(
            negative_doc_logits,
            batch["negative_labels"],
            batch["supervision_mask"],
        )
        l1_loss = (
            model.positive_head.weight.abs().mean()
            + model.negative_head.weight.abs().mean()
        )
        loss = positive_loss + negative_loss + l1_weight * l1_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else 0.0


@torch.no_grad()
def score_split(
    *,
    model: TokenValueProbe,
    cache_dir: Path,
    layer: int,
    split_index: dict[int, list[int]],
    metadata_rows: list[dict[str, object]],
    device: torch.device,
    aggregation: str,
    temperature: float,
    topk: int,
) -> list[dict[str, object]]:
    model.eval()
    metadata_by_shard_offset = {
        (int(row["shard"]), int(row["offset"])): row for row in metadata_rows
    }
    scored: list[dict[str, object]] = []
    for shard, offsets in sorted(split_index.items()):
        batch = load_shard(cache_dir, shard, layer, offsets, device)
        positive_token_logits, negative_token_logits = model(batch["hidden"])
        positive_scores = torch.sigmoid(
            aggregate(positive_token_logits, batch["attention_mask"], aggregation, temperature, topk)
        ).cpu()
        negative_scores = torch.sigmoid(
            aggregate(negative_token_logits, batch["attention_mask"], aggregation, temperature, topk)
        ).cpu()
        target_indices = batch["target_indices"].cpu().tolist()
        for local_index, offset in enumerate(offsets):
            row = metadata_by_shard_offset[(shard, offset)]
            target_index = int(target_indices[local_index])
            score_row = {
                "example_id": row["example_id"],
                "split": row["split"],
                "target_value": row["target_value"],
                "relation": row["relation"],
                "positive_score": float(positive_scores[local_index, target_index]),
                "negative_score": float(negative_scores[local_index, target_index]),
                "all_positive_scores": {
                    value_id: float(positive_scores[local_index, value_index])
                    for value_index, value_id in enumerate(VALUE_IDS)
                },
                "all_negative_scores": {
                    value_id: float(negative_scores[local_index, value_index])
                    for value_index, value_id in enumerate(VALUE_IDS)
                },
            }
            for key in (
                "text_type",
                "template_id",
                "semantic_pattern",
                "paraphrase_group_id",
                "noise_group_id",
                "pair_id",
            ):
                if row.get(key):
                    score_row[key] = row[key]
            scored.append(score_row)
    return scored


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def train_layer(args: argparse.Namespace, layer: int, cache_config: dict[str, object], index: dict[str, dict[int, list[int]]]) -> dict[str, object]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    sample_shard = next(iter(index["train"]))
    sample_offsets = index["train"][sample_shard][:1]
    sample_batch = load_shard(args.cache_dir, sample_shard, layer, sample_offsets, device)
    hidden_size = int(sample_batch["hidden"].shape[-1])
    del sample_batch

    model = TokenValueProbe(hidden_size=hidden_size, num_values=len(VALUE_IDS), dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metadata_rows = read_jsonl(args.cache_dir / "metadata.jsonl")
    layer_dir = args.output_dir / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    best_metric = -math.inf
    best_payload: dict[str, object] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            cache_dir=args.cache_dir,
            layer=layer,
            split_index=index["train"],
            device=device,
            aggregation=args.aggregation,
            temperature=args.temperature,
            topk=args.topk,
            l1_weight=args.l1_weight,
            seed=args.seed + epoch + layer * 1000,
        )
        val_scores = score_split(
            model=model,
            cache_dir=args.cache_dir,
            layer=layer,
            split_index=index["val"],
            metadata_rows=metadata_rows,
            device=device,
            aggregation=args.aggregation,
            temperature=args.temperature,
            topk=args.topk,
        )
        val_metrics = macro_metrics(val_scores)
        val_metrics.update(diagonal_dominance(val_scores))
        metric = float(val_metrics.get(args.selection_metric, 0.0))
        epoch_payload = {"layer": layer, "epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(epoch_payload)
        print(json.dumps(epoch_payload, ensure_ascii=False, sort_keys=True))
        if metric > best_metric:
            best_metric = metric
            best_payload = epoch_payload
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hidden_size": hidden_size,
                    "value_ids": list(VALUE_IDS),
                    "layer": layer,
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
    test_scores = score_split(
        model=model,
        cache_dir=args.cache_dir,
        layer=layer,
        split_index=index["test"],
        metadata_rows=metadata_rows,
        device=device,
        aggregation=args.aggregation,
        temperature=args.temperature,
        topk=args.topk,
    )
    test_metrics = macro_metrics(test_scores)
    test_metrics.update(diagonal_dominance(test_scores))
    write_json(layer_dir / "test_scores.json", test_scores)
    write_json(layer_dir / "test_metrics.json", test_metrics)
    return {
        "layer": layer,
        "best_epoch": best_payload["epoch"] if best_payload else None,
        "best_val_metric": best_metric,
        **{f"val_{key}": value for key, value in (best_payload or {}).items() if key not in {"layer", "epoch"}},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen34b_token_probe/activations"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen34b_token_probe/probes"),
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--l1-weight", type=float, default=1e-4)
    parser.add_argument("--aggregation", choices=("logmeanexp", "topk_mean"), default="logmeanexp")
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--selection-metric", default="macro_auc")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=20260524)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_config, index = load_cache_index(args.cache_dir)
    layers = parse_layers(args.layers, list(map(int, cache_config["cached_layers"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for layer in layers:
        results.append(train_layer(args, layer, cache_config, index))

    metrics_path = args.output_dir / "layer_metrics.csv"
    if results:
        fieldnames = sorted({key for row in results for key in row})
        with metrics_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    write_json(args.output_dir / "layer_metrics.json", results)
    best = max(results, key=lambda row: float(row.get("best_val_metric", 0.0))) if results else {}
    write_json(args.output_dir / "best_layer.json", best)
    print(json.dumps({"metrics_path": str(metrics_path), "best": best}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
