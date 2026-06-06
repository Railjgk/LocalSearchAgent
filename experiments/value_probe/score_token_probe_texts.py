"""Score arbitrary texts with a trained token-level value probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.value_probe.cache_qwen_activations import LayerCapture, tensor_dtype  # noqa: E402
from experiments.value_probe.constants import VALUE_IDS  # noqa: E402
from experiments.value_probe.data import read_jsonl  # noqa: E402
from experiments.value_probe.modeling import TokenValueProbe, aggregate, require_torch  # noqa: E402


torch, _nn = require_torch()


def require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("score_token_probe_texts.py requires transformers.") from exc
    return AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen3_4b_cn10_probe_100_v7/probes/layer_11/probe.pt"),
    )
    parser.add_argument("--layer", type=int, default=-1, help="Defaults to layer stored in checkpoint.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--offload-folder", type=Path, default=Path("experiments/value_probe_runs/offload"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--opposite-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_text_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    output = []
    for index, row in enumerate(rows, start=1):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        output.append(
            {
                "example_id": str(row.get("example_id") or f"input-{index:04d}"),
                "text": text,
                "note": row.get("note"),
                "likely_values": row.get("likely_values", []),
            }
        )
    if not output:
        raise ValueError(f"No text rows found in {path}")
    return output


def ranked(scores: dict[str, float], top_k: int) -> list[dict[str, float | str]]:
    return [
        {"value_id": value_id, "score": score}
        for value_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    ]


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    rows = load_text_rows(args.input)

    checkpoint = torch.load(args.probe, map_location="cpu", weights_only=False)
    value_ids = list(checkpoint.get("value_ids") or VALUE_IDS)
    layer = int(checkpoint["layer"] if args.layer < 0 else args.layer)
    training_args = checkpoint.get("training_args", {}) if isinstance(checkpoint.get("training_args"), dict) else {}
    aggregation = str(training_args.get("aggregation", "logmeanexp"))
    temperature = float(training_args.get("temperature", 0.35))
    topk = int(training_args.get("topk", 6))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=tensor_dtype(torch, args.dtype),
        device_map=args.device_map,
        offload_folder=str(args.offload_folder),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.eval()

    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = TokenValueProbe(
        hidden_size=int(checkpoint["hidden_size"]),
        num_values=len(value_ids),
        dropout=0.0,
    ).to(probe_device)
    probe.load_state_dict(checkpoint["state_dict"])
    probe.eval()

    capture = LayerCapture(model, [layer])
    scored_rows: list[dict[str, Any]] = []
    for batch_start in range(0, len(rows), args.batch_size):
        batch_rows = rows[batch_start : batch_start + args.batch_size]
        encoded = tokenizer(
            [row["text"] for row in batch_rows],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            capture.clear()
            model(**encoded, output_hidden_states=False, use_cache=False)
            hidden = capture.outputs[layer].detach().to(probe_device)
            attention_mask = encoded["attention_mask"].to(probe_device)
            positive_token_logits, negative_token_logits = probe(hidden)
            positive_scores = torch.sigmoid(
                aggregate(positive_token_logits, attention_mask, aggregation, temperature, topk)
            ).detach().cpu()
            negative_scores = torch.sigmoid(
                aggregate(negative_token_logits, attention_mask, aggregation, temperature, topk)
            ).detach().cpu()

        for local_index, row in enumerate(batch_rows):
            positive = {
                value_id: float(positive_scores[local_index, value_index])
                for value_index, value_id in enumerate(value_ids)
            }
            opposite = {
                value_id: float(negative_scores[local_index, value_index])
                for value_index, value_id in enumerate(value_ids)
            }
            signed = {
                value_id: positive[value_id] - opposite[value_id]
                for value_id in value_ids
            }
            scored_rows.append(
                {
                    **row,
                    "positive_scores": positive,
                    "opposite_scores": opposite,
                    "signed_scores": signed,
                    "activated_positive": [
                        value_id for value_id, score in positive.items() if score >= args.positive_threshold
                    ],
                    "activated_opposite": [
                        value_id for value_id, score in opposite.items() if score >= args.opposite_threshold
                    ],
                    "top_positive": ranked(positive, args.top_k),
                    "top_opposite": ranked(opposite, args.top_k),
                    "top_signed": ranked(signed, args.top_k),
                }
            )
        print(f"scored {min(batch_start + len(batch_rows), len(rows))}/{len(rows)}")

    capture.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "probe": str(args.probe),
        "model": args.model,
        "layer": layer,
        "count": len(scored_rows),
    }
    (args.output.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
