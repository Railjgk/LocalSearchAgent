"""Score one text with a trained fluent-level value probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.value_probe.cache_qwen_activations import LayerCapture, tensor_dtype
from experiments.value_probe.constants import VALUE_IDS
from experiments.value_probe.fluent_alignment import (
    aggregate_token_scores_to_words,
    normalize_for_alignment,
    tokenizer_token_spans,
    word_char_spans,
)
from experiments.value_probe.modeling import FluentValueProbe, require_torch


torch, _nn = require_torch()


def require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("score_fluent_level.py requires transformers.") from exc
    return AutoModelForCausalLM, AutoTokenizer


def parse_tokens(raw_tokens: str) -> list[str]:
    payload = json.loads(raw_tokens)
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError("--tokens-json must be a JSON string array")
    return payload


def model_token_words(token_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words = []
    for span in token_spans:
        if not span.get("is_supervised"):
            continue
        words.append(
            {
                "token_index": span["token_index"],
                "token": span["token"],
                "score": 0,
                "start": span["start"],
                "end": span["end"],
            }
        )
    return words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--target-value", choices=VALUE_IDS, required=True)
    parser.add_argument("--tokens-json", default="", help='Optional fluent words, e.g. ["少","排队"].')
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen3_4b_token_probe/fluent_level_probes/layer_29/probe.pt"),
    )
    parser.add_argument("--layer", type=int, default=-1, help="Defaults to layer stored in checkpoint.")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen3_4b_token_probe/offload"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    checkpoint = torch.load(args.probe, map_location="cpu", weights_only=False)
    layer = int(checkpoint["layer"] if args.layer < 0 else args.layer)
    value_ids = list(checkpoint.get("value_ids") or VALUE_IDS)
    value_index = value_ids.index(args.target_value)
    score_scale = float(checkpoint.get("score_scale", 6.0))

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

    probe = FluentValueProbe(
        hidden_size=int(checkpoint["hidden_size"]),
        num_values=len(value_ids),
        dropout=0.0,
        score_scale=score_scale,
    )
    probe.load_state_dict(checkpoint["state_dict"])
    probe.eval()

    capture = LayerCapture(model, [layer])
    encoded = tokenizer(
        [args.text],
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.no_grad():
        capture.clear()
        model(**encoded, output_hidden_states=False, use_cache=False)
        hidden = capture.outputs[layer].detach().to(next(probe.parameters()).device)
        scores = probe(hidden)[0, :, value_index].detach().cpu().tolist()

    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    attention_mask = encoded["attention_mask"][0].detach().cpu().tolist()
    source = normalize_for_alignment(args.text)
    token_spans = tokenizer_token_spans(
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        source=source,
    )
    model_token_scores = []
    for span, score in zip(token_spans, scores, strict=True):
        if not span.get("is_supervised"):
            continue
        model_token_scores.append(
            {
                "token_index": span["token_index"],
                "token": span["token"],
                "score": float(score),
            }
        )

    if args.tokens_json:
        words = parse_tokens(args.tokens_json)
        _, word_spans = word_char_spans(
            {
                "example_id": "input",
                "text": args.text,
                "tokens": [{"token": token, "score": 0} for token in words],
            }
        )
    else:
        word_spans = model_token_words(token_spans)
    word_scores = aggregate_token_scores_to_words(
        word_spans=word_spans,
        token_spans=token_spans,
        token_scores=scores,
    )
    for item in word_scores:
        item.pop("gold_score", None)

    capture.close()
    print(
        json.dumps(
            {
                "text": args.text,
                "target_value": args.target_value,
                "model": args.model,
                "layer": layer,
                "probe": str(args.probe),
                "word_scores": word_scores,
                "model_token_scores": model_token_scores,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
