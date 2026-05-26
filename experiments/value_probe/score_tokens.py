"""Score one text with a trained token-level value probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.value_probe.cache_qwen_activations import LayerCapture, tensor_dtype
from experiments.value_probe.constants import VALUE_IDS
from experiments.value_probe.modeling import TokenValueProbe, aggregate, require_torch


def require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("score_tokens.py requires transformers.") from exc
    return AutoModelForCausalLM, AutoTokenizer


torch, _nn = require_torch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen3_4b_token_probe/probes/layer_22/probe.pt"),
    )
    parser.add_argument("--layer", type=int, default=-1, help="Defaults to layer stored in the checkpoint.")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--offload-folder", type=Path, default=Path("experiments/value_probe_runs/qwen3_4b_token_probe/offload"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--aggregation", choices=("logmeanexp", "topk_mean"), default="logmeanexp")
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    checkpoint = torch.load(args.probe, map_location="cpu", weights_only=False)
    layer = int(checkpoint["layer"] if args.layer < 0 else args.layer)
    value_ids = list(checkpoint.get("value_ids") or VALUE_IDS)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=tensor_dtype(torch, args.dtype),
        device_map=args.device_map,
        offload_folder=str(args.offload_folder),
        local_files_only=args.local_files_only,
    )
    model.eval()

    probe = TokenValueProbe(
        hidden_size=int(checkpoint["hidden_size"]),
        num_values=len(value_ids),
        dropout=0.0,
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
        attention_mask = encoded["attention_mask"].to(hidden.device)
        positive_token_logits, negative_token_logits = probe(hidden)
        positive_doc = torch.sigmoid(
            aggregate(positive_token_logits, attention_mask, args.aggregation, args.temperature, args.top_k)
        )[0]
        negative_doc = torch.sigmoid(
            aggregate(negative_token_logits, attention_mask, args.aggregation, args.temperature, args.top_k)
        )[0]
        positive_tokens = torch.sigmoid(positive_token_logits[0]).cpu()
        negative_tokens = torch.sigmoid(negative_token_logits[0]).cpu()

    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    mask = encoded["attention_mask"][0].detach().cpu().tolist()
    special_ids = set(tokenizer.all_special_ids)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    values = {}
    for value_index, value_id in enumerate(value_ids):
        candidates = []
        negative_candidates = []
        for token_index, (token_id, token, is_valid) in enumerate(zip(input_ids, tokens, mask, strict=True)):
            if not is_valid or token_id in special_ids:
                continue
            clean_token = tokenizer.decode([token_id], skip_special_tokens=True)
            clean_token = clean_token if clean_token.strip() else token.replace("Ġ", "").replace("▁", "")
            candidates.append(
                {
                    "token_index": token_index,
                    "token": clean_token,
                    "score": float(positive_tokens[token_index, value_index]),
                }
            )
            negative_candidates.append(
                {
                    "token_index": token_index,
                    "token": clean_token,
                    "score": float(negative_tokens[token_index, value_index]),
                }
            )
        values[value_id] = {
            "positive_score": float(positive_doc[value_index].cpu()),
            "negative_score": float(negative_doc[value_index].cpu()),
            "positive_evidence": sorted(candidates, key=lambda item: item["score"], reverse=True)[: args.top_k],
            "negative_evidence": sorted(negative_candidates, key=lambda item: item["score"], reverse=True)[: args.top_k],
        }

    capture.close()
    print(
        json.dumps(
            {
                "text": args.text,
                "model": args.model,
                "layer": layer,
                "probe": str(args.probe),
                "values": values,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
