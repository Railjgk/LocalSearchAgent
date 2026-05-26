"""Cache Qwen hidden states for token-level value probe training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.value_probe.constants import VALUE_IDS
from experiments.value_probe.data import read_jsonl


def require_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Caching activations requires torch and transformers. "
            "Install them in the active environment before running this script."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def parse_layers(raw_layers: str, num_layers: int) -> list[int]:
    if raw_layers == "all":
        return list(range(num_layers))
    layers: set[int] = set()
    for part in raw_layers.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    invalid = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"Invalid layers for model with {num_layers} layers: {invalid}")
    return sorted(layers)


def transformer_layers(model):
    """Return the decoder block list for common causal LM wrappers."""

    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        current = model
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return current
    raise AttributeError("Could not locate transformer block list on model.")


def first_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


class LayerCapture:
    """Capture selected decoder block outputs without returning all hidden states."""

    def __init__(self, model, layers: list[int]) -> None:
        self.outputs = {}
        self.handles = []
        blocks = transformer_layers(model)
        for layer in layers:
            self.handles.append(blocks[layer].register_forward_hook(self._make_hook(layer)))

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            self.outputs[layer] = first_tensor(output).detach()

        return hook

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen34b_token_probe/dataset/all.jsonl"),
    )
    parser.add_argument("--model", default="Qwen/Qwen-34B")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen34b_token_probe/activations"),
    )
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--layers", default="all", help="all, comma list, or inclusive ranges like 0-7,20")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional debug limit after split filtering.")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--offload-folder", type=Path, default=Path("experiments/value_probe_runs/offload"))
    parser.add_argument(
        "--capture-method",
        choices=("hooks", "output_hidden_states"),
        default="hooks",
        help="hooks captures only requested layers; output_hidden_states returns every layer.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tensor_dtype(torch, raw_dtype: str):
    if raw_dtype == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[raw_dtype]


def main() -> int:
    args = parse_args()
    torch, AutoModelForCausalLM, AutoTokenizer = require_dependencies()
    rows = read_jsonl(args.dataset)
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]
    if args.max_rows:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError(f"No rows selected from {args.dataset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "cache_config.json"
    if config_path.exists() and not args.overwrite:
        raise FileExistsError(f"{config_path} exists. Use --overwrite to replace this cache.")

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
    num_layers = int(model.config.num_hidden_layers)
    layers = parse_layers(args.layers, num_layers)
    capture = LayerCapture(model, layers) if args.capture_method == "hooks" else None

    if args.overwrite:
        for old_path in args.output_dir.glob("shard_*"):
            if old_path.is_dir():
                for child in old_path.iterdir():
                    child.unlink()
                old_path.rmdir()

    metadata_rows = []
    for batch_start in range(0, len(rows), args.batch_size):
        batch_rows = rows[batch_start : batch_start + args.batch_size]
        texts = [str(row["text"]) for row in batch_rows]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            if capture:
                capture.clear()
                model(**encoded, output_hidden_states=False, use_cache=False)
                hidden_by_layer = capture.outputs
            else:
                output = model(**encoded, output_hidden_states=True, use_cache=False)
                hidden_by_layer = {
                    layer: output.hidden_states[layer + 1].detach()
                    for layer in layers
                }

        shard_index = batch_start // args.batch_size
        shard_dir = args.output_dir / f"shard_{shard_index:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        cpu_payload = {
            "row_start": batch_start,
            "row_count": len(batch_rows),
            "input_ids": encoded["input_ids"].detach().cpu(),
            "attention_mask": encoded["attention_mask"].detach().cpu(),
            "positive_labels": torch.tensor([row["positive_labels"] for row in batch_rows], dtype=torch.float32),
            "negative_labels": torch.tensor([row["negative_labels"] for row in batch_rows], dtype=torch.float32),
            "supervision_mask": torch.tensor([row["supervision_mask"] for row in batch_rows], dtype=torch.float32),
            "target_indices": torch.tensor([row["target_index"] for row in batch_rows], dtype=torch.long),
        }
        torch.save(cpu_payload, shard_dir / "meta.pt")
        for layer in layers:
            hidden = hidden_by_layer[layer].to(torch.float16).cpu()
            torch.save(hidden, shard_dir / f"layer_{layer:02d}.pt")
        for offset, row in enumerate(batch_rows):
            metadata_rows.append(
                {
                    "row_index": batch_start + offset,
                    "shard": shard_index,
                    "offset": offset,
                    "example_id": row["example_id"],
                    "split": row["split"],
                    "target_value": row["target_value"],
                    "relation": row["relation"],
                    "text": row["text"],
                }
            )
        del hidden_by_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"cached shard {shard_index:05d} rows {batch_start}-{batch_start + len(batch_rows) - 1}")

    if capture:
        capture.close()

    (args.output_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )
    config = {
        "model": args.model,
        "dataset": str(args.dataset),
        "split": args.split,
        "num_rows": len(rows),
        "num_layers": num_layers,
        "cached_layers": layers,
        "value_ids": list(VALUE_IDS),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
