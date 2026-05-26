"""Serve a local HTML app for sentence-level and fluent-level value probes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.value_probe.cache_qwen_activations import LayerCapture, tensor_dtype
from experiments.value_probe.constants import VALUE_IDS
from experiments.value_probe.data import read_jsonl
from experiments.value_probe.fluent_alignment import (
    aggregate_token_scores_to_words,
    normalize_for_alignment,
    tokenizer_token_spans,
    word_char_spans,
)
from experiments.value_probe.modeling import FluentValueProbe, TokenValueProbe, aggregate, require_torch


torch, _nn = require_torch()


def require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("serve_probe_app.py requires transformers.") from exc
    return AutoModelForCausalLM, AutoTokenizer


VALUE_NAMES = {
    "family_care": "家庭照顾",
    "health": "健康",
    "convenience": "便利",
    "cost_sensitivity": "预算敏感",
}


def build_fluent_vocab(path: Path) -> set[str]:
    vocab: set[str] = set()
    for row in read_jsonl(path):
        for token in row.get("tokens", []):
            normalized = normalize_for_alignment(str(token.get("token", "")))
            if normalized:
                vocab.add(normalized)
    return vocab


def auto_segment(text: str, vocab: set[str], max_token_chars: int = 6) -> list[str]:
    """Segment punctuation-stripped Chinese text with a training-token lexicon."""

    source = normalize_for_alignment(text)
    tokens: list[str] = []
    cursor = 0
    while cursor < len(source):
        char = source[cursor]
        if char.isascii() and char.isalnum():
            end = cursor + 1
            while end < len(source) and source[end].isascii() and source[end].isalnum():
                end += 1
            tokens.append(source[cursor:end])
            cursor = end
            continue
        best = ""
        max_end = min(len(source), cursor + max_token_chars)
        for end in range(max_end, cursor, -1):
            candidate = source[cursor:end]
            if candidate in vocab:
                best = candidate
                break
        if not best:
            best = char
        tokens.append(best)
        cursor += len(best)
    return tokens


class ProbeRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        AutoModelForCausalLM, AutoTokenizer = require_transformers()
        self.args = args
        self.value_ids = list(VALUE_IDS)
        self.vocab = build_fluent_vocab(args.fluent_data)

        self.sentence_checkpoint = torch.load(args.sentence_probe, map_location="cpu", weights_only=False)
        self.fluent_checkpoint = torch.load(args.fluent_probe, map_location="cpu", weights_only=False)
        self.sentence_layer = int(args.sentence_layer if args.sentence_layer >= 0 else self.sentence_checkpoint["layer"])
        self.fluent_layer = int(args.fluent_layer if args.fluent_layer >= 0 else self.fluent_checkpoint["layer"])

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=tensor_dtype(torch, args.dtype),
            device_map=args.device_map,
            offload_folder=str(args.offload_folder),
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        self.model.eval()

        probe_device = torch.device(args.probe_device if args.probe_device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sentence_probe = TokenValueProbe(
            hidden_size=int(self.sentence_checkpoint["hidden_size"]),
            num_values=len(self.value_ids),
            dropout=0.0,
        ).to(probe_device)
        self.sentence_probe.load_state_dict(self.sentence_checkpoint["state_dict"])
        self.sentence_probe.eval()

        self.fluent_probe = FluentValueProbe(
            hidden_size=int(self.fluent_checkpoint["hidden_size"]),
            num_values=len(self.value_ids),
            dropout=0.0,
            score_scale=float(self.fluent_checkpoint.get("score_scale", 6.0)),
        ).to(probe_device)
        self.fluent_probe.load_state_dict(self.fluent_checkpoint["state_dict"])
        self.fluent_probe.eval()

        layers = sorted({self.sentence_layer, self.fluent_layer})
        self.capture = LayerCapture(self.model, layers)

    def close(self) -> None:
        self.capture.close()

    def predict(self, text: str, tokens: list[str] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        text = text.strip()
        if not text:
            raise ValueError("text is required")
        tokens = tokens or auto_segment(text, self.vocab)
        source = normalize_for_alignment(text)
        _, word_spans = word_char_spans(
            {
                "example_id": "input",
                "text": text,
                "tokens": [{"token": token, "score": 0} for token in tokens],
            }
        )

        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with torch.no_grad():
            self.capture.clear()
            self.model(**encoded, output_hidden_states=False, use_cache=False)

            sentence_hidden = self.capture.outputs[self.sentence_layer].detach().to(next(self.sentence_probe.parameters()).device)
            fluent_hidden = self.capture.outputs[self.fluent_layer].detach().to(next(self.fluent_probe.parameters()).device)
            attention_mask = encoded["attention_mask"].to(next(self.sentence_probe.parameters()).device)

            positive_logits, negative_logits = self.sentence_probe(sentence_hidden)
            positive_doc = torch.sigmoid(
                aggregate(
                    positive_logits,
                    attention_mask,
                    self.args.aggregation,
                    self.args.temperature,
                    self.args.topk,
                )
            )[0].detach().cpu()
            negative_doc = torch.sigmoid(
                aggregate(
                    negative_logits,
                    attention_mask,
                    self.args.aggregation,
                    self.args.temperature,
                    self.args.topk,
                )
            )[0].detach().cpu()
            fluent_scores = self.fluent_probe(fluent_hidden)[0].detach().cpu()

        input_ids = encoded["input_ids"][0].detach().cpu().tolist()
        mask = encoded["attention_mask"][0].detach().cpu().tolist()
        token_spans = tokenizer_token_spans(
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=mask,
            source=source,
        )

        sentence_level = []
        fluent_level: dict[str, list[dict[str, Any]]] = {}
        for value_index, value_id in enumerate(self.value_ids):
            positive = float(positive_doc[value_index])
            negative = float(negative_doc[value_index])
            sentence_level.append(
                {
                    "value_id": value_id,
                    "value_name": VALUE_NAMES[value_id],
                    "positive_score": positive,
                    "negative_score": negative,
                    "signed_score": 6.0 * (positive - negative),
                }
            )
            word_scores = aggregate_token_scores_to_words(
                word_spans=word_spans,
                token_spans=token_spans,
                token_scores=fluent_scores[:, value_index].tolist(),
            )
            fluent_level[value_id] = [
                {
                    "token_index": int(item["token_index"]),
                    "token": item["token"],
                    "predicted_score": float(item["predicted_score"]),
                }
                for item in word_scores
            ]

        sentence_level.sort(key=lambda item: abs(float(item["signed_score"])), reverse=True)
        return {
            "text": text,
            "tokens": tokens,
            "model": self.args.model,
            "sentence_layer": self.sentence_layer,
            "fluent_layer": self.fluent_layer,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "sentence_level": sentence_level,
            "fluent_level": fluent_level,
        }


def json_response(handler: BaseHTTPRequestHandler, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def html_response(handler: BaseHTTPRequestHandler, path: Path) -> None:
    body = path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(runtime: ProbeRuntime, html_path: Path):
    docs_root = REPO_ROOT / "docs"

    def resolve_html_path(raw_path: str) -> Path | None:
        path = unquote(raw_path.split("?", 1)[0])
        if path in {"/", "/index.html"}:
            return html_path
        if path.startswith("/docs/"):
            candidate = (REPO_ROOT / path.lstrip("/")).resolve()
        else:
            candidate = (docs_root / path.lstrip("/")).resolve()
        if docs_root in candidate.parents and candidate.suffix == ".html" and candidate.exists():
            return candidate
        return None

    class ProbeHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()

        def do_HEAD(self) -> None:
            if resolve_html_path(self.path):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                return
            if self.path == "/api/health":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                return
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

        def do_GET(self) -> None:
            request_html_path = resolve_html_path(self.path)
            if request_html_path:
                html_response(self, request_html_path)
                return
            if self.path == "/api/health":
                json_response(
                    self,
                    {
                        "ok": True,
                        "model": runtime.args.model,
                        "sentence_layer": runtime.sentence_layer,
                        "fluent_layer": runtime.fluent_layer,
                    },
                )
                return
            json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/predict":
                json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                content_length = int(self.headers.get("Content-Length") or "0")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                text = str(payload.get("text") or "")
                raw_tokens = payload.get("tokens")
                tokens = raw_tokens if isinstance(raw_tokens, list) and all(isinstance(token, str) for token in raw_tokens) else None
                json_response(self, runtime.predict(text, tokens=tokens))
            except Exception as exc:  # noqa: BLE001
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

    return ProbeHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--html", type=Path, default=Path("docs/qwen3-4b-token-value-probe-training-report.html"))
    parser.add_argument(
        "--sentence-probe",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen3_4b_token_probe/probes/layer_11/probe.pt"),
    )
    parser.add_argument(
        "--fluent-probe",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen3_4b_token_probe/fluent_level_probes/layer_29/probe.pt"),
    )
    parser.add_argument(
        "--fluent-data",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen3_4b_token_probe/fluent_level/all.jsonl"),
    )
    parser.add_argument("--sentence-layer", type=int, default=-1)
    parser.add_argument("--fluent-layer", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--probe-device", default="")
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=Path("experiments/value_probe_runs/qwen3_4b_token_probe/offload"),
    )
    parser.add_argument("--aggregation", choices=("logmeanexp", "topk_mean"), default="logmeanexp")
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = ProbeRuntime(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime, args.html))
    print(f"serving value probe app at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        runtime.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
