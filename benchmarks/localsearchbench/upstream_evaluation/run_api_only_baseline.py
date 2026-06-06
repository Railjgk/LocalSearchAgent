#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API-only baseline:
 - For each question, call the LLM API directly (no RAG, no WebSearch)
 - Ask the model to output a trajectory (if applicable) and a final answer
 - Save outputs in agent_results-compatible JSON (metadata + results[])

Usage:
  python run_api_only_baseline.py --dataset path/to/data.json --output-dir ./api_only_outputs
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_utils import get_friday_client, ConfigManager


def load_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        for key in ("data", "questions", "records", "items", "results"):
            if isinstance(raw.get(key), list):
                entries = raw.get(key)
                break
        else:
            entries = [raw]
    else:
        entries = []
    normalized = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        q = e.get("query") or e.get("question") or e.get("title") or e.get("prompt")
        if not q:
            continue
        normalized.append({
            "query": q,
            "ground_truth": e.get("ground_truth") or e.get("box") or e.get("reference_answer"),
            "raw": e
        })
    return normalized


def build_messages_for_api_only(query: str, system_prompt: str) -> List[Dict[str, str]]:
    user = (
        f"问题: {query}\n\n"
        "请仅依靠自身生成。\n"
        "请同时输出：\n"
        "1) trajectory: 一个包含每一步检索/思路的列表（如果问题需要多跳检索则列出每一步的查询意图），\n"
        "2) final_answer: 基于上述 trajectory 的最终答案。\n"
        "输出格式要求：一个单行或多行的 JSON 对象，例如:\n"
        "{\"trajectory\": [\"step1\", \"step2\"], \"final_answer\": \"...\"}\n"
        "只输出 JSON，不要额外解释。"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def worker_proc(worker_id: int, items: List[Dict[str, Any]], args):
    # create Friday client with API key manager option
    client = get_friday_client(model_name=args.model, rpm=args.rpm, config_path=args.config, use_api_key_manager=args.use_multi_tokens)

    # if using multi tokens, subset keys for this worker (stable per-worker subset)
    def _subset_api_keys(api_keys: List[str], worker_id: int, worker_count: int) -> List[str]:
        if not api_keys:
            return []
        if worker_count <= 1:
            return api_keys
        subset = [k for i, k in enumerate(api_keys) if (i % worker_count) == worker_id]
        if not subset:
            subset = [api_keys[worker_id % len(api_keys)]]
        return subset

    if args.use_multi_tokens:
        try:
            cfg = ConfigManager(config_path=args.config)
            all_keys = cfg.get_api_keys()
            subset = _subset_api_keys(all_keys, worker_id, args.parallel_workers)
            if subset:
                client.api_keys = subset
                client.api_token = subset[0]
        except Exception:
            # fallback: leave client as-is
            pass
    local_results = []
    for entry in items:
        idx = entry["_global_idx"]
        q = entry["query"]
        gt = entry.get("ground_truth")
        t0 = time.time()
        messages = build_messages_for_api_only(q, args.system_prompt or "你是一个智能助手，请给出结构化输出。")
        try:
            response_text, cost_time, token_info = client.single_request(messages, temperature=0.0, max_tokens=args.max_tokens)
            success = True
        except Exception as e:
            response_text = ""
            cost_time = 0.0
            token_info = {}
            success = False

        processing_time = time.time() - t0

        # try to parse JSON from model response
        traj = None
        final_answer = None
        try:
            parsed = json.loads(response_text.strip())
            traj = parsed.get("trajectory")
            final_answer = parsed.get("final_answer") or parsed.get("answer") or parsed.get("final")
        except Exception:
            # fallback: store raw response_text
            final_answer = response_text

        res = {
            "question_id": idx,
            "question": q,
            "ground_truth": gt,
            "final_response": final_answer,
            "trajectory": traj,
            "success": success,
            "processing_time": processing_time,
            "tool_calls": [],  # intentionally empty for API-only ablation
            "conversation_history": [
                {
                    "round": 1,
                    "llm_response": response_text,
                    "token_info": token_info,
                    "cost_time": cost_time
                }
            ],
            "judge_scores": None
        }
        local_results.append(res)
        print(f"[W{worker_id}] [{idx}] success={success} time={processing_time:.2f}s")
    return local_results


def main():
    parser = argparse.ArgumentParser(description="Run API-only baseline (no RAG, no WebSearch)")
    parser.add_argument("--dataset", required=True, help="Path to dataset (json)")
    parser.add_argument("--output-dir", default="./api_only_outputs", help="Output directory")
    parser.add_argument("--model", default="deepseek-v31-meituan", help="LLM model name")
    parser.add_argument("--max-tokens", type=int, default=1600, help="LLM max tokens")
    parser.add_argument("--rpm", type=int, default=None, help="LLM rpm limit (optional)")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--system-prompt", type=str, default=None, help="Optional system prompt to override")
    parser.add_argument("--use-multi-tokens", action="store_true", help="Use multiple API tokens from config/model_rpm.yaml")
    parser.add_argument("--config", type=str, default="config/model_rpm.yaml", help="Config path for API keys and RPM")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to run (useful for debugging)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]
    print(f"📂 加载 {len(dataset)} 个问题")

    # dispatch work
    parallel = args.parallel_workers if args.parallel_workers and args.parallel_workers > 1 else 1
    results = []
    if parallel == 1:
        items = [{"query": it["query"], "ground_truth": it.get("ground_truth"), "_global_idx": i} for i, it in enumerate(dataset, 1)]
        results = worker_proc(0, items, args)
    else:
        buckets = [[] for _ in range(parallel)]
        for i, it in enumerate(dataset, 1):
            buckets[i % parallel].append({"query": it["query"], "ground_truth": it.get("ground_truth"), "_global_idx": i})
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = [ex.submit(worker_proc, wid, buckets[wid], args) for wid in range(parallel)]
            for fut in as_completed(futures):
                results.extend(fut.result())

    start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

    agent_output = {
        "metadata": {
            "dataset": dataset_name,
            "timestamp": start_ts,
            "total_questions": len(results),
            "model": args.model,
            "ablation": "api_only"
        },
        "results": results
    }
    out_path = os.path.join(args.output_dir, f"{dataset_name}_agent_results_{start_ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(agent_output, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成，agent results: {out_path}")


if __name__ == "__main__":
    main()


