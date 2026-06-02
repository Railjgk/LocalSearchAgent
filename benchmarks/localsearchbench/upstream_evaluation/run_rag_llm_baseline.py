#!/usr/bin/env python3
"""
Pure RAG + LLM baseline:
 - For each question, run one RAG search using the question text as the query
 - Feed the retrieved merchant context to the LLM to generate an answer
 - Save outputs in a judged-like JSON format (metadata + results[])

Usage:
  python run_rag_llm_baseline.py --dataset path/to/data.json --output-dir ./baseline_outputs
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from rag_agent import RAGAgent
from llm_utils import get_friday_client, ConfigManager, FridayClient
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a dataset file. Supports JSON list or single JSON object containing list under keys."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        # common wrappers
        for key in ("data", "questions", "records", "items", "results"):
            if isinstance(raw.get(key), list):
                entries = raw.get(key)
                break
        else:
            # try to treat dict as single entry
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


def build_llm_messages(query: str, rag_context: str) -> List[Dict[str, str]]:
    """Simple message template: system + user with context."""
    system = (
        "你是一个专业的本地搜索助手。请基于下面的检索到的商户信息（RAG上下文）回答用户的查询，"
    )
    user = (
        f"用户查询: {query}\n\n检索到的商户信息:\n{rag_context}\n\n"
        "注意：请在最终答案中明确列出你推荐的商家及其对应的 POI 标识（使用上下文中的 `poi_id` 或 `id` 字段）。"
        " 不允许编造商家或 POI；若上下文中没有 POI，请说明无法提供 POI。"
        "请基于以上信息直接给出最终推荐答案（只输出结果，不要输出额外调试信息）。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def main():
    parser = argparse.ArgumentParser(description="Run pure RAG + LLM baseline")
    parser.add_argument("--dataset", required=True, help="Path to dataset (json)")
    parser.add_argument("--output-dir", default="./baseline_outputs", help="Output directory")
    parser.add_argument("--model", default="deepseek-v32-meituan", help="LLM model name")
    parser.add_argument("--rag-index-path", default=None, help="RAG index path (optional)")
    parser.add_argument("--rag-top-k", type=int, default=5, help="RAG 返回结果数量 (默认: 5)")
    parser.add_argument("--embedding-model-path", type=str, default=None, help="嵌入模型路径（可选，覆盖RAG默认）")
    parser.add_argument("--reranker-model-path", type=str, default=None, help="重排序模型路径（可选，覆盖RAG默认）")
    parser.add_argument("--max-tokens", type=int, default=1600, help="LLM max tokens")
    parser.add_argument("--rpm", type=int, default=None, help="LLM rpm limit (optional)")
    parser.add_argument("--config", default="config/model_rpm.yaml", help="配置文件路径 (默认: config/model_rpm.yaml)")
    parser.add_argument("--parallel-workers", type=int, default=1, help="并行工作线程数（默认1）")
    parser.add_argument("--direct-api", action="store_true", help="Use direct HTTP API calls (bypass FridayClient)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset)
    print(f"📂 加载 {len(dataset)} 个问题")

    # init RAG (shared) and a lock for thread-safe access
    rag = RAGAgent(
        index_path=args.rag_index_path,
        embedding_model_path=args.embedding_model_path,
        reranker_model_path=args.reranker_model_path,
        use_reranker=True
    )
    rag_lock = threading.Lock()

    start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: List[Dict[str, Any]] = []
    search_details: List[Dict[str, Any]] = []
    results_lock = threading.Lock()

    # helper to split api keys (stable per-worker subset)
    def _subset_api_keys(api_keys: List[str], worker_id: int, worker_count: int) -> List[str]:
        if not api_keys:
            return []
        if worker_count <= 1:
            return api_keys
        subset = [k for i, k in enumerate(api_keys) if (i % worker_count) == worker_id]
        if not subset:
            subset = [api_keys[worker_id % len(api_keys)]]
        return subset

    # worker processing function
    def worker_proc(worker_id: int, items: List[Dict[str, Any]]):
        # prepare client or direct API keys
        client = None
        api_base = None
        api_keys = None
        if args.direct_api:
            try:
                cfg = ConfigManager()
                api_base = cfg.get_base_url()
                if api_base.endswith('/chat/completions'):
                    api_base = api_base[: -len('/chat/completions')]
                all_keys = cfg.get_api_keys()
                api_keys = _subset_api_keys(all_keys, worker_id, args.parallel_workers)
            except Exception:
                api_base = None
                api_keys = None
        else:
            # create FridayClient exactly like evaluate_trajectories
            try:
                cfg = ConfigManager(config_path=args.config)
                base = cfg.get_base_url()
            except Exception:
                base = None
            client = FridayClient(
                model_name=args.model,
                api_url=base or "https://api.example.com/v1/openai/native",
                api_token=None,
                temperature=0.0,
                max_tokens=args.max_tokens,
                timeout=300,
                max_retries=5,
                config_path=args.config,
                use_api_key_manager=True,
            )
            try:
                cfg = ConfigManager(config_path=args.config)
                all_keys = cfg.get_api_keys()
                subset = _subset_api_keys(all_keys, worker_id, args.parallel_workers)
                if subset:
                    client.api_keys = subset
                    client.api_token = subset[0]
            except Exception:
                pass

        local_results = []
        local_search = []
        for entry in items:
            idx = entry["_global_idx"]
            q = entry["query"]
            gt = entry.get("ground_truth")
            t0 = time.time()

            # RAG search (thread-safe) - guard against RAG internal failures
            try:
                with rag_lock:
                    rag_res = rag.search(q, top_k=args.rag_top_k)
                rag_context = rag_res.get("context", "")
            except Exception as e:
                import traceback as _tb
                tb = _tb.format_exc()
                logger = __import__("logging").getLogger(__name__)
                logger.error("❌ RAG检索失败: %s", e)
                logger.error("❌ 完整错误信息:\n%s", tb)
                # build a minimal failed rag_res to keep output format consistent
                rag_res = {
                    "success": False,
                    "query": q,
                    "results": [],
                    "context": "",
                    "error": str(e),
                    "traceback": tb
                }
                rag_context = ""

            messages = build_llm_messages(q, rag_context)
            # call LLM
            response_text = ""
            cost_time = 0.0
            token_info = {}
            success = False

            if args.direct_api and api_base and api_keys:
                # rotate through keys; try each once
                for ak in api_keys:
                    payload = {
                        "model": args.model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": args.max_tokens
                    }
                    headers = {"Authorization": f"Bearer {ak}", "Content-Type": "application/json"}
                    try:
                        resp = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=300)
                        cost_time = time.time() - t0
                        if resp.status_code == 200:
                            data = resp.json()
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if isinstance(choice.get("message"), dict):
                                    response_text = choice["message"].get("content", "")
                                else:
                                    response_text = choice.get("text", "")
                                token_info = {"prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0), "completion_tokens": data.get("usage", {}).get("completion_tokens", 0), "total_tokens": data.get("usage", {}).get("total_tokens", 0)}
                                success = True
                                break
                            else:
                                # try next key
                                continue
                        else:
                            # if 400, log payload and response for debugging and stop trying
                            if resp.status_code == 400:
                                import logging as _logging
                                _logger = _logging.getLogger(__name__)
                                _logger.error("direct-api 400 Bad Request. payload: %s", json.dumps(payload, ensure_ascii=False))
                                _logger.error("direct-api response: %s", resp.text)
                                break
                            else:
                                continue
                    except Exception:
                        # try next key
                        continue
            elif client:
                try:
                    response_text, cost_time, token_info = client.single_request(messages, temperature=0.0, max_tokens=args.max_tokens)
                    success = True
                except Exception:
                    response_text = ""
                    cost_time = 0.0
                    token_info = {}
                    success = False

            processing_time = time.time() - t0

            res = {
                "question_id": idx,
                "question": q,
                "ground_truth": gt,
                "final_response": response_text,
                "success": success,
                "processing_time": processing_time,
                "tool_calls": [
                    {
                        "round": 1,
                        "tool_type": "rag",
                        "query": q,
                        "result": rag_res,
                        "timestamp": datetime.now().isoformat()
                    }
                ],
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

            search_record = {
                "question_id": idx,
                "round": 1,
                "tool_type": "rag",
                "query": q,
                "timestamp": datetime.now().isoformat(),
                "success": rag_res.get("success", False),
                "rag": {
                    "total_results": rag_res.get("total_results", 0),
                    "results": rag_res.get("results", []),
                    "context": rag_res.get("context", "")
                }
            }

            local_results.append(res)
            local_search.append(search_record)
            print(f"[W{worker_id}] [{idx}] success={success} time={processing_time:.2f}s")

        with results_lock:
            results.extend(local_results)
            search_details.extend(local_search)

    # dispatch work (round-robin)
    parallel = args.parallel_workers if args.parallel_workers and args.parallel_workers > 1 else 1
    if parallel == 1:
        items = [{"query": it["query"], "ground_truth": it.get("ground_truth"), "_global_idx": i} for i, it in enumerate(dataset, 1)]
        worker_proc(0, items)
    else:
        buckets = [[] for _ in range(parallel)]
        for i, it in enumerate(dataset, 1):
            buckets[i % parallel].append({"query": it["query"], "ground_truth": it.get("ground_truth"), "_global_idx": i})
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = [ex.submit(worker_proc, wid, buckets[wid]) for wid in range(parallel)]
            for fut in as_completed(futures):
                fut.result()

    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

    # agent_results file (compatible with evaluate_trajectories load_agent_results)
    agent_output = {
        "metadata": {
            "dataset": dataset_name,
            "timestamp": start_ts,
            "total_questions": len(results),
            "model": args.model
        },
        "results": results
    }
    agent_out_path = os.path.join(args.output_dir, f"{dataset_name}_agent_results_{start_ts}.json")
    with open(agent_out_path, "w", encoding="utf-8") as f:
        json.dump(agent_output, f, ensure_ascii=False, indent=2)

    # search_details file (compatible with evaluate_trajectories load_search_details)
    search_out_path = os.path.join(args.output_dir, f"{dataset_name}_agent_results_{start_ts}_search_details.json")
    with open(search_out_path, "w", encoding="utf-8") as f:
        json.dump(search_details, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成，agent results: {agent_out_path}")
    print(f"✅ 完成，search details: {search_out_path}")


if __name__ == "__main__":
    main()


