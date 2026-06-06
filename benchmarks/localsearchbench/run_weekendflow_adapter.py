#!/usr/bin/env python3
"""Run the local WeekendFlow graph on LocalSearchBench samples.

This adapter writes the upstream LocalSearchBench `agent_results` shape so the
official summarizers and LLM-judge scripts can be reused. It is intentionally a
compatibility harness, not a claim that WeekendFlow already has the same
merchant-scale RAG/Web tool environment as LocalPlayground.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graph import get_graph


DEFAULT_DATASET = (
    Path(__file__).resolve().parent / "data" / "localsearchbench_train.json"
)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    for key in ("data", "items", "records", "questions"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return [item for item in data[key] if isinstance(item, dict)]
    raise ValueError(f"Unsupported dataset shape: {path}")


def compact_selected_plan(plan: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "plan_id",
        "title",
        "total_price",
        "total_distance_km",
        "weighted_score",
        "optimization_score",
        "objective_vector",
        "execution_ready",
        "timeline",
        "action_hints",
        "why_selected",
    )
    return {field: plan.get(field) for field in fields if field in plan}


def build_final_response(state: dict[str, Any]) -> str:
    if state.get("final_share_message"):
        return str(state["final_share_message"])
    if state.get("explanation_text"):
        return str(state["explanation_text"])
    selected_plan = state.get("selected_plan") or {}
    if selected_plan:
        return json.dumps(compact_selected_plan(selected_plan), ensure_ascii=False)
    issues = state.get("candidate_generation_issues") or []
    if issues:
        return json.dumps({"status": "no_plan", "issues": issues}, ensure_ascii=False)
    return "未生成可执行计划。"


def build_trajectory(state: dict[str, Any]) -> list[dict[str, Any]]:
    selected_plan = state.get("selected_plan") or {}
    return [
        {
            "stage": "intent_parser",
            "query": state.get("user_input"),
            "intent": state.get("intent"),
            "constraints": state.get("constraints"),
        },
        {
            "stage": "candidate_generation",
            "query": "local mock supply candidate generation",
            "candidates_count": len(state.get("candidates") or []),
            "filtered_candidates_count": len(state.get("filtered_candidates") or []),
            "issues": state.get("candidate_generation_issues") or [],
        },
        {
            "stage": "plan_optimizer",
            "query": selected_plan.get("title") or selected_plan.get("plan_id"),
            "selected_plan": compact_selected_plan(selected_plan),
        },
    ]


def run_one(graph: Any, sample: dict[str, Any], question_id: int) -> dict[str, Any]:
    question = sample.get("question") or sample.get("query") or ""
    started = time.time()
    result: dict[str, Any] = {
        "question_id": question_id,
        "id": sample.get("id"),
        "city": sample.get("city"),
        "difficulty": sample.get("difficulty"),
        "question": question,
        "ground_truth": sample.get("ground_truth"),
        "reference_answer": sample.get("reference_answer") or sample.get("answer"),
        "search_path": sample.get("search_path") or sample.get("multi_hop_search_path"),
        "tool_calls": [],
        "conversation_history": [],
        "trajectory": [],
        "final_response": "",
        "processing_time": 0.0,
        "success": False,
        "error": None,
        "judge_scores": None,
    }
    try:
        state = graph.invoke({"user_input": question, "user_id": "localsearchbench"})
        result["final_response"] = build_final_response(state)
        result["trajectory"] = build_trajectory(state)
        result["conversation_history"] = [
            {
                "round": 1,
                "llm_response": result["final_response"],
                "token_info": {},
                "cost_time": result["processing_time"],
            }
        ]
        result["weekendflow_state"] = {
            "scene_type": state.get("scene_type"),
            "intent": state.get("intent"),
            "constraints": state.get("constraints"),
            "selected_plan": compact_selected_plan(state.get("selected_plan") or {}),
            "candidate_generation_issues": state.get("candidate_generation_issues") or [],
            "execution_log": state.get("execution_log") or [],
        }
        result["success"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["processing_time"] = time.time() - started
        if result["conversation_history"]:
            result["conversation_history"][0]["cost_time"] = result["processing_time"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WeekendFlow on LocalSearchBench.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/localsearchbench/results"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--city", action="append", help="Filter by Chinese city name; can repeat.")
    parser.add_argument("--difficulty", action="append", help="Filter by difficulty, e.g. L3 or L4.")
    parser.add_argument("--mock-data-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.mock_data_dir:
        os.environ["WF_MOCK_DATA_DIR"] = str(args.mock_data_dir.resolve())

    samples = load_dataset(args.dataset)
    if args.city:
        cities = set(args.city)
        samples = [sample for sample in samples if sample.get("city") in cities]
    if args.difficulty:
        difficulties = {item.upper() for item in args.difficulty}
        samples = [
            sample
            for sample in samples
            if str(sample.get("difficulty", "")).upper() in difficulties
        ]
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    graph = get_graph()
    results = [run_one(graph, sample, index + 1) for index, sample in enumerate(samples)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"weekendflow_localsearchbench_agent_results_{timestamp}.json"
    payload = {
        "metadata": {
            "dataset": args.dataset.stem,
            "timestamp": timestamp,
            "total_questions": len(results),
            "model": "weekendflow-local-graph",
            "adapter": "benchmarks/localsearchbench/run_weekendflow_adapter.py",
            "note": "Compatibility output only; local graph does not use the official 1.3M merchant RAG index.",
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
