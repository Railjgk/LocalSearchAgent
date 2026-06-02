#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe WeekendFlow B against LocalSearchBench samples.

This is a diagnostic script, not a benchmark submission runner. It asks:
1. What itinerary shape does B infer from each question?
2. Does the current activity+restaurant pair planner cover that shape?
3. Where does runtime concentrate on large local supply data?
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run import build_initial_state
from src.nodes.b_poi_rag import b_poi_rag_node
from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.b_replan_loop import b_replan_loop_node
from src.nodes.explainability import explainability_node
from src.nodes.intent_parser import intent_parser_node
from src.nodes.memory_manager import memory_manager_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.scenario_planner import scenario_planner_node


DATASET_URL = (
    "https://huggingface.co/datasets/localsearchbench/localsearchbench/"
    "resolve/main/data/train-00000-of-00001.parquet"
)
DEFAULT_DATASET_PATH = (
    REPO_ROOT
    / "experiments"
    / "artifacts"
    / "localsearchbench"
    / "train-00000-of-00001.parquet"
)
DEFAULT_REPORT_JSON = (
    REPO_ROOT
    / "experiments"
    / "artifacts"
    / "localsearchbench"
    / "weekendflow_b_probe.json"
)
DEFAULT_MOCK_DATA_DIR = (
    REPO_ROOT
    / "experiments"
    / "mock_data"
    / "gaode_supply_shanghai_v2_20260527_full"
)

NODE_PIPELINE = (
    ("intent_parser", intent_parser_node),
    ("memory_manager", memory_manager_node),
    ("scenario_planner", scenario_planner_node),
    ("b_poi_rag", b_poi_rag_node),
    ("candidate_generator", candidate_generator_node),
    ("constraint_filter", constraint_filter_node),
    ("plan_optimizer", plan_optimizer_node),
    ("b_replan_loop", b_replan_loop_node),
    ("explainability", explainability_node),
)


@contextmanager
def _env_override(key: str, value: str | None):
    previous = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def ensure_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    response = requests.get(DATASET_URL, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def _parse_answer_sequence(answer: str) -> list[str]:
    boxed = re.search(r"\\boxed\{([^}]+)\}", answer or "")
    if boxed:
        text = boxed.group(1)
    else:
        text = answer or ""
    parts = re.split(r"→|->|，|；", text)
    result: list[str] = []
    for part in parts:
        cleaned = re.sub(r"参考答案[:：]|最终参考答案[:：]", "", part).strip()
        cleaned = re.sub(r"（.*?）", "", cleaned).strip()
        if len(cleaned) >= 2 and cleaned not in result:
            result.append(cleaned)
    return result[:8]


def _parse_search_hops(search_path: str) -> list[dict[str, Any]]:
    hops: list[dict[str, Any]] = []
    pattern = re.compile(r"第([一二三四五六七八九十]+)跳:\s*搜索\"([^\"]+)\"")
    for match in pattern.finditer(search_path or ""):
        hops.append({"hop": match.group(1), "query": match.group(2)})
    return hops


def _run_b_pipeline(question: str, *, user_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = build_initial_state(question, user_id=user_id, payment_ui_mode="auto")
    steps: list[dict[str, Any]] = []
    for node_name, node in NODE_PIPELINE:
        start = time.time()
        updates = node(state) or {}
        state.update(updates)
        steps.append(
            {
                "node": node_name,
                "duration_sec": round(time.time() - start, 3),
                "update_keys": sorted(updates.keys()),
            }
        )
    return state, steps


def _timeline_summary(state: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = (state.get("selected_plan") or {}).get("timeline") or []
    return [
        {
            "time": item.get("time"),
            "type": item.get("type"),
            "activity": item.get("activity"),
            "poi_id": item.get("poi_id"),
        }
        for item in timeline
    ]


def _shape_gap(
    blueprint: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    poi_timeline = [item for item in timeline if item.get("poi_id")]
    expected_roles = [item.get("role") for item in blueprint.get("node_intents", [])]
    unsupported_roles = blueprint.get("unsupported_roles", []) or []
    template_mode = blueprint.get("template_mode")
    missing_role_count = max(0, len(expected_roles) - len(poi_timeline))
    return {
        "template_mode": template_mode,
        "expected_role_count": len(expected_roles),
        "actual_poi_node_count": len(poi_timeline),
        "missing_role_count": missing_role_count,
        "unsupported_roles": unsupported_roles,
        "requires_rag": bool(blueprint.get("requires_rag")),
        "pair_planner_insufficient": bool(
            template_mode == "multi_node"
            or unsupported_roles
            or missing_role_count > 0
        ),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = ensure_dataset(Path(args.dataset_path))
    df = pd.read_parquet(dataset_path)
    if args.city:
        df = df[df["City"].eq(args.city)]
    if args.difficulty:
        df = df[df["Difficulty"].eq(args.difficulty)]
    df = df.iloc[args.offset : args.offset + args.limit]

    results: list[dict[str, Any]] = []
    mock_data_dir = str(Path(args.mock_data_dir).resolve()) if args.mock_data_dir else None
    enable_rag = bool(getattr(args, "enable_rag", False))
    with (
        _env_override("WF_MOCK_DATA_DIR", mock_data_dir),
        _env_override("WF_B_RAG_DATA_DIR", mock_data_dir if enable_rag else None),
        _env_override("WF_B_RAG_ENABLED", "1" if enable_rag else None),
    ):
        for row_index, row in df.iterrows():
            question = str(row["Question"])
            start = time.time()
            try:
                state, steps = _run_b_pipeline(
                    question,
                    user_id=f"lsb_probe_{row_index}",
                )
                blueprint = (
                    state.get("b_itinerary_blueprint")
                    or (state.get("constraints") or {}).get("b_itinerary_blueprint")
                    or {}
                )
                timeline = _timeline_summary(state)
                selected_plan = state.get("selected_plan") or {}
                result = {
                    "dataset_index": int(row_index),
                    "city": row["City"],
                    "difficulty": row["Difficulty"],
                    "hop_count": int(row["Hop Count"]),
                    "question": question,
                    "benchmark_hops": _parse_search_hops(str(row["Multi-hop search path"])),
                    "benchmark_answer_sequence": _parse_answer_sequence(str(row["Answer"])),
                    "scene_type": state.get("scene_type"),
                    "scenario_activities": state.get("scenario_activities") or [],
                    "b_itinerary_blueprint": blueprint,
                    "b_rag_candidate_coverage": state.get("b_rag_candidate_coverage") or {},
                    "candidate_generation_issues": state.get("candidate_generation_issues") or [],
                    "candidates_count": len(state.get("candidates") or []),
                    "filtered_candidates_count": len(state.get("filtered_candidates") or []),
                    "selected_plan_status": selected_plan.get("plan_status"),
                    "selected_plan_planner_mode": selected_plan.get("planner_mode"),
                    "selected_plan_plan_shape": selected_plan.get("plan_shape"),
                    "selected_plan_execution_ready": selected_plan.get("execution_ready"),
                    "selected_plan_planning_days": selected_plan.get("planning_days"),
                    "selected_plan_id": selected_plan.get("plan_id"),
                    "action_hints_count": len(selected_plan.get("action_hints") or []),
                    "route_summary": selected_plan.get("route") or {},
                    "timeline": timeline,
                    "shape_gap": _shape_gap(blueprint, timeline),
                    "optimization_score": state.get("optimization_score"),
                    "node_timings": steps,
                    "duration_sec": round(time.time() - start, 3),
                }
            except Exception as exc:  # pragma: no cover - diagnostic script
                result = {
                    "dataset_index": int(row_index),
                    "question": question,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                    "duration_sec": round(time.time() - start, 3),
                }
            results.append(result)

    summary = _summarize_results(results)
    return {
        "source": "localsearchbench/localsearchbench",
        "dataset_path": str(dataset_path),
        "mock_data_dir": mock_data_dir,
        "enable_rag": enable_rag,
        "city": args.city,
        "offset": args.offset,
        "limit": args.limit,
        "summary": summary,
        "results": results,
    }


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    errored = sum(1 for item in results if item.get("error_type"))
    gaps = [item.get("shape_gap") or {} for item in results if not item.get("error_type")]
    pair_insufficient = sum(1 for gap in gaps if gap.get("pair_planner_insufficient"))
    requires_rag = sum(1 for gap in gaps if gap.get("requires_rag"))
    avg_duration = (
        round(sum(float(item.get("duration_sec") or 0) for item in results) / total, 3)
        if total
        else 0.0
    )
    slow_cases = [
        item.get("dataset_index")
        for item in results
        if float(item.get("duration_sec") or 0) >= 20
    ]
    unsupported_roles: dict[str, int] = {}
    for gap in gaps:
        for role in gap.get("unsupported_roles") or []:
            unsupported_roles[role] = unsupported_roles.get(role, 0) + 1
    return {
        "total": total,
        "errored": errored,
        "pair_planner_insufficient": pair_insufficient,
        "requires_rag": requires_rag,
        "avg_duration_sec": avg_duration,
        "slow_case_indexes": slow_cases,
        "unsupported_role_counts": unsupported_roles,
    }


def write_reports(report: dict[str, Any], json_path: Path) -> tuple[Path, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = json_path.with_suffix(".md")

    lines = [
        "# WeekendFlow B LocalSearchBench Probe",
        "",
        "## Summary",
        "",
    ]
    for key, value in report["summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Cases", ""])
    for item in report["results"]:
        lines.extend(
            [
                f"### #{item.get('dataset_index')} {item.get('difficulty', '')} / {item.get('hop_count', '')} hops",
                "",
                f"- Question: {item.get('question')}",
            ]
        )
        if item.get("error_type"):
            lines.append(f"- Error: {item.get('error_type')} {item.get('error')}")
            lines.append("")
            continue
        blueprint = item.get("b_itinerary_blueprint") or {}
        lines.extend(
            [
                f"- Blueprint: {blueprint.get('template_mode')} / {blueprint.get('planning_horizon')} / nodes={blueprint.get('node_count')}",
                f"- Unsupported roles: {blueprint.get('unsupported_roles')}",
                f"- Requires RAG: {blueprint.get('requires_rag')}",
                f"- Shape gap: {item.get('shape_gap')}",
                f"- Plan status: {item.get('selected_plan_status') or 'executable_or_empty'} / days={item.get('selected_plan_planning_days')}",
                f"- Timeline POIs: {' -> '.join(str(node.get('activity')) for node in item.get('timeline', []) if node.get('poi_id'))}",
                f"- Timeline skeleton: {' -> '.join(str(node.get('activity')) for node in item.get('timeline', []))}",
                f"- Benchmark answer: {' -> '.join(item.get('benchmark_answer_sequence', []))}",
                f"- Duration: {item.get('duration_sec')} sec",
                "",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe WeekendFlow B on LocalSearchBench.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--mock-data-dir", default=str(DEFAULT_MOCK_DATA_DIR))
    parser.add_argument("--city", default="上海")
    parser.add_argument("--difficulty", default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--enable-rag", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_probe(args)
    json_path, md_path = write_reports(report, Path(args.report_out))
    print(
        json.dumps(
            {
                "json_report": str(json_path),
                "md_report": str(md_path),
                "summary": report["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
