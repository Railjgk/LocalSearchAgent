#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate B's multi-node itinerary planning with local POI RAG enabled.

This runner is intentionally diagnostic.  It reports whether B decomposes the
request into the right itinerary nodes, whether local RAG covers those nodes,
and whether the selected plan/timeline follows the intended shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nodes.b_poi_rag import b_poi_rag_node
from src.nodes.candidate_generator import candidate_generator_node
from src.nodes.constraint_filter import constraint_filter_node
from src.nodes.plan_optimizer import plan_optimizer_node
from src.nodes.b_replan_loop import b_replan_loop_node
from src.nodes.explainability import explainability_node


DEFAULT_CASES = REPO_ROOT / "experiments" / "b_rag_itinerary_eval_cases.jsonl"
DEFAULT_MOCK_DATA_DIR = (
    REPO_ROOT
    / "experiments"
    / "mock_data"
    / "gaode_supply_shanghai_v2_20260527_full"
)
DEFAULT_REPORT_OUT = (
    REPO_ROOT
    / "experiments"
    / "artifacts"
    / "b_rag_itinerary_eval_report.json"
)

PIPELINE = (
    ("b_poi_rag", b_poi_rag_node),
    ("candidate_generator", candidate_generator_node),
    ("constraint_filter", constraint_filter_node),
    ("plan_optimizer", plan_optimizer_node),
    ("b_replan_loop", b_replan_loop_node),
    ("explainability", explainability_node),
)


@contextmanager
def _env_context(overrides: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in overrides}
    for key, value in overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"case line {line_no} must be a JSON object")
            cases.append(item)
    return cases


def _run_pipeline(case: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = {
        "user_input": case["user_input"],
        "scene_type": case.get("scene_type", "solo"),
        "constraints": deepcopy(case.get("constraints", {}) or {}),
        "user_profile": deepcopy(case.get("user_profile", {}) or {}),
        "scenario_activities": deepcopy(case.get("scenario_activities", []) or []),
        "execution_log": [],
    }
    steps: list[dict[str, Any]] = []
    for node_name, node in PIPELINE:
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


def _roles_from_blueprint(state: dict[str, Any]) -> list[str]:
    blueprint = state.get("b_itinerary_blueprint") or (state.get("constraints") or {}).get("b_itinerary_blueprint") or {}
    return [
        str(item.get("role"))
        for item in blueprint.get("node_intents", []) or []
        if item.get("role")
    ]


def _timeline_poi_nodes(selected_plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in selected_plan.get("timeline", []) or []
        if item.get("poi_id")
    ]


def _missing_roles_from_coverage(state: dict[str, Any]) -> set[str]:
    coverage = state.get("b_rag_candidate_coverage") or {}
    blueprint = state.get("b_itinerary_blueprint") or {}
    node_by_id = {
        str(item.get("node_id")): item
        for item in blueprint.get("node_intents", []) or []
        if item.get("node_id")
    }
    missing = set(coverage.get("unsupported_missing_roles") or [])
    for node_id in coverage.get("missing_node_ids") or []:
        intent = node_by_id.get(str(node_id), {})
        if intent.get("role"):
            missing.add(str(intent.get("role")))
    return missing


def _validate_case(case: dict[str, Any], state: dict[str, Any]) -> list[str]:
    expected = case.get("expected", {}) or {}
    errors: list[str] = []
    selected_plan = state.get("selected_plan") or {}
    blueprint = state.get("b_itinerary_blueprint") or (state.get("constraints") or {}).get("b_itinerary_blueprint") or {}
    coverage = state.get("b_rag_candidate_coverage") or {}
    roles = _roles_from_blueprint(state)

    expected_template = expected.get("template_mode")
    if expected_template and blueprint.get("template_mode") != expected_template:
        errors.append(
            f"template_mode expected {expected_template}, got {blueprint.get('template_mode')}"
        )

    expected_horizon = expected.get("planning_horizon")
    if expected_horizon and blueprint.get("planning_horizon") != expected_horizon:
        errors.append(
            f"planning_horizon expected {expected_horizon}, got {blueprint.get('planning_horizon')}"
        )

    for role in expected.get("must_include_roles", []) or []:
        if role not in roles:
            errors.append(f"missing blueprint role: {role}")

    min_covered = expected.get("min_rag_covered_nodes")
    if min_covered is not None and int(coverage.get("covered_node_count") or 0) < int(min_covered):
        errors.append(
            f"RAG covered_node_count expected >= {min_covered}, got {coverage.get('covered_node_count')}"
        )

    known_missing = set(expected.get("known_missing_roles", []) or [])
    actual_missing = _missing_roles_from_coverage(state)
    for role in known_missing:
        if role not in actual_missing:
            errors.append(f"expected RAG/data gap for role {role}, actual missing roles={sorted(actual_missing)}")

    allowed_modes = set(expected.get("allowed_planner_modes", []) or [])
    planner_mode = selected_plan.get("planner_mode")
    if allowed_modes and planner_mode not in allowed_modes:
        errors.append(f"planner_mode expected one of {sorted(allowed_modes)}, got {planner_mode}")

    allowed_status = set(expected.get("allowed_plan_status", []) or [])
    plan_status = selected_plan.get("plan_status")
    if allowed_status and plan_status not in allowed_status:
        errors.append(f"plan_status expected one of {sorted(allowed_status)}, got {plan_status}")

    expected_scope = expected.get("expected_execution_scope")
    if expected_scope and selected_plan.get("execution_scope") != expected_scope:
        errors.append(
            f"execution_scope expected {expected_scope}, got {selected_plan.get('execution_scope')}"
        )

    if "expected_execution_ready" in expected:
        expected_ready = bool(expected.get("expected_execution_ready"))
        if bool(selected_plan.get("execution_ready")) != expected_ready:
            errors.append(
                f"execution_ready expected {expected_ready}, got {selected_plan.get('execution_ready')}"
            )

    min_timeline_nodes = expected.get("min_timeline_poi_nodes")
    if min_timeline_nodes is not None:
        actual = len(_timeline_poi_nodes(selected_plan))
        if actual < int(min_timeline_nodes):
            errors.append(f"timeline POI node count expected >= {min_timeline_nodes}, got {actual}")

    return errors


def _summarize_case(case: dict[str, Any], state: dict[str, Any], steps: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    blueprint = state.get("b_itinerary_blueprint") or (state.get("constraints") or {}).get("b_itinerary_blueprint") or {}
    coverage = state.get("b_rag_candidate_coverage") or {}
    selected_plan = state.get("selected_plan") or {}
    evidence = state.get("b_rag_candidate_evidence") or {}
    node_evidence = evidence.get("node_evidence", []) if isinstance(evidence, dict) else []
    return {
        "case_id": case.get("case_id"),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "user_input": case.get("user_input"),
        "expected_notes": (case.get("expected") or {}).get("notes"),
        "blueprint": {
            "template_mode": blueprint.get("template_mode"),
            "planning_horizon": blueprint.get("planning_horizon"),
            "planning_days": blueprint.get("planning_days"),
            "roles": _roles_from_blueprint(state),
            "unsupported_roles": blueprint.get("unsupported_roles", []),
            "requires_rag": blueprint.get("requires_rag"),
        },
        "rag": {
            "covered_node_count": coverage.get("covered_node_count"),
            "required_node_count": coverage.get("required_node_count"),
            "covered_node_ids": coverage.get("covered_node_ids"),
            "missing_node_ids": coverage.get("missing_node_ids"),
            "missing_roles": sorted(_missing_roles_from_coverage(state)),
            "all_nodes_covered": coverage.get("all_nodes_covered"),
            "retrieval_trace": evidence.get("retrieval_trace", []),
            "node_retrieval": [
                {
                    "node_id": block.get("node_id"),
                    "role": block.get("role"),
                    "coverage_status": block.get("coverage_status"),
                    "candidate_count": len(block.get("candidates", []) or []),
                    "retriever": (block.get("retrieval_meta") or {}).get("retriever"),
                    "candidate_pool_size": (block.get("retrieval_meta") or {}).get("candidate_pool_size"),
                    "fallback_full_scan": (block.get("retrieval_meta") or {}).get("fallback_full_scan"),
                    "matched_terms": (block.get("retrieval_meta") or {}).get("matched_terms", [])[:12],
                }
                for block in node_evidence
                if isinstance(block, dict)
            ],
        },
        "candidate_counts": {
            "candidates": len(state.get("candidates", []) or []),
            "filtered_candidates": len(state.get("filtered_candidates", []) or []),
        },
        "selected_plan": {
            "plan_id": selected_plan.get("plan_id"),
            "plan_status": selected_plan.get("plan_status"),
            "planner_mode": selected_plan.get("planner_mode"),
            "plan_shape": selected_plan.get("plan_shape"),
            "execution_ready": selected_plan.get("execution_ready"),
            "execution_scope": selected_plan.get("execution_scope"),
            "benchmark_ready": selected_plan.get("benchmark_ready"),
            "timeline": selected_plan.get("timeline", []),
        },
        "steps": steps,
        "execution_log_tail": (state.get("execution_log") or [])[-8:],
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    cases = _load_cases(Path(args.cases))
    mock_data_dir = Path(args.mock_data_dir).resolve()
    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    overrides = {
        "WF_B_RAG_ENABLED": "1" if args.enable_rag else "0",
        "WF_B_RAG_DATA_DIR": str(mock_data_dir),
        "WF_MOCK_DATA_DIR": str(mock_data_dir),
        "WF_B_RAG_TOP_K": str(args.top_k),
    }
    if args.disable_longcat:
        overrides["WF_LONGCAT_DISABLE_DOTENV"] = "1"

    results: list[dict[str, Any]] = []
    start = time.time()
    with _env_context(overrides):
        for case in cases:
            state, steps = _run_pipeline(case)
            errors = _validate_case(case, state)
            results.append(_summarize_case(case, state, steps, errors))

    passed = sum(1 for item in results if item["status"] == "PASS")
    report = {
        "runner": "run_b_rag_itinerary_eval",
        "cases_path": str(Path(args.cases)),
        "mock_data_dir": str(mock_data_dir),
        "enable_rag": args.enable_rag,
        "top_k": args.top_k,
        "summary": {
            "case_count": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results), 4) if results else 0.0,
            "duration_sec": round(time.time() - start, 3),
        },
        "results": results,
    }
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B+RAG itinerary diagnostics")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--mock-data-dir", default=str(DEFAULT_MOCK_DATA_DIR))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_OUT))
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--enable-rag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-longcat", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    report = run_eval(args)
    summary = report["summary"]
    print("=" * 80)
    print("WeekendFlow B+RAG Itinerary Eval")
    print("=" * 80)
    print(f"cases={summary['case_count']} passed={summary['passed']} failed={summary['failed']} pass_rate={summary['pass_rate']:.1%}")
    for item in report["results"]:
        roles = ", ".join(item["blueprint"]["roles"])
        missing = ", ".join(item["rag"]["missing_roles"])
        print(f"[{item['status']}] {item['case_id']} roles=[{roles}] missing=[{missing}]")
        for error in item["errors"]:
            print(f"  - {error}")
    print(f"Report written to: {Path(args.report_out)}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
