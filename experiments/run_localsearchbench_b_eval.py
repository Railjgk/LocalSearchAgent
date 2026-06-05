#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate WeekendFlow B framework coverage on LocalSearchBench.

This runner is intentionally stricter than the lightweight probe. It converts
LocalSearchBench rows into product-facing B metrics:

- Did B infer the right itinerary shape?
- Which supply domains block an executable plan?
- Did selected POIs overlap with benchmark answers?
- Is the plan executable enough to hand off to C?

It does not submit to the official benchmark. It is a local engineering harness
for making B/RAG improvements measurable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_localsearchbench_b_probe import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DEFAULT_MOCK_DATA_DIR,
    run_probe,
)


DEFAULT_REPORT_JSON = (
    REPO_ROOT
    / "experiments"
    / "artifacts"
    / "localsearchbench"
    / "weekendflow_b_eval.json"
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _norm_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、,.!?;；:：\"'“”‘’（）()\[\]{}<>《》\-_/\\|]+", "", text)
    return text


def _loose_name_match(expected: str, actual: str) -> bool:
    expected_norm = _norm_text(expected)
    actual_norm = _norm_text(actual)
    if not expected_norm or not actual_norm:
        return False
    if expected_norm in actual_norm or actual_norm in expected_norm:
        return True
    # Most LocalSearchBench answers are merchant names. A short shared fragment
    # is useful for diagnostics but not treated as official correctness.
    min_len = min(len(expected_norm), len(actual_norm))
    if min_len < 4:
        return False
    shared = 0
    for size in range(min(8, min_len), 3, -1):
        for index in range(0, len(expected_norm) - size + 1):
            if expected_norm[index : index + size] in actual_norm:
                shared = max(shared, size)
                break
        if shared:
            break
    return shared >= 4


def _timeline_pois(item: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        node
        for node in _as_list(item.get("timeline"))
        if isinstance(node, dict) and node.get("poi_id")
    ]


def _answer_match_metrics(item: dict[str, Any]) -> dict[str, Any]:
    answers = [str(value) for value in _as_list(item.get("benchmark_answer_sequence")) if str(value).strip()]
    actual_names = [str(node.get("activity") or "") for node in _timeline_pois(item)]
    matched_answers: list[str] = []
    unmatched_answers: list[str] = []
    for answer in answers:
        if any(_loose_name_match(answer, actual) for actual in actual_names):
            matched_answers.append(answer)
        else:
            unmatched_answers.append(answer)
    return {
        "answer_count": len(answers),
        "matched_answer_count": len(matched_answers),
        "answer_recall": round(len(matched_answers) / len(answers), 3) if answers else None,
        "matched_answers": matched_answers,
        "unmatched_answers": unmatched_answers,
        "actual_poi_names": actual_names,
    }


def _role_metrics(item: dict[str, Any]) -> dict[str, Any]:
    blueprint = item.get("b_itinerary_blueprint") or {}
    coverage = item.get("b_rag_candidate_coverage") or {}
    node_intents = [
        intent
        for intent in _as_list(blueprint.get("node_intents"))
        if isinstance(intent, dict)
    ]
    roles = [str(intent.get("role")) for intent in node_intents if intent.get("role")]
    blueprint_unsupported_roles = [
        str(role)
        for role in _as_list(blueprint.get("unsupported_roles"))
        if str(role)
    ]
    if coverage.get("all_nodes_covered") or coverage.get("unsupported_roles_covered"):
        unsupported_roles: list[str] = []
    else:
        rag_missing_roles = {
            str(role)
            for role in _as_list(coverage.get("unsupported_missing_roles"))
            if str(role)
        }
        unsupported_roles = [
            role
            for role in blueprint_unsupported_roles
            if not rag_missing_roles or role in rag_missing_roles
        ]
    actual_poi_count = len(_timeline_pois(item))
    expected_count = len(roles)
    return {
        "expected_roles": roles,
        "expected_role_count": expected_count,
        "actual_poi_node_count": actual_poi_count,
        "role_coverage": round(min(actual_poi_count, expected_count) / expected_count, 3) if expected_count else 0.0,
        "unsupported_roles": unsupported_roles,
        "blueprint_unsupported_roles": blueprint_unsupported_roles,
        "unsupported_role_count": len(unsupported_roles),
        "supported_role_count": max(0, expected_count - len(unsupported_roles)),
        "requires_rag": bool(blueprint.get("requires_rag")),
    }


def _failure_categories(item: dict[str, Any], role_metrics: dict[str, Any], answer_metrics: dict[str, Any]) -> list[str]:
    if item.get("error_type"):
        return ["pipeline_error"]

    categories: list[str] = []
    unsupported_roles = role_metrics["unsupported_roles"]
    for role in unsupported_roles:
        categories.append(f"rag_supply_gap:{role}")

    blueprint = item.get("b_itinerary_blueprint") or {}
    rag_metadata = item.get("b_poi_rag_metadata") or {}
    if blueprint.get("named_entities") and not rag_metadata.get("named_event_location_anchors"):
        categories.append("exact_entity_retrieval_gap")

    if role_metrics["actual_poi_node_count"] < role_metrics["expected_role_count"]:
        categories.append("itinerary_shape_gap")

    if item.get("selected_plan_status") == "needs_rag_candidate_evidence":
        categories.append("candidate_evidence_gap")
    if item.get("selected_plan_status") == "time_adjustment_required":
        categories.append("time_adjustment_required")

    if item.get("selected_plan_execution_ready") is not True:
        categories.append("execution_not_ready")

    answer_recall = answer_metrics.get("answer_recall")
    if answer_recall is not None and answer_recall < 1.0:
        categories.append("answer_mismatch")

    return sorted(set(categories)) or ["pass_or_product_ready"]


def _weekendflow_product_metrics(
    item: dict[str, Any],
    role_metrics: dict[str, Any],
    answer_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Metrics closer to WeekendFlow than exact benchmark answer matching.

    LocalSearchBench exact answers are useful for retrieval alignment, but
    WeekendFlow also needs truthful partial plans, low-latency planning, and
    C-handoffable core actions when long-tail domains are not executable yet.
    """

    status = str(item.get("selected_plan_status") or "")
    action_count = int(item.get("action_hints_count") or 0)
    role_coverage = float(role_metrics.get("role_coverage") or 0.0)
    answer_recall = float(answer_metrics.get("answer_recall") or 0.0)
    execution_ready = item.get("selected_plan_execution_ready") is True
    time_adjustment_ready = (
        status == "time_adjustment_required"
        and role_coverage >= 1.0
        and item.get("selected_plan_execution_scope") == "partial"
    )
    core_handoff_ready = (
        action_count > 0
        and status in {"partial_executable", "executable_or_empty"}
        and role_metrics.get("actual_poi_node_count", 0) > 0
    )
    truthful_partial_ready = (
        status == "partial_executable"
        and role_coverage >= 1.0
        and item.get("selected_plan_execution_scope") == "partial"
    )
    latency_sec = float(item.get("duration_sec") or 0.0)
    latency_ok = latency_sec <= 15.0
    score = (
        0.25 * role_coverage
        + 0.20 * float(core_handoff_ready)
        + 0.20 * float(execution_ready)
        + 0.15 * answer_recall
        + 0.10 * float(latency_ok)
        + 0.05 * float(truthful_partial_ready)
        + 0.05 * float(time_adjustment_ready)
    )
    return {
        "core_handoff_ready": core_handoff_ready,
        "truthful_partial_ready": truthful_partial_ready,
        "time_adjustment_ready": time_adjustment_ready,
        "fully_executable_ready": execution_ready,
        "latency_ok": latency_ok,
        "latency_sec": latency_sec,
        "action_hints_count": action_count,
        "partial_missing_roles": item.get("selected_plan_partial_missing_roles") or [],
        "non_executable_node_count": item.get("selected_plan_non_executable_node_count") or 0,
        "weekendflow_product_score": round(score, 3),
    }


def _score_item(item: dict[str, Any]) -> dict[str, Any]:
    role = _role_metrics(item)
    answer = _answer_match_metrics(item)
    categories = _failure_categories(item, role, answer)
    wf_metrics = _weekendflow_product_metrics(item, role, answer)
    execution_ready = item.get("selected_plan_execution_ready") is True
    planning_ready = (
        role["expected_role_count"] > 0
        and role["role_coverage"] >= 1.0
        and not role["unsupported_roles"]
    )
    product_ready = (
        execution_ready
        and planning_ready
    )
    answer_recall = answer.get("answer_recall")
    benchmark_answer_ready = answer_recall == 1.0 if answer_recall is not None else False
    return {
        "dataset_index": item.get("dataset_index"),
        "city": item.get("city"),
        "difficulty": item.get("difficulty"),
        "hop_count": item.get("hop_count"),
        "question": item.get("question"),
        "planner_mode": item.get("selected_plan_planner_mode"),
        "plan_status": item.get("selected_plan_status") or "executable_or_empty",
        "execution_ready": execution_ready,
        "planning_ready": planning_ready,
        "product_ready": product_ready,
        "benchmark_answer_ready": benchmark_answer_ready,
        "role_metrics": role,
        "answer_metrics": answer,
        "weekendflow_product_metrics": wf_metrics,
        "non_executable_nodes": item.get("selected_plan_non_executable_nodes") or [],
        "failure_categories": categories,
        "duration_sec": item.get("duration_sec"),
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _group_summary(scored: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        groups[str(item.get(key) or "unknown")].append(item)

    result = {}
    for group_key, rows in sorted(groups.items()):
        result[group_key] = _summarize_scored(rows, include_breakdowns=False)
    return result


def _c_execution_gap_counter(scored: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in scored:
        for node in _as_list(item.get("non_executable_nodes")):
            if not isinstance(node, dict):
                continue
            if str(node.get("blocker") or "").strip() == "营业时间不满足深夜/夜宵需求":
                continue
            role = str(node.get("role") or "").strip()
            domain = str(node.get("supply_domain") or "").strip()
            counter[role or domain or "unknown"] += 1
    return counter


def _summarize_scored(scored: list[dict[str, Any]], *, include_breakdowns: bool = True) -> dict[str, Any]:
    total = len(scored)
    role_coverages = [float(item["role_metrics"]["role_coverage"]) for item in scored]
    wf_scores = [
        float(item["weekendflow_product_metrics"]["weekendflow_product_score"])
        for item in scored
    ]
    answer_recalls = [
        float(item["answer_metrics"]["answer_recall"])
        for item in scored
        if item["answer_metrics"]["answer_recall"] is not None
    ]
    category_counter: Counter[str] = Counter()
    unsupported_counter: Counter[str] = Counter()
    c_execution_counter = _c_execution_gap_counter(scored)
    for item in scored:
        category_counter.update(item["failure_categories"])
        unsupported_counter.update(item["role_metrics"]["unsupported_roles"])

    summary = {
        "total": total,
        "planning_ready_count": sum(1 for item in scored if item["planning_ready"]),
        "planning_ready_rate": round(sum(1 for item in scored if item["planning_ready"]) / total, 3) if total else 0.0,
        "product_ready_count": sum(1 for item in scored if item["product_ready"]),
        "product_ready_rate": round(sum(1 for item in scored if item["product_ready"]) / total, 3) if total else 0.0,
        "benchmark_answer_ready_count": sum(1 for item in scored if item["benchmark_answer_ready"]),
        "benchmark_answer_ready_rate": round(sum(1 for item in scored if item["benchmark_answer_ready"]) / total, 3) if total else 0.0,
        "execution_ready_count": sum(1 for item in scored if item["execution_ready"]),
        "execution_ready_rate": round(sum(1 for item in scored if item["execution_ready"]) / total, 3) if total else 0.0,
        "wf_core_handoff_ready_count": sum(
            1 for item in scored if item["weekendflow_product_metrics"]["core_handoff_ready"]
        ),
        "wf_core_handoff_ready_rate": round(
            sum(1 for item in scored if item["weekendflow_product_metrics"]["core_handoff_ready"]) / total,
            3,
        ) if total else 0.0,
        "wf_truthful_partial_ready_count": sum(
            1 for item in scored if item["weekendflow_product_metrics"]["truthful_partial_ready"]
        ),
        "wf_truthful_partial_ready_rate": round(
            sum(1 for item in scored if item["weekendflow_product_metrics"]["truthful_partial_ready"]) / total,
            3,
        ) if total else 0.0,
        "wf_time_adjustment_ready_count": sum(
            1 for item in scored if item["weekendflow_product_metrics"].get("time_adjustment_ready")
        ),
        "wf_time_adjustment_ready_rate": round(
            sum(1 for item in scored if item["weekendflow_product_metrics"].get("time_adjustment_ready")) / total,
            3,
        ) if total else 0.0,
        "wf_avg_product_score": _mean(wf_scores),
        "avg_role_coverage": _mean(role_coverages),
        "avg_answer_recall": _mean(answer_recalls),
        "needs_rag_count": sum(1 for item in scored if item["role_metrics"]["requires_rag"]),
        "needs_rag_rate": round(sum(1 for item in scored if item["role_metrics"]["requires_rag"]) / total, 3) if total else 0.0,
    }
    if include_breakdowns:
        summary["failure_category_counts"] = dict(category_counter.most_common())
        summary["unsupported_role_counts"] = dict(unsupported_counter.most_common())
        summary["c_execution_gap_counts"] = dict(c_execution_counter.most_common())
        summary["by_city"] = _group_summary(scored, "city")
        summary["by_difficulty"] = _group_summary(scored, "difficulty")
        summary["action_backlog"] = _build_action_backlog(
            unsupported_counter,
            category_counter,
            c_execution_counter,
        )
    return summary


def _build_action_backlog(
    unsupported_counter: Counter[str],
    category_counter: Counter[str],
    c_execution_counter: Counter[str],
) -> list[dict[str, Any]]:
    role_owner = {
        "lodging": "RAG/POI supply",
        "convenience_store": "RAG/POI supply",
        "souvenir_shopping": "RAG/POI supply",
        "parking": "RAG/POI supply + route/parking verifier",
        "beauty_cosmetics": "RAG/POI supply",
        "flower_shop": "RAG/POI supply",
        "wellness_massage": "RAG/POI supply",
    }
    backlog: list[dict[str, Any]] = []
    for role, count in unsupported_counter.most_common():
        backlog.append(
            {
                "priority": len(backlog) + 1,
                "type": "supply_domain_gap",
                "role": role,
                "count": count,
                "owner_hint": role_owner.get(role, "B/RAG jointly define role contract"),
                "next_step": "Add RAG candidate retrieval and evidence schema for this node role.",
            }
        )
    for role, count in c_execution_counter.most_common():
        backlog.append(
            {
                "priority": len(backlog) + 1,
                "type": "c_execution_contract_gap",
                "role": role,
                "count": count,
                "owner_hint": "C mock/API execution layer",
                "next_step": "Define action_hints and mock execution state for this B itinerary role.",
            }
        )
    if category_counter.get("exact_entity_retrieval_gap"):
        backlog.append(
            {
                "priority": len(backlog) + 1,
                "type": "exact_entity_retrieval",
                "count": category_counter["exact_entity_retrieval_gap"],
                "owner_hint": "RAG + B verifier",
                "next_step": "Resolve named events/venues to coordinates before B itinerary optimization.",
            }
        )
    if category_counter.get("answer_mismatch"):
        backlog.append(
            {
                "priority": len(backlog) + 1,
                "type": "benchmark_answer_alignment",
                "count": category_counter["answer_mismatch"],
                "owner_hint": "B eval harness",
                "next_step": "After RAG is wired, compare selected POIs against benchmark answer sequence.",
            }
        )
    return backlog


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    probe_report = run_probe(args)
    scored = [_score_item(item) for item in probe_report["results"]]
    return {
        "source": probe_report["source"],
        "dataset_path": probe_report["dataset_path"],
        "mock_data_dir": probe_report["mock_data_dir"],
        "enable_rag": probe_report.get("enable_rag"),
        "deterministic_parser": probe_report.get("deterministic_parser"),
        "city": args.city,
        "difficulty": args.difficulty,
        "offset": args.offset,
        "limit": args.limit,
        "summary": _summarize_scored(scored),
        "cases": scored,
    }


def write_reports(report: dict[str, Any], json_path: Path) -> tuple[Path, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    summary = report["summary"]
    lines = [
        "# WeekendFlow B LocalSearchBench Eval",
        "",
        "## Summary",
        "",
        f"- total: {summary['total']}",
        f"- planning_ready_rate: {summary['planning_ready_rate']}",
        f"- product_ready_rate: {summary['product_ready_rate']}",
        f"- benchmark_answer_ready_rate: {summary['benchmark_answer_ready_rate']}",
        f"- execution_ready_rate: {summary['execution_ready_rate']}",
        f"- wf_core_handoff_ready_rate: {summary['wf_core_handoff_ready_rate']}",
        f"- wf_truthful_partial_ready_rate: {summary['wf_truthful_partial_ready_rate']}",
        f"- wf_time_adjustment_ready_rate: {summary['wf_time_adjustment_ready_rate']}",
        f"- wf_avg_product_score: {summary['wf_avg_product_score']}",
        f"- avg_role_coverage: {summary['avg_role_coverage']}",
        f"- avg_answer_recall: {summary['avg_answer_recall']}",
        f"- needs_rag_rate: {summary['needs_rag_rate']}",
        "",
        "## Failure Categories",
        "",
    ]
    for key, count in summary.get("failure_category_counts", {}).items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Unsupported Role Backlog", ""])
    if summary.get("c_execution_gap_counts"):
        lines.extend(["", "## C Execution Contract Gaps", ""])
        for key, count in summary.get("c_execution_gap_counts", {}).items():
            lines.append(f"- {key}: {count}")
        lines.append("")
    for item in summary.get("action_backlog", []):
        lines.append(
            f"- P{item['priority']} {item['type']} / {item.get('role', '-')}: "
            f"{item['count']} cases -> {item['next_step']}"
        )
    lines.extend(["", "## Cases", ""])
    for item in report["cases"]:
        role = item["role_metrics"]
        answer = item["answer_metrics"]
        wf = item["weekendflow_product_metrics"]
        lines.extend(
            [
                f"### #{item.get('dataset_index')} {item.get('city')} {item.get('difficulty')} / {item.get('hop_count')} hops",
                "",
                f"- Question: {item.get('question')}",
                f"- Roles: {' -> '.join(role['expected_roles'])}",
                f"- Unsupported roles: {role['unsupported_roles']}",
                f"- Role coverage: {role['role_coverage']}",
                f"- Planning ready: {item['planning_ready']}",
                f"- Execution ready: {item['execution_ready']}",
                f"- Product ready: {item['product_ready']}",
                f"- WeekendFlow product score: {wf['weekendflow_product_score']}",
                f"- Core handoff ready: {wf['core_handoff_ready']}",
                f"- Truthful partial ready: {wf['truthful_partial_ready']}",
                f"- Partial missing roles: {' -> '.join(wf['partial_missing_roles'])}",
                f"- Non-executable nodes: {' -> '.join(str(node.get('role') or node.get('name') or node.get('supply_domain')) for node in item.get('non_executable_nodes', []))}",
                f"- Benchmark answer ready: {item['benchmark_answer_ready']}",
                f"- Answer recall: {answer['answer_recall']}",
                f"- Actual POIs: {' -> '.join(answer['actual_poi_names'])}",
                f"- Unmatched benchmark answers: {' -> '.join(answer['unmatched_answers'])}",
                f"- Failure categories: {', '.join(item['failure_categories'])}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WeekendFlow B on LocalSearchBench.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--mock-data-dir", default=str(DEFAULT_MOCK_DATA_DIR))
    parser.add_argument("--city", default="上海")
    parser.add_argument("--difficulty", default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--enable-rag", action="store_true")
    parser.add_argument(
        "--deterministic-parser",
        action="store_true",
        help="Disable A/B LongCat calls so benchmark regressions isolate B deterministic planning.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_eval(args)
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
