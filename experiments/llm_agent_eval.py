#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM-driven case generation and evaluation harness for WeekendFlow.

The harness is intentionally outside `src/`: it treats the current agent as a
black box, generates user-facing requests from partial profiles, runs the
existing graph through `run.py`, and asks an LLM judge to compare rough expected
outcomes with actual results. Local fallback generation/scoring keeps the
pipeline testable when no API key is configured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nodes.longcat_client import (  # noqa: E402
    DEFAULT_LONGCAT_BASE_URL,
    DEFAULT_LONGCAT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    LongCatConfig,
    chat_completion,
    sanitize_longcat_error,
)


DEFAULT_ARTIFACT_DIR = REPO_ROOT / "experiments" / "artifacts" / "llm_agent_eval"
DEFAULT_CASES_PATH = DEFAULT_ARTIFACT_DIR / "cases.jsonl"
DEFAULT_RUNS_PATH = DEFAULT_ARTIFACT_DIR / "runs.jsonl"
DEFAULT_REPORT_PATH = DEFAULT_ARTIFACT_DIR / "report.json"
HORIZONS = ("short", "one_day", "two_day")


CASE_GENERATION_SYSTEM_PROMPT = """\
你是本地生活 Agent 的自动评测数据生成器。

目标：根据“用户部分画像”生成短时、一天、两天三类本地生活需求，并给出粗粒度期望结果。期望结果不能写死具体 POI、商户名、地址或榜单答案，只描述活动/餐饮/交通/时间/预算/风险等语义级目标。

必须输出严格 JSON 对象，不要 Markdown，不要解释。顶层格式：
{
  "cases": [
    {
      "horizon": "short|one_day|two_day",
      "user_request": "自然语言用户需求，必须包含画像中对本次任务可见的关键信息",
      "expected": {
        "intent_summary": "一句话概括真实需求",
        "must_satisfy": ["硬约束，越可判定越好"],
        "should_satisfy": ["软偏好"],
        "avoid": ["应避免的体验或风险"],
        "expected_activity_roles": ["语义角色，如 parent_child_activity, light_meal, rest_stop"],
        "time_budget": {"start_hint": "...", "duration_hours": 0, "schedule_flexibility": "low|medium|high"},
        "money_budget": {"amount": 0, "strictness": "low|medium|high"},
        "result_shape": {"min_nodes": 1, "max_nodes": 5, "needs_reservation_or_purchase": true},
        "success_criteria": [
          {"name": "可量化指标名", "weight": 0.2, "rubric": "如何判断好坏"}
        ]
      },
      "difficulty_tags": ["reasoning", "constraint", "personalization"]
    }
  ]
}

生成要求：
- 每个 horizon 至少 1 条 case，除非用户要求更少。
- 需求要像真实用户说话，可不完整、有偏好冲突或隐含约束，但不能离开画像。
- `must_satisfy` 不少于 3 条，`success_criteria` 权重总和约等于 1。
- 不要要求具体 POI；不要让期望答案依赖固定商户。
"""


RUN_EVALUATION_SYSTEM_PROMPT = """\
你是本地生活 Agent 的评测裁判。你会看到一个用户画像生成 case、粗粒度期望结果、以及 Agent 实际输出摘要。请比较实际输出是否满足语义期望，而不是匹配具体 POI。

必须输出严格 JSON 对象，不要 Markdown，不要解释。顶层格式：
{
  "scores": {
    "intent_fit": 0,
    "constraint_satisfaction": 0,
    "itinerary_shape": 0,
    "personalization": 0,
    "feasibility_execution": 0,
    "explanation_quality": 0,
    "overall": 0
  },
  "pass": false,
  "failure_categories": ["intent_miss|constraint_violation|shape_gap|personalization_gap|execution_gap|explanation_gap|unsafe_or_unpleasant|supply_gap"],
  "met_expectations": ["已满足的粗粒度期望"],
  "missed_expectations": ["未满足或证据不足的期望"],
  "evidence": ["引用 actual_summary 中可核验的短证据"],
  "improvement_hints": ["可用于下一轮 agent 改进的具体建议"],
  "confidence": 0.0
}

评分规则：
- 每个维度 0-100，overall 为加权综合分。
- 只惩罚语义错误、约束冲突、行程形状不对、执行不可用、解释缺失等；不要因为没有命中特定 POI 扣分。
- 若 actual_summary 证据不足，要在 missed_expectations 中说明“证据不足”，并适度扣分。
- 输出的 failure_categories 要便于自动聚合定位改进方向。
"""


DEFAULT_PROFILES = [
    {
        "profile_id": "family_health_budget_shanghai",
        "city": "上海",
        "partial_profile": {
            "group": "两位大人和一个 5 岁孩子",
            "preferences": ["孩子需要安全有趣", "伴侣近期控制热量", "不喜欢长时间排队"],
            "constraints": ["预算中等", "尽量少换乘", "下午更有空"],
        },
    },
    {
        "profile_id": "friends_citywalk_rainy",
        "city": "上海",
        "partial_profile": {
            "group": "4 个朋友",
            "preferences": ["喜欢拍照", "想要轻松聊天", "有人不能吃辣"],
            "constraints": ["可能下雨", "预算人均 200 左右", "晚上不想太晚结束"],
        },
    },
]


AgentRunner = Callable[[dict[str, Any]], dict[str, Any]]


def parse_jsonish(text: str) -> dict[str, Any]:
    """Parse a JSON object from plain text or a fenced LLM response."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM response did not contain a JSON object")


def _read_float_env(keys: Iterable[str], default: float) -> float:
    for key in keys:
        raw_value = os.getenv(key)
        if not raw_value:
            continue
        try:
            return float(raw_value)
        except ValueError:
            continue
    return default


def _read_int_env(keys: Iterable[str], default: int) -> int:
    for key in keys:
        raw_value = os.getenv(key)
        if not raw_value:
            continue
        try:
            return int(raw_value)
        except ValueError:
            continue
    return default


def load_eval_llm_config() -> LongCatConfig:
    """Load an OpenAI-compatible LLM config for the eval harness."""

    api_key = (
        os.getenv("LLM_AGENT_EVAL_API_KEY")
        or os.getenv("LONGCAT_API_KEY")
        or os.getenv("LONGCAT_APP_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "Set LLM_AGENT_EVAL_API_KEY, LONGCAT_API_KEY, or LONGCAT_APP_KEY before using --call-llm"
        )

    return LongCatConfig(
        api_key=api_key,
        base_url=(
            os.getenv("LLM_AGENT_EVAL_BASE_URL")
            or os.getenv("LONGCAT_BASE_URL")
            or DEFAULT_LONGCAT_BASE_URL
        ).strip().rstrip("/"),
        model=(
            os.getenv("LLM_AGENT_EVAL_MODEL")
            or os.getenv("LONGCAT_MODEL")
            or DEFAULT_LONGCAT_MODEL
        ).strip(),
        timeout_seconds=_read_float_env(
            ("LLM_AGENT_EVAL_TIMEOUT_SECONDS", "LONGCAT_TIMEOUT_SECONDS"),
            DEFAULT_TIMEOUT_SECONDS,
        ),
        max_tokens=_read_int_env(("LLM_AGENT_EVAL_MAX_TOKENS", "LONGCAT_MAX_TOKENS"), 2600),
        temperature=_read_float_env(
            ("LLM_AGENT_EVAL_TEMPERATURE", "LONGCAT_TEMPERATURE"),
            DEFAULT_TEMPERATURE,
        ),
    )


def call_llm_json(system_prompt: str, payload: dict[str, Any], config: LongCatConfig | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the configured LLM and parse a JSON-object response."""

    config = config or load_eval_llm_config()
    response = chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        config=config,
    )
    return parse_jsonish(response["content"]), {
        "provider": "longcat_openai_compatible",
        "model": response.get("model") or config.model,
        "usage": response.get("usage", {}),
        "finish_reason": response.get("finish_reason"),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(item)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def load_profiles(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return deepcopy(DEFAULT_PROFILES)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("profiles", [])
    if not isinstance(data, list):
        raise ValueError("Profile file must be a JSON list, JSONL file, or {'profiles': [...]}")
    return [item for item in data if isinstance(item, dict)]


def stable_case_id(profile_id: str, horizon: str, user_request: str) -> str:
    digest = hashlib.sha1(f"{profile_id}|{horizon}|{user_request}".encode("utf-8")).hexdigest()[:10]
    return f"llmcase_{profile_id}_{horizon}_{digest}"


def _fallback_expected(horizon: str) -> dict[str, Any]:
    duration = {"short": 4, "one_day": 9, "two_day": 30}.get(horizon, 4)
    max_nodes = {"short": 2, "one_day": 4, "two_day": 6}.get(horizon, 3)
    roles = {
        "short": ["primary_activity", "meal_or_drink"],
        "one_day": ["morning_activity", "lunch", "afternoon_activity", "dinner_or_rest"],
        "two_day": ["day1_activity", "lodging_or_rest", "day2_activity", "meal"],
    }.get(horizon, ["primary_activity"])
    return {
        "intent_summary": f"{horizon} 本地生活安排",
        "must_satisfy": ["符合同行人画像", "控制总时长和预算", "减少排队和绕路"],
        "should_satisfy": ["兼顾体验丰富度", "给出可执行时间顺序", "解释推荐理由"],
        "avoid": ["高风险不可执行安排", "明显违背饮食或同行人限制", "过度密集行程"],
        "expected_activity_roles": roles,
        "time_budget": {
            "start_hint": "按用户自然语言需求推断",
            "duration_hours": duration,
            "schedule_flexibility": "medium",
        },
        "money_budget": {"amount": 600 if horizon == "short" else 1200, "strictness": "medium"},
        "result_shape": {
            "min_nodes": 1,
            "max_nodes": max_nodes,
            "needs_reservation_or_purchase": horizon != "short",
        },
        "success_criteria": [
            {"name": "intent_fit", "weight": 0.25, "rubric": "活动类型和人群匹配"},
            {"name": "constraints", "weight": 0.25, "rubric": "时间、预算、饮食和排队约束可满足"},
            {"name": "shape", "weight": 0.2, "rubric": "节点数量和顺序符合 horizon"},
            {"name": "execution", "weight": 0.2, "rubric": "能进入工具执行或明确说明不可执行原因"},
            {"name": "explanation", "weight": 0.1, "rubric": "解释包含个性化理由"},
        ],
    }


def fallback_generate_cases(profile: dict[str, Any], cases_per_horizon: int) -> list[dict[str, Any]]:
    """Create deterministic smoke-test cases when no generation LLM is used."""

    profile_id = str(profile.get("profile_id") or "profile")
    city = str(profile.get("city") or "上海")
    partial = profile.get("partial_profile") or {}
    group = str(partial.get("group") or "我和朋友")
    preferences = "，".join(str(item) for item in partial.get("preferences", [])[:3])
    constraints = "，".join(str(item) for item in partial.get("constraints", [])[:3])
    request_templates = {
        "short": f"今天下午在{city}{group}想轻松玩 3-4 小时，偏好{preferences}，注意{constraints}",
        "one_day": f"帮我安排明天在{city}{group}的一日行程，偏好{preferences}，注意{constraints}",
        "two_day": f"这个周末想在{city}安排{group}的两天本地生活计划，偏好{preferences}，注意{constraints}",
    }
    cases: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for index in range(cases_per_horizon):
            user_request = request_templates[horizon]
            if cases_per_horizon > 1:
                user_request = f"{user_request}。这是第 {index + 1} 个备选风格，请稍微有区分。"
            cases.append(
                normalize_case(
                    {
                        "profile": profile,
                        "horizon": horizon,
                        "user_request": user_request,
                        "expected": _fallback_expected(horizon),
                        "difficulty_tags": ["fallback", "personalization", horizon],
                        "generation_metadata": {"mode": "local_fallback"},
                    },
                    profile,
                )
            )
    return cases


def normalize_case(raw_case: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    profile_id = str(profile.get("profile_id") or raw_case.get("profile_id") or "profile")
    horizon = str(raw_case.get("horizon") or "short")
    if horizon not in HORIZONS:
        horizon = "short"
    user_request = str(raw_case.get("user_request") or "").strip()
    expected = raw_case.get("expected") if isinstance(raw_case.get("expected"), dict) else {}
    case = {
        "case_id": raw_case.get("case_id") or stable_case_id(profile_id, horizon, user_request),
        "profile_id": profile_id,
        "profile": profile,
        "horizon": horizon,
        "user_request": user_request,
        "expected": expected or _fallback_expected(horizon),
        "difficulty_tags": list(raw_case.get("difficulty_tags") or []),
        "generation_metadata": raw_case.get("generation_metadata") or {},
    }
    if not case["user_request"]:
        raise ValueError(f"Generated case for profile {profile_id} has empty user_request")
    return case


def generate_cases(
    profiles: list[dict[str, Any]],
    *,
    cases_per_horizon: int,
    call_llm: bool,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for profile in profiles:
        if not call_llm:
            cases.extend(fallback_generate_cases(profile, cases_per_horizon))
            continue
        payload = {
            "profile": profile,
            "horizons": list(HORIZONS),
            "cases_per_horizon": cases_per_horizon,
            "non_poi_expected_result": True,
        }
        try:
            generated, metadata = call_llm_json(CASE_GENERATION_SYSTEM_PROMPT, payload)
            raw_cases = generated.get("cases") or []
            if not isinstance(raw_cases, list) or not raw_cases:
                raise ValueError("generation LLM returned no cases")
            for raw_case in raw_cases:
                if not isinstance(raw_case, dict):
                    continue
                item = normalize_case(raw_case, profile)
                item["generation_metadata"] = metadata
                cases.append(item)
        except Exception as exc:  # pragma: no cover - exercised only with live API issues
            fallback_cases = fallback_generate_cases(profile, cases_per_horizon)
            for item in fallback_cases:
                item["generation_metadata"] = {
                    "mode": "local_fallback_after_llm_error",
                    "error": sanitize_longcat_error(exc),
                }
            cases.extend(fallback_cases)
    return cases


def summarize_actual_state(final_state: dict[str, Any], steps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    selected_plan = final_state.get("selected_plan") or {}
    timeline = []
    for item in selected_plan.get("timeline") or []:
        if not isinstance(item, dict):
            continue
        timeline.append(
            {
                "time": item.get("time"),
                "activity": item.get("activity"),
                "poi_id": item.get("poi_id"),
                "duration_min": item.get("duration_min"),
                "notes": item.get("notes") or [],
            }
        )
    nodes = []
    for node in selected_plan.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        nodes.append(
            {
                "type": node.get("type"),
                "name": node.get("name"),
                "category": node.get("category") or node.get("restaurant_category"),
                "tags": node.get("tags") or [],
                "price": node.get("price"),
                "queue_time_min": node.get("queue_time_min"),
                "rating": node.get("rating"),
            }
        )
    return {
        "scene_type": final_state.get("scene_type"),
        "constraints": final_state.get("constraints") or {},
        "scenario_activities": final_state.get("scenario_activities") or [],
        "selected_plan": {
            "plan_id": selected_plan.get("plan_id") or selected_plan.get("id"),
            "status": selected_plan.get("status"),
            "timeline": timeline,
            "nodes": nodes,
            "total_distance_km": selected_plan.get("total_distance_km"),
            "estimated_total_price": selected_plan.get("estimated_total_price")
            or (selected_plan.get("budget") or {}).get("total_price"),
        },
        "optimization_score": final_state.get("optimization_score"),
        "explanation_text": final_state.get("explanation_text"),
        "execution_status": final_state.get("execution_status"),
        "payment_status": final_state.get("payment_status"),
        "action_sequence_count": len(final_state.get("action_sequence") or []),
        "tool_result_keys": sorted((final_state.get("tool_results") or {}).keys()),
        "retry_count": final_state.get("retry_count"),
        "final_share_message": final_state.get("final_share_message"),
        "execution_log_tail": (final_state.get("execution_log") or [])[-8:],
        "step_count": len(steps or []),
    }


def default_agent_runner(case: dict[str, Any]) -> dict[str, Any]:
    """Run the existing agent graph without modifying its implementation."""

    from run import build_initial_state, run_with_trace

    initial_state = build_initial_state(
        case["user_request"],
        user_id=str(case.get("profile_id") or "eval_user"),
        payment_ui_mode="auto",
    )
    final_state, steps = run_with_trace(initial_state)
    return {
        "case_id": case["case_id"],
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "case": case,
        "actual_summary": summarize_actual_state(final_state, steps),
        "final_state": final_state,
    }


def run_cases(cases: list[dict[str, Any]], runner: AgentRunner = default_agent_runner) -> list[dict[str, Any]]:
    return [runner(case) for case in cases]


def _text_blob(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


def heuristic_evaluate_run(run_record: dict[str, Any]) -> dict[str, Any]:
    """Deterministic baseline scorer for CI and offline development."""

    case = run_record["case"]
    expected = case.get("expected") or {}
    actual = run_record.get("actual_summary") or {}
    actual_text = _text_blob(actual)
    musts = [str(item) for item in expected.get("must_satisfy", [])]
    shoulds = [str(item) for item in expected.get("should_satisfy", [])]
    avoid = [str(item) for item in expected.get("avoid", [])]
    timeline = ((actual.get("selected_plan") or {}).get("timeline") or [])
    result_shape = expected.get("result_shape") or {}
    min_nodes = int(result_shape.get("min_nodes") or 1)
    max_nodes = int(result_shape.get("max_nodes") or 6)

    met_musts = [item for item in musts if any(token in actual_text for token in re.split(r"[，,、\s]+", item) if len(token) >= 2)]
    avoided = [item for item in avoid if item and item not in actual_text]
    execution_ok = actual.get("execution_status") in {"completed", "success", "paid", "pending_confirmation"}
    shape_ok = min_nodes <= len(timeline) <= max_nodes
    explanation_ok = bool(str(actual.get("explanation_text") or actual.get("final_share_message") or "").strip())

    intent_fit = 55 + min(35, len(met_musts) * 10)
    constraint_satisfaction = 50 + min(30, len(met_musts) * 8) + min(20, len(avoided) * 6)
    itinerary_shape = 90 if shape_ok else max(20, 70 - abs(len(timeline) - min_nodes) * 15)
    personalization = 45 + min(35, len(met_musts) * 8) + min(20, len(shoulds) * 2)
    feasibility_execution = 85 if execution_ok else (60 if timeline else 25)
    explanation_quality = 80 if explanation_ok else 30
    overall = round(
        intent_fit * 0.2
        + constraint_satisfaction * 0.25
        + itinerary_shape * 0.2
        + personalization * 0.15
        + feasibility_execution * 0.15
        + explanation_quality * 0.05,
        1,
    )

    failure_categories = []
    if not met_musts:
        failure_categories.append("constraint_violation")
    if not shape_ok:
        failure_categories.append("shape_gap")
    if not execution_ok:
        failure_categories.append("execution_gap")
    if not explanation_ok:
        failure_categories.append("explanation_gap")
    if overall < 70 and "constraint_violation" not in failure_categories:
        failure_categories.append("personalization_gap")

    return {
        "case_id": case["case_id"],
        "horizon": case.get("horizon"),
        "profile_id": case.get("profile_id"),
        "scores": {
            "intent_fit": round(intent_fit, 1),
            "constraint_satisfaction": round(constraint_satisfaction, 1),
            "itinerary_shape": round(itinerary_shape, 1),
            "personalization": round(personalization, 1),
            "feasibility_execution": round(feasibility_execution, 1),
            "explanation_quality": round(explanation_quality, 1),
            "overall": overall,
        },
        "pass": overall >= 70,
        "failure_categories": sorted(set(failure_categories)) or ["pass_or_product_ready"],
        "met_expectations": met_musts[:5],
        "missed_expectations": [item for item in musts if item not in met_musts][:5],
        "evidence": ["local heuristic scorer; use --call-llm for semantic judge"],
        "improvement_hints": ["Inspect missed_expectations and actual_summary for the highest-count failure categories."],
        "confidence": 0.45,
        "evaluation_metadata": {"mode": "local_heuristic"},
    }


def normalize_evaluation(raw_eval: dict[str, Any], run_record: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    case = run_record["case"]
    scores = raw_eval.get("scores") if isinstance(raw_eval.get("scores"), dict) else {}
    normalized_scores: dict[str, float] = {}
    for key in (
        "intent_fit",
        "constraint_satisfaction",
        "itinerary_shape",
        "personalization",
        "feasibility_execution",
        "explanation_quality",
        "overall",
    ):
        try:
            normalized_scores[key] = max(0.0, min(100.0, float(scores.get(key, 0))))
        except (TypeError, ValueError):
            normalized_scores[key] = 0.0
    if not normalized_scores["overall"]:
        normalized_scores["overall"] = round(
            statistics.mean(value for key, value in normalized_scores.items() if key != "overall"),
            1,
        )
    return {
        "case_id": case["case_id"],
        "horizon": case.get("horizon"),
        "profile_id": case.get("profile_id"),
        "scores": normalized_scores,
        "pass": bool(raw_eval.get("pass")) if "pass" in raw_eval else normalized_scores["overall"] >= 70,
        "failure_categories": list(raw_eval.get("failure_categories") or []),
        "met_expectations": list(raw_eval.get("met_expectations") or []),
        "missed_expectations": list(raw_eval.get("missed_expectations") or []),
        "evidence": list(raw_eval.get("evidence") or []),
        "improvement_hints": list(raw_eval.get("improvement_hints") or []),
        "confidence": raw_eval.get("confidence", 0.0),
        "evaluation_metadata": metadata,
    }


def evaluate_runs(runs: list[dict[str, Any]], *, call_llm: bool) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for run_record in runs:
        if not call_llm:
            evaluations.append(heuristic_evaluate_run(run_record))
            continue
        payload = {
            "case": run_record["case"],
            "actual_summary": run_record.get("actual_summary") or {},
            "rough_expected_not_specific_poi": True,
        }
        try:
            raw_eval, metadata = call_llm_json(RUN_EVALUATION_SYSTEM_PROMPT, payload)
            evaluations.append(normalize_evaluation(raw_eval, run_record, metadata))
        except Exception as exc:  # pragma: no cover - exercised only with live API issues
            fallback = heuristic_evaluate_run(run_record)
            fallback["evaluation_metadata"] = {
                "mode": "local_heuristic_after_llm_error",
                "error": sanitize_longcat_error(exc),
            }
            evaluations.append(fallback)
    return evaluations


def aggregate_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(evaluations)
    overall_scores = [float(item["scores"].get("overall", 0.0)) for item in evaluations]
    category_counter: Counter[str] = Counter()
    horizon_scores: dict[str, list[float]] = defaultdict(list)
    for item in evaluations:
        category_counter.update(item.get("failure_categories") or [])
        horizon_scores[str(item.get("horizon") or "unknown")].append(float(item["scores"].get("overall", 0.0)))

    return {
        "total": total,
        "pass_count": sum(1 for item in evaluations if item.get("pass")),
        "pass_rate": round(sum(1 for item in evaluations if item.get("pass")) / total, 3) if total else 0.0,
        "avg_overall": round(statistics.mean(overall_scores), 2) if overall_scores else 0.0,
        "min_overall": round(min(overall_scores), 2) if overall_scores else 0.0,
        "max_overall": round(max(overall_scores), 2) if overall_scores else 0.0,
        "by_horizon": {
            key: {
                "count": len(values),
                "avg_overall": round(statistics.mean(values), 2) if values else 0.0,
                "pass_rate": round(
                    sum(
                        1
                        for item in evaluations
                        if str(item.get("horizon") or "unknown") == key and item.get("pass")
                    )
                    / len(values),
                    3,
                )
                if values
                else 0.0,
            }
            for key, values in sorted(horizon_scores.items())
        },
        "failure_category_counts": dict(category_counter.most_common()),
    }


def build_report(cases: list[dict[str, Any]], runs: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(cases),
        "run_count": len(runs),
        "summary": aggregate_evaluations(evaluations),
        "evaluations": evaluations,
    }


def write_report(report: dict[str, Any], path: Path) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = path.with_suffix(".md")
    summary = report["summary"]
    lines = [
        "# LLM Agent Eval Report",
        "",
        "## Summary",
        "",
        f"- total: {summary['total']}",
        f"- pass_rate: {summary['pass_rate']}",
        f"- avg_overall: {summary['avg_overall']}",
        f"- min_overall: {summary['min_overall']}",
        f"- max_overall: {summary['max_overall']}",
        "",
        "## By Horizon",
        "",
    ]
    for horizon, item in summary.get("by_horizon", {}).items():
        lines.append(f"- {horizon}: count={item['count']}, pass_rate={item['pass_rate']}, avg={item['avg_overall']}")
    lines.extend(["", "## Failure Categories", ""])
    for category, count in summary.get("failure_category_counts", {}).items():
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Improvement Hints", ""])
    for item in report["evaluations"]:
        hints = item.get("improvement_hints") or []
        if not hints:
            continue
        lines.append(f"- {item['case_id']} ({item.get('horizon')}): {hints[0]}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate, run, and evaluate WeekendFlow agent cases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate profile-driven eval cases.")
    generate.add_argument("--profiles", type=Path)
    generate.add_argument("--cases-out", type=Path, default=DEFAULT_CASES_PATH)
    generate.add_argument("--cases-per-horizon", type=int, default=1)
    generate.add_argument("--call-llm", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run generated cases through the existing agent.")
    run_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    run_parser.add_argument("--runs-out", type=Path, default=DEFAULT_RUNS_PATH)
    run_parser.add_argument("--limit", type=int, default=0)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate agent runs against rough expectations.")
    evaluate.add_argument("--runs", type=Path, default=DEFAULT_RUNS_PATH)
    evaluate.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    evaluate.add_argument("--call-llm", action="store_true")

    pipeline = subparsers.add_parser("pipeline", help="Generate cases, run agent, and evaluate.")
    pipeline.add_argument("--profiles", type=Path)
    pipeline.add_argument("--cases-out", type=Path, default=DEFAULT_CASES_PATH)
    pipeline.add_argument("--runs-out", type=Path, default=DEFAULT_RUNS_PATH)
    pipeline.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    pipeline.add_argument("--cases-per-horizon", type=int, default=1)
    pipeline.add_argument("--limit", type=int, default=0)
    pipeline.add_argument("--call-llm", action="store_true")
    pipeline.add_argument(
        "--call-llm-eval-only",
        action="store_true",
        help="Use fallback generation but LLM judge evaluation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "generate":
        profiles = load_profiles(args.profiles)
        cases = generate_cases(
            profiles,
            cases_per_horizon=args.cases_per_horizon,
            call_llm=args.call_llm,
        )
        write_jsonl(args.cases_out, cases)
        print(json.dumps({"cases_out": str(args.cases_out), "case_count": len(cases)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run":
        cases = load_jsonl(args.cases)
        if args.limit:
            cases = cases[: args.limit]
        runs = run_cases(cases)
        write_jsonl(args.runs_out, runs)
        print(json.dumps({"runs_out": str(args.runs_out), "run_count": len(runs)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate":
        runs = load_jsonl(args.runs)
        evaluations = evaluate_runs(runs, call_llm=args.call_llm)
        cases = [item["case"] for item in runs]
        report = build_report(cases, runs, evaluations)
        json_path, md_path = write_report(report, args.report_out)
        print(json.dumps({"report": str(json_path), "markdown": str(md_path), "summary": report["summary"]}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "pipeline":
        profiles = load_profiles(args.profiles)
        cases = generate_cases(
            profiles,
            cases_per_horizon=args.cases_per_horizon,
            call_llm=args.call_llm and not args.call_llm_eval_only,
        )
        if args.limit:
            cases = cases[: args.limit]
        write_jsonl(args.cases_out, cases)
        runs = run_cases(cases)
        write_jsonl(args.runs_out, runs)
        evaluations = evaluate_runs(runs, call_llm=args.call_llm or args.call_llm_eval_only)
        report = build_report(cases, runs, evaluations)
        json_path, md_path = write_report(report, args.report_out)
        print(json.dumps({"cases": str(args.cases_out), "runs": str(args.runs_out), "report": str(json_path), "markdown": str(md_path), "summary": report["summary"]}, ensure_ascii=False, indent=2))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
