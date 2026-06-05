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
import time
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
DEFAULT_HARD_SEED_CASES_PATH = REPO_ROOT / "experiments" / "eval_seed_cases" / "hard_realistic_cases.jsonl"
HORIZONS = ("short", "one_day", "two_day")
EVAL_COMPONENTS = ("intent", "memory", "planner", "execution", "explanation")
COMPONENT_STATUSES = ("pass", "weak", "fail")
ALLOWED_GENERATION_PROVIDERS = {
    "longcat_openai_compatible",
    "codex_seeded_realistic_case",
    "codex_architecture_challenge_case",
}


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
        "poi_reference": [
          {
            "name": "宽泛偏好名，如 healthy_dining",
            "expectation": "用于校对实际 POI 的语义期望，如餐饮应体现健康、低卡、少油或轻食",
            "evidence_terms": ["actual_summary 中可作为证据的宽泛词，不要写死具体 POI 名"]
          }
        ],
        "time_budget": {"start_hint": "...", "duration_hours": 0, "schedule_flexibility": "low|medium|high"},
        "money_budget": {"amount": 0, "strictness": "low|medium|high"},
        "result_shape": {"min_nodes": 1, "max_primary_nodes": 5, "needs_reservation_or_purchase": true},
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
- `poi_reference` 要把用户偏好转成宽泛但可被实际 POI 校对的期望，例如“饮食健康/低卡”“儿童友好/安全”“低排队风险/可预约”“少换乘/路线短”。不要写死具体 POI、商户名、地址或榜单答案。
- `result_shape.max_primary_nodes` 只是行程复杂度参考，只约束主要活动/餐饮/住宿节点；转场、步行、休息、缓冲时间不算 primary node。
"""


RUN_EVALUATION_SYSTEM_PROMPT = """\
你是本地生活 Agent 的评测裁判。你会看到一个用户画像生成 case、粗粒度期望结果、以及 Agent 实际输出摘要。请根据 reference 检查实际输出中的 POI、路线、预算、执行和解释证据是否满足语义期望，而不是匹配固定 POI。

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
  "component_findings": [
    {
      "component": "intent|memory|planner|execution|explanation",
      "status": "pass|weak|fail",
      "finding": "对该组件表现的简短判断",
      "evidence": "引用 actual_summary.component_summaries 中的可核验证据"
    }
  ],
  "met_expectations": ["已满足的粗粒度期望"],
  "missed_expectations": ["未满足或证据不足的期望"],
  "evidence": ["引用 actual_summary 中可核验的短证据"],
  "improvement_hints": ["可用于下一轮 agent 改进的具体建议"],
  "confidence": 0.0
}

评分规则：
- 每个维度 0-100，overall 为加权综合分。
- `poi_reference` 是核心 reference：它通常是“饮食健康”“儿童友好”“低排队风险”“少换乘”等宽泛期望。请用 actual_summary 里的 POI 名称、类别、tags、notes、价格、距离、排队、预约/执行结果和解释文本作为证据判断。
- 不要因为没有命中特定 POI 扣分；只有当实际 POI 证据不支持 reference，或证据不足时才扣分。
- `result_shape.max_primary_nodes` 只是复杂度提示。转场、步行、休息、缓冲节点不应导致 shape_gap；只有主要活动/餐饮/住宿节点明显过少、过多或顺序违背需求时才判 shape_gap。
- `actual_summary.component_summaries` 是辅助诊断证据。整体评分仍以最终用户需求是否满足为准，但 component_findings 要指出 intent、memory、planner、execution 或 explanation 哪一环最可能造成问题。
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


def load_eval_llm_retry_config() -> tuple[int, float]:
    """Load retry settings for eval-only LLM calls."""

    retries = _read_int_env(("LLM_AGENT_EVAL_RETRIES", "LONGCAT_RETRIES"), 0)
    backoff_seconds = _read_float_env(
        ("LLM_AGENT_EVAL_RETRY_BACKOFF_SECONDS", "LONGCAT_RETRY_BACKOFF_SECONDS"),
        2.0,
    )
    return max(0, retries), max(0.0, backoff_seconds)


def load_generation_attempts(default: int = 3) -> int:
    """Load profile/horizon backfill attempts for LLM case generation."""

    return max(1, _read_int_env(("LLM_AGENT_EVAL_GENERATION_ATTEMPTS",), default))


def call_llm_json(system_prompt: str, payload: dict[str, Any], config: LongCatConfig | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the configured LLM and parse a JSON-object response."""

    config = config or load_eval_llm_config()
    retries, backoff_seconds = load_eval_llm_retry_config()
    attempts = retries + 1
    errors: list[str] = []
    response: dict[str, Any] | None = None
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    parsed: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = chat_completion(messages, config=config)
            parsed = parse_jsonish(response["content"])
            break
        except Exception as exc:
            errors.append(sanitize_longcat_error(exc))
            if attempt >= attempts:
                raise
            time.sleep(backoff_seconds * attempt)
    else:  # pragma: no cover - loop always breaks or raises
        raise RuntimeError("LLM call failed without an exception")

    assert response is not None
    assert parsed is not None
    return parsed, {
        "provider": "longcat_openai_compatible",
        "model": response.get("model") or config.model,
        "usage": response.get("usage", {}),
        "finish_reason": response.get("finish_reason"),
        "attempt_count": len(errors) + 1,
        "retry_errors": errors,
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


def _fallback_poi_reference(profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    profile = profile or {}
    partial = profile.get("partial_profile") if isinstance(profile.get("partial_profile"), dict) else {}
    profile_text = _text_blob(partial)
    references = [
        {
            "name": "profile_fit",
            "expectation": "POI 和行程安排应匹配同行人画像与当前任务偏好",
            "evidence_terms": ["亲子", "儿童友好", "family", "kid_friendly", "朋友", "拍照", "聊天"],
        },
        {
            "name": "budget_control",
            "expectation": "方案总价应符合用户预算表达，或在预算不明确时体现价格克制",
            "evidence_terms": ["预算", "价格", "总预算", "estimated_total_price", "cost", "人均"],
        },
        {
            "name": "route_and_queue_control",
            "expectation": "路线、排队和预约风险应被控制，避免明显绕路或长时间等待",
            "evidence_terms": ["排队风险低", "排队", "queue_time", "路线", "距离", "可预约", "少换乘"],
        },
    ]
    if any(term in profile_text for term in ("孩子", "亲子", "儿童", "family")):
        references.append(
            {
                "name": "child_friendly_activity",
                "expectation": "活动 POI 应体现儿童友好、安全、有趣或低强度",
                "evidence_terms": ["儿童友好", "亲子", "孩子", "安全", "低强度", "kid_friendly"],
            }
        )
    if any(term in profile_text for term in ("热量", "健康", "轻食", "低卡", "不吃辣", "少油")):
        references.append(
            {
                "name": "healthy_or_restricted_dining",
                "expectation": "餐饮 POI 应体现健康、低卡、少油、轻食或满足饮食限制",
                "evidence_terms": ["健康", "低卡", "低脂", "少油", "轻食", "不辣", "饮食"],
            }
        )
    if any(term in profile_text for term in ("下雨", "雨")):
        references.append(
            {
                "name": "weather_robustness",
                "expectation": "POI 和路线应适应下雨风险，优先室内、近距离或可调整安排",
                "evidence_terms": ["室内", "下雨", "雨天", "可调整", "近距离"],
            }
        )
    return references


def _fallback_expected(horizon: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    duration = {"short": 4, "one_day": 9, "two_day": 30}.get(horizon, 4)
    max_primary_nodes = {"short": 2, "one_day": 4, "two_day": 6}.get(horizon, 3)
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
        "poi_reference": _fallback_poi_reference(profile),
        "time_budget": {
            "start_hint": "按用户自然语言需求推断",
            "duration_hours": duration,
            "schedule_flexibility": "medium",
        },
        "money_budget": {"amount": 600 if horizon == "short" else 1200, "strictness": "medium"},
        "result_shape": {
            "min_nodes": 1,
            "max_primary_nodes": max_primary_nodes,
            "needs_reservation_or_purchase": horizon != "short",
        },
        "success_criteria": [
            {"name": "intent_fit", "weight": 0.25, "rubric": "活动类型和人群匹配"},
            {"name": "poi_reference_fit", "weight": 0.25, "rubric": "实际 POI 证据支持宽泛 reference"},
            {"name": "constraints", "weight": 0.2, "rubric": "时间、预算、饮食和排队约束可满足"},
            {"name": "shape", "weight": 0.1, "rubric": "主要节点复杂度和顺序符合 horizon"},
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
                        "expected": _fallback_expected(horizon, profile),
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


def is_llm_generated_case(case: dict[str, Any]) -> bool:
    metadata = case.get("generation_metadata") if isinstance(case.get("generation_metadata"), dict) else {}
    return metadata.get("provider") == "longcat_openai_compatible"


def is_allowed_generated_case(case: dict[str, Any]) -> bool:
    metadata = case.get("generation_metadata") if isinstance(case.get("generation_metadata"), dict) else {}
    return metadata.get("provider") in ALLOWED_GENERATION_PROVIDERS


def load_seed_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = load_jsonl(path)
    cases: list[dict[str, Any]] = []
    for row in rows:
        profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
        if not profile:
            profile = {
                "profile_id": row.get("profile_id") or "codex_seed_profile",
                "city": row.get("city") or "上海",
                "partial_profile": row.get("partial_profile") or {},
            }
        item = normalize_case(row, profile)
        metadata = item.get("generation_metadata") if isinstance(item.get("generation_metadata"), dict) else {}
        item["generation_metadata"] = {
            "provider": "codex_seeded_realistic_case",
            "model": "codex",
            "source": str(path),
            **metadata,
        }
        cases.append(item)
    return cases


def load_external_cases(
    path: Path | None,
    *,
    provider: str,
    model: str = "codex",
    source: str | None = None,
) -> list[dict[str, Any]]:
    if path is None:
        return []
    if provider not in ALLOWED_GENERATION_PROVIDERS:
        raise ValueError(f"Unsupported external case provider: {provider}")
    rows = load_jsonl(path)
    cases: list[dict[str, Any]] = []
    for row in rows:
        profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
        if not profile:
            profile = {
                "profile_id": row.get("profile_id") or f"{provider}_profile",
                "city": row.get("city") or "上海",
                "partial_profile": row.get("partial_profile") or {},
            }
        item = normalize_case(row, profile)
        metadata = item.get("generation_metadata") if isinstance(item.get("generation_metadata"), dict) else {}
        item["generation_metadata"] = {
            "provider": provider,
            "model": model,
            "source": source or str(path),
            **metadata,
        }
        cases.append(item)
    return cases


def append_seed_cases(cases: list[dict[str, Any]], seed_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(cases)
    seen_ids = {str(case.get("case_id")) for case in merged if case.get("case_id")}
    for case in seed_cases:
        case_id = str(case.get("case_id") or "")
        if case_id and case_id in seen_ids:
            continue
        merged.append(case)
        if case_id:
            seen_ids.add(case_id)
    return merged


def _case_bucket_key(case: dict[str, Any]) -> tuple[str, str]:
    return str(case.get("profile_id") or "profile"), str(case.get("horizon") or "short")


def _has_enough_cases(
    cases_by_bucket: dict[tuple[str, str], list[dict[str, Any]]],
    profile_id: str,
    cases_per_horizon: int,
) -> bool:
    return all(len(cases_by_bucket.get((profile_id, horizon), [])) >= cases_per_horizon for horizon in HORIZONS)


def _needed_horizons(
    cases_by_bucket: dict[tuple[str, str], list[dict[str, Any]]],
    profile_id: str,
    cases_per_horizon: int,
) -> list[str]:
    return [
        horizon
        for horizon in HORIZONS
        if len(cases_by_bucket.get((profile_id, horizon), [])) < cases_per_horizon
    ]


def _append_unique_llm_cases(
    cases_by_bucket: dict[tuple[str, str], list[dict[str, Any]]],
    raw_cases: list[Any],
    profile: dict[str, Any],
    metadata: dict[str, Any],
    allowed_horizons: set[str],
) -> int:
    accepted = 0
    seen_ids = {
        str(case.get("case_id"))
        for bucket_cases in cases_by_bucket.values()
        for case in bucket_cases
        if case.get("case_id")
    }
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            continue
        item = normalize_case(raw_case, profile)
        if item["horizon"] not in allowed_horizons:
            continue
        if item["case_id"] in seen_ids:
            continue
        item["generation_metadata"] = metadata
        cases_by_bucket[_case_bucket_key(item)].append(item)
        seen_ids.add(item["case_id"])
        accepted += 1
    return accepted


def generate_llm_cases_with_backfill(
    profile: dict[str, Any],
    *,
    cases_per_horizon: int,
    max_generation_attempts: int,
) -> list[dict[str, Any]]:
    profile_id = str(profile.get("profile_id") or "profile")
    cases_by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []

    for attempt in range(1, max(1, max_generation_attempts) + 1):
        needed = _needed_horizons(cases_by_bucket, profile_id, cases_per_horizon)
        if not needed:
            break
        remaining_by_horizon = {
            horizon: cases_per_horizon - len(cases_by_bucket.get((profile_id, horizon), []))
            for horizon in needed
        }
        payload = {
            "profile": profile,
            "horizons": needed,
            "cases_per_horizon": max(remaining_by_horizon.values()),
            "remaining_cases_by_horizon": remaining_by_horizon,
            "non_poi_expected_result": True,
            "generation_attempt": attempt,
            "strict_no_fallback": True,
            "deduplicate_against_case_ids": [
                case.get("case_id")
                for bucket_cases in cases_by_bucket.values()
                for case in bucket_cases
            ],
        }
        try:
            generated, metadata = call_llm_json(CASE_GENERATION_SYSTEM_PROMPT, payload)
            raw_cases = generated.get("cases") or []
            if not isinstance(raw_cases, list) or not raw_cases:
                raise ValueError("generation LLM returned no cases")
            metadata = {
                **metadata,
                "generation_attempt": attempt,
                "requested_horizons": needed,
                "strict_no_fallback": True,
            }
            accepted = _append_unique_llm_cases(
                cases_by_bucket,
                raw_cases,
                profile,
                metadata,
                set(needed),
            )
            if accepted == 0:
                errors.append(f"attempt {attempt}: accepted no usable cases for horizons {needed}")
        except Exception as exc:  # pragma: no cover - exercised only with live API issues
            errors.append(f"attempt {attempt}: {sanitize_longcat_error(exc)}")

    if not _has_enough_cases(cases_by_bucket, profile_id, cases_per_horizon):
        missing = {
            horizon: cases_per_horizon - len(cases_by_bucket.get((profile_id, horizon), []))
            for horizon in HORIZONS
            if len(cases_by_bucket.get((profile_id, horizon), [])) < cases_per_horizon
        }
        raise RuntimeError(
            "LLM case generation did not fill all required buckets for "
            f"{profile_id}; missing={missing}; errors={errors}"
        )

    cases: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        cases.extend(cases_by_bucket[(profile_id, horizon)][:cases_per_horizon])
    return cases


def generate_cases(
    profiles: list[dict[str, Any]],
    *,
    cases_per_horizon: int,
    call_llm: bool,
    strict_llm_generation: bool = False,
    max_generation_attempts: int | None = None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    max_generation_attempts = max_generation_attempts or load_generation_attempts()
    for profile in profiles:
        if not call_llm:
            cases.extend(fallback_generate_cases(profile, cases_per_horizon))
            continue
        if strict_llm_generation:
            cases.extend(
                generate_llm_cases_with_backfill(
                    profile,
                    cases_per_horizon=cases_per_horizon,
                    max_generation_attempts=max_generation_attempts,
                )
            )
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


def _compact_sequence(items: Iterable[Any], limit: int = 8) -> list[Any]:
    compacted = []
    for item in items:
        compacted.append(item)
        if len(compacted) >= limit:
            break
    return compacted


def _compact_value_memory(items: list[Any]) -> list[dict[str, Any]]:
    compacted = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "value_id": item.get("value_id"),
                "label": item.get("label"),
                "score": item.get("score"),
                "confidence": item.get("confidence"),
                "source": item.get("source"),
                "ttl": item.get("ttl"),
                "planning_effect": item.get("planning_effect"),
                "evidence": _compact_sequence(item.get("evidence") or [], 3),
            }
        )
        if len(compacted) >= 8:
            break
    return compacted


def _compact_action_sequence(items: list[Any]) -> list[dict[str, Any]]:
    actions = []
    for item in items:
        if not isinstance(item, dict):
            continue
        actions.append(
            {
                "step": item.get("step"),
                "action_type": item.get("action_type"),
                "name": item.get("name"),
                "poi_id": item.get("poi_id"),
                "merchant_id": item.get("merchant_id"),
                "product_id": item.get("product_id"),
                "deal_id": item.get("deal_id"),
                "time": item.get("time"),
                "requires_reservation": item.get("requires_reservation"),
            }
        )
        if len(actions) >= 10:
            break
    return actions


def _compact_tool_results(tool_results: dict[str, Any]) -> list[dict[str, Any]]:
    compacted = []
    for key, value in sorted(tool_results.items()):
        if not isinstance(value, dict):
            compacted.append({"key": key, "success": None, "status": type(value).__name__})
            continue
        data = value.get("data") if isinstance(value.get("data"), dict) else {}
        compacted.append(
            {
                "key": key,
                "action": value.get("action"),
                "name": value.get("name"),
                "success": value.get("success"),
                "status": data.get("status") or value.get("status"),
                "failure_reason": data.get("failure_reason") or value.get("failure_reason"),
                "time": data.get("time") or value.get("time"),
                "reservation_id": data.get("reservation_id"),
                "order_id": data.get("order_id"),
            }
        )
        if len(compacted) >= 12:
            break
    return compacted


def _top_objective_scores(objective_vector: dict[str, Any], limit: int = 6) -> dict[str, Any]:
    numeric_items = []
    other_items: dict[str, Any] = {}
    for key, value in objective_vector.items():
        try:
            numeric_items.append((key, float(value)))
        except (TypeError, ValueError):
            other_items[key] = value
    numeric_items.sort(key=lambda item: item[1], reverse=True)
    return {key: round(value, 3) for key, value in numeric_items[:limit]} | other_items


def _planner_shape_from_blueprint(blueprint: dict[str, Any]) -> dict[str, Any] | None:
    if not blueprint:
        return None
    shape = {
        "template_mode": blueprint.get("template_mode"),
        "node_count": blueprint.get("node_count"),
        "route_pattern": blueprint.get("route_pattern"),
        "unsupported_roles": blueprint.get("unsupported_roles"),
    }
    return {key: value for key, value in shape.items() if value not in (None, [], {})} or None


def summarize_component_state(final_state: dict[str, Any], steps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build small component-level evidence bundles for automated diagnosis."""

    intent = final_state.get("intent") if isinstance(final_state.get("intent"), dict) else {}
    constraints = final_state.get("constraints") if isinstance(final_state.get("constraints"), dict) else {}
    blueprint = (
        constraints.get("b_itinerary_blueprint")
        if isinstance(constraints.get("b_itinerary_blueprint"), dict)
        else {}
    )
    user_profile = final_state.get("user_profile") if isinstance(final_state.get("user_profile"), dict) else {}
    selected_plan = final_state.get("selected_plan") if isinstance(final_state.get("selected_plan"), dict) else {}
    candidate_diagnostics = (
        final_state.get("candidate_recall_diagnostics")
        if isinstance(final_state.get("candidate_recall_diagnostics"), dict)
        else {}
    )
    execution_contract = (
        selected_plan.get("execution_contract")
        if isinstance(selected_plan.get("execution_contract"), dict)
        else {}
    )
    why_selected = (
        selected_plan.get("why_selected")
        if isinstance(selected_plan.get("why_selected"), dict)
        else {}
    )
    tool_results = (
        final_state.get("tool_results")
        if isinstance(final_state.get("tool_results"), dict)
        else {}
    )
    step_trace = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        updates = step.get("updates") if isinstance(step.get("updates"), dict) else {}
        summary = step.get("state_summary") if isinstance(step.get("state_summary"), dict) else {}
        step_trace.append(
            {
                "node": step.get("node"),
                "update_keys": sorted(updates.keys()),
                "selected_plan_id": summary.get("selected_plan_id"),
                "execution_status": summary.get("execution_status"),
                "payment_status": summary.get("payment_status"),
            }
        )

    return {
        "intent": {
            "task_type": intent.get("task_type") or constraints.get("task_type"),
            "scene": intent.get("scene") or final_state.get("scene_type"),
            "goal": intent.get("goal"),
            "people_count": intent.get("people_count") or constraints.get("people_count"),
            "time": intent.get("time") or constraints.get("time_window"),
            "location": intent.get("location") or constraints.get("location"),
            "budget": intent.get("budget") or constraints.get("budget"),
            "hard_tags": _compact_sequence(constraints.get("hard_tags") or [], 10),
            "soft_tags": _compact_sequence(constraints.get("soft_tags") or [], 12),
            "avoid": _compact_sequence(constraints.get("avoid") or [], 8),
            "missing_slots": _compact_sequence(
                intent.get("missing_slots") or constraints.get("missing_slots") or [],
                8,
            ),
            "confidence": intent.get("confidence") or constraints.get("confidence"),
        },
        "memory": {
            "memory_policy": constraints.get("memory_policy"),
            "retrieved_memory_ids": _compact_sequence(user_profile.get("retrieved_memory_ids") or [], 12),
            "short_term_memory_count": len(final_state.get("short_term_memory") or []),
            "short_term_memory_sample": _compact_sequence(final_state.get("short_term_memory") or [], 3),
            "active_value_ids": _compact_sequence(constraints.get("active_value_ids") or [], 12),
            "value_memory": _compact_value_memory(final_state.get("value_memory") or []),
            "stable_profile": user_profile.get("stable_profile") or {},
        },
        "planner": {
            "planning_horizon": selected_plan.get("planning_horizon") or blueprint.get("planning_horizon"),
            "planning_days": selected_plan.get("planning_days") or blueprint.get("planning_days"),
            "plan_shape": selected_plan.get("plan_shape") or _planner_shape_from_blueprint(blueprint),
            "candidate_counts": candidate_diagnostics.get("counts") or {},
            "semantic_groups": _compact_sequence(candidate_diagnostics.get("semantic_groups") or [], 10),
            "requirements": candidate_diagnostics.get("requirements") or {},
            "candidate_generation_issues": _compact_sequence(
                final_state.get("candidate_generation_issues") or [],
                8,
            ),
            "selected_plan_id": selected_plan.get("plan_id"),
            "supply_identity": selected_plan.get("supply_identity") or {},
            "timeline": _compact_sequence(selected_plan.get("timeline") or [], 8),
            "total_price": selected_plan.get("total_price")
            or (selected_plan.get("budget") or {}).get("total_price"),
            "total_distance_km": selected_plan.get("total_distance_km"),
            "total_duration_min": selected_plan.get("total_duration_min"),
            "weighted_score": selected_plan.get("weighted_score"),
            "objective_vector_top": _top_objective_scores(selected_plan.get("objective_vector") or {}),
            "execution_ready": selected_plan.get("execution_ready"),
            "execution_contract_ready": execution_contract.get("ready"),
            "action_hint_types": [
                item.get("action_type")
                for item in selected_plan.get("action_hints") or []
                if isinstance(item, dict)
            ],
            "top_reasons": _compact_sequence(why_selected.get("top_reasons") or [], 6),
            "tradeoffs": _compact_sequence(why_selected.get("tradeoffs") or [], 6),
            "alternative_plans_count": len(final_state.get("alternative_plans") or []),
        },
        "execution": {
            "execution_status": final_state.get("execution_status"),
            "payment_status": final_state.get("payment_status"),
            "retry_count": final_state.get("retry_count"),
            "action_sequence": _compact_action_sequence(final_state.get("action_sequence") or []),
            "tool_results": _compact_tool_results(tool_results),
            "failed_tools": [
                item["key"]
                for item in _compact_tool_results(tool_results)
                if item.get("success") is False or item.get("failure_reason")
            ],
            "execution_log_tail": _compact_sequence((final_state.get("execution_log") or [])[-8:], 8),
        },
        "trace": {
            "step_count": len(steps or []),
            "nodes": step_trace,
        },
    }


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
        "component_summaries": summarize_component_state(final_state, steps),
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


def actual_summary_for_evaluation(run_record: dict[str, Any]) -> dict[str, Any]:
    """Return actual_summary enriched with component evidence when possible."""

    actual = run_record.get("actual_summary") if isinstance(run_record.get("actual_summary"), dict) else {}
    if actual.get("component_summaries"):
        return actual
    final_state = run_record.get("final_state") if isinstance(run_record.get("final_state"), dict) else {}
    if not final_state:
        return actual
    enriched = dict(actual)
    enriched["component_summaries"] = summarize_component_state(final_state)
    return enriched


def _text_blob(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


def _primary_timeline_items(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    auxiliary_terms = ("转场", "休息", "步行", "交通", "缓冲", "等待")
    primary_items = []
    for item in timeline:
        if not isinstance(item, dict):
            continue
        activity = str(item.get("activity") or "")
        if item.get("poi_id") is None and any(term in activity for term in auxiliary_terms):
            continue
        primary_items.append(item)
    return primary_items


def _reference_met(reference: dict[str, Any], actual_text: str) -> bool:
    terms = [str(term).strip().lower() for term in reference.get("evidence_terms") or []]
    terms.extend(token for token in re.split(r"[，,、\s/]+", str(reference.get("expectation") or "").lower()) if len(token) >= 2)
    return any(term and term in actual_text for term in terms)


def _must_met(expectation: str, actual_text: str, met_reference_names: set[str]) -> bool:
    if any(token in actual_text for token in re.split(r"[，,、\s/]+", expectation.lower()) if len(token) >= 2):
        return True
    semantic_aliases = {
        "同行人画像": {"profile_fit", "child_friendly_activity"},
        "预算": {"budget_control"},
        "排队": {"route_and_queue_control"},
        "绕路": {"route_and_queue_control"},
        "饮食": {"healthy_or_restricted_dining"},
        "健康": {"healthy_or_restricted_dining"},
        "下雨": {"weather_robustness"},
    }
    return any(alias in expectation and names & met_reference_names for alias, names in semantic_aliases.items())


def _component_finding(component: str, status: str, finding: str, evidence: str) -> dict[str, str]:
    if component not in EVAL_COMPONENTS:
        component = "planner"
    if status not in COMPONENT_STATUSES:
        status = "weak"
    return {
        "component": component,
        "status": status,
        "finding": finding,
        "evidence": evidence,
    }


def _heuristic_component_findings(
    actual: dict[str, Any],
    *,
    met_musts: list[str],
    reference_fit: float,
    shape_ok: bool,
    execution_ok: bool,
    explanation_ok: bool,
) -> list[dict[str, str]]:
    components = actual.get("component_summaries") if isinstance(actual.get("component_summaries"), dict) else {}
    intent_summary = components.get("intent") if isinstance(components.get("intent"), dict) else {}
    memory_summary = components.get("memory") if isinstance(components.get("memory"), dict) else {}
    planner_summary = components.get("planner") if isinstance(components.get("planner"), dict) else {}
    execution_summary = components.get("execution") if isinstance(components.get("execution"), dict) else {}

    intent_status = "pass" if met_musts else "weak"
    memory_status = "pass" if memory_summary.get("value_memory") or memory_summary.get("retrieved_memory_ids") else "weak"
    planner_status = "pass" if shape_ok and reference_fit >= 0.5 else "weak"
    execution_status = "pass" if execution_ok else "fail"
    explanation_status = "pass" if explanation_ok else "fail"

    return [
        _component_finding(
            "intent",
            intent_status,
            "Intent parsing appears aligned with matched must-satisfy signals." if met_musts else "Intent or constraint evidence is weak.",
            json.dumps(
                {
                    "scene": intent_summary.get("scene") or actual.get("scene_type"),
                    "matched_musts": met_musts[:4],
                    "missing_slots": intent_summary.get("missing_slots"),
                },
                ensure_ascii=False,
            ),
        ),
        _component_finding(
            "memory",
            memory_status,
            "Memory evidence is available for personalization." if memory_status == "pass" else "Memory evidence is sparse in the summary.",
            json.dumps(
                {
                    "retrieved_memory_ids": memory_summary.get("retrieved_memory_ids", [])[:6],
                    "active_value_ids": memory_summary.get("active_value_ids", [])[:6],
                    "short_term_memory_count": memory_summary.get("short_term_memory_count"),
                },
                ensure_ascii=False,
            ),
        ),
        _component_finding(
            "planner",
            planner_status,
            "Planner selected a shape-compatible plan with enough reference evidence."
            if planner_status == "pass"
            else "Planner evidence is incomplete or shape/reference fit is weak.",
            json.dumps(
                {
                    "selected_plan_id": planner_summary.get("selected_plan_id")
                    or (actual.get("selected_plan") or {}).get("plan_id"),
                    "reference_fit": round(reference_fit, 3),
                    "shape_ok": shape_ok,
                    "candidate_counts": planner_summary.get("candidate_counts", {}),
                },
                ensure_ascii=False,
            ),
        ),
        _component_finding(
            "execution",
            execution_status,
            "Execution completed or reached a user-confirmable state." if execution_ok else "Execution did not complete successfully.",
            json.dumps(
                {
                    "execution_status": execution_summary.get("execution_status") or actual.get("execution_status"),
                    "payment_status": execution_summary.get("payment_status") or actual.get("payment_status"),
                    "failed_tools": execution_summary.get("failed_tools", []),
                },
                ensure_ascii=False,
            ),
        ),
        _component_finding(
            "explanation",
            explanation_status,
            "Explanation or final share message is present." if explanation_ok else "Explanation and final share message are missing.",
            "explanation_text/final_share_message present" if explanation_ok else "no explanation_text or final_share_message",
        ),
    ]


def heuristic_evaluate_run(run_record: dict[str, Any]) -> dict[str, Any]:
    """Deterministic baseline scorer for CI and offline development."""

    case = run_record["case"]
    expected = case.get("expected") or {}
    actual = actual_summary_for_evaluation(run_record)
    actual_text = _text_blob(actual)
    musts = [str(item) for item in expected.get("must_satisfy", [])]
    shoulds = [str(item) for item in expected.get("should_satisfy", [])]
    avoid = [str(item) for item in expected.get("avoid", [])]
    timeline = ((actual.get("selected_plan") or {}).get("timeline") or [])
    primary_timeline = _primary_timeline_items(timeline)
    result_shape = expected.get("result_shape") or {}
    min_nodes = int(result_shape.get("min_nodes") or 1)
    max_primary_nodes = int(result_shape.get("max_primary_nodes") or result_shape.get("max_nodes") or 6)

    poi_references = [item for item in expected.get("poi_reference", []) if isinstance(item, dict)]
    met_references = [item for item in poi_references if _reference_met(item, actual_text)]
    met_reference_names = {str(item.get("name") or item.get("expectation") or "") for item in met_references}
    met_musts = [item for item in musts if _must_met(item, actual_text, met_reference_names)]
    avoided = [item for item in avoid if item and item not in actual_text]
    execution_ok = actual.get("execution_status") in {"completed", "success", "paid", "pending_confirmation"}
    shape_ok = min_nodes <= len(primary_timeline) <= max_primary_nodes
    explanation_ok = bool(str(actual.get("explanation_text") or actual.get("final_share_message") or "").strip())
    reference_fit = len(met_references) / len(poi_references) if poi_references else 1.0

    intent_fit = 55 + min(25, len(met_musts) * 8) + min(15, len(met_references) * 4)
    constraint_satisfaction = 50 + min(25, len(met_musts) * 7) + min(15, len(avoided) * 5) + round(reference_fit * 10)
    itinerary_shape = 90 if shape_ok else max(35, 70 - abs(len(primary_timeline) - min_nodes) * 15)
    personalization = 45 + min(25, len(met_musts) * 6) + min(25, len(met_references) * 5) + min(10, len(shoulds) * 2)
    feasibility_execution = 85 if execution_ok else (60 if timeline else 25)
    explanation_quality = 80 if explanation_ok else 30
    overall = round(
        intent_fit * 0.2
        + constraint_satisfaction * 0.25
        + itinerary_shape * 0.1
        + personalization * 0.25
        + feasibility_execution * 0.15
        + explanation_quality * 0.05,
        1,
    )

    failure_categories = []
    if not met_musts:
        failure_categories.append("constraint_violation")
    if poi_references and reference_fit < 0.5:
        failure_categories.append("personalization_gap")
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
        "component_findings": _heuristic_component_findings(
            actual,
            met_musts=met_musts,
            reference_fit=reference_fit,
            shape_ok=shape_ok,
            execution_ok=execution_ok,
            explanation_ok=explanation_ok,
        ),
        "met_expectations": (met_musts + [str(item.get("expectation") or item.get("name")) for item in met_references])[:6],
        "missed_expectations": (
            [item for item in musts if item not in met_musts]
            + [str(item.get("expectation") or item.get("name")) for item in poi_references if item not in met_references]
        )[:6],
        "evidence": ["local heuristic scorer; use --call-llm for semantic judge"],
        "improvement_hints": ["Inspect missed_expectations and actual_summary for the highest-count failure categories."],
        "confidence": 0.45,
        "evaluation_metadata": {"mode": "local_heuristic"},
    }


def normalize_component_findings(raw_findings: Any) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not isinstance(raw_findings, list):
        return findings
    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        component = str(item.get("component") or "").strip()
        status = str(item.get("status") or "").strip()
        finding = str(item.get("finding") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        findings.append(_component_finding(component, status, finding, evidence))
    return findings


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
        "component_findings": normalize_component_findings(raw_eval.get("component_findings")),
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
            "actual_summary": actual_summary_for_evaluation(run_record),
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
    component_status_counter: dict[str, Counter[str]] = defaultdict(Counter)
    horizon_scores: dict[str, list[float]] = defaultdict(list)
    for item in evaluations:
        category_counter.update(item.get("failure_categories") or [])
        for finding in item.get("component_findings") or []:
            if not isinstance(finding, dict):
                continue
            component = str(finding.get("component") or "unknown")
            status = str(finding.get("status") or "weak")
            component_status_counter[component][status] += 1
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
        "component_status_counts": {
            component: dict(counter.most_common())
            for component, counter in sorted(component_status_counter.items())
        },
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
    lines.extend(["", "## Component Status", ""])
    for component, counts in summary.get("component_status_counts", {}).items():
        count_text = ", ".join(f"{status}={count}" for status, count in counts.items())
        lines.append(f"- {component}: {count_text}")
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
    generate.add_argument("--seed-cases", type=Path, default=None)
    generate.add_argument("--cases-per-horizon", type=int, default=1)
    generate.add_argument("--call-llm", action="store_true")
    generate.add_argument(
        "--strict-llm-generation",
        action="store_true",
        help="Fail instead of writing fallback cases when LLM generation cannot fill all buckets.",
    )
    generate.add_argument(
        "--llm-generation-attempts",
        type=int,
        default=None,
        help="Profile/horizon backfill attempts for strict LLM generation.",
    )

    run_parser = subparsers.add_parser("run", help="Run generated cases through the existing agent.")
    run_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    run_parser.add_argument("--runs-out", type=Path, default=DEFAULT_RUNS_PATH)
    run_parser.add_argument("--limit", type=int, default=0)

    merge_cases = subparsers.add_parser("merge-cases", help="Normalize and merge generated case JSONL files.")
    merge_cases.add_argument("--base-cases", type=Path, required=True)
    merge_cases.add_argument("--extra-cases", type=Path, required=True)
    merge_cases.add_argument("--cases-out", type=Path, required=True)
    merge_cases.add_argument("--extra-provider", required=True)
    merge_cases.add_argument("--extra-model", default="codex")
    merge_cases.add_argument("--extra-source", default=None)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate agent runs against rough expectations.")
    evaluate.add_argument("--runs", type=Path, default=DEFAULT_RUNS_PATH)
    evaluate.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    evaluate.add_argument("--call-llm", action="store_true")

    pipeline = subparsers.add_parser("pipeline", help="Generate cases, run agent, and evaluate.")
    pipeline.add_argument("--profiles", type=Path)
    pipeline.add_argument("--cases-out", type=Path, default=DEFAULT_CASES_PATH)
    pipeline.add_argument("--runs-out", type=Path, default=DEFAULT_RUNS_PATH)
    pipeline.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    pipeline.add_argument("--seed-cases", type=Path, default=None)
    pipeline.add_argument("--cases-per-horizon", type=int, default=1)
    pipeline.add_argument("--limit", type=int, default=0)
    pipeline.add_argument("--call-llm", action="store_true")
    pipeline.add_argument(
        "--strict-llm-generation",
        action="store_true",
        help="Fail instead of writing fallback cases when LLM generation cannot fill all buckets.",
    )
    pipeline.add_argument("--llm-generation-attempts", type=int, default=None)
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
            strict_llm_generation=args.strict_llm_generation,
            max_generation_attempts=args.llm_generation_attempts,
        )
        cases = append_seed_cases(cases, load_seed_cases(args.seed_cases))
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

    if args.command == "merge-cases":
        base_cases = load_jsonl(args.base_cases)
        extra_cases = load_external_cases(
            args.extra_cases,
            provider=args.extra_provider,
            model=args.extra_model,
            source=args.extra_source,
        )
        merged_cases = append_seed_cases(base_cases, extra_cases)
        write_jsonl(args.cases_out, merged_cases)
        print(
            json.dumps(
                {
                    "cases_out": str(args.cases_out),
                    "base_count": len(base_cases),
                    "extra_count": len(extra_cases),
                    "merged_count": len(merged_cases),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
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
            strict_llm_generation=args.strict_llm_generation,
            max_generation_attempts=args.llm_generation_attempts,
        )
        cases = append_seed_cases(cases, load_seed_cases(args.seed_cases))
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
