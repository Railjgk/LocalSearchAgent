#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ROUNDS="${ROUNDS:-1}"
CASES_PER_HORIZON="${CASES_PER_HORIZON:-1}"
ROUND_TIMEOUT_MINUTES="${ROUND_TIMEOUT_MINUTES:-20}"
EVAL_TIMEOUT_SECONDS="${LLM_AGENT_EVAL_TIMEOUT_SECONDS:-180}"
EVAL_RETRIES="${LLM_AGENT_EVAL_RETRIES:-2}"
EVAL_RETRY_BACKOFF_SECONDS="${LLM_AGENT_EVAL_RETRY_BACKOFF_SECONDS:-5}"
GENERATION_ATTEMPTS="${LLM_AGENT_EVAL_GENERATION_ATTEMPTS:-3}"
GENERATION_FAILURE_SLEEP_SECONDS="${GENERATION_FAILURE_SLEEP_SECONDS:-60}"
MAX_POST_FIX_RERUN_SECONDS="${MAX_POST_FIX_RERUN_SECONDS:-180}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiments/artifacts/llm_agent_eval/automation}"
INCLUDE_HARD_SEED_CASES="${INCLUDE_HARD_SEED_CASES:-1}"
HARD_SEED_CASES_PATH="${HARD_SEED_CASES_PATH:-experiments/eval_seed_cases/hard_realistic_cases.jsonl}"
SEED_ONLY_ON_LLM_GENERATION_FAILURE="${SEED_ONLY_ON_LLM_GENERATION_FAILURE:-1}"
INCLUDE_CODEX_ARCH_CASES="${INCLUDE_CODEX_ARCH_CASES:-1}"
CODEX_ARCH_CASE_COUNT="${CODEX_ARCH_CASE_COUNT:-3}"
CODEX_CASE_GENERATION_TIMEOUT_MINUTES="${CODEX_CASE_GENERATION_TIMEOUT_MINUTES:-8}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

export LLM_AGENT_EVAL_TIMEOUT_SECONDS="$EVAL_TIMEOUT_SECONDS"
export LLM_AGENT_EVAL_RETRIES="$EVAL_RETRIES"
export LLM_AGENT_EVAL_RETRY_BACKOFF_SECONDS="$EVAL_RETRY_BACKOFF_SECONDS"
export LLM_AGENT_EVAL_GENERATION_ATTEMPTS="$GENERATION_ATTEMPTS"

mkdir -p "$ARTIFACT_ROOT"

run_logged() {
  local log_path="$1"
  shift
  {
    printf '## command\n'
    printf '%q ' "$@"
    printf '\n\n## output\n'
    "$@"
  } >"$log_path" 2>&1
}

validate_llm_generated_cases() {
  local cases_path="$1"
  python - "$cases_path" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit("no generated cases")
providers = Counter(
    (row.get("generation_metadata") or {}).get("provider")
    or (row.get("generation_metadata") or {}).get("mode")
    or "unknown"
    for row in rows
)
models = Counter((row.get("generation_metadata") or {}).get("model") or "unknown" for row in rows)
print(json.dumps({"case_count": len(rows), "providers": dict(providers), "models": dict(models)}, ensure_ascii=False))
allowed_providers = {
    "longcat_openai_compatible",
    "codex_seeded_realistic_case",
    "codex_architecture_challenge_case",
}
bad = [
    row.get("case_id")
    for row in rows
    if (row.get("generation_metadata") or {}).get("provider") not in allowed_providers
]
if bad:
    raise SystemExit(f"fallback or unknown generated cases are not allowed: {bad}")
PY
}

write_failed_round_status() {
  local iter_dir="$1"
  local round_index="$2"
  local phase="$3"
  local exit_status="$4"
  local details_path="$5"
  cat >"$iter_dir/status.json" <<EOF
{
  "round_index": $round_index,
  "finished_at": "$(date -Iseconds)",
  "status": "failed",
  "phase": "$phase",
  "exit_status": $exit_status,
  "details_path": "$details_path",
  "continue_next_round": true
}
EOF
}

write_seed_only_generation() {
  local seed_cases_path="$1"
  local cases_out="$2"
  local outage_status_path="$3"
  local failed_generation_log="$4"
  python - "$seed_cases_path" "$cases_out" "$outage_status_path" "$failed_generation_log" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

from experiments.llm_agent_eval import load_seed_cases, write_jsonl

seed_cases_path = Path(sys.argv[1])
cases_out = Path(sys.argv[2])
outage_status_path = Path(sys.argv[3])
failed_generation_log = Path(sys.argv[4])

cases = load_seed_cases(seed_cases_path)
if not cases:
    raise SystemExit(f"no seed cases available at {seed_cases_path}")

write_jsonl(cases_out, cases)
providers = Counter(
    (case.get("generation_metadata") or {}).get("provider", "unknown")
    for case in cases
)
payload = {
    "mode": "codex_seed_only_after_llm_generation_failure",
    "case_count": len(cases),
    "providers": dict(providers),
    "seed_cases_path": str(seed_cases_path),
    "failed_generation_log": str(failed_generation_log),
    "local_fallback_used": False,
    "continue_to_codex_architecture_cases": True,
}
outage_status_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

write_codex_architecture_case_prompt() {
  local iter_dir="$1"
  local prompt_path="$2"
  local case_count="$3"
  cat >"$prompt_path" <<EOF
You are the pre-run architecture-aware case generator for LocalSearchAgent automation.

Goal:
- Analyze the current LocalSearchAgent architecture and generate challenging but realistic user cases that are likely to expose module-level defects.
- These cases will be run through the full agent before the later Codex judge/optimizer phase.

Hard requirements:
- Do not modify source code, tests, docs outside this iteration directory, or runtime state files.
- Do not run heuristic or rule-based evaluation. Do not call \`experiments/llm_agent_eval.py evaluate\`.
- Inspect architecture and current cases before generating examples.
- Generate exactly ${case_count} cases.
- Write strict JSONL to: \`$iter_dir/codex_architecture_cases.raw.jsonl\`
- Write a short architecture analysis report to: \`$iter_dir/codex_architecture_case_report.md\`
- Each JSONL line must be one complete JSON object. No Markdown fences in the JSONL file.

Read these inputs:
- Existing generated cases: \`$iter_dir/cases.jsonl\`
- Main runner and graph: \`run.py\`, \`src/graph.py\`, \`src/state.py\`
- A-stage: \`src/nodes/intent_parser.py\`, \`src/nodes/memory_manager.py\`, \`src/nodes/scenario_planner.py\`
- B-stage: \`src/nodes/b_itinerary_blueprint.py\`, \`src/nodes/b_poi_rag.py\`, \`src/nodes/candidate_generator.py\`, \`src/nodes/constraint_filter.py\`, \`src/nodes/plan_optimizer.py\`, \`src/nodes/b_replan_loop.py\`
- C-stage: \`src/nodes/tool_router.py\`, \`src/nodes/mock_api_layer.py\`, \`src/nodes/execution_manager.py\`, \`src/nodes/b_repair_planner.py\`, \`src/nodes/payment_layer.py\`, \`src/nodes/share_generator.py\`

Case design requirements:
- Cases must be realistic local-life user requests in Chinese, not synthetic checklist prompts.
- Each case must target at least two likely module boundaries, such as:
  - ambiguous or conflicting intent slots,
  - memory personalization vs current-turn constraints,
  - multi-node/time-anchor planning,
  - two-day or full-day shape preservation,
  - RAG supply gaps or execution-contract honesty,
  - tool routing, partial execution recovery, payment/share-message faithfulness.
- Include reference expectations that are semantic and evidence-based, not fixed POI answers.
- Do not duplicate the existing generated or seed cases.
- Keep the set small but sharp. Prefer cases that would teach the optimizer what to fix if the agent fails.

Required JSON object schema per line:
{
  "case_id": "codexarch_<stable_slug>",
  "profile_id": "codex_arch_<profile_slug>",
  "profile": {
    "profile_id": "codex_arch_<profile_slug>",
    "city": "上海",
    "partial_profile": {
      "group": "...",
      "preferences": ["..."],
      "constraints": ["..."]
    }
  },
  "horizon": "short|one_day|two_day",
  "user_request": "...",
  "expected": {
    "intent_summary": "...",
    "must_satisfy": ["at least 4 concrete semantic requirements"],
    "should_satisfy": ["..."],
    "avoid": ["..."],
    "expected_activity_roles": ["..."],
    "poi_reference": [
      {
        "name": "...",
        "expectation": "...",
        "evidence_terms": ["..."]
      }
    ],
    "time_budget": {"start_hint": "...", "duration_hours": 0, "schedule_flexibility": "low|medium|high"},
    "money_budget": {"amount": 0, "strictness": "low|medium|high"},
    "result_shape": {"min_nodes": 1, "max_primary_nodes": 6, "needs_reservation_or_purchase": true},
    "success_criteria": [
      {"name": "component_or_goal", "weight": 0.2, "rubric": "..."}
    ]
  },
  "difficulty_tags": ["codex_architecture", "..."],
  "architecture_targets": [
    {"component": "intent|memory|planner|execution|share", "risk": "...", "expected_signal": "..."}
  ]
}

The \`success_criteria\` weights should approximately sum to 1.0.
EOF
}

run_codex_architecture_case_generation() {
  local iter_dir="$1"
  local prompt_path="$iter_dir/codex_architecture_case_prompt.md"
  write_codex_architecture_case_prompt "$iter_dir" "$prompt_path" "$CODEX_ARCH_CASE_COUNT"

  timeout --kill-after=60s "${CODEX_CASE_GENERATION_TIMEOUT_MINUTES}m" \
    codex \
      --ask-for-approval never \
      exec \
      --cd "$ROOT_DIR" \
      --sandbox danger-full-access \
      --output-last-message "$iter_dir/codex_architecture_case_last_message.md" \
      - <"$prompt_path" >"$iter_dir/codex_architecture_case_exec.log" 2>&1
}

write_compact_run_summary() {
  local cases_path="$1"
  local runs_path="$2"
  local out_json="$3"
  local out_md="$4"
  python - "$cases_path" "$runs_path" "$out_json" "$out_md" <<'PY'
import json
import sys
from pathlib import Path

cases_path, runs_path, out_json, out_md = [Path(arg) for arg in sys.argv[1:]]
cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
runs = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
case_by_id = {case["case_id"]: case for case in cases}

items = []
for run in runs:
    case = case_by_id.get(run.get("case_id"), run.get("case", {}))
    actual = run.get("actual_summary") or {}
    components = actual.get("component_summaries") or {}
    plan = actual.get("selected_plan") or {}
    planner = components.get("planner") or {}
    execution = components.get("execution") or {}
    intent = components.get("intent") or {}
    memory = components.get("memory") or {}
    final_state = run.get("final_state") or {}
    selected_plan = final_state.get("selected_plan") or {}
    constraints = final_state.get("constraints") or {}
    blueprint = constraints.get("b_itinerary_blueprint") or {}
    items.append(
        {
            "case_id": run.get("case_id"),
            "profile_id": case.get("profile_id"),
            "horizon": case.get("horizon"),
            "generation_provider": (case.get("generation_metadata") or {}).get("provider"),
            "difficulty_tags": case.get("difficulty_tags") or [],
            "architecture_targets": case.get("architecture_targets") or [],
            "user_request": case.get("user_request"),
            "expected": {
                "must_satisfy": (case.get("expected") or {}).get("must_satisfy", []),
                "should_satisfy": (case.get("expected") or {}).get("should_satisfy", []),
                "avoid": (case.get("expected") or {}).get("avoid", []),
                "poi_reference": (case.get("expected") or {}).get("poi_reference", []),
                "result_shape": (case.get("expected") or {}).get("result_shape", {}),
            },
            "intent": {
                "scene": intent.get("scene"),
                "people_count": intent.get("people_count"),
                "time": intent.get("time"),
                "budget": intent.get("budget"),
                "hard_tags": intent.get("hard_tags"),
                "soft_tags": intent.get("soft_tags"),
                "avoid": intent.get("avoid"),
                "missing_slots": intent.get("missing_slots"),
            },
            "memory": {
                "retrieved_memory_ids": memory.get("retrieved_memory_ids"),
                "active_value_ids": memory.get("active_value_ids"),
                "value_memory": memory.get("value_memory"),
                "stable_profile": memory.get("stable_profile"),
            },
            "planner": {
                "planning_horizon": (
                    selected_plan.get("planning_horizon")
                    or planner.get("planning_horizon")
                    or blueprint.get("planning_horizon")
                ),
                "planning_days": (
                    selected_plan.get("planning_days")
                    or planner.get("planning_days")
                    or blueprint.get("planning_days")
                ),
                "plan_shape": selected_plan.get("plan_shape") or planner.get("plan_shape"),
                "selected_plan_id": planner.get("selected_plan_id") or plan.get("plan_id"),
                "timeline": plan.get("timeline", []),
                "total_price": plan.get("estimated_total_price") or planner.get("total_price"),
                "total_distance_km": plan.get("total_distance_km") or planner.get("total_distance_km"),
                "candidate_counts": planner.get("candidate_counts"),
                "candidate_generation_issues": planner.get("candidate_generation_issues"),
                "supply_identity": planner.get("supply_identity"),
                "tradeoffs": planner.get("tradeoffs"),
            },
            "execution": {
                "execution_status": execution.get("execution_status") or actual.get("execution_status"),
                "payment_status": execution.get("payment_status") or actual.get("payment_status"),
                "retry_count": execution.get("retry_count") or actual.get("retry_count"),
                "failed_tools": execution.get("failed_tools"),
                "tool_results": execution.get("tool_results"),
                "action_sequence": execution.get("action_sequence"),
            },
            "explanation_text": actual.get("explanation_text"),
            "final_share_message": actual.get("final_share_message"),
            "execution_log_tail": actual.get("execution_log_tail", []),
        }
    )

payload = {"case_count": len(cases), "run_count": len(runs), "items": items}
out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

lines = ["# Compact Run Summary", "", f"- cases: {len(cases)}", f"- runs: {len(runs)}", ""]
for item in items:
    lines.extend(
        [
            f"## {item['case_id']} ({item.get('horizon')})",
            "",
            f"- request: {item.get('user_request')}",
            f"- generation_provider: {item.get('generation_provider')} difficulty_tags={item.get('difficulty_tags')}",
            f"- architecture_targets: {json.dumps(item.get('architecture_targets'), ensure_ascii=False)}",
            f"- expected_must: {item['expected'].get('must_satisfy')}",
            f"- intent: {json.dumps(item.get('intent'), ensure_ascii=False)}",
            f"- memory_ids: {item.get('memory', {}).get('retrieved_memory_ids')}",
            f"- planner_horizon: {item.get('planner', {}).get('planning_horizon')} days={item.get('planner', {}).get('planning_days')}",
            f"- timeline: {[(node.get('time'), node.get('activity'), node.get('poi_id')) for node in item.get('planner', {}).get('timeline', [])]}",
            f"- execution: {item.get('execution', {}).get('execution_status')} failed_tools={item.get('execution', {}).get('failed_tools')}",
            f"- final_share_len: {len(item.get('final_share_message') or '')}",
            "",
        ]
    )
out_md.write_text("\n".join(lines), encoding="utf-8")
PY
}

write_codex_prompt() {
  local iter_dir="$1"
  local prompt_path="$2"
  cat >"$prompt_path" <<EOF
You are running an automated LocalSearchAgent optimization round.

Hard requirements:
- Do not use heuristic or rule-based evaluation. Do not call \`experiments/llm_agent_eval.py evaluate\`.
- Judge the run qualitatively as Codex, using the LLM-generated cases and complete agent run state.
- Use the current branch/worktree. Preserve unrelated user changes. Do not revert existing dirty files unless they are your own failed edits in this round.
- This is end-to-end LocalSearchAgent optimization, not the standalone B policy optimization loop.
- Make a small, focused code improvement on the current branch if the run exposes a clear defect.
- Save all analysis and validation outputs in this iteration directory: \`$iter_dir\`.
- Probe-training architecture is protected. Do not alter probe model architecture, training loop, loss/objective implementation, layer-selection mechanics, or probe serving/evaluation architecture. You may add, remove, or generate value training items, value categories, labels, prompts, or data files if a probe-related issue is directly relevant.
- Write \`codex_judge_report.md\` and \`codex_judge_report.json\` before making code changes, so the diagnostic record survives even if validation later stalls.

Inputs:
- cases: \`$iter_dir/cases.jsonl\`
- pre-fix runs: \`$iter_dir/runs.jsonl\`
- compact summary JSON: \`$iter_dir/compact_run_summary.json\`
- compact summary Markdown: \`$iter_dir/compact_run_summary.md\`
- generation log: \`$iter_dir/generate.log\`
- run log: \`$iter_dir/run.log\`

Required outputs:
- \`$iter_dir/codex_judge_report.md\`: case-by-case Codex diagnostic evaluation with component diagnosis.
- \`$iter_dir/codex_judge_report.json\`: machine-readable summary with case ids, pass/fail/warn, scores or qualitative levels, and top failure clusters.
- \`$iter_dir/optimization_summary.md\`: what you changed, why, files touched, and validation results.
- If you modify code, run a targeted validation. Prefer pure Python checks and targeted tests available in this repo. If a test tool is unavailable, record that.
- Do not run unbounded post-fix reruns. If you rerun the generated cases, use \`timeout ${MAX_POST_FIX_RERUN_SECONDS}s python experiments/llm_agent_eval.py run --cases "$iter_dir/cases.jsonl" --runs-out "$iter_dir/post_fix_runs.jsonl"\` and record timeout or partial results. Do not run heuristic evaluation.
- Keep validation scoped. The Codex phase has a ${ROUND_TIMEOUT_MINUTES}-minute outer timeout, and long-running validation should be stopped and recorded rather than left live.

Suggested evaluation focus:
- Whether intent horizon and constraints survive into planning.
- Whether memory helps or pollutes personalization.
- Whether planner output matches short/one_day/two_day shape.
- Whether execution recovers from tool failures.
- Whether user-facing final output is present and faithful.

Finish with a concise final message summarizing the round status.
EOF
}

round_index=1
while [[ "$ROUNDS" == "0" || "$round_index" -le "$ROUNDS" ]]; do
  stamp="$(date +%Y%m%d_%H%M%S)"
  iter_dir="$ARTIFACT_ROOT/round_${stamp}"
  mkdir -p "$iter_dir"

  cat >"$iter_dir/metadata.json" <<EOF
{
  "round_index": $round_index,
  "started_at": "$(date -Iseconds)",
  "cases_per_horizon": $CASES_PER_HORIZON,
  "round_timeout_minutes": $ROUND_TIMEOUT_MINUTES,
  "eval_timeout_seconds": $EVAL_TIMEOUT_SECONDS,
  "eval_retries": $EVAL_RETRIES,
  "eval_retry_backoff_seconds": $EVAL_RETRY_BACKOFF_SECONDS,
  "generation_attempts": $GENERATION_ATTEMPTS,
  "include_hard_seed_cases": "$INCLUDE_HARD_SEED_CASES",
  "hard_seed_cases_path": "$HARD_SEED_CASES_PATH",
  "seed_only_on_llm_generation_failure": "$SEED_ONLY_ON_LLM_GENERATION_FAILURE",
  "include_codex_architecture_cases": "$INCLUDE_CODEX_ARCH_CASES",
  "codex_architecture_case_count": $CODEX_ARCH_CASE_COUNT,
  "codex_case_generation_timeout_minutes": $CODEX_CASE_GENERATION_TIMEOUT_MINUTES,
  "max_post_fix_rerun_seconds": $MAX_POST_FIX_RERUN_SECONDS,
  "probe_training_architecture_protected": true,
  "judge": "codex_exec",
  "heuristic_eval_allowed": false
}
EOF

  generation_cmd=(
    python experiments/llm_agent_eval.py generate \
      --call-llm \
      --strict-llm-generation \
      --llm-generation-attempts "$GENERATION_ATTEMPTS" \
      --cases-per-horizon "$CASES_PER_HORIZON" \
      --cases-out "$iter_dir/cases.jsonl"
  )
  if [[ "$INCLUDE_HARD_SEED_CASES" != "0" && -f "$HARD_SEED_CASES_PATH" ]]; then
    generation_cmd+=(--seed-cases "$HARD_SEED_CASES_PATH")
  fi

  generation_status=0
  run_logged "$iter_dir/generate.log" "${generation_cmd[@]}" || generation_status=$?

  if [[ "$generation_status" != "0" ]]; then
    seed_only_status=1
    if [[ "$SEED_ONLY_ON_LLM_GENERATION_FAILURE" != "0" && "$INCLUDE_HARD_SEED_CASES" != "0" && -f "$HARD_SEED_CASES_PATH" ]]; then
      if run_logged "$iter_dir/seed_only_generation.log" \
          write_seed_only_generation \
            "$HARD_SEED_CASES_PATH" \
            "$iter_dir/cases.jsonl" \
            "$iter_dir/llm_generation_outage_seed_only.json" \
            "$iter_dir/generate.log"; then
        seed_only_status=0
      else
        seed_only_status=$?
      fi
    fi
    if [[ "$seed_only_status" != "0" ]]; then
      write_failed_round_status "$iter_dir" "$round_index" "generate" "$generation_status" "$iter_dir/generate.log"
      echo "Round $round_index generation failed with status=$generation_status. See $iter_dir/generate.log"
      round_index=$((round_index + 1))
      if [[ "$ROUNDS" == "0" && "$GENERATION_FAILURE_SLEEP_SECONDS" != "0" ]]; then
        sleep "$GENERATION_FAILURE_SLEEP_SECONDS"
      fi
      continue
    fi
    echo "Round $round_index LLM generation failed with status=$generation_status; continuing with Codex seed cases. See $iter_dir/llm_generation_outage_seed_only.json"
  fi

  if [[ "$INCLUDE_CODEX_ARCH_CASES" != "0" ]]; then
    codex_case_status=0
    run_codex_architecture_case_generation "$iter_dir" || codex_case_status=$?
    if [[ "$codex_case_status" != "0" ]]; then
      write_failed_round_status "$iter_dir" "$round_index" "codex_architecture_case_generation" "$codex_case_status" "$iter_dir/codex_architecture_case_exec.log"
      echo "Round $round_index Codex architecture case generation failed with status=$codex_case_status. See $iter_dir/codex_architecture_case_exec.log"
      round_index=$((round_index + 1))
      if [[ "$ROUNDS" == "0" && "$GENERATION_FAILURE_SLEEP_SECONDS" != "0" ]]; then
        sleep "$GENERATION_FAILURE_SLEEP_SECONDS"
      fi
      continue
    fi
    if [[ ! -s "$iter_dir/codex_architecture_cases.raw.jsonl" ]]; then
      write_failed_round_status "$iter_dir" "$round_index" "codex_architecture_case_generation" 1 "$iter_dir/codex_architecture_cases.raw.jsonl"
      echo "Round $round_index Codex architecture case generation produced no cases. See $iter_dir/codex_architecture_case_exec.log"
      round_index=$((round_index + 1))
      if [[ "$ROUNDS" == "0" && "$GENERATION_FAILURE_SLEEP_SECONDS" != "0" ]]; then
        sleep "$GENERATION_FAILURE_SLEEP_SECONDS"
      fi
      continue
    fi
    cp "$iter_dir/cases.jsonl" "$iter_dir/cases.before_codex_architecture.jsonl"
    merge_codex_case_status=0
    run_logged "$iter_dir/merge_codex_architecture_cases.log" \
      python experiments/llm_agent_eval.py merge-cases \
        --base-cases "$iter_dir/cases.before_codex_architecture.jsonl" \
        --extra-cases "$iter_dir/codex_architecture_cases.raw.jsonl" \
        --cases-out "$iter_dir/cases.jsonl" \
        --extra-provider codex_architecture_challenge_case \
        --extra-model codex \
        --extra-source "$iter_dir/codex_architecture_case_report.md" || merge_codex_case_status=$?
    if [[ "$merge_codex_case_status" != "0" ]]; then
      write_failed_round_status "$iter_dir" "$round_index" "merge_codex_architecture_cases" "$merge_codex_case_status" "$iter_dir/merge_codex_architecture_cases.log"
      echo "Round $round_index failed to merge Codex architecture cases with status=$merge_codex_case_status. See $iter_dir/merge_codex_architecture_cases.log"
      round_index=$((round_index + 1))
      if [[ "$ROUNDS" == "0" && "$GENERATION_FAILURE_SLEEP_SECONDS" != "0" ]]; then
        sleep "$GENERATION_FAILURE_SLEEP_SECONDS"
      fi
      continue
    fi
  fi

  generation_validation_status=0
  validate_llm_generated_cases "$iter_dir/cases.jsonl" >"$iter_dir/generation_metadata_check.jsonl" 2>&1 || generation_validation_status=$?
  if [[ "$generation_validation_status" != "0" ]]; then
    write_failed_round_status "$iter_dir" "$round_index" "validate_generation" "$generation_validation_status" "$iter_dir/generation_metadata_check.jsonl"
    echo "Round $round_index generated non-LLM cases with status=$generation_validation_status. See $iter_dir/generation_metadata_check.jsonl"
    round_index=$((round_index + 1))
    if [[ "$ROUNDS" == "0" && "$GENERATION_FAILURE_SLEEP_SECONDS" != "0" ]]; then
      sleep "$GENERATION_FAILURE_SLEEP_SECONDS"
    fi
    continue
  fi

  run_logged "$iter_dir/run.log" \
    python experiments/llm_agent_eval.py run \
      --cases "$iter_dir/cases.jsonl" \
      --runs-out "$iter_dir/runs.jsonl"

  write_compact_run_summary \
    "$iter_dir/cases.jsonl" \
    "$iter_dir/runs.jsonl" \
    "$iter_dir/compact_run_summary.json" \
    "$iter_dir/compact_run_summary.md"

  prompt_path="$iter_dir/codex_prompt.md"
  write_codex_prompt "$iter_dir" "$prompt_path"

  codex_status=0
  timeout --kill-after=60s "${ROUND_TIMEOUT_MINUTES}m" \
    codex \
      --ask-for-approval never \
      exec \
      --cd "$ROOT_DIR" \
      --sandbox danger-full-access \
      --output-last-message "$iter_dir/codex_last_message.md" \
      - <"$prompt_path" >"$iter_dir/codex_exec.log" 2>&1 || codex_status=$?

  cat >"$iter_dir/status.json" <<EOF
{
  "round_index": $round_index,
  "finished_at": "$(date -Iseconds)",
  "codex_exit_status": $codex_status,
  "timed_out": $([[ "$codex_status" == "124" || "$codex_status" == "137" ]] && echo true || echo false)
}
EOF

  if [[ "$codex_status" != "0" ]]; then
    echo "Round $round_index ended with codex_status=$codex_status. See $iter_dir/codex_exec.log"
    exit "$codex_status"
  fi

  echo "Round $round_index completed: $iter_dir"
  round_index=$((round_index + 1))
done
