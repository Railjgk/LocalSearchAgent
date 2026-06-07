#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ROUNDS="${ROUNDS:-1}"
CASES_PER_HORIZON="${CASES_PER_HORIZON:-1}"
ROUND_TIMEOUT_MINUTES="${ROUND_TIMEOUT_MINUTES:-20}"
PLAN_TIMEOUT_MINUTES="${PLAN_TIMEOUT_MINUTES:-8}"
CRITIQUE_TIMEOUT_MINUTES="${CRITIQUE_TIMEOUT_MINUTES:-8}"
IMPLEMENT_TIMEOUT_MINUTES="${IMPLEMENT_TIMEOUT_MINUTES:-20}"
ADOPTION_TIMEOUT_MINUTES="${ADOPTION_TIMEOUT_MINUTES:-8}"
EVAL_TIMEOUT_SECONDS="${LLM_AGENT_EVAL_TIMEOUT_SECONDS:-180}"
EVAL_RETRIES="${LLM_AGENT_EVAL_RETRIES:-2}"
EVAL_RETRY_BACKOFF_SECONDS="${LLM_AGENT_EVAL_RETRY_BACKOFF_SECONDS:-5}"
GENERATION_ATTEMPTS="${LLM_AGENT_EVAL_GENERATION_ATTEMPTS:-3}"
GENERATION_FAILURE_SLEEP_SECONDS="${GENERATION_FAILURE_SLEEP_SECONDS:-60}"
MAX_POST_FIX_RERUN_SECONDS="${MAX_POST_FIX_RERUN_SECONDS:-180}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-experiments/artifacts/llm_agent_eval/automation}"
AUTOMATION_DEADLINE_AT="${AUTOMATION_DEADLINE_AT:-}"
INCLUDE_HARD_SEED_CASES="${INCLUDE_HARD_SEED_CASES:-1}"
HARD_SEED_CASES_PATH="${HARD_SEED_CASES_PATH:-experiments/eval_seed_cases/hard_realistic_cases.jsonl}"
SEED_ONLY_ON_LLM_GENERATION_FAILURE="${SEED_ONLY_ON_LLM_GENERATION_FAILURE:-1}"
INCLUDE_CODEX_ARCH_CASES="${INCLUDE_CODEX_ARCH_CASES:-1}"
CODEX_ARCH_CASE_COUNT="${CODEX_ARCH_CASE_COUNT:-3}"
CODEX_CASE_GENERATION_TIMEOUT_MINUTES="${CODEX_CASE_GENERATION_TIMEOUT_MINUTES:-8}"
SKIP_B_REFACTOR="${SKIP_B_REFACTOR:-0}"

load_env_file() {
  local env_path="$1"
  eval "$(
    python - "$env_path" <<'PY'
import re
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1])
pattern = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    match = pattern.match(line)
    if not match:
        continue
    key, value = match.groups()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    if key.endswith(("_API_KEY", "_APP_KEY")) and any(ord(ch) > 127 for ch in value):
        continue
    print(f"export {key}={shlex.quote(value)}")
PY
  )"
}

if [[ -f ".env" ]]; then
  load_env_file ".env"
fi

sanitize_sensitive_env() {
  eval "$(
    python - <<'PY'
import os

keys = [
    "LLM_AGENT_EVAL_API_KEY",
    "LONGCAT_API_KEY",
    "LONGCAT_APP_KEY",
    "WF_A_LLM_API_KEY",
    "WF_A_LLM_APP_KEY",
    "GAODE_API_KEY",
    "GAODE_WEATHER_API_KEY",
]
for key in keys:
    value = os.environ.get(key, "")
    if not value:
        continue
    if any(ord(ch) > 127 for ch in value):
        print(f"unset {key}")
PY
  )"
}

sanitize_sensitive_env

has_eval_llm_key() {
  [[ -n "${LLM_AGENT_EVAL_API_KEY:-}${LONGCAT_API_KEY:-}${LONGCAT_APP_KEY:-}" ]]
}

export LLM_AGENT_EVAL_TIMEOUT_SECONDS="$EVAL_TIMEOUT_SECONDS"
export LLM_AGENT_EVAL_RETRIES="$EVAL_RETRIES"
export LLM_AGENT_EVAL_RETRY_BACKOFF_SECONDS="$EVAL_RETRY_BACKOFF_SECONDS"
export LLM_AGENT_EVAL_GENERATION_ATTEMPTS="$GENERATION_ATTEMPTS"

mkdir -p "$ARTIFACT_ROOT"

AUTOMATION_DEADLINE_EPOCH=""
if [[ -n "$AUTOMATION_DEADLINE_AT" ]]; then
  AUTOMATION_DEADLINE_EPOCH="$(date -d "$AUTOMATION_DEADLINE_AT" +%s)"
fi

seconds_until_deadline() {
  if [[ -z "$AUTOMATION_DEADLINE_EPOCH" ]]; then
    echo 0
    return
  fi
  local now
  now="$(date +%s)"
  echo $((AUTOMATION_DEADLINE_EPOCH - now))
}

deadline_reached() {
  if [[ -z "$AUTOMATION_DEADLINE_EPOCH" ]]; then
    return 1
  fi
  [[ "$(seconds_until_deadline)" -le 0 ]]
}

phase_timeout_seconds() {
  local max_minutes="$1"
  local max_seconds=$((max_minutes * 60))
  if [[ -z "$AUTOMATION_DEADLINE_EPOCH" ]]; then
    echo "$max_seconds"
    return
  fi
  local remaining
  remaining="$(seconds_until_deadline)"
  if [[ "$remaining" -le 0 ]]; then
    echo 0
  elif [[ "$remaining" -lt "$max_seconds" ]]; then
    echo "$remaining"
  else
    echo "$max_seconds"
  fi
}

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

run_codex_phase() {
  local prompt_path="$1"
  local log_path="$2"
  local last_message_path="$3"
  local timeout_minutes="$4"
  local timeout_seconds
  timeout_seconds="$(phase_timeout_seconds "$timeout_minutes")"
  if [[ "$timeout_seconds" -le 0 ]]; then
    echo "deadline reached before codex phase: $prompt_path" >"$log_path"
    return 124
  fi

  timeout --kill-after=60s "${timeout_seconds}s" \
    codex \
      --ask-for-approval never \
      exec \
      --cd "$ROOT_DIR" \
      --sandbox danger-full-access \
      --output-last-message "$last_message_path" \
      - <"$prompt_path" >"$log_path" 2>&1
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

  run_codex_phase \
    "$prompt_path" \
    "$iter_dir/codex_architecture_case_exec.log" \
    "$iter_dir/codex_architecture_case_last_message.md" \
    "$CODEX_CASE_GENERATION_TIMEOUT_MINUTES"
}

write_compact_run_summary() {
  local cases_path="$1"
  local runs_path="$2"
  local out_json="$3"
  local out_md="$4"
  python experiments/llm_agent_eval.py compact-summary \
    --cases "$cases_path" \
    --runs "$runs_path" \
    --summary-out "$out_json" \
    --markdown-out "$out_md"
}

collect_previous_round_feedback() {
  local iter_dir="$1"
  python - "$iter_dir" <<'PY'
import json
import sys
from pathlib import Path

current = Path(sys.argv[1])
root = current.parent
round_dirs = [
    path for path in root.glob("round_*")
    if path.is_dir() and path.name != current.name
]
round_dirs.sort(key=lambda path: path.name)
items = []
for path in round_dirs[-4:]:
    item = {"round_dir": str(path)}
    for name in (
        "status.json",
        "candidate_diff_guard.json",
        "adoption_decision.json",
    ):
        artifact = path / name
        if not artifact.exists():
            continue
        try:
            item[name] = json.loads(artifact.read_text(encoding="utf-8"))
        except Exception as exc:
            item[name] = {"unreadable": str(exc)}
    if len(item) > 1:
        items.append(item)

if not items:
    print("No previous completed round feedback is available.")
else:
    print(json.dumps(items, ensure_ascii=False, indent=2))
PY
}

write_refactor_plan_prompt() {
  local iter_dir="$1"
  local prompt_path="$2"
  local previous_feedback
  previous_feedback="$(collect_previous_round_feedback "$iter_dir")"
  local scope_constraints=""
  if [[ "$SKIP_B_REFACTOR" != "0" ]]; then
    scope_constraints=$(cat <<'SCOPE'
- Current scope mode: non-B backend refactor only.
- Do not propose edits to B-stage or B-consumer modules already handled in prior rounds:
  `src/nodes/b_itinerary_blueprint.py`, `src/nodes/b_poi_rag.py`,
  `src/nodes/candidate_generator.py`, `src/nodes/constraint_filter.py`,
  `src/nodes/plan_optimizer.py`, `src/nodes/b_replan_loop.py`,
  `src/nodes/explainability.py`, `src/nodes/share_generator.py`,
  and their B-focused tests.
- Prefer A-stage intent/memory/scenario planning, graph/orchestration, tool routing,
  execution manager, payment layer, eval/report automation, or backend test hygiene.
- Do not change frontend interfaces, frontend files, mock data, or C mock API fields.
SCOPE
)
  fi
  cat >"$prompt_path" <<EOF
You are Planner Agent A for an automated LocalSearchAgent refactor round.

Goal:
- Generate one small, engineering-grade refactor plan from the observed eval run.
- Do not modify repository source code. Only write artifacts inside \`$iter_dir\`.

Hard requirements:
- Do not use heuristic or rule-based evaluation. Do not call \`experiments/llm_agent_eval.py evaluate\`.
- This is end-to-end LocalSearchAgent optimization, not the standalone B policy optimization loop.
- The plan must be small enough for one independent implementer to finish in one bounded round.
- Keep B-stage Chinese semantics as the primary flow. English canonical values may only be compatibility indexes.
- Do not propose scoring-weight changes unless the current eval artifacts show concrete evidence that scoring weights are the defect.
- Do not propose edits to mock data, frontend files, or C mock API fields.
- Do not design LongCat critique as direct plan mutation. Critique must become a bounded replan request controlled by the automation.
- Probe-training architecture is protected. Do not alter probe model architecture, training loop, loss/objective implementation, layer-selection mechanics, or probe serving/evaluation architecture. You may add, remove, or generate value training items, value categories, labels, prompts, or data files if a probe-related issue is directly relevant.
$scope_constraints

Inputs:
- cases: \`$iter_dir/cases.jsonl\`
- pre-fix runs: \`$iter_dir/runs.jsonl\`
- compact summary JSON: \`$iter_dir/compact_run_summary.json\`
- compact summary Markdown: \`$iter_dir/compact_run_summary.md\`
- generation log: \`$iter_dir/generate.log\`
- run log: \`$iter_dir/run.log\`
- recent previous round feedback, including guard/adoption rejections:
\`\`\`json
$previous_feedback
\`\`\`

Use previous feedback to avoid repeating rejected plans. In particular, do not
propose a fallback that exposes rejected/non-executable POIs through
recommendation-style explanation text, execution-ready semantics, or "already
booked" wording. A rejected candidate may only become bounded evidence for a
safe replan request unless the post-fix run proves it is executable and truthful.

Required outputs:
- \`$iter_dir/refactor_plan.md\`: concise diagnosis and proposed refactor.
- \`$iter_dir/refactor_plan.json\`: strict JSON with this schema:
  {
    "plan_id": "round-local stable id",
    "diagnosis": [{"case_id": "...", "component": "intent|memory|planner|execution|explanation", "issue": "...", "evidence": "..."}],
    "refactor_goal": "...",
    "owned_paths": ["relative/source/path.py"],
    "forbidden_paths_acknowledged": true,
    "implementation_steps": ["small step 1", "small step 2"],
    "validation_plan": ["targeted command or eval rerun"],
    "acceptance_signals": ["observable pre/post improvement"],
    "risk_notes": ["..."],
    "no_goals": ["..."]
  }

Suggested evaluation focus:
- Whether intent horizon and constraints survive into planning.
- Whether memory helps or pollutes personalization.
- Whether planner output matches short/one_day/two_day shape.
- Whether execution recovers from tool failures.
- Whether user-facing final output is present and faithful.

Finish with a concise final message that names the plan id and owned paths.
EOF
}

write_plan_critique_prompt() {
  local iter_dir="$1"
  local prompt_path="$2"
  local scope_constraints=""
  if [[ "$SKIP_B_REFACTOR" != "0" ]]; then
    scope_constraints=$(cat <<'SCOPE'
- Current scope mode: non-B backend refactor only.
- Reject or constrain plans that edit B-stage/B-consumer files:
  `src/nodes/b_itinerary_blueprint.py`, `src/nodes/b_poi_rag.py`,
  `src/nodes/candidate_generator.py`, `src/nodes/constraint_filter.py`,
  `src/nodes/plan_optimizer.py`, `src/nodes/b_replan_loop.py`,
  `src/nodes/explainability.py`, `src/nodes/share_generator.py`,
  and B-focused tests.
- Keep review focused on A-stage, orchestration, tool routing, execution manager,
  payment layer, eval/report automation, or backend test hygiene.
SCOPE
)
  fi
  cat >"$prompt_path" <<EOF
You are Critic Agent B for an automated LocalSearchAgent refactor round.

Goal:
- Evaluate Planner Agent A's plan. Do not modify source code and do not edit the plan file.
- Produce critique only; the automation controller will convert it into a bounded replan request.

Hard requirements:
- Do not use heuristic or rule-based evaluation. Do not call \`experiments/llm_agent_eval.py evaluate\`.
- Do not modify repository source code. Only write artifacts inside \`$iter_dir\`.
- Keep Chinese semantics primary in B-stage; English canonical values are compatibility indexes only.
- Reject or constrain any proposal to change scoring weights unless eval evidence specifically points to scoring weights.
- Reject or constrain edits to mock data, frontend files, or C mock API fields.
- Critique must not directly rewrite the plan. Provide bounded requests and constraints only.
$scope_constraints

Inputs:
- plan markdown: \`$iter_dir/refactor_plan.md\`
- plan JSON: \`$iter_dir/refactor_plan.json\`
- compact summary JSON: \`$iter_dir/compact_run_summary.json\`
- compact summary Markdown: \`$iter_dir/compact_run_summary.md\`
- cases: \`$iter_dir/cases.jsonl\`
- pre-fix runs: \`$iter_dir/runs.jsonl\`

Required outputs:
- \`$iter_dir/codex_judge_report.md\`: case-by-case diagnostic evaluation and critique of the plan.
- \`$iter_dir/codex_judge_report.json\`: strict JSON with:
  {
    "decision": "approve|request_bounded_replan|reject",
    "top_failure_clusters": ["..."],
    "plan_strengths": ["..."],
    "plan_risks": ["..."],
    "bounded_replan_constraints": ["..."],
    "allowed_focus": ["..."],
    "blocked_changes": ["..."],
    "must_validate": ["..."]
  }

Finish with a concise final message naming the decision.
EOF
}

write_bounded_replan_request() {
  local iter_dir="$1"
  python - "$iter_dir/refactor_plan.json" "$iter_dir/codex_judge_report.json" "$iter_dir/bounded_replan_request.json" <<'PY'
import json
import sys
from pathlib import Path

plan_path, critique_path, out_path = [Path(arg) for arg in sys.argv[1:]]

def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}

plan = load_json(plan_path)
critique = load_json(critique_path)
owned_paths = [str(path) for path in plan.get("owned_paths", []) if isinstance(path, str)]
blocked = [
    "experiments/mock_data/**",
    "frontend/**",
    "web/**",
    "src/nodes/mock_api_layer.py",
    "src/tools/execution_mock_api.py",
    "C mock API field/schema changes",
    "scoring weight changes without explicit eval evidence",
    "Chinese-to-English canonical main-flow rewrites",
]
skip_b = bool(int(__import__("os").environ.get("SKIP_B_REFACTOR", "0") or "0"))
if skip_b:
    blocked.extend([
        "B-stage/B-consumer files while SKIP_B_REFACTOR=1",
        "src/nodes/b_itinerary_blueprint.py",
        "src/nodes/b_poi_rag.py",
        "src/nodes/candidate_generator.py",
        "src/nodes/constraint_filter.py",
        "src/nodes/plan_optimizer.py",
        "src/nodes/b_replan_loop.py",
        "src/nodes/explainability.py",
        "src/nodes/share_generator.py",
        "B-focused tests while SKIP_B_REFACTOR=1",
    ])
request = {
    "source_plan_id": plan.get("plan_id"),
    "critic_decision": critique.get("decision", "request_bounded_replan"),
    "refactor_goal": plan.get("refactor_goal"),
    "allowed_owned_paths": owned_paths[:6],
    "implementation_steps": plan.get("implementation_steps", [])[:5],
    "critic_constraints": critique.get("bounded_replan_constraints", [])[:8],
    "allowed_focus": critique.get("allowed_focus", [])[:6],
    "blocked_changes": blocked + [str(item) for item in critique.get("blocked_changes", []) if isinstance(item, str)],
    "must_validate": list(dict.fromkeys(
        [str(item) for item in plan.get("validation_plan", []) if isinstance(item, str)]
        + [str(item) for item in critique.get("must_validate", []) if isinstance(item, str)]
    ))[:8],
    "controller_rules": [
        "Implement one focused refactor only.",
        "Do not edit files outside allowed_owned_paths unless required for a directly related test.",
        "Do not change mock data, frontend files, C mock API field/schema files, or scoring weights.",
        "Preserve Chinese-first B-stage semantics.",
        "When SKIP_B_REFACTOR=1, do not edit B-stage/B-consumer modules or B-focused tests.",
        "Run targeted validation and let controller run post-fix eval before adoption.",
    ],
}
out_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"bounded_replan_request": str(out_path), "allowed_owned_paths": request["allowed_owned_paths"]}, ensure_ascii=False))
PY
}

write_refactor_implementation_prompt() {
  local iter_dir="$1"
  local prompt_path="$2"
  local scope_constraints=""
  if [[ "$SKIP_B_REFACTOR" != "0" ]]; then
    scope_constraints=$(cat <<'SCOPE'
- Current scope mode: non-B backend refactor only.
- Do not edit B-stage/B-consumer modules or B-focused tests:
  `src/nodes/b_itinerary_blueprint.py`, `src/nodes/b_poi_rag.py`,
  `src/nodes/candidate_generator.py`, `src/nodes/constraint_filter.py`,
  `src/nodes/plan_optimizer.py`, `src/nodes/b_replan_loop.py`,
  `src/nodes/explainability.py`, `src/nodes/share_generator.py`,
  `tests/test_b_*`, `tests/test_multinode_slot_alignment.py`,
  `tests/test_share_generator_non_executable.py`.
- Implement only A-stage, orchestration, tool routing, execution/payment, eval/report
  automation, or backend test hygiene changes supported by the bounded request.
SCOPE
)
  fi
  cat >"$prompt_path" <<EOF
You are independent Implementer Agent C for an automated LocalSearchAgent refactor round.

Goal:
- Implement one small, focused, engineering-grade refactor from the bounded request.
- You are independent from Planner Agent A and Critic Agent B. Use their artifacts as inputs, but verify with the repository.

Hard requirements:
- Make source changes only if there is a clear defect and bounded request supports it.
- Preserve unrelated user changes. Do not revert dirty files you did not create in this implementation phase.
- Keep Chinese semantics primary in B-stage. English canonical values may only be compatibility indexes.
- Do not change scoring weights unless the bounded request explicitly includes eval evidence for scoring weights.
- Do not edit mock data, frontend files, \`src/nodes/mock_api_layer.py\`, \`src/tools/execution_mock_api.py\`, or C mock API fields/schemas.
- Do not call \`experiments/llm_agent_eval.py evaluate\`.
- Do not run unbounded post-fix reruns; the controller runs the adoption eval.
- Keep the change small. Do not attempt to fix every issue in the report.
$scope_constraints

Inputs:
- bounded request: \`$iter_dir/bounded_replan_request.json\`
- plan markdown: \`$iter_dir/refactor_plan.md\`
- plan JSON: \`$iter_dir/refactor_plan.json\`
- critic report: \`$iter_dir/codex_judge_report.md\`
- critic JSON: \`$iter_dir/codex_judge_report.json\`
- compact summary: \`$iter_dir/compact_run_summary.md\`

Required outputs:
- \`$iter_dir/optimization_summary.md\`: what changed, why, touched files, validation run, and residual risks.
- \`$iter_dir/optimization_summary.json\`: strict JSON with:
  {
    "changed": true,
    "touched_files": ["..."],
    "validation": [{"command": "...", "status": "pass|fail|not_run", "notes": "..."}],
    "expected_eval_signal": "...",
    "residual_risks": ["..."]
  }

Finish with a concise final message summarizing whether code changed.
EOF
}

write_adoption_prompt() {
  local iter_dir="$1"
  local prompt_path="$2"
  local scope_constraints=""
  if [[ "$SKIP_B_REFACTOR" != "0" ]]; then
    scope_constraints=$(cat <<'SCOPE'
- Current scope mode: non-B backend refactor only.
- Reject if candidate files include B-stage/B-consumer modules or B-focused tests while
  `SKIP_B_REFACTOR=1`.
SCOPE
)
  fi
  cat >"$prompt_path" <<EOF
You are Adoption Judge Agent D for an automated LocalSearchAgent refactor round.

Goal:
- Decide whether to adopt the implementer's candidate changes after post-fix eval run.
- Do not modify repository source code. Only write artifacts inside \`$iter_dir\`.

Hard requirements:
- Do not call \`experiments/llm_agent_eval.py evaluate\`.
- Use the pre-fix and post-fix agent runs qualitatively. Prefer adoption only when the candidate is coherent, guarded, and not worse on the generated cases.
- Reject changes that modify mock data, frontend files, C mock API fields/schemas, scoring weights without eval evidence, or Chinese-first B-stage semantics.
- Reject if post-fix run failed, timed out without useful evidence, or candidate touched protected paths.
$scope_constraints

Inputs:
- plan: \`$iter_dir/refactor_plan.json\`
- critique: \`$iter_dir/codex_judge_report.json\`
- implementation summary: \`$iter_dir/optimization_summary.json\`
- candidate diff guard: \`$iter_dir/candidate_diff_guard.json\`
- pre-fix summary: \`$iter_dir/compact_run_summary.json\`
- post-fix summary: \`$iter_dir/post_fix_compact_run_summary.json\`
- post-fix run log: \`$iter_dir/post_fix_run.log\`

Required outputs:
- \`$iter_dir/adoption_decision.md\`: concise evidence and decision.
- \`$iter_dir/adoption_decision.json\`: strict JSON with:
  {
    "adopt": true,
    "decision": "adopt|reject",
    "reason": "...",
    "evidence": ["..."],
    "regressions": ["..."],
    "next_round_focus": ["..."]
  }

Finish with a concise final message naming the decision.
EOF
}

save_impl_baseline() {
  local iter_dir="$1"
  git diff --binary >"$iter_dir/pre_impl_worktree.patch"
  git diff --name-only | sort >"$iter_dir/pre_impl_changed_files.txt"
  git ls-files --others --exclude-standard | sort >"$iter_dir/pre_impl_untracked_files.txt"
}

restore_impl_baseline() {
  local iter_dir="$1"
  git diff --binary >"$iter_dir/rejected_candidate_worktree.patch"
  git ls-files --others --exclude-standard | sort >"$iter_dir/post_impl_untracked_files.txt"

  if [[ -s "$iter_dir/rejected_candidate_worktree.patch" ]]; then
    git apply -R "$iter_dir/rejected_candidate_worktree.patch" || return $?
  fi
  if [[ -s "$iter_dir/pre_impl_worktree.patch" ]]; then
    git apply "$iter_dir/pre_impl_worktree.patch" || return $?
  fi

  python - "$iter_dir/pre_impl_untracked_files.txt" "$iter_dir/post_impl_untracked_files.txt" "$iter_dir" <<'PY'
import os
import shutil
import sys
from pathlib import Path

pre_path, post_path, iter_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
pre = set(pre_path.read_text(encoding="utf-8").splitlines()) if pre_path.exists() else set()
post = set(post_path.read_text(encoding="utf-8").splitlines()) if post_path.exists() else set()
iter_prefix = str(iter_dir).rstrip("/") + "/"
for rel in sorted(post - pre, reverse=True):
    if not rel or rel.startswith(iter_prefix):
        continue
    path = Path(rel)
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()
PY
}

guard_candidate_diff() {
  local iter_dir="$1"
  git diff --name-only | sort >"$iter_dir/post_impl_changed_files.txt"
  git ls-files --others --exclude-standard | sort >"$iter_dir/post_impl_untracked_files.txt"
  git diff -- . ':(exclude)experiments/artifacts/llm_agent_eval/automation/**' >"$iter_dir/candidate_full_diff.patch" || true

  python - \
    "$iter_dir/pre_impl_changed_files.txt" \
    "$iter_dir/post_impl_changed_files.txt" \
    "$iter_dir/pre_impl_untracked_files.txt" \
    "$iter_dir/post_impl_untracked_files.txt" \
    "$iter_dir/pre_impl_worktree.patch" \
    "$iter_dir/candidate_full_diff.patch" \
    "$iter_dir/candidate_changed_files.txt" \
    "$iter_dir/candidate_diff_guard.json" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

(
    pre_path,
    post_path,
    pre_untracked_path,
    post_untracked_path,
    pre_diff_path,
    post_diff_path,
    changed_out_path,
    out_path,
) = [Path(arg) for arg in sys.argv[1:]]
pre = set(pre_path.read_text(encoding="utf-8").splitlines()) if pre_path.exists() else set()
post = set(post_path.read_text(encoding="utf-8").splitlines()) if post_path.exists() else set()
pre_untracked = set(pre_untracked_path.read_text(encoding="utf-8").splitlines()) if pre_untracked_path.exists() else set()
post_untracked = set(post_untracked_path.read_text(encoding="utf-8").splitlines()) if post_untracked_path.exists() else set()


def diff_blocks(text):
    blocks = {}
    current = None
    lines = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            if current is not None:
                blocks[current] = "\n".join(lines)
            parts = line.split()
            current = parts[-1][2:] if len(parts) >= 4 and parts[-1].startswith("b/") else None
            lines = [line] if current else []
            continue
        if current is not None:
            lines.append(line)
    if current is not None:
        blocks[current] = "\n".join(lines)
    return blocks


pre_diff = diff_blocks(pre_diff_path.read_text(encoding="utf-8", errors="replace") if pre_diff_path.exists() else "")
post_diff = diff_blocks(post_diff_path.read_text(encoding="utf-8", errors="replace") if post_diff_path.exists() else "")
candidate_files = sorted(
    {path for path in post if post_diff.get(path, "") != pre_diff.get(path, "")}
    | (post - pre)
    | (post_untracked - pre_untracked)
)
changed_out_path.write_text("\n".join(candidate_files) + ("\n" if candidate_files else ""), encoding="utf-8")

protected_prefixes = (
    "experiments/mock_data/",
    "frontend/",
    "web/",
    "src/frontend/",
)
protected_suffixes = (
    ".tsx",
    ".jsx",
    ".css",
    ".scss",
)
protected_exact = {
    "src/nodes/mock_api_layer.py",
    "src/tools/execution_mock_api.py",
}
if os.environ.get("SKIP_B_REFACTOR", "0") != "0":
    protected_prefixes = protected_prefixes + (
        "tests/test_b_",
    )
    protected_exact.update(
        {
            "src/nodes/b_itinerary_blueprint.py",
            "src/nodes/b_poi_rag.py",
            "src/nodes/candidate_generator.py",
            "src/nodes/constraint_filter.py",
            "src/nodes/plan_optimizer.py",
            "src/nodes/b_replan_loop.py",
            "src/nodes/explainability.py",
            "src/nodes/share_generator.py",
            "tests/test_multinode_slot_alignment.py",
            "tests/test_share_generator_non_executable.py",
        }
    )
ignored_files = {
    "experiments/run_llm_codex_automation.sh",
}
violations = []
for path in candidate_files:
    if path in ignored_files:
        continue
    if path in protected_exact or path.startswith(protected_prefixes) or path.endswith(protected_suffixes):
        violations.append({"path": path, "reason": "protected path"})

diff_text = post_diff_path.read_text(encoding="utf-8", errors="replace") if post_diff_path.exists() else ""
current_file = None
score_weight_hits = []
for line in diff_text.splitlines():
    if line.startswith("diff --git "):
        parts = line.split()
        current_file = parts[-1][2:] if len(parts) >= 4 and parts[-1].startswith("b/") else None
        continue
    if not current_file or current_file in ignored_files or current_file not in candidate_files:
        continue
    if line.startswith(("+++", "---")) or not line.startswith(("+", "-")):
        continue
    text = line[1:].strip()
    if re.match(r"""["']weights["']\s*:\s*\{\s*\}\s*,?$""", text):
        continue
    if re.search(r"\b(score_weights?|scoring_weights?|SCORING_WEIGHTS|WEIGHT_CONFIG|SCORING_CONFIG)\b", text):
        score_weight_hits.append({"path": current_file, "line": line[:240]})
        continue
    if re.search(r"\bweights?\b", text) and re.search(r"\b(score|scoring|rank|ranking|penalt|bonus)\b", text):
        score_weight_hits.append({"path": current_file, "line": line[:240]})

if score_weight_hits:
    violations.append({"path": "scoring", "reason": "candidate changed scoring-weight-like lines", "hits": score_weight_hits[:10]})

payload = {
    "candidate_files": candidate_files,
    "protected_violations": violations,
    "passed": not violations,
}
out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
if violations:
    raise SystemExit(json.dumps(payload, ensure_ascii=False))
print(json.dumps(payload, ensure_ascii=False))
PY
}

adoption_is_accepted() {
  local iter_dir="$1"
  python - "$iter_dir/adoption_decision.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"cannot read adoption decision: {exc}")
decision = str(data.get("decision", "")).lower()
adopt = bool(data.get("adopt")) or decision == "adopt"
raise SystemExit(0 if adopt else 1)
PY
}

write_round_status() {
  local iter_dir="$1"
  local round_index="$2"
  local status="$3"
  local phase="$4"
  local exit_status="$5"
  cat >"$iter_dir/status.json" <<EOF
{
  "round_index": $round_index,
  "finished_at": "$(date -Iseconds)",
  "status": "$status",
  "phase": "$phase",
  "exit_status": $exit_status,
  "continue_next_round": true
}
EOF
}

round_index=1
while [[ "$ROUNDS" == "0" || "$round_index" -le "$ROUNDS" ]]; do
  if deadline_reached; then
    echo "Automation deadline reached at $(date -Iseconds); stopping before round $round_index."
    break
  fi

  stamp="$(date +%Y%m%d_%H%M%S)"
  iter_dir="$ARTIFACT_ROOT/round_${stamp}"
  mkdir -p "$iter_dir"

  cat >"$iter_dir/metadata.json" <<EOF
{
  "round_index": $round_index,
  "started_at": "$(date -Iseconds)",
  "cases_per_horizon": $CASES_PER_HORIZON,
  "round_timeout_minutes": $ROUND_TIMEOUT_MINUTES,
  "plan_timeout_minutes": $PLAN_TIMEOUT_MINUTES,
  "critique_timeout_minutes": $CRITIQUE_TIMEOUT_MINUTES,
  "implement_timeout_minutes": $IMPLEMENT_TIMEOUT_MINUTES,
  "adoption_timeout_minutes": $ADOPTION_TIMEOUT_MINUTES,
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
  "skip_b_refactor": "$SKIP_B_REFACTOR",
  "max_post_fix_rerun_seconds": $MAX_POST_FIX_RERUN_SECONDS,
  "automation_deadline_at": "$AUTOMATION_DEADLINE_AT",
  "probe_training_architecture_protected": true,
  "planner": "codex_exec",
  "critic": "codex_exec",
  "implementer": "codex_exec",
  "adoption_judge": "codex_exec",
  "heuristic_eval_allowed": false
}
EOF

  generation_status=0
  if has_eval_llm_key; then
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
    run_logged "$iter_dir/generate.log" "${generation_cmd[@]}" || generation_status=$?
  else
    generation_status=90
    {
      printf '## command\n'
      printf 'skip LLM generation because no usable eval LLM API key is configured\n\n'
      printf '## output\n'
      printf 'Using seed-only generation path when available.\n'
    } >"$iter_dir/generate.log"
  fi

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

  plan_prompt_path="$iter_dir/refactor_plan_prompt.md"
  write_refactor_plan_prompt "$iter_dir" "$plan_prompt_path"
  plan_status=0
  run_codex_phase \
    "$plan_prompt_path" \
    "$iter_dir/refactor_plan_exec.log" \
    "$iter_dir/refactor_plan_last_message.md" \
    "$PLAN_TIMEOUT_MINUTES" || plan_status=$?
  if [[ "$plan_status" != "0" || ! -s "$iter_dir/refactor_plan.json" ]]; then
    write_round_status "$iter_dir" "$round_index" "failed" "refactor_plan" "$plan_status"
    echo "Round $round_index planner failed with status=$plan_status. See $iter_dir/refactor_plan_exec.log"
    round_index=$((round_index + 1))
    continue
  fi

  critique_prompt_path="$iter_dir/plan_critique_prompt.md"
  write_plan_critique_prompt "$iter_dir" "$critique_prompt_path"
  critique_status=0
  run_codex_phase \
    "$critique_prompt_path" \
    "$iter_dir/plan_critique_exec.log" \
    "$iter_dir/plan_critique_last_message.md" \
    "$CRITIQUE_TIMEOUT_MINUTES" || critique_status=$?
  if [[ "$critique_status" != "0" || ! -s "$iter_dir/codex_judge_report.json" ]]; then
    write_round_status "$iter_dir" "$round_index" "failed" "plan_critique" "$critique_status"
    echo "Round $round_index critic failed with status=$critique_status. See $iter_dir/plan_critique_exec.log"
    round_index=$((round_index + 1))
    continue
  fi

  bounded_status=0
  run_logged "$iter_dir/bounded_replan_request.log" write_bounded_replan_request "$iter_dir" || bounded_status=$?
  if [[ "$bounded_status" != "0" || ! -s "$iter_dir/bounded_replan_request.json" ]]; then
    write_round_status "$iter_dir" "$round_index" "failed" "bounded_replan_request" "$bounded_status"
    echo "Round $round_index failed to build bounded replan request. See $iter_dir/bounded_replan_request.log"
    round_index=$((round_index + 1))
    continue
  fi

  save_impl_baseline "$iter_dir"

  implement_prompt_path="$iter_dir/refactor_implementation_prompt.md"
  write_refactor_implementation_prompt "$iter_dir" "$implement_prompt_path"
  implement_status=0
  run_codex_phase \
    "$implement_prompt_path" \
    "$iter_dir/refactor_implementation_exec.log" \
    "$iter_dir/refactor_implementation_last_message.md" \
    "$IMPLEMENT_TIMEOUT_MINUTES" || implement_status=$?
  if [[ "$implement_status" != "0" ]]; then
    restore_status=0
    restore_impl_baseline "$iter_dir" || restore_status=$?
    write_round_status "$iter_dir" "$round_index" "rejected" "implementation" "$implement_status"
    echo "Round $round_index implementation failed with status=$implement_status; restore_status=$restore_status. See $iter_dir/refactor_implementation_exec.log"
    round_index=$((round_index + 1))
    continue
  fi

  guard_status=0
  guard_candidate_diff "$iter_dir" >"$iter_dir/candidate_diff_guard.log" 2>&1 || guard_status=$?
  if [[ "$guard_status" != "0" ]]; then
    restore_status=0
    restore_impl_baseline "$iter_dir" || restore_status=$?
    write_round_status "$iter_dir" "$round_index" "rejected" "candidate_diff_guard" "$guard_status"
    echo "Round $round_index candidate rejected by diff guard; restore_status=$restore_status. See $iter_dir/candidate_diff_guard.json"
    round_index=$((round_index + 1))
    continue
  fi

  post_fix_status=0
  post_fix_timeout="$MAX_POST_FIX_RERUN_SECONDS"
  if [[ -n "$AUTOMATION_DEADLINE_EPOCH" ]]; then
    remaining_seconds="$(seconds_until_deadline)"
    if [[ "$remaining_seconds" -le 0 ]]; then
      post_fix_timeout=0
    elif [[ "$remaining_seconds" -lt "$post_fix_timeout" ]]; then
      post_fix_timeout="$remaining_seconds"
    fi
  fi
  if [[ "$post_fix_timeout" -le 0 ]]; then
    restore_status=0
    restore_impl_baseline "$iter_dir" || restore_status=$?
    write_round_status "$iter_dir" "$round_index" "rejected" "post_fix_eval_deadline" 124
    echo "Round $round_index reached deadline before post-fix eval; restore_status=$restore_status."
    break
  fi
  run_logged "$iter_dir/post_fix_run.log" \
    timeout --kill-after=30s "${post_fix_timeout}s" \
      python experiments/llm_agent_eval.py run \
        --cases "$iter_dir/cases.jsonl" \
        --runs-out "$iter_dir/post_fix_runs.jsonl" || post_fix_status=$?
  if [[ "$post_fix_status" != "0" || ! -s "$iter_dir/post_fix_runs.jsonl" ]]; then
    restore_status=0
    restore_impl_baseline "$iter_dir" || restore_status=$?
    write_round_status "$iter_dir" "$round_index" "rejected" "post_fix_eval_run" "$post_fix_status"
    echo "Round $round_index post-fix eval run failed with status=$post_fix_status; restore_status=$restore_status. See $iter_dir/post_fix_run.log"
    round_index=$((round_index + 1))
    continue
  fi

  write_compact_run_summary \
    "$iter_dir/cases.jsonl" \
    "$iter_dir/post_fix_runs.jsonl" \
    "$iter_dir/post_fix_compact_run_summary.json" \
    "$iter_dir/post_fix_compact_run_summary.md"

  adoption_prompt_path="$iter_dir/adoption_prompt.md"
  write_adoption_prompt "$iter_dir" "$adoption_prompt_path"
  adoption_status=0
  run_codex_phase \
    "$adoption_prompt_path" \
    "$iter_dir/adoption_exec.log" \
    "$iter_dir/adoption_last_message.md" \
    "$ADOPTION_TIMEOUT_MINUTES" || adoption_status=$?
  if [[ "$adoption_status" != "0" || ! -s "$iter_dir/adoption_decision.json" ]]; then
    restore_status=0
    restore_impl_baseline "$iter_dir" || restore_status=$?
    write_round_status "$iter_dir" "$round_index" "rejected" "adoption_judge" "$adoption_status"
    echo "Round $round_index adoption judge failed with status=$adoption_status; restore_status=$restore_status. See $iter_dir/adoption_exec.log"
    round_index=$((round_index + 1))
    continue
  fi

  if adoption_is_accepted "$iter_dir"; then
    write_round_status "$iter_dir" "$round_index" "accepted" "adoption_judge" 0
    echo "Round $round_index accepted candidate changes: $iter_dir"
  else
    restore_status=0
    restore_impl_baseline "$iter_dir" || restore_status=$?
    write_round_status "$iter_dir" "$round_index" "rejected" "adoption_judge" 1
    echo "Round $round_index rejected candidate changes; restore_status=$restore_status. See $iter_dir/adoption_decision.json"
  fi

  round_index=$((round_index + 1))
done
