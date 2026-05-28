# WeekendFlow B Policy Optimizer

This is the safe Route-C path for WeekendFlow B. The planner remains deterministic at runtime; the LLM is only used offline to inspect eval traces and propose small `planner_policy.yaml` mutations.

## Why This Exists

Fixed weights are useful for MVP, but local-life planning differs sharply across family, couple, low-budget, rainy-day, and relaxed solo scenarios. The policy optimizer gives B a controlled improvement loop:

1. Run offline B eval cases.
2. Package failures, selected plans, objective vectors, plan-quality evidence, and explanations.
3. Ask the LLM to propose small policy/config changes.
4. Apply only safe policy paths.
5. Re-run eval with the mutated policy.
6. Accept only if guardrails pass.

The loop must not edit core code, touch A/C contracts, or auto-merge changes.

## Main Entry Point

```powershell
python experiments/gepa_policy_loop.py `
  --cases experiments/b_eval_cases_gaode_v2.jsonl `
  --mock-dir experiments/mock_data/gaode_supply_shanghai_v2_20260527_full `
  --call-llm `
  --label shanghai_policy_probe
```

Without `--call-llm`, the script only writes the reflection package for manual review.

## Safe Mutation Rules

LLM proposals are filtered by `experiments/reflect_and_mutate.py`. Only planner-policy configuration paths are allowed:

- `defaults.*`
- `candidate_generation.*`
- `template_policy.*`
- `scene_weights.*`
- `score_thresholds.*`
- `penalties.*`
- `bonuses.*`
- `preference_mapping.*`
- `hard_constraints.*`
- `alternative_plan_policy.*`
- `explainability_policy.*`
- `offline_eval_targets.*`

Unsafe proposals are skipped and recorded under `mutation_meta.rejected_changes`.

## Guardrails

Mutated policies are rejected if they:

- regress any passing eval case,
- drop pass rate below the configured minimum,
- drop execution-ready rate below the configured minimum,
- reduce average optimization score beyond the allowed threshold,
- reduce protected objective averages such as `group_fit`, `route`, `availability`, `experience`, or `budget` beyond the allowed threshold.

This keeps Route C useful for iteration without letting the optimizer become an uncontrolled code-changing agent.
