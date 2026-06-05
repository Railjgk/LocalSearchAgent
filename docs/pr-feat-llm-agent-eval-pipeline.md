# PR Draft: Improve agenteval automation and A/B handoff robustness

Compare URL:

https://github.com/Railjgk/LocalSearchAgent/compare/main...feat/llm-agent-eval-pipeline?expand=1

## Title

Improve agenteval automation and A/B handoff robustness

## Summary

- Adds a repeatable `experiments/run_llm_codex_automation.sh` loop for full LocalSearchAgent agenteval runs, qualitative Codex judging, and focused iterative fixes.
- Adds hard seeded eval cases so automation can continue when strict LongCat case generation fails with quota errors, while avoiding heuristic evaluation fallback.
- Improves A-stage intent and memory handling for current-turn overrides, companion counts, time anchors, service anchors, dietary constraints, accessibility constraints, destination-city handoff, and negated historical preferences.
- Improves B-stage blueprint/candidate behavior around negated roles, existing service anchors, slot alignment, destination-city supply guards, current B RAG updates, and latest city-data routing.
- Adds A-stage handoff documentation in `docs/a-stage-intent-memory-contract.md`.
- Merges latest `origin/dev/b-rag-itinerary-optimization` so the branch includes current B RAG itinerary optimization and city supply data.

## Validation

```bash
python -m compileall -q src experiments/llm_agent_eval.py experiments/run_llm_codex_automation.sh tests
bash -n experiments/run_llm_codex_automation.sh
git diff --check --cached
env PYTHONPATH=. uvx --from pytest --with requests pytest -q \
  tests/test_intent_parser_airport_transfer.py \
  tests/test_intent_parser_round_20260605.py \
  tests/test_intent_parser_people_count.py \
  tests/test_child_exclusion_order.py \
  tests/test_no_child_date_memory_override.py \
  tests/test_b_itinerary_blueprint.py \
  tests/test_b_supported_multinode_itinerary.py \
  tests/test_b_semantics.py \
  tests/test_b_requirement_compiler.py \
  tests/test_city_data_router.py \
  tests/test_ab_destination_city.py \
  tests/test_execution_mock_api.py \
  tests/test_llm_agent_eval.py
```

Observed focused pytest result:

```text
201 passed in 4.49s
```

## Notes

- Full automation was run through round 28 before syncing the latest B branch.
- LongCat generation is currently quota-limited; the automation continues with seed-only cases plus Codex architecture challenge cases and records `llm_generation_outage_seed_only.json`.
- `experiments/llm_agent_eval.py evaluate` was not used for the iterative judge loop.
- The local environment does not include GitHub CLI or a GitHub token, so PR creation must be completed through the compare URL above.
