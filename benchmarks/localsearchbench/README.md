# LocalSearchBench Local Notes

This directory prepares the LocalSearchBench dataset and records how the local
WeekendFlow project can connect to the benchmark.

## Local Files

- `data/train-00000-of-00001.parquet`: official Hugging Face parquet artifact.
- `data/localsearchbench_train.json`: JSON conversion compatible with the upstream evaluation scripts.
- `data/localsearchbench_train.jsonl`: JSONL conversion for local experiments.
- `data/dataset_summary.json`: schema and split counts.
- `upstream_evaluation/`: copied official evaluation toolkit from `localsearchbench/localsearchbench`.
- `prepare_dataset.py`: reproducible downloader/converter.
- `run_weekendflow_adapter.py`: local compatibility harness that exports WeekendFlow runs as `agent_results` JSON.

## Dataset Facts

- Source: `https://huggingface.co/datasets/localsearchbench/localsearchbench`
- Paper: `https://arxiv.org/abs/2512.07436`
- Official code: `https://github.com/localsearchbench/localsearchbench`
- Rows: 900
- Cities: 北京、上海、广州、深圳、杭州、苏州、成都、重庆、武汉, each with 100 questions.
- Difficulty: 600 L3 and 300 L4.
- Hop count: 540 three-hop, 270 four-hop, 90 five-hop.
- Fields: `Hop Count`, `Difficulty`, `City`, `Question`, `Multi-hop search path`, `Answer`.

## Reproduce Data Prep

The converter needs `pandas` and a parquet engine such as `pyarrow`.

```bash
uv run --with pandas --with pyarrow benchmarks/localsearchbench/prepare_dataset.py
```

## Smoke Run The Local Adapter

```bash
python benchmarks/localsearchbench/run_weekendflow_adapter.py \
  --dataset benchmarks/localsearchbench/data/localsearchbench_train.json \
  --city 上海 \
  --limit 5
```

The output is written under `benchmarks/localsearchbench/results/` in the
official `metadata + results[]` shape. This allows the upstream summarizers and
trajectory judge to parse local outputs.

## Integration Judgment

Direct, leaderboard-equivalent evaluation is not valid yet. LocalSearchBench is
a merchant-scale search benchmark: the official setup expects an agent that can
issue `<rag>...</rag>` and `<web_search>...</web_search>` calls, retrieve from a
1.3M POI index across 9 cities and 6 local-life categories, then answer with
grounded POI evidence. The current local project is a WeekendFlow planner over
mock/local Gaode supply, primarily optimized for activity plus restaurant plans
and downstream execution/payment flows.

The safest integration path is staged:

1. Dataset-level compatibility: done here by converting parquet to JSON/JSONL.
2. Output-level compatibility: done here by exporting WeekendFlow runs as
   upstream-compatible `agent_results` JSON.
3. Retrieval-layer compatibility: still needed. Add a `LocalSearchBenchRAG`
   adapter that exposes `search(query) -> {success, results, context,
   total_results}` with the same contract as upstream `rag_agent.RAGAgent`.
4. Agent-loop compatibility: either adapt WeekendFlow to emit `<rag>`/`<web_search>`
   tool calls, or wrap WeekendFlow as a planner inside the upstream
   `AgentEvaluationPipeline._process_single_question` loop.
5. Metric compatibility: reuse upstream answer judge and trajectory judge only
   after local answers cite POIs returned by the benchmark RAG context. Otherwise
   correctness/faithfulness scores will mostly measure missing retrieval
   coverage rather than planning quality.

Practical first benchmark slice: run Shanghai-only samples against the current
Gaode/mock supply to measure planning-format coverage, not official answer
quality. For official-style scoring, ingest or mount the benchmark merchant
index first, then map local candidate generation to that RAG source.

## Coverage Gap Seen From The Dataset

The benchmark questions are broader than the current local planner surface:

- The 3151 serialized hops include many restaurant queries, but also hotels,
  homestays, convenience stores, specialty shops, parking lots, transit-adjacent
  requests, entertainment, shopping, errands, and mixed chains.
- Current local supply and tests focus on Shanghai WeekendFlow scenarios:
  activity plus restaurant pair selection, constraints, route/availability,
  execution readiness, and payment.
- Therefore, a direct run can be useful as a diagnostic harness, but should be
  reported as `WeekendFlow compatibility smoke`, not as a LocalSearchBench
  leaderboard result.
