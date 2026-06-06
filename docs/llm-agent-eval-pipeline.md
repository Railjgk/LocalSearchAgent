# LLM Agent Eval Pipeline

## 目标

在不改动现有 `src/` agent 框架的前提下，新增一个黑盒评测台：

1. LLM 根据用户部分画像生成短时、一天、两天三类用户需求。
2. 每个 case 附带粗粒度期望结果，只描述语义角色、时间/预算/人群/风险等，不绑定具体 POI。
3. pipeline 调用现有 `run.py` 入口执行 agent。
4. LLM judge 将粗粒度期望和实际输出摘要比较，给出 0-100 分、是否通过、失败类别、改进建议。
5. 聚合报告输出 pass rate、平均分、horizon 分组和失败类别计数，供自动化迭代 agent 使用。

实现入口：`experiments/llm_agent_eval.py`。

## 数据流

```text
profiles.json/jsonl
  -> generate
     -> cases.jsonl
        {
          profile,
          horizon,
          user_request,
          expected: {
            must_satisfy,
            should_satisfy,
            avoid,
            expected_activity_roles,
            time_budget,
            money_budget,
            result_shape,
            success_criteria
          }
        }
  -> run
     -> existing run.build_initial_state + run.run_with_trace
     -> runs.jsonl
        {
          case,
          actual_summary,
          final_state
        }
  -> evaluate
     -> LLM judge / local fallback scorer
     -> report.json + report.md
        {
          summary: {
            pass_rate,
            avg_overall,
            by_horizon,
            failure_category_counts
          },
          evaluations: [...]
        }
```

## Prompt 设计

### Case 生成器

系统 prompt 存在 `CASE_GENERATION_SYSTEM_PROMPT`。核心约束：

- 输入是用户部分画像、目标 horizon、每类 case 数量。
- 输出严格 JSON。
- 需求必须像真实用户自然语言，且包含本次对 agent 可见的画像信息。
- 期望结果不能写死具体 POI、商户名、地址或榜单答案。
- `must_satisfy`、`should_satisfy`、`avoid`、`expected_activity_roles`、`result_shape` 和 `success_criteria` 用于后续量化评测。

### 运行评测器

系统 prompt 存在 `RUN_EVALUATION_SYSTEM_PROMPT`。核心评分维度：

- `intent_fit`
- `constraint_satisfaction`
- `itinerary_shape`
- `personalization`
- `feasibility_execution`
- `explanation_quality`
- `overall`

failure category 固定为可聚合标签，例如 `intent_miss`、`constraint_violation`、`shape_gap`、`personalization_gap`、`execution_gap`、`supply_gap`。

## CLI

离线 smoke test：

```bash
python experiments/llm_agent_eval.py pipeline --limit 1
```

使用 LLM 生成和评测：

```bash
export LONGCAT_API_KEY=...
python experiments/llm_agent_eval.py pipeline --call-llm --cases-per-horizon 2
```

只用 LLM judge，case 生成走本地 fallback：

```bash
python experiments/llm_agent_eval.py pipeline --call-llm-eval-only --limit 6
```

分阶段运行：

```bash
python experiments/llm_agent_eval.py generate --call-llm --cases-out experiments/artifacts/llm_agent_eval/cases.jsonl
python experiments/llm_agent_eval.py run --cases experiments/artifacts/llm_agent_eval/cases.jsonl
python experiments/llm_agent_eval.py evaluate --call-llm --runs experiments/artifacts/llm_agent_eval/runs.jsonl
```

## 自动化改进接口

自动化改进循环可以读取 `report.json`：

- 用 `summary.avg_overall` 和 `summary.pass_rate` 做整体 guardrail。
- 用 `summary.by_horizon` 判断短时/一天/两天哪类退化。
- 用 `summary.failure_category_counts` 定位改进方向。
- 用每个 evaluation 的 `missed_expectations` 和 `improvement_hints` 生成下一轮 planner、RAG、执行层改动候选。

