# 本地生活规划 Benchmark 设计

## 范围

这个 benchmark 面向 WeekendFlow/LocalSearchAgent 的本地生活规划能力，不依赖官方 LocalSearchBench。它评估的是完整规划代理在真实用户式请求下的端到端表现，以及中间组件是否给出了可解释、可调试的证据。

核心能力范围：

- intent：识别任务类型、人群、时间、地点、预算、风险和缺失槽位。
- memory：检索并使用短期记忆、长期画像、同行人画像和 value memory。
- planner：生成候选、过滤约束、选择 POI/路线/时间/预算合理的计划，并给出备选和 tradeoff。
- execution：生成 action sequence，完成预约、下单、路线/库存/券等工具闭环。
- explanation：把选择理由、约束满足情况和执行结果讲清楚。

## 数据形态

每条 benchmark case 使用 JSONL。推荐 schema：

```json
{
  "case_id": "llmcase_profile_horizon_hash",
  "schema_version": "local_life_planning_bench.v1",
  "split": "dev|regression|heldout",
  "city": "上海",
  "horizon": "short|one_day|two_day",
  "profile_id": "family_health_budget_shanghai",
  "profile": {},
  "user_request": "自然语言用户需求",
  "expected": {
    "intent_summary": "语义级真实需求",
    "must_satisfy": [],
    "should_satisfy": [],
    "avoid": [],
    "poi_reference": [],
    "result_shape": {},
    "component_expectations": {
      "intent": [],
      "memory": [],
      "planner": [],
      "execution": [],
      "explanation": []
    },
    "success_criteria": []
  },
  "difficulty_tags": [],
  "generation_metadata": {
    "generator": "llm|human|fallback",
    "model": "",
    "created_at": ""
  }
}
```

`expected` 仍然不绑定具体 POI。`poi_reference` 用宽泛语义校对实际 POI，例如 `儿童友好`、`轻食/低卡`、`少换乘`、`可预约`、`低排队风险`。`component_expectations` 只作为诊断期望，不强制替代整体端到端评分。

## Split 策略

- `dev`：快速迭代集。允许 LLM 持续生成新 case，允许人工清洗，主要用于发现问题。
- `regression`：稳定回归集。从 dev 中晋升，case 内容和 expected 固定。每次自动优化必须跑。
- `heldout`：隐藏评测集。用于阶段性验收，不参与 Codex 日常改代码决策。

建议起步规模：

- dev：60-120 条，覆盖 3 个 horizon、4-6 类画像、常见失败模式。
- regression：30-60 条，优先放历史 bug、业务核心路径、执行闭环 case。
- heldout：30 条以上，保持更强的自然语言多样性。

## 覆盖矩阵

每批数据至少按这些维度打标签并统计覆盖：

- horizon：`short`、`one_day`、`two_day`。
- profile：亲子、朋友、情侣、独行、长辈同行、饮食限制。
- constraint：预算、时间窗、距离、排队、天气、饮食禁忌、交通方式。
- workflow：只规划、规划加预约、规划加购买、失败后重试、需要用户确认。
- planner challenge：偏好冲突、供应不足、路线折中、低预算、体验优先、多节点行程。
- memory challenge：当前输入优先、历史画像优先、短期记忆覆盖、错误记忆抑制。

## 评估产物

`experiments/llm_agent_eval.py` 的产物继续保留三层：

- `cases.jsonl`：benchmark 输入与 reference。
- `runs.jsonl`：完整 `final_state`，用于深挖问题。
- `report.json`/`report.md`：评估聚合，给 automation 读取。

`actual_summary.component_summaries` 是 automation 的主要诊断入口。它应该包含 intent、memory、planner、execution、trace 的紧凑证据，避免 Codex 每次都处理完整 `final_state`。

报告至少聚合：

- overall：pass rate、avg/min/max overall。
- by horizon：不同时间跨度得分。
- failure categories：定位需求没满足的类别。
- component status：`intent|memory|planner|execution|explanation` 的 `pass|weak|fail` 计数。
- improvement hints：可用于下一轮代码分析的具体建议。

## 自动化优化流程

Codex automation 使用这条闭环：

1. 运行 eval：生成或读取固定 benchmark cases，执行完整 agent，产出报告。
2. 分析报告：优先看 failed/weak case、低分 horizon、失败组件和 missed expectations。
3. 定位代码：根据 component status 映射到 A intent、memory、planner、execution、share/explanation 相关模块。
4. 修改代码：只针对当前失败簇做小步修复，不混入 B planner 独立优化链路。
5. 回归验证：重跑 regression，并比较 pass rate、component status、关键失败样例。
6. 接受或回滚：只有整体分数不下降、核心组件不退化、历史失败不复发时接受修改。

B planner 的单独优化仍保留在独立 eval/policy loop 中。本 benchmark 的 automation 面向完整 LocalSearchAgent 端到端流程，必要时可以指出 planner 问题，但不把 B policy mutation 当作默认修复方式。

## 晋升规则

dev case 晋升到 regression 需要满足：

- 用户需求真实、不依赖固定商户答案。
- expected 足够判定，且不写死具体 POI。
- 至少一次完整 agent run 有可读 `component_summaries`。
- LLM judge 或人工审核确认 reference 合理。
- 该 case 覆盖了一个重要业务能力、历史失败模式或高频用户场景。

## 近期落地顺序

1. 扩展 `actual_summary`，固化组件诊断证据。
2. 生成 30 条 dev cases，覆盖三类 horizon 和基础画像。
3. 用 LLM judge 跑一轮，人工抽查 10 条 reference 和 judge 结论。
4. 从 dev 中选 15 条稳定 case 作为第一版 regression。
5. 将 automation 的分析模板固定为：失败聚类、组件定位、候选修复、回归比较。
