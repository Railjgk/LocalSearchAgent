# 自动化重构报告

生成时间：2026-06-07 09:15 CST

## 总览

本次按要求完成 4 轮自动化重构闭环：每轮先生成重构计划，再由独立 critic 给出评估或 bounded replan 约束，随后由独立实现 agent 改代码，并通过 post-fix eval 决定是否采纳。

- 第 1 轮：采纳，多节点候选按中文时间骨架顺序构造。
- 第 2 轮：拒绝，可选下午补充活动省略 fallback 未带来 eval 改善并降低部分候选证据数，已自动回滚。
- 第 3 轮：采纳，B-stage POI RAG 增加中文标签召回与 no-supply 诊断。
- 第 4 轮：采纳，bounded schedule repair target 补全 zero-fit/positive-drift 中文节点，并保持 missing evidence 非执行语义。
- 最终人工接管修复：多节点 partial candidate 若仍缺住宿等阻断性节点，不再被优化器包装成 `partial_executable`，改回非执行中文骨架和 B-stage replan request。

未改动 frontend、mock data、C mock API 字段或 schema；未调整 scoring 权重；未把中文主语义改成英文 canonical 主流程。

## 代码改动

### `src/nodes/candidate_generator.py`

- 新增 `_multinode_skeleton_slot_order` 和 `_order_multinode_intents_by_skeleton`。
- 多节点候选构造现在以 `time_skeleton.days[].slots` 的 day/start 顺序为准，而不是仅依赖 `node_intents` 原始顺序。
- 对缺失节点和 `plan_template` 也使用同一中文时间骨架顺序，减少 beam construction 与用户中文行程顺序漂移。

### `src/nodes/b_poi_rag.py`

- 新增中文标签召回 guard，仅在 `cultural_photo`、`talk_show` 严格检索为零时触发。
- `文化体验/拍照` 需要来自 POI identity 字段的文化/展览/博物馆/摄影等证据，并排除 SPA、按摩、健身等 false positive。
- `脱口秀/演出` 需要演出/剧场/儿童剧/亲子剧等身份字段证据，并排除商场、会议、livehouse 等不安全泛化。
- 召回结果只作为候选证据，不会变成已推荐、已预订或 C-stage 可执行动作。
- 写入 `chinese_label_recall_guard`、触发词、候选数、匹配词数量等诊断，便于区分“缺供给”和“下游时段失败”。

### `src/nodes/plan_optimizer.py`

- schedule repair request 现在会从 `slot_window_diagnostics` 补入 `slot_fit == 0` 且 `slot_drift > 0` 的中文节点。
- 补入仅限已有 bounded schedule repair 且原因是时间骨架/候选时段不匹配，不会把宠物友好证据不足等非 schedule blocker 改写成 schedule repair。
- `missing_node_evidence` 被放入 `missing_evidence_target_nodes`，不混入可执行 target。
- `preserve_existing_slot_fit_nodes` 继续保留 `slot_fit > 0` 节点，如 `文化体验/拍照`、`指定餐饮`。
- 新增保护：多节点 partial candidate 若缺失住宿等阻断性角色，优化器返回完整非执行 skeleton 和 replan request，不生成残缺 action hints。

## 测试与 Eval

通过的本地验证：

- `git diff --check`
- `python -m py_compile`：`experiments/llm_agent_eval.py`、`src/nodes/b_poi_rag.py`、`src/nodes/candidate_generator.py`、`src/nodes/plan_optimizer.py`、`src/nodes/execution_manager.py`、`src/nodes/share_generator.py`、`src/nodes/tool_router.py`
- `pytest tests/test_b_poi_rag_node.py tests/test_b_replan_loop.py tests/test_multinode_slot_alignment.py tests/test_b_multinode_skeleton_plan.py -q`：47 passed
- `pytest tests/test_execution_manager_no_actions.py tests/test_share_generator_non_executable.py tests/test_llm_agent_eval.py tests/test_b_itinerary_blueprint.py tests/test_tool_router_partial_guidance.py tests/test_tool_router_share_generator.py -q`：110 passed

最终 eval 目录：

- `experiments/artifacts/llm_agent_eval/automation/final_dev_new_20260607_091555`

最终 compact summary：

| case | raw/plan candidates | status | blocker |
| --- | ---: | --- | --- |
| `codexseed_family_elder_child_urgent_short` | 0/0 | `failed/no_executable_actions` | 缺少满足硬约束的具体候选和确认依据 |
| `codexseed_friends_rain_budget_citywalk_one_day` | 18/18 | `failed/no_executable_actions` | 时间骨架与候选营业/时段不匹配 |
| `codexseed_couple_pet_parking_anniversary_two_day` | 18/18 | `failed/no_executable_actions` | 缺少活动和餐厅均宠物友好的证据 |
| `codexseed_family_fixed_anchors_full_day` | 18/18 | `failed/no_executable_actions` | 时间骨架与候选营业/时段不匹配 |

关键抽查：

- friends case 的 schedule repair targets 为 `城市漫步/市集`、`下午补充活动`。
- friends case 的 `文化体验/拍照`、`指定餐饮` 保留为 slot-fit evidence metadata。
- fixed-anchor case 保留 `13:30-15:00 午睡/休息` 为 protected non-executable anchor。
- `脱口秀/演出` 只记录为 `missing_node_evidence`，没有变成可执行动作。
- 最终 runs 中关键 case 的 `action_sequence` 长度为 0。

## 约束遵守

- 未修改 mock data。
- 未修改前端。
- 未修改 C mock API 字段或 schema。
- 未修改 scoring 权重。
- 未调用 `experiments/llm_agent_eval.py evaluate`。
- LongCat/critic 输出只被控制器转成 bounded replan request，没有直接改 plan。
- 中文标签和中文行程语义保持主流程；英文 role 只作为 compatibility index。
