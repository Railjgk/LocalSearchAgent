# WeekendFlow B 重构交接给 A

## 当前分支

- 远端仓库：`https://github.com/Railjgk/LocalSearchAgent.git`
- 远端分支：`codex/refactor-b-structure`
- 基线：`origin/main`
- 本机工作目录仅供参考：`C:\Users\daixi\weekendflow-refactor-b-clean`
- 本轮原则：只做 B 代码结构重构，不改前端、不改数据、不提交 key、不改 C mock 数据

## A 在自己电脑上的获取方式

如果本地已有仓库：

```powershell
git fetch origin
git switch -c codex/refactor-b-structure origin/codex/refactor-b-structure
```

如果本地已经存在同名分支：

```powershell
git fetch origin
git switch codex/refactor-b-structure
git pull --ff-only
```

如果是从零开始：

```powershell
git clone https://github.com/Railjgk/LocalSearchAgent.git
cd LocalSearchAgent
git fetch origin
git switch -c codex/refactor-b-structure origin/codex/refactor-b-structure
```

检查本轮重构提交：

```powershell
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
```

如果 A 使用 sparse checkout 且看不到 `docs/`，可以执行：

```powershell
git sparse-checkout add docs
```

## 已完成的重构

这轮目标是把 B 从“大文件里混合策略、规则、评分、RAG、执行合同”的状态，拆成更清晰的工程模块，同时保持行为不变。

已经拆出的模块：

- `src/nodes/b_route_geometry.py`：路线几何、坐标距离、路径估算基础函数
- `src/nodes/b_route_facts.py`：route facts 构建，统一路线摘要字段
- `src/nodes/b_candidate_policy.py`：候选生成 budget、top-k、policy/env 配置
- `src/nodes/b_multinode_policy.py`：多节点 itinerary 可行性与 fallback 策略
- `src/nodes/b_execution_scope.py`：B 到 C 的执行范围规则
- `src/nodes/b_plan_templates.py`：半天/全天/两天等 plan template 选择
- `src/nodes/b_weather_scoring.py`：天气对候选的 scoring signal
- `src/nodes/b_restaurant_roles.py`：餐厅角色识别和角色评分
- `src/nodes/b_sequence_policy.py`：先活动后吃饭/先吃饭后活动的 sequence preference
- `src/nodes/b_time_slots.py`：时间槽、行程时间、默认 start/end helper
- `src/nodes/b_text_match.py`：文本匹配、语义索引、轻量 cache
- `src/nodes/b_local_food_guardrails.py`：上海本帮菜/本地菜 guardrail
- `src/nodes/b_replan_filter.py`：AI replan 时避开旧 plan/旧商家组合
- `src/nodes/b_score_policy.py`：场景动态权重、阈值、惩罚、policy YAML 读取
- `src/nodes/b_plan_critic_bridge.py`：LongCat critic 输出到 B bounded replan request 的桥接

核心效果：

- `candidate_generator.py` 已经明显瘦身，但仍保留主流程编排。
- `plan_optimizer.py` 已经移出 score policy、C 执行范围、critic bridge，但仍保留主要评分函数和 selected_plan 构建。
- B 的 Agent 设计更清楚：规则/OR 处理硬约束，score policy 处理可解释排序，LongCat critic 只生成有边界的 replan request，不直接修改执行代码。

## 已跑验证

最近一轮验证：

```powershell
python -m compileall src\nodes tests\test_b_plan_critic_bridge.py
python -m pytest tests\test_b_plan_critic_bridge.py tests\test_b_ai_plan_critic.py tests\test_b_replan_loop.py tests\test_b_replan_filter.py tests\test_b_score_policy.py tests\test_b_execution_scope.py -q
python -m pytest tests\test_b_plan_critic_bridge.py tests\test_b_score_policy.py tests\test_b_replan_filter.py tests\test_b_local_food_guardrails.py tests\test_b_text_match.py tests\test_b_time_slots.py tests\test_b_sequence_policy.py tests\test_b_restaurant_roles.py tests\test_b_weather_scoring.py tests\test_b_weather_context.py tests\test_b_plan_templates.py tests\test_b_execution_scope.py tests\test_b_multinode_policy.py tests\test_b_candidate_policy.py tests\test_b_route_geometry.py tests\test_b_route_facts.py tests\test_b_itinerary_blueprint.py tests\test_b_poi_rag_node.py tests\test_b_requirement_compiler.py -q
python experiments\run_b_eval.py
```

结果：

- 相关 B 单测：`168 passed`
- B offline eval：`27/28`
- 唯一失败仍是历史基线问题：`friends_social_nearby`，总距离 `8.22km`，超过 eval 要求 `<= 8.0km`，并缺少 `nearby_route` trait。这个不是本轮重构引入的新失败。

## A 接手时建议怎么继续

继续遵守“每次只切一小块、补测试、跑 eval、再 commit”的节奏。不要一次性重写 `candidate_generator.py` 或 `plan_optimizer.py`。

推荐下一步顺序：

1. 抽 `candidate_generator.py` 里的 strict node identity 规则
   - 目标模块：`src/nodes/b_node_identity.py`
   - 当前内容包括：公园/景区识别、住宿识别、KTV 误召回过滤、citywalk/flower shop/bar 等 strict role guardrail。
   - 这块对多城市和多日 itinerary 很重要，但中文词表很多，迁移时要小心编码。
   - 注意：PowerShell 可能把中文显示成乱码，不要以终端显示为准；用 `Path.read_text(encoding="utf-8")` 或 `repr()` 检查真实内容。

2. 抽 `candidate_generator.py` 里的 explicit requirement filtering
   - 目标模块：`src/nodes/b_requirement_filters.py`
   - 包括 `_explicit_activity_requirements`、`_explicit_restaurant_requirements`、requirement token expand、activity/restaurant requirement filter。
   - 这块是“用户说烤肉/亲子/手作时不要误召回”的核心。

3. 抽 `plan_optimizer.py` 的 score components
   - 目标模块：`src/nodes/b_score_components.py`
   - 可以先抽 `_score_route`、`_score_budget`、`_score_time_fit`、`_score_availability`、`_score_experience`。
   - 第一轮只搬代码，不改公式；后续再谈优化公式。

4. 谨慎处理 execution contract
   - `plan_optimizer.py` 里的 action hints 和 execution contract 逻辑和 C 强相关。
   - 如果要抽，建议叫 `src/nodes/b_execution_contract.py`。
   - 抽之前最好和 C 确认字段，不要在重构时顺手改字段名。

5. 最后再清理 import/private alias
   - 当前很多新模块在 `candidate_generator.py` 和 `plan_optimizer.py` 中用 `_xxx` alias 保持兼容。
   - 这对测试和旧调用比较安全；等结构稳定后再统一公开接口。

## 不建议做的事

- 不要把中文语义重新改回英文 canonical 主信息流。现在 B 的方向是中文优先，英文 canonical 只做兼容索引。
- 不要在重构里顺手调 scoring 权重，除非有 eval 证据。
- 不要改 mock data、前端、C mock API 字段。
- 不要把 LongCat critic 设计成“直接改 plan”。当前更安全的设计是：LLM 给 critique，B 把 critique 约束成 bounded replan request。
- 不要一次性修改 RAG、候选生成、optimizer 三块；容易难以定位回归。

## 给设计文档可用的一句话

B 被重构为“硬约束过滤 + 可解释评分策略 + RAG 候选召回 + LLM critic 有界重规划”的分层架构：算法负责可验证的时间、路线、预算、库存等约束，LongCat 负责识别难以穷举的体验一致性和场景风险，但它只能触发受控 replan request，不能直接改核心执行逻辑。
