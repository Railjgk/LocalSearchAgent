# WeekendFlow LocalSearchAgent

WeekendFlow LocalSearchAgent 是一个本地生活行程规划与执行验证原型。项目以 LangGraph 风格的状态流为骨架，把用户自然语言需求拆成 A/B/C 三个阶段：

- A-stage：理解用户意图、抽取约束、读取和更新偏好记忆。
- B-stage：从本地 POI 供给中生成候选方案，过滤约束，选择可解释的行程。
- C-stage：把选中的行程转换成可执行动作，并通过 mock API 校验订座、库存、优惠、路线、支付和分享闭环。

当前代码重点服务于周末、本地生活、亲子、朋友聚会、情侣约会、轻度 citywalk 和微度假场景。仓库里同时包含上海、北京、青岛的高德 POI mock supply 数据和用于继续补充城市数据的脚本。

## 功能概览

### 1. 自然语言需求理解

入口接收中文用户输入，例如：

```text
今晚和朋友想吃烤肉，再找一个轻松的活动，不要太远。
```

A-stage 节点会抽取和归一化：

- 场景类型：亲子、朋友、情侣、个人等。
- 时间、预算、人数、距离、排队时间等硬约束。
- 餐饮偏好、活动偏好、氛围偏好、规避项。
- 用户长期/短期偏好记忆。

相关代码：

- `src/nodes/intent_parser.py`
- `src/nodes/memory_manager.py`
- `src/nodes/scenario_planner.py`
- `src/memory/`

### 2. POI 候选生成和方案优化

B-stage 从 `experiments/mock_data` 下的本地供给数据读取活动和餐饮 POI，结合语义标签、天气、距离、路线估算和场景模板生成方案。它的目标不是只找热门 POI，而是把用户的语义需求映射到合适供给，例如：

- `烤肉`、`烧烤`、`羊肉串` 不应误配成 `火锅`。
- `看展 + 咖啡` 不应退化成普通正餐。
- 亲子场景要考虑儿童友好、排队、停车、轻餐等字段。
- 城市特色数据要和对应城市路由，不混用上海、北京、青岛 POI。

相关代码：

- `src/nodes/candidate_generator.py`
- `src/nodes/constraint_filter.py`
- `src/nodes/plan_optimizer.py`
- `src/nodes/explainability.py`
- `src/nodes/b_semantics.py`
- `src/nodes/b_requirement_compiler.py`
- `src/nodes/b_ai_plan_critic.py`
- `src/nodes/b_repair_planner.py`

### 3. C-stage mock 执行闭环

C-stage 把 B-stage 选中的方案转换成执行动作，并使用本地 mock API 验证：

- 活动票或服务是否可订。
- 餐厅是否有可预约时段。
- 优惠、产品、商户关系是否一致。
- 路线是否可行。
- 价格、库存、排队和替换策略是否能被执行层解释。
- 支付和最终分享文案是否可以生成。

相关代码：

- `src/nodes/tool_router.py`
- `src/nodes/mock_api_layer.py`
- `src/nodes/execution_manager.py`
- `src/nodes/payment_layer.py`
- `src/nodes/share_generator.py`
- `src/tools/execution_mock_api.py`
- `src/tools/mock_apis.py`

### 4. 多城市 POI 数据

项目使用高德 Web 服务拉取真实 POI 身份字段，再由本地规则补齐 WeekendFlow 需要的 mock 字段。高德负责真实观测字段，例如名称、地址、经纬度、电话、行政区、类别；本地脚本负责生成产品、优惠、库存、预约、排队、家庭友好、健康标签、解释证据等字段。

当前主要数据目录：

- `experiments/mock_data/gaode_supply_shanghai_v2_20260527_full/`
- `experiments/mock_data/gaode_supply_beijing_v1/`
- `experiments/mock_data/gaode_supply_qingdao_v1/`
- `experiments/mock_data/c_execution/`

北京和青岛的统一下载入口：

```powershell
py experiments/run_gaode_city_poi.py --cities 北京 青岛 --max-calls-per-city 800
```

只看计划、不调用高德 API：

```powershell
py experiments/run_gaode_city_poi.py --dry-run-plan --cities 北京 青岛 --max-calls-per-city 20
```

只补城市特色类目：

```powershell
py experiments/run_gaode_city_poi.py --special-only --cities 北京 青岛 --max-calls-per-city 500
```

高德 key 只应保存在本地环境或 GitHub Secrets，不要提交到仓库：

```text
GAODE_API_KEY=你的高德 Web 服务 key
```

本地推荐放在 `.env.local`，该文件已被 `.gitignore` 忽略。

## 技术实现

### LangGraph 状态流

核心工作流在 `src/graph.py` 中定义。项目优先使用 `langgraph.graph.StateGraph`，当运行环境没有安装 LangGraph 时，会退化到 `SequentialGraph`，按同样节点顺序顺序执行。这让轻量本地 demo 和完整 LangGraph 环境都能运行。

全局状态定义在 `src/state.py`。`PlanState` 是跨阶段共享的 TypedDict，包含用户输入、A-stage 意图、B-stage 候选和选中方案、C-stage 执行结果、支付状态、重试历史和最终分享信息。

### 本地 mock supply

`experiments/build_supply_from_gaode_v2.py` 是高德 POI 数据构建主脚本，支持：

- YAML 配置驱动的搜索计划。
- text search 和 around search 两类高德接口。
- resume：已成功的 `plan_id` 不重复调用。
- raw POI 记录落盘。
- 去重和 mock 字段生成。
- `--rebuild-from-raw` 从现有 raw 数据重建输出。
- build、coverage、dedupe、quota 报告。

`experiments/run_gaode_city_poi.py` 是多城市包装器。它不重复实现爬取逻辑，只生成北京/青岛城市区域、网格和特色类目 overlay，然后调用 `build_supply_from_gaode_v2.py`。

### 语义匹配

B-stage 的语义匹配集中在 `src/nodes/b_semantics.py`、`src/nodes/b_utils.py` 和 `src/nodes/taxonomy.py`。系统会把用户口语表达扩展到内部语义组，再和 POI 标签、类别、关键词、体验类型、餐饮类型等字段做匹配。

典型例子：

- `烤肉`、`烧烤`、`烤串`、`羊肉串` 映射到 barbecue 语义组。
- `桌游`、`密室`、`剧本杀` 映射到社交活动语义组。
- `轻食`、`沙拉`、`低卡` 映射到健康/轻餐语义组。
- `看展`、`博物馆`、`美术馆` 映射到 exhibition 语义组。

### 记忆系统

`src/memory/` 提供本地长期记忆能力，包括：

- 结构化 memory schema。
- 本地 JSON/文件存储。
- 语义化检索接口。
- 图索引和偏好策略。
- 记忆写入和 TTL 策略。

目前记忆系统用于把用户偏好转化为可计算的 planning effect，而不是简单保留聊天文本。

### 执行 mock API

`src/tools/execution_mock_api.py` 使用 mock supply 数据模拟真实 C-stage 执行服务，包括：

- 订座和订单状态。
- 活动票、餐饮产品、商户、优惠券关系。
- 可用时段和库存。
- 路线检查。
- 价格变化和失败原因。
- 替换策略和重试闭环。

后端入口会根据显式 `--city` 参数或用户输入里的城市词自动选择 POI 数据目录。当前内置映射：

```text
上海 -> experiments/mock_data/gaode_supply_shanghai_v2_20260527_full
北京 -> experiments/mock_data/gaode_supply_beijing_v1
青岛 -> experiments/mock_data/gaode_supply_qingdao_v1
```

无法识别城市时，执行层默认读取 `experiments/mock_data`，保证旧 demo 和测试行为不变。调试时也可以通过环境变量手动切换数据目录：

```powershell
$env:WF_MOCK_DATA_DIR="experiments/mock_data/gaode_supply_qingdao_v1"
```

## 目录结构

```text
src/
  graph.py                  工作流定义
  state.py                  跨阶段共享状态
  nodes/                    A/B/C 各阶段节点
  tools/                    C-stage mock API
  memory/                   本地记忆系统

experiments/
  build_supply_from_gaode_v2.py      高德供给构建主脚本
  run_gaode_city_poi.py              北京/青岛统一 POI runner
  validate_gaode_supply_run.py       POI 数据质量校验
  check_mock_data_quality.py         mock 字段质量检查
  gaode_supply_extraction_config.yaml
  mock_data/                         本地 POI 和 C-stage 状态数据

docs/
  设计记录、评估报告和实验说明

tests/
  单元测试和回归测试

run_backend_cli.py          推荐的命令行后端入口
run.py                      调试入口，保留更多 trace 输出
```

## 快速运行

### 运行完整规划链路

```powershell
py run_backend_cli.py "今晚和朋友想吃烤肉，再找一个轻松的活动，不要太远"
```

显式指定城市：

```powershell
py run_backend_cli.py --city 青岛 "周末想吃海鲜，再安排一个海边 citywalk"
```

输出 JSON：

```powershell
py run_backend_cli.py --json "今天下午一个人想看展，再找个安静咖啡店坐一会儿"
```

从标准输入读取：

```powershell
echo "周末带孩子找亲子活动，晚饭要清淡一点" | py run_backend_cli.py --stdin
```

### 校验 POI 数据

```powershell
py experiments/validate_gaode_supply_run.py --data-dir experiments/mock_data/gaode_supply_qingdao_v1 --skip-eval
```

### 运行测试

仓库根目录有项目级测试：

```powershell
py -m pytest tests
```

如果修改 `libs/` 下的任一库，按 `AGENTS.md` 要求进入对应库目录运行：

```powershell
make format
make lint
make test
```

## GitHub Actions

`.github/workflows/nightly_gaode_city_poi.yml` 定义了夜间自动更新北京/青岛 POI 的工作流：

- 同步远程 `main`。
- 新建自动化分支。
- 调用 `experiments/run_gaode_city_poi.py` 下载 POI。
- 提交生成数据。
- 创建 PR。

云端运行需要在 GitHub 仓库 Secrets 中配置 `GAODE_API_KEY`。本地 `.env.local` 不会被 GitHub Actions 读取。

## 安全和数据注意事项

- 不要提交 `.env`、`.env.local` 或任何 API key。
- 高德 raw 请求日志会保留 URL，但脚本会把 key 写成 `<redacted>`。
- `experiments/artifacts/` 默认被忽略，适合放本地运行摘要和临时报告。
- 大型 POI 数据文件会显著增加仓库体积，提交前应确认确实需要版本化。
- 多城市数据消费时建议按 `city -> data_dir` 路由，避免上海、北京、青岛 POI 混用。

## 当前重构约定

- 统一使用 `experiments/run_gaode_city_poi.py` 作为北京/青岛 POI 下载入口。
- 旧的 `run_gaode_city_nightly.py` 和 `run_gaode_city_speciality.py` 已合并删除。
- 核心 A/B/C 运行逻辑和 mock 数据格式不变。
- 测试文件暂时保留，用于后续回归验证。
