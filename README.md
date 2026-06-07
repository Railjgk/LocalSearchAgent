# WeekendFlow LocalSearchAgent

WeekendFlow LocalSearchAgent 是一个本地生活行程规划与执行验证原型。项目以 LangGraph 风格的状态流为骨架，把用户自然语言需求拆成 A/B/C 三个阶段：

- A-stage：理解用户意图、抽取约束、读取和更新偏好记忆，并识别亲子、朋友、情侣、个人、商务等场景。
- B-stage：从本地 POI 供给中做 RAG 检索、候选生成、约束过滤、方案优化和解释。
- C-stage：把选中的方案转换成可执行动作，并通过本地 mock API 校验订座、库存、优惠、路线、支付和分享闭环。

当前代码重点服务于周末、本地生活、亲子、朋友聚会、情侣约会、轻度 citywalk 和微度假场景。仓库里包含上海、北京、青岛的高德 POI mock supply 数据，以及用于继续补充城市数据、运行 B-stage/LocalSearchBench 评估的脚本。

## 环境配置

### Conda

根目录当前没有项目级 `pyproject.toml`、`uv.lock` 或 `requirements.txt`。建议先用 conda 固定 Python 版本，再用 `uv` 创建本地虚拟环境并安装 `libs/` 中的 LangGraph 相关库：

```bash
conda create -n weekendflow python=3.11 -y
conda activate weekendflow
python -m pip install -U pip uv

uv venv .venv
source .venv/bin/activate
make install

# 根项目运行、测试和实验脚本常用工具
uv pip install pytest requests pyyaml pandas pyarrow numpy scikit-learn
```

Windows PowerShell 可把激活命令替换为：

```powershell
conda create -n weekendflow python=3.11 -y
conda activate weekendflow
python -m pip install -U pip uv
uv venv .venv
.\.venv\Scripts\Activate.ps1
make install
uv pip install pytest requests pyyaml pandas pyarrow numpy scikit-learn
```

### uv

根目录 `Makefile` 的 `make install` 会创建 `.venv`，并把 `libs/*` 下带 `pyproject.toml` 的库以 editable 方式安装进虚拟环境。它主要服务 LangGraph monorepo 子库，不会自动声明 WeekendFlow 根应用的所有实验依赖，所以根项目脚本需要的测试/数据处理包仍建议显式安装。

常用命令：

```bash
uv venv .venv
source .venv/bin/activate
make install
uv pip install pytest requests pyyaml pandas pyarrow numpy scikit-learn
```

### Docker

仓库根目录当前没有 `Dockerfile`，也没有项目级 Docker Compose 文件。需要容器化时，可以基于下面模板在本地临时创建 Dockerfile；不要把它理解为仓库已提供的官方 Docker 入口。

```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir -U pip uv
COPY . .
RUN uv venv /opt/weekendflow \
    && . /opt/weekendflow/bin/activate \
    && make install \
    && uv pip install pytest requests pyyaml pandas pyarrow numpy scikit-learn

ENV PATH="/opt/weekendflow/bin:${PATH}"
CMD ["python", "run_backend_cli.py", "--json", "周末想吃烤肉，再安排一个轻松活动"]
```

如需挂载本地环境变量和 mock data，优先通过 `--env-file`、`-e` 和 bind mount 注入，不要把 `.env` 或 API key 写进镜像。

## 必要环境变量

不配置任何外部 key 时，项目仍可使用本地 mock supply 跑 deterministic demo。以下变量按用途选择配置：

- `GAODE_API_KEY`：高德 Web 服务 key，用于 POI 拉取、路线或天气相关能力；`.env.example` 只提供占位示例。
- `GAODE_WEATHER_API_KEY`：天气接口专用 key；未设置时会回退到 `GAODE_API_KEY`。
- `WF_WEATHER_ENABLED`：是否启用天气上下文，默认启用；没有 key 时返回 `missing_api_key` fallback。
- `WF_MOCK_DATA_DIR`：B/C 默认 mock supply 数据目录覆盖，例如 `experiments/mock_data/gaode_supply_qingdao_v1`。
- `WF_B_RAG_DATA_DIR`：B-stage RAG 数据目录覆盖；未设置时通常跟随 `WF_MOCK_DATA_DIR`。
- `WF_C_MOCK_DATA_DIR`：C-stage mock API 专用数据目录覆盖。
- `WF_C_EXECUTION_STATE_DIR`：C-stage 可变执行状态目录覆盖，适合测试或多人并行 demo 时隔离状态。
- `WF_A_LLM_ENABLED` 或 `WF_A_AI_ENABLED`：启用 A-stage LLM 意图解析。
- `WF_A_LLM_API_KEY`、`WF_A_LLM_APP_KEY`、`WF_A_LLM_BASE_URL`、`WF_A_LLM_MODEL`、`WF_A_LLM_TIMEOUT_SECONDS`、`WF_A_LLM_MAX_TOKENS`、`WF_A_LLM_TEMPERATURE`：A-stage 专用 OpenAI-compatible/LongCat 配置。
- `WF_B_AI_ENABLED`：启用 B-stage AI 辅助能力。
- `WF_B_AI_REQUIREMENT_COMPILER_ENABLED`、`WF_B_AI_SEMANTIC_HINTS_ENABLED`、`WF_B_AI_PLAN_CRITIC_ENABLED`、`WF_B_AI_REPAIR_PLANNER_ENABLED`、`WF_B_AI_EXPLANATION_ON_RAG`：B-stage AI 子能力开关。
- `LONGCAT_API_KEY`、`LONGCAT_APP_KEY`、`LONGCAT_BASE_URL`、`LONGCAT_MODEL`、`LONGCAT_TIMEOUT_SECONDS`、`LONGCAT_MAX_TOKENS`、`LONGCAT_TEMPERATURE`：A/B 共用的 LongCat OpenAI-compatible 配置。
- `WF_PLANNER_POLICY_PATH`：覆盖 planner policy 文件路径，默认使用仓库内策略配置。

本地可把密钥放在 `.env.local` 或 shell 环境里。`src/nodes/longcat_client.py` 会在非测试场景读取 `.env.local` 和 `.env`，但这些文件不应提交。

## 使用方法

### CLI 运行

推荐入口是 `run_backend_cli.py`，它会运行完整 A/B/C 链路并输出适合前端或脚本消费的摘要：

```bash
python run_backend_cli.py "今晚和朋友想吃烤肉，再找一个轻松的活动，不要太远"
```

显式指定城市：

```bash
python run_backend_cli.py --city 青岛 "周末想吃海鲜，再安排一个海边 citywalk"
```

当前内置城市路由支持 `上海`、`北京`、`青岛`，也识别 `上海市`、`北京市`、`青岛市`、`shanghai`、`beijing`、`qingdao`。显式 `--city` 优先级高于用户输入中的城市词。

### JSON 输出

```bash
python run_backend_cli.py --json "今天下午一个人想看展，再找个安静咖啡店坐一会儿"
```

JSON 摘要包含 `requested_city`、`mock_data_dir`、`scene_type`、`constraints`、`timeline`、`explanation`、`action_sequence`、`execution_status`、`payment_status`、`retry_history`、`final_share_message` 和最多 3 个 `alternative_plans`。

### stdin

```bash
echo "周末带孩子找亲子活动，晚饭要清淡一点" | python run_backend_cli.py --stdin
```

### 调试入口

`run.py` 会打印更完整的节点日志，并默认写入 `docs/run-status-test-*` 记录；只想看终端输出时使用 `--no-trace`：

```bash
python run.py "下午和朋友 citywalk，4个人，预算600" --no-trace
```

### C-stage 执行状态

`run_backend_cli.py` 默认会隔离并恢复 C-stage mutable state，避免 demo 后改动 `experiments/mock_data/c_execution/*.json`。需要保留订座、订单、优惠、路线等状态变化时，显式传入：

```bash
python run_backend_cli.py --persist-execution-state "今晚订个双人餐，再买一个活动票"
```

## 测试命令

根项目测试：

```bash
python -m pytest tests test_b_pipeline.py
```

运行单个测试文件：

```bash
python -m pytest tests/test_city_data_router.py
```

按 `AGENTS.md` 约定，如果修改 `libs/` 下任一库，需要进入对应库目录运行：

```bash
cd libs/langgraph
make format
make lint
make test
```

特定 pytest 文件或参数可通过 `TEST` 变量传入：

```bash
TEST=tests/test_pregel.py make test
```

根目录 `make test` 会遍历 `libs/*` 中带 Makefile 的库，不等同于运行 WeekendFlow 根项目的 `tests/`。

## 组件说明

### 工作流编排

`src/graph.py` 定义完整节点顺序：

```text
intent_parser -> memory_manager -> scenario_planner
-> b_poi_rag -> candidate_generator -> constraint_filter -> plan_optimizer
-> b_replan_loop -> explainability
-> tool_router -> mock_api_layer -> execution_manager
-> repair_planner -> b_ai_trace -> payment_layer -> share_generator
```

项目优先使用 `langgraph.graph.StateGraph`。当运行环境没有安装 LangGraph 时，会退化到 `SequentialGraph`，按同样节点顺序顺序执行。全局状态定义在 `src/state.py`，初始状态构造在 `src/initial_state.py`。

### A-stage：意图、记忆、场景

- `src/nodes/intent_parser.py`：解析自然语言输入，抽取人数、时间、预算、城市、地点、餐饮偏好、活动偏好、否定约束、同行关系等字段；可选择启用 A-stage LLM。
- `src/nodes/memory_manager.py` 和 `src/memory/`：维护本地长期/短期记忆，把历史偏好、价值倾向和反馈转成可计算的 planning effect，而不是简单保存聊天文本。
- `src/nodes/scenario_planner.py`：把意图和记忆归一到场景模板，例如亲子半天、朋友本地探索、情侣轻约会、商务接待、微度假等，并生成 B-stage 可使用的路线和活动提示。

### B-stage：RAG、候选、过滤、优化、解释

- `src/nodes/b_poi_rag.py`：从 `experiments/mock_data` 或 `WF_B_RAG_DATA_DIR` / `WF_MOCK_DATA_DIR` 指定的数据目录读取活动、餐厅和多城市 supply，支持多节点 itinerary 的 POI 证据检索。
- `src/nodes/b_requirement_compiler.py`、`src/nodes/b_itinerary_blueprint.py`、`src/nodes/b_rag_contract.py`：把 A-stage 输出编译为 B-stage 可执行的需求、时间骨架和 RAG handoff contract。
- `src/nodes/candidate_generator.py`：从本地 supply 和 C mock API 适配层生成候选活动/餐厅/行程节点。
- `src/nodes/constraint_filter.py`：过滤预算、距离、排队、儿童友好、饮食禁忌、城市数据混用、时间窗等硬约束。
- `src/nodes/plan_optimizer.py` 和 `src/nodes/b_plan_quality.py`：综合偏好匹配、路线、价格、可履约性、体验质量、天气适配和风险进行打分。
- `src/nodes/b_replan_loop.py`、`src/nodes/b_repair_planner.py`、`src/nodes/b_ai_plan_critic.py`：在候选不足、计划部分可执行或 AI critic 发现风险时进行修复或重排。
- `src/nodes/explainability.py`、`src/nodes/b_ai_hints.py`、`src/nodes/b_ai_trace.py`：生成推荐理由、权衡说明、AI 调用 trace 和可解释证据。
- `src/nodes/b_semantics.py`、`src/nodes/b_utils.py`、`src/nodes/taxonomy.py`：维护本地生活语义组，例如烤肉/烧烤、看展/博物馆、轻食/低卡、桌游/密室/剧本杀等。

### C-stage：tool router、mock API、execution、payment、share

- `src/nodes/tool_router.py`：把 B-stage `selected_plan.action_hints` 转换为 C-stage 动作序列，例如订座、买票、购买套餐、路线检查。
- `src/nodes/mock_api_layer.py`：在请求级别对 C mock 数据目录做路由，调用本地 mock API。
- `src/tools/execution_mock_api.py`：模拟订座、活动票、餐饮产品、商户、优惠券、库存、路线、订单和执行状态；会读写 `c_execution` 状态文件。
- `src/nodes/execution_manager.py`：汇总工具结果，判断执行成功、部分成功、失败及重试需要。
- `src/nodes/payment_layer.py`：模拟支付订单和支付状态。
- `src/nodes/share_generator.py`：生成最终可分享文案，并在执行失败或部分可执行时给出相应话术。

## Mock data 与多城市路由

主要数据目录：

- `experiments/mock_data/gaode_supply_shanghai_v2_20260527_full/`：当前较大的上海供给快照，活动和餐厅大文件使用 JSONL shards。
- `experiments/mock_data/gaode_supply_beijing_v1/`：北京供给。
- `experiments/mock_data/gaode_supply_qingdao_v1/`：青岛供给。
- `experiments/mock_data/c_execution/`：C-stage 可变执行状态。

内置城市路由在 `src/city_data_router.py`：

```text
上海 -> experiments/mock_data/gaode_supply_shanghai_v2_20260527_full
北京 -> experiments/mock_data/gaode_supply_beijing_v1
青岛 -> experiments/mock_data/gaode_supply_qingdao_v1
```

无法识别城市时，执行层默认读取 `experiments/mock_data`，保证旧 demo 和测试行为不变。调试时也可以手动切换：

```bash
export WF_MOCK_DATA_DIR="experiments/mock_data/gaode_supply_qingdao_v1"
python run_backend_cli.py --json "想吃海鲜，再安排海边散步"
```

北京和青岛的统一下载入口：

```bash
python experiments/run_gaode_city_poi.py --cities 北京 青岛 --max-calls-per-city 800
```

只看计划、不调用高德 API：

```bash
python experiments/run_gaode_city_poi.py --dry-run-plan --cities 北京 青岛 --max-calls-per-city 20
```

只补城市特色类目：

```bash
python experiments/run_gaode_city_poi.py --special-only --cities 北京 青岛 --max-calls-per-city 500
```

数据质量检查：

```bash
python experiments/check_mock_data_quality.py
python experiments/validate_gaode_supply_run.py --data-dir experiments/mock_data/gaode_supply_qingdao_v1 --skip-eval
```

高德 raw POI 主要提供名称、地址、经纬度、电话、行政区和类别等真实观测字段；WeekendFlow 本地脚本补齐产品、优惠、库存、预约、排队、家庭友好、健康标签、解释证据等 mock 字段。不要把这些 mock 字段误认为真实线上库存或真实交易状态。

## Experiments 与 Eval

`experiments/` 包含供给构建、B-stage 评估、策略迭代和 value probe 实验脚本：

- `experiments/build_supply_from_gaode_v2.py`：高德 POI 供给构建主脚本，支持配置驱动搜索、resume、raw POI 落盘、去重、mock 字段生成和 `--rebuild-from-raw`。
- `experiments/run_gaode_city_poi.py`：北京/青岛多城市包装器，生成城市区域、网格和特色类目 overlay 后调用构建脚本。
- `experiments/run_b_eval.py`、`experiments/run_b_rag_itinerary_eval.py`、`experiments/run_localsearchbench_b_probe.py`、`experiments/run_localsearchbench_rag_probe.py`：B-stage 和 RAG/LocalSearchBench 评估入口。
- `experiments/planner_policy.yaml`、`experiments/gepa_policy_loop.py`、`experiments/run_policy_iteration.py`：规划策略和迭代实验。
- `benchmarks/localsearchbench/`：LocalSearchBench 适配器、数据和上游评估脚本。

示例：

```bash
python experiments/run_b_eval.py --mock-data-dir experiments/mock_data/gaode_supply_qingdao_v1
python benchmarks/localsearchbench/run_weekendflow_adapter.py
```

具体参数以脚本 `--help` 为准。

## libs 与上游 LangGraph monorepo

仓库是从 LangGraph monorepo 初始化的，`UPSTREAM.md` 保留了上游来源和 MIT License 说明。`libs/` 下是上游 LangGraph 相关库，而不是 WeekendFlow 业务节点：

- `libs/checkpoint`：LangGraph checkpointer 基础接口。
- `libs/checkpoint-postgres`：Postgres checkpoint saver。
- `libs/checkpoint-sqlite`：SQLite checkpoint saver。
- `libs/cli`：LangGraph 官方命令行工具。
- `libs/langgraph`：构建 stateful multi-actor agents 的核心框架。
- `libs/prebuilt`：预构建 agent/tool API。
- `libs/sdk-js`：LangGraph REST API 的 JS/TS SDK。
- `libs/sdk-py`：LangGraph Server API 的 Python SDK。

修改 `libs/` 任一库时，按 `AGENTS.md` 在对应库目录执行 `make format`、`make lint`、`make test`。WeekendFlow 本身的 A/B/C 业务代码主要在 `src/`、`experiments/`、`benchmarks/` 和 `tests/`。

## 安全和数据注意事项

- 不要提交 `.env`、`.env.local`、API key、token、真实用户隐私数据或带密钥的请求日志。
- 高德 raw 请求日志应确认 key 已被 `<redacted>` 处理后再提交。
- `.env.example` 只能放占位变量，例如 `GAODE_API_KEY=your_gaode_web_service_key`。
- `experiments/mock_data/c_execution/*.json` 是 C-stage 可变状态，demo 或测试可能改变订座、订单、优惠、路线和 execution state。日常运行优先使用 `run_backend_cli.py` 的默认隔离模式，或设置 `WF_C_EXECUTION_STATE_DIR` 指向临时目录。
- `--persist-execution-state` 会保留 C-stage 状态变化，提交前应仔细检查这些 JSON 是否确实需要版本化。
- mock data 中的库存、价格、套餐、优惠、营业和可订字段是本地模拟数据，不代表真实线上履约承诺。
- 多城市数据消费时应使用 `city -> data_dir` 路由，避免上海、北京、青岛 POI 混用。
- 大型 POI 数据文件会显著增加仓库体积，提交前确认确实需要版本化，并优先使用既有 shard 结构。

## GitHub Actions

`.github/workflows/nightly_gaode_city_poi.yml` 定义了夜间自动更新北京/青岛 POI 的工作流：同步远程 `main`、新建自动化分支、调用 `experiments/run_gaode_city_poi.py`、提交生成数据并创建 PR。

云端运行需要在 GitHub 仓库 Secrets 中配置 `GAODE_API_KEY`。本地 `.env.local` 不会被 GitHub Actions 读取。
