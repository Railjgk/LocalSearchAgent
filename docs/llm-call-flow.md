# 项目大模型调用全流程与使用说明

本文档说明 WeekendFlow/LocalSearchAgent 当前的大模型调用链路、环境变量配置、运行方式和故障回退策略。

## 总览

项目的大模型能力默认关闭。未显式启用时，A/B/C 全链路都走确定性规则、离线数据和 mock API，不会调用外部模型。

启用后，所有模型请求统一通过 `src/nodes/longcat_client.py` 的 OpenAI-compatible Chat Completions 客户端发出。当前默认接入 LongCat 平台的 OpenAI format：

- Base URL: `https://api.longcat.chat/openai`
- Chat Completions URL: `https://api.longcat.chat/openai/v1/chat/completions`
- Auth: `Authorization: Bearer <API_KEY>`
- 默认模型: `LongCat-Flash-Chat`
- 官方文档: https://longcat.chat/platform/docs/

底层请求体形态：

```json
{
  "model": "LongCat-Flash-Chat",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "stream": false,
  "max_tokens": 700,
  "temperature": 0.2
}
```

## 调用链路

### 1. 通用客户端

文件: `src/nodes/longcat_client.py`

职责：

- 读取共享 LongCat/OpenAI-compatible 配置。
- 组装 `/v1/chat/completions` 请求。
- 兼容 `https://api.longcat.chat` 根地址和 `https://api.longcat.chat/openai` OpenAI format 地址。
- 解析 OpenAI Chat Completions 响应里的 `choices[0].message.content`、`usage`、`finish_reason`。
- API 错误、空响应、非 JSON 响应都会抛出 `LongCatAPIError`，调用方负责回退。
- 错误上报前会通过 `sanitize_longcat_error` 隐去 `LONGCAT_API_KEY` 和 `LONGCAT_APP_KEY`。

### 2. A 阶段意图解析

文件: `src/nodes/intent_parser.py`

入口：

- `intent_parser_node`
- `maybe_parse_intent_with_llm`
- `load_a_llm_config`

开关：

- `WF_A_LLM_ENABLED=1` 或 `WF_A_AI_ENABLED=1`

配置优先级：

1. A 专属变量：`WF_A_LLM_API_KEY`、`WF_A_LLM_APP_KEY`、`WF_A_LLM_BASE_URL`、`WF_A_LLM_MODEL`、`WF_A_LLM_TIMEOUT_SECONDS`、`WF_A_LLM_MAX_TOKENS`、`WF_A_LLM_TEMPERATURE`
2. 共享 LongCat 变量：`LONGCAT_API_KEY`、`LONGCAT_APP_KEY`、`LONGCAT_BASE_URL`、`LONGCAT_MODEL`、`LONGCAT_TIMEOUT_SECONDS`、`LONGCAT_MAX_TOKENS`、`LONGCAT_TEMPERATURE`
3. 代码默认值

请求内容：

- System prompt 要求模型只返回结构化 JSON intent。
- User message 是 JSON payload，包含：
  - `user_input`
  - 规则解析得到的 `baseline_intent`
  - `allowed_scene_types`

输出写入状态：

- `intent`
- `constraints`
- `scene_type`
- `a_llm_intent`

`a_llm_intent` 成功时包含：

```json
{
  "enabled": true,
  "provider": "longcat",
  "api_format": "openai",
  "base_url": "https://api.longcat.chat/openai",
  "model": "LongCat-Flash-Chat",
  "success": true,
  "fallback": false,
  "finish_reason": "stop",
  "usage": {"total_tokens": 88}
}
```

回退策略：

- 未启用：只用规则解析，不写 `a_llm_intent`。
- 启用但缺 key：保留规则解析结果，`a_llm_intent.reason=missing_api_key`。
- API 报错、非 JSON、JSON schema 不可用：保留规则解析结果，`a_llm_intent.fallback=true`。

### 3. B 阶段语义 hint

文件: `src/nodes/b_ai_hints.py`

入口：

- `apply_b_semantic_hints`
- `generate_b_semantic_hints`

调用位置：

- `src/nodes/candidate_generator.py`
- 在候选召回/排序前增强 `constraints`、`user_profile` 和 `scenario_activities`。

开关：

- 默认跟随 `WF_B_AI_ENABLED`
- 可用 `WF_B_AI_SEMANTIC_HINTS_ENABLED=0/1` 单独覆盖

请求内容：

- System prompt 要求返回 JSON only，字段包括：
  - `soft_tags`
  - `avoid_tags`
  - `activity_intent_tags`
  - `restaurant_intent_tags`
  - `route_priority`
  - `budget_priority`
  - `confidence`
  - `evidence`
- User message 是 JSON payload，包含：
  - `user_input`
  - `scene_type`
  - `constraints`
  - `user_profile`
  - `scenario_activities`
  - `allowed_tags`

输出写入状态：

- `b_ai_semantic_hints`

成功 metadata 中包含 `hints`，失败时保留原规则约束和候选生成逻辑。

### 4. B 阶段解释文案

文件: `src/nodes/explainability.py`

入口：

- `explainability_node`
- `_maybe_generate_ai_explanation`

调用位置：

- B 阶段选出 `selected_plan` 后，用模型润色最终推荐理由。

开关：

- `WF_B_AI_ENABLED=1`

请求内容：

- System prompt 限定模型不能改方案、商家、价格、路线、评分或库存，只能基于 payload 写中文解释。
- User message 是 JSON payload，包含：
  - `scene_type`
  - `user_input`
  - 精简后的 `constraints`
  - `user_profile`
  - `selected_plan`
  - `optimization_score`
  - 前 2 个 `alternative_plans`
  - 确定性解释 `deterministic_explanation`

模型返回 JSON：

```json
{
  "explanation_text": "推荐理由...",
  "risk_notes": ["..."],
  "next_best_action": "..."
}
```

输出写入状态：

- `explanation_text`
- `b_ai_explanation`

回退策略：

- 未启用或失败时继续使用确定性解释文案。
- API key 会在错误信息中脱敏。

## 环境变量配置

推荐先复制或直接编辑本地 `.env`，然后通过 `source` 加载：

```bash
set -a && source .env && set +a
```

### 只启用 A 阶段 LLM 意图解析

```bash
export WF_A_LLM_ENABLED=1
export WF_A_LLM_API_KEY="你的 LongCat API Key"
export WF_A_LLM_BASE_URL="https://api.longcat.chat/openai"
export WF_A_LLM_MODEL="LongCat-Flash-Chat"
export WF_A_LLM_TIMEOUT_SECONDS=20
export WF_A_LLM_MAX_TOKENS=900
export WF_A_LLM_TEMPERATURE=0.2
```

也可以让 A 复用共享 key：

```bash
export WF_A_LLM_ENABLED=1
export LONGCAT_API_KEY="你的 LongCat API Key"
```

### 只启用 B 阶段语义 hint 和解释文案

```bash
export WF_B_AI_ENABLED=1
export LONGCAT_API_KEY="你的 LongCat API Key"
export LONGCAT_BASE_URL="https://api.longcat.chat/openai"
export LONGCAT_MODEL="LongCat-Flash-Chat"
export LONGCAT_TIMEOUT_SECONDS=20
export LONGCAT_MAX_TOKENS=700
export LONGCAT_TEMPERATURE=0.2
```

只想启用或关闭 B 语义 hint，可以单独设置：

```bash
export WF_B_AI_SEMANTIC_HINTS_ENABLED=1
```

或：

```bash
export WF_B_AI_SEMANTIC_HINTS_ENABLED=0
```

### 同时启用 A 和 B

```bash
export WF_A_LLM_ENABLED=1
export WF_B_AI_ENABLED=1
export LONGCAT_API_KEY="你的 LongCat API Key"
export LONGCAT_BASE_URL="https://api.longcat.chat/openai"
export LONGCAT_MODEL="LongCat-Flash-Chat"
```

A 阶段需要独立模型或参数时，再覆盖 `WF_A_LLM_*`：

```bash
export WF_A_LLM_MODEL="LongCat-Flash-Thinking-2601"
export WF_A_LLM_MAX_TOKENS=1200
export WF_A_LLM_TEMPERATURE=0.1
```

## 运行示例

命令行运行完整链路：

```bash
set -a && source .env && set +a
python run.py "今晚想和对象散步吃轻食，预算500，别去太挤的商场"
```

不写 trace 文件：

```bash
set -a && source .env && set +a
python run.py --no-trace "下午和朋友 citywalk，4个人，预算600"
```

从标准输入读取：

```bash
echo "今天下午想和老婆孩子出去玩，孩子5岁，老婆最近在减脂，别太远" | python run.py --stdin
```

只验证 A 阶段节点：

```bash
python - <<'PY'
from src.nodes.intent_parser import intent_parser_node

result = intent_parser_node({
    "user_input": "今晚想和对象散步吃轻食，预算500，别去太挤的商场"
})
print(result["scene_type"])
print(result.get("a_llm_intent"))
PY
```

## 状态字段观察

运行后可重点观察这些字段：

- `a_llm_intent`: A 阶段 LLM 意图解析状态、模型、token 用量、是否 fallback。
- `b_ai_semantic_hints`: B 阶段语义 hint 状态和生成的 hint。
- `b_ai_explanation`: B 阶段解释文案模型调用状态。
- `execution_log`: 每个节点是否应用模型结果或触发回退。
- `tool_results.intent_parser_prompt`: A 阶段规则 prompt 留痕。

## 常见问题

### 开关开了但没有模型调用

检查是否设置了 API key：

```bash
echo "$WF_A_LLM_ENABLED"
echo "$WF_B_AI_ENABLED"
test -n "$WF_A_LLM_API_KEY" || test -n "$LONGCAT_API_KEY"
```

A 阶段启用但没有 key 时会回退到规则解析，并写入：

```json
{"reason": "missing_api_key", "fallback": true}
```

B 阶段启用但没有 key 时会回退到确定性候选生成和解释。

### Base URL 应该填什么

推荐填：

```bash
https://api.longcat.chat/openai
```

代码也兼容：

```bash
https://api.longcat.chat
https://api.longcat.chat/openai/v1
https://api.longcat.chat/openai/v1/chat/completions
```

### 模型输出不是 JSON 怎么办

A 阶段、B hint、B explanation 都会先尝试剥离 Markdown fence，再截取 JSON object 解析。仍解析失败时会自动 fallback，不中断主流程。

### 会不会泄露 API key

错误信息会经过脱敏处理，覆盖：

- `LONGCAT_API_KEY`
- `LONGCAT_APP_KEY`
- `WF_A_LLM_API_KEY`
- `WF_A_LLM_APP_KEY`

不要把 `.env` 提交到仓库。

## 验证命令

相关单测：

```bash
PYTHONPATH=. uvx --from pytest --with requests pytest \
  tests/test_longcat_client.py \
  tests/test_weekendflow_demo.py \
  tests/test_b_ai_hints.py \
  tests/test_b_ai_explainability.py \
  -q
```

轻量静态检查：

```bash
uvx ruff check \
  src/nodes/longcat_client.py \
  src/nodes/intent_parser.py \
  src/nodes/b_ai_hints.py \
  src/nodes/explainability.py
```

