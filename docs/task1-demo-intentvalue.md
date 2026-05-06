# task1/demo-intentvalue 实现说明

本文档说明 `task1/demo-intentvalue` 分支的 A 阶段 demo：基于 prompt 包装的 intent parser，以及带 Value 的 memory 表示。

## 实现范围

当前分支只展示输入到 memory 的处理链路：

```text
user_input
  -> intent_parser_node
  -> memory_manager_node
  -> intent + memory + constraints
```

不包含候选生成、约束过滤、方案优化、工具调用和执行闭环。后续 B/C 阶段应读取 `constraints`，同时可用 `intent` 和 `memory` 做解释或调试。

## 核心设计

本 demo 将 A 阶段输出拆成三层：

1. `intent`：本轮任务意图，回答“用户这一次想做什么”。
2. `memory`：用户画像和值，回答“这个用户长期/短期倾向是什么”。
3. `constraints`：给后续 planner 使用的合并结果，回答“哪些字段可以直接进入过滤、打分、解释”。

原则：

- 当前输入优先级最高。
- Memory 只补充默认值、偏好、权重和惩罚项。
- 隐含表达必须结构化，例如“老婆减肥”转成 `low_calorie`、`light_food`。
- Value 必须带 `score`、`confidence`、`ttl`、`source`、`evidence`。

## 主要文件

- `src/state.py`：定义 `PlanState` 和 `ValueMemoryItem`。
- `src/nodes/intent_parser.py`：prompt 包装 + 轻量规则解析，输出 `intent` 和初始 `constraints`。
- `src/nodes/memory_manager.py`：加载结构化 memory，把 value 合并进 planner constraints。
- `src/graph.py`：两节点图流程，优先使用 LangGraph；未安装 `langgraph` 时使用 fallback runner。
- `src/demo.py`：命令行 demo。
- `tests/test_weekendflow_demo.py`：A 阶段 smoke 测试。

## Intent Parser

入口：

```python
def intent_parser_node(state: PlanState) -> dict[str, Any]:
    ...
```

输入：

```python
{
    "user_id": "u001",
    "user_input": "今天下午想和老婆孩子出去玩几个小时，别离家太远，孩子5岁，老婆最近在减肥。"
}
```

输出：

```python
{
    "intent": {...},
    "constraints": {...},
    "scene_type": "family",
    "tool_results": {
        "intent_parser_prompt": "..."
    },
    "execution_log": [...]
}
```

当前 demo 使用 prompt 模板声明目标，并用规则解析保证离线可运行。支持的主要映射：

- `老婆/妻子/太太/媳妇` -> `people.role = wife`
- `减肥/控卡/低脂/少油` -> `state = dieting`, `low_calorie`, `light_food`
- `孩子/小孩/5岁` -> `people.role = child`, `kid_friendly`, `low_intensity`
- `别太远/别离家太远/附近/近一点/离家近` -> `nearby`, `max_distance_km = 8.0`
- `今天下午/下午` -> `today_afternoon`, `duration_range = [4, 6]`
- `排队/等位/人多` -> `long_queue`, `crowded`

### Intent 表示

Intent 完整刻画本轮任务：

```python
{
    "task_type": "local_life_plan",
    "goal": "安排一次本地生活出行计划",
    "scene": "family",
    "time": {
        "window": "today_afternoon",
        "duration_range": [4, 6],
        "start_time": None,
        "end_time": None
    },
    "people": [
        {"role": "self", "needs": []},
        {
            "role": "wife",
            "state": "dieting",
            "needs": ["low_calorie", "light_food"]
        },
        {
            "role": "child",
            "age": 5,
            "needs": ["kid_friendly", "low_intensity"]
        }
    ],
    "location": {
        "origin": "home",
        "distance_preference": "nearby",
        "max_distance_km": 8.0,
        "transport_mode": "unknown"
    },
    "budget": {
        "amount": None,
        "type": None,
        "sensitivity": "unknown"
    },
    "planning_preferences": {
        "activity_type": ["light_activity", "parent_child"],
        "food_type": ["light_food", "low_calorie"],
        "pace": "relaxed"
    },
    "constraints": {
        "hard": ["kid_friendly"],
        "soft": ["light_food", "low_calorie", "low_intensity"],
        "avoid": ["long_queue", "too_far"]
    },
    "people_count": 3,
    "missing_slots": ["budget", "transport_mode"],
    "confidence": {
        "wife_dieting": 0.9,
        "child_age": 0.95,
        "time_window": 0.85,
        "scene": 0.92,
        "distance": 0.85
    },
    "raw_text": "..."
}
```

必须关注的 intent 要素：

- `task_type`：本地生活规划、订餐、找活动等任务类型。
- `scene`：family、couple、friends、low_budget 等场景。
- `time`：时间窗口、持续时长、明确或缺失的开始结束时间。
- `people`：同行人、关系、年龄、状态和需求。
- `location`：出发地、距离偏好、最大距离、交通方式。
- `budget`：预算数值、预算类型、价格敏感度。
- `planning_preferences`：活动、饮食、节奏偏好。
- `constraints.hard`：后续不能轻易违反的硬约束。
- `constraints.soft`：后续用于加分或扣分的软偏好。
- `constraints.avoid`：明确或隐含规避项。
- `missing_slots`：需要追问或默认处理的信息。
- `confidence`：隐含解析的置信度。

## Memory Manager

入口：

```python
def memory_manager_node(state: PlanState) -> dict[str, Any]:
    ...
```

输入：

```python
{
    "intent": {...},
    "constraints": {...},
    "user_id": "u001",
    "user_input": "..."
}
```

输出：

```python
{
    "memory": {...},
    "constraints": {...},
    "value_memory": [...],
    "short_term_memory": [...],
    "tool_results": {
        "memory_manager": {
            "memory": {...},
            "merged_constraints": {...}
        }
    },
    "execution_log": [...]
}
```

### Memory 表示

Memory 完整刻画用户画像：

```python
{
    "user_id": "u001",
    "stable_profile": {
        "home_area": "unknown",
        "consumption_level": "middle",
        "default_transport": "drive_or_taxi"
    },
    "companion_profile": {
        "child": {
            "age": 5,
            "needs": ["kid_friendly", "low_intensity"],
            "confidence": 0.9,
            "source": "historical_profile"
        },
        "wife": {
            "state": "dieting",
            "needs": ["low_calorie", "light_food"],
            "ttl": "short_term",
            "confidence": 0.8,
            "source": "recent_user_input"
        }
    },
    "preference_profile": {
        "food": ["light_food", "japanese"],
        "activity": ["indoor", "parent_child", "light_activity"],
        "avoid": ["long_queue", "crowded_mall"]
    },
    "history_feedback": [
        {
            "plan_id": "p001",
            "positive": ["kid_happy"],
            "negative": ["too_crowded", "too_far"]
        }
    ],
    "derived_defaults": {
        "max_distance_km": 8.0,
        "max_queue_time_min": 15,
        "preferred_duration_hours": [4, 6]
    },
    "value_profile": [...]
}
```

必须关注的 memory 要素：

- `stable_profile`：长期稳定画像，如消费层级、默认交通、常用区域。
- `companion_profile`：常见同行人画像，如孩子年龄、伴侣短期饮食状态。
- `preference_profile`：显式偏好和规避项。
- `history_feedback`：历史方案的正负反馈。
- `derived_defaults`：可进入规划的默认阈值。
- `value_profile`：价值维度及其证据。
- `confidence`：画像或推断的置信度。
- `source`：来源，区分当前输入、历史反馈、长期画像。
- `ttl`：短期状态和长期偏好的生命周期。

## Value 维度

当前 demo 使用 4 个 value 维度：

| value_id | 中文含义 | score | confidence | ttl | source | 用途 |
| --- | --- | ---: | ---: | --- | --- | --- |
| `family_care` | 儿童优先和家庭舒适 | `0.92` | `0.90` | `long_term` | `historical_profile` | 提升群体适配，要求亲子友好 |
| `health` | 健康饮食 | `0.82` | `0.78` | `short_term` | `recent_user_input` | 偏好低卡、轻食 |
| `convenience` | 少排队和少折腾 | `0.76` | `0.84` | `long_term` | `history_feedback` | 惩罚长排队、拥挤商场、远距离 |
| `cost_sensitivity` | 中等预算 | `0.45` | `0.55` | `long_term` | `stable_profile` | 控制预算但不强制最低价 |

Value 单项结构：

```python
{
    "value_id": "family_care",
    "label": "儿童优先和家庭舒适",
    "score": 0.92,
    "confidence": 0.9,
    "ttl": "long_term",
    "source": "historical_profile",
    "planning_effect": "increase group_fit and require kid_friendly activities",
    "evidence": ["常与5岁孩子同行", "历史反馈偏好低强度活动"]
}
```

## Constraints 兼容层

`constraints` 是给 B 阶段使用的合并结果。它来自 `intent`，再由 `memory` 补充默认值、规避项和值权重。

示例：

```python
{
    "task_type": "local_life_plan",
    "scene": "family",
    "time_window": "today_afternoon",
    "duration_range": [4, 6],
    "companions": [
        {
            "role": "wife",
            "state": "dieting",
            "needs": ["low_calorie", "light_food"]
        },
        {
            "role": "child",
            "age": 5,
            "needs": ["kid_friendly", "low_intensity"]
        }
    ],
    "people_count": 3,
    "origin": "home",
    "distance_preference": "nearby",
    "max_distance_km": 8.0,
    "transport_mode": "drive_or_taxi",
    "budget": None,
    "hard_tags": ["kid_friendly"],
    "soft_tags": ["light_food", "low_calorie", "low_intensity"],
    "avoid": ["crowded_mall", "long_queue", "too_far"],
    "value_weights": {
        "family_care": 0.92,
        "health": 0.82,
        "convenience": 0.76,
        "cost_sensitivity": 0.45
    },
    "value_confidence": {
        "family_care": 0.9,
        "health": 0.78,
        "convenience": 0.84,
        "cost_sensitivity": 0.55
    },
    "score_weights": {
        "group_fit": 0.392,
        "availability": 0.238,
        "route": 0.238,
        "health": 0.182,
        "budget": 0.1225,
        "experience": 0.1
    },
    "max_queue_time_min": 15,
    "memory_policy": "explicit_current_input_first"
}
```

合并规则：

- 当前输入中的显式距离、时间、同行人优先。
- Memory 可补充 `transport_mode`、`avoid`、`max_queue_time_min` 等默认值。
- Memory 可把 value 转成 `value_weights` 和 `score_weights`。
- Memory 不直接替用户选择方案，只影响后续过滤、打分和解释。

## Graph 接口

`src/graph.py` 暴露：

```python
def get_graph() -> Any:
    return build_graph()
```

调用：

```python
from src.graph import get_graph

graph = get_graph()
result = graph.invoke(
    {
        "user_id": "u001",
        "user_input": "今天下午想和老婆孩子出去玩几个小时，别离家太远，孩子5岁，老婆最近在减肥。"
    }
)
```

无论是否安装 LangGraph，调用方都使用 `invoke` 接口。

## 运行示例

命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m src.demo
```

默认输入：

```text
今天下午想和老婆孩子出去玩几个小时，别离家太远，孩子5岁，老婆最近在减肥。
```

输出顶层结构：

```json
{
  "intent": {},
  "memory": {},
  "constraints": {},
  "short_term_memory": [],
  "execution_log": []
}
```

关键输出节选：

```json
{
  "intent": {
    "scene": "family",
    "time": {
      "window": "today_afternoon",
      "duration_range": [4, 6]
    },
    "location": {
      "origin": "home",
      "distance_preference": "nearby",
      "max_distance_km": 8.0
    },
    "missing_slots": ["budget", "transport_mode"]
  },
  "memory": {
    "stable_profile": {
      "consumption_level": "middle",
      "default_transport": "drive_or_taxi"
    },
    "value_profile": [
      {
        "value_id": "family_care",
        "score": 0.92,
        "confidence": 0.9,
        "ttl": "long_term"
      }
    ]
  },
  "constraints": {
    "hard_tags": ["kid_friendly"],
    "soft_tags": ["light_food", "low_calorie", "low_intensity"],
    "avoid": ["crowded_mall", "long_queue", "too_far"],
    "max_queue_time_min": 15,
    "memory_policy": "explicit_current_input_first"
  }
}
```

## 测试

如果环境安装了 `pytest`：

```bash
python -m pytest tests/test_weekendflow_demo.py
```

当前环境没有 `pytest` 时，可以使用 smoke test：

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from src.graph import get_graph

result = get_graph().invoke({
    "user_id": "u001",
    "user_input": "今天下午想和老婆孩子出去玩几个小时，别离家太远，孩子5岁，老婆最近在减肥。",
})

assert result["scene_type"] == "family"
assert result["intent"]["location"]["distance_preference"] == "nearby"
assert result["constraints"]["value_weights"]["family_care"] == 0.92
assert result["memory"]["companion_profile"]["child"]["age"] == 5
assert result["constraints"]["memory_policy"] == "explicit_current_input_first"
print("A-stage rebuilt smoke test passed")
PY
```

## 后续模块接口约定

B 阶段建议优先读取：

- `constraints.scene`
- `constraints.time_window`
- `constraints.duration_range`
- `constraints.companions`
- `constraints.hard_tags`
- `constraints.soft_tags`
- `constraints.avoid`
- `constraints.max_distance_km`
- `constraints.max_queue_time_min`
- `constraints.transport_mode`
- `constraints.value_weights`
- `constraints.value_confidence`
- `constraints.score_weights`
- `constraints.missing_slots`

用于解释或调试时读取：

- `intent`：说明当前输入如何被理解。
- `memory`：说明用户画像和值从哪里来。
- `memory.value_profile[].evidence`：生成“为什么这样推荐”的证据。

当前约定：显式当前输入 > 短期 memory > 长期 memory > 派生默认值。
