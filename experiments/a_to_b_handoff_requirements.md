# WeekendFlow A -> B Handoff Requirements

版本：v0.1
负责人视角：B 戴部分对 A 阶段的输入需求
状态：草案，可发给 A 同学对齐

## 1. 背景

B 现在已经接入了更完整的本地生活供给侧 mock 数据，包括：

- 真实/半真实 POI 底座：活动、餐厅、坐标、商圈、类目。
- B 本地供给字段：商家、产品、券、库存、排队、可预约时段、健康菜单、堂食能力、路线估算。
- B optimizer：会综合偏好、亲子/低卡匹配、路线、预算、库存、体验、风险做排序。

目前 B 的主要剩余问题不是“没有数据”，而是 A 传入 B 的意图信号还不够结构化。有些 case 虽然能出方案，但 B 需要靠 `scenario_activities` 的泛化标签猜用户到底要什么，导致偏好匹配分偏低。

## 2. 对接目标

A 需要把自然语言需求拆成稳定、可计算的 handoff state，让 B 不再从一句话里二次猜测。

B 不要求 A 排序 POI，也不要求 A 生成具体商家。A 只负责：

- 理解用户是谁、和谁出行、什么时候出行。
- 提取硬约束、软偏好、规避项。
- 把中文表达映射成 B 能直接消费的 canonical tags。
- 给出路线起点/时间窗等规划上下文。

B 负责：

- 召回活动/餐厅候选。
- 检查距离、排队、预算、时长、低龄儿童、低卡/轻食等硬约束。
- 结合产品、券、库存、路线和体验分做 optimizer 排序。
- 输出 `selected_plan.action_hints` 给 C。

## 3. A 必须输出的字段

### 3.1 顶层字段

```json
{
  "scene_type": "family",
  "constraints": {},
  "user_profile": {},
  "scenario_activities": [],
  "scenario_template": {},
  "route_pattern_hints": {}
}
```

### 3.2 `scene_type`

必须是以下枚举之一：

- `family`
- `friends`
- `couple`
- `low_budget`
- `solo`

不要传中文场景名给 B。中文可以保留在解释字段里，但规划字段要 canonical。

### 3.3 `constraints`

建议结构：

```json
{
  "scene": "family",
  "people_count": 3,
  "companions": [
    {"role": "wife", "state": "dieting", "needs": ["low_calorie", "light_food"]},
    {"role": "child", "age": 5, "needs": ["kid_friendly", "low_intensity"]}
  ],
  "child_age": 5,
  "mom_diet": "low_calorie",
  "budget": 520,
  "max_distance_km": 8,
  "max_queue_time_min": 25,
  "duration_range": [4, 6],
  "start_time": "14:00",
  "time_window": "today_afternoon",
  "route_origin": "121.508,31.308",
  "route_mode": "driving",
  "hard_tags": ["kid_friendly"],
  "soft_tags": ["low_intensity", "low_calorie", "light_food", "nearby"],
  "avoid": ["long_queue", "crowded_mall"],
  "planning_preferences": {
    "activity_type": ["parent_child", "indoor", "light_activity"],
    "food_type": ["low_calorie", "light_food", "low_oil"],
    "emotion_type": ["relaxation"],
    "atmosphere_type": ["quiet"],
    "experience_type": ["hands_on_parent_child"],
    "restaurant_type": ["dine_in"]
  }
}
```

字段要求：

- `people_count`：必须尽量给出。没有则 B 会根据 companions 推断，但不如 A 明确。
- `companions`：建议保留角色、年龄、状态、needs。B 会用 child/wife/dieting 等字段推断亲子和低卡需求。
- `child_age`：如果有孩子且年龄可识别，必须给。6 岁及以下会触发 B 的低龄儿童友好硬约束。
- `mom_diet`：如果出现减肥、控卡、低脂、少油、清淡，统一给 `low_calorie`。
- `budget`：总预算，不是人均预算。若 A 只识别人均，请加 `budget_type: "per_person"`，但当前 B 默认按总预算处理。
- `max_distance_km`：A 可以按“附近/别太远/跨区也可以”给不同默认值。
- `max_queue_time_min`：用户明确“别排队/少排队/能直接去”时要降低。
- `duration_range`：当前 B 支持小时格式 `[4, 6]` 或分钟格式 `[240, 360]`。
- `start_time`：使用 `HH:MM`，例如 `14:00`。这会影响 B 的活动和餐厅 slot 排期。
- `route_origin`：如果 A 能拿到用户起点坐标，尽量传 `"lng,lat"`。没有则 B 会退回离线路线/坐标估算。
- `hard_tags`：必须满足，不满足会被 B hard filter。
- `soft_tags`：偏好项，会影响召回和排序。
- `avoid`：规避项。比如 `long_queue`、`crowded_mall`、`high_calorie`、`takeaway_only`、`few_reviews`、`new_merchant`。

## 4. A 应使用的 canonical tags

### 4.1 活动类

| 用户表达 | A 输出 tag |
| --- | --- |
| 亲子、孩子、小朋友 | `parent_child`, `kid_friendly` |
| 低强度、不累、轻松 | `light_activity`, `low_intensity` |
| 室内、下雨天 | `indoor` |
| 手作、陶艺、画画 | `handcraft`, `hands_on_parent_child` |
| Citywalk、逛街、本地生活 | `citywalk`, `local_culture`, `local_market` |
| 市集 | `local_market`, `local_culture` |
| 微度假、温泉、康养、放松 | `micro_vacation`, `wellness`, `relaxation`, `healing` |
| 朋友聚会、社交 | `group_activity`, `social`, `group_friendly` |
| 约会、情侣、氛围 | `date_activity`, `romantic`, `atmosphere` |

### 4.2 餐饮类

| 用户表达 | A 输出 tag |
| --- | --- |
| 轻食、低卡、减肥、清淡 | `low_calorie`, `light_food`, `low_oil` |
| 少油、少盐 | `low_oil`, `low_sugar` |
| 健康、有机、食材好 | `healthy`, `fresh_ingredients`, `vegetable_rich` |
| 火锅、热闹、朋友局 | `hotpot`, `social` |
| 本帮菜、本地口味 | `regional_home_cuisine`, `local_flavor` |
| 堂食、订座 | `dine_in` |
| 外带、打包 | `takeaway_only` |

### 4.3 风险/规避类

| 用户表达 | A 输出 tag |
| --- | --- |
| 不想排队、人少点 | `long_queue` 放入 `avoid` |
| 不想去商场、人挤 | `crowded_mall` 放入 `avoid` |
| 不要太油、不想高热量 | `high_calorie` 放入 `avoid` |
| 要靠谱、别踩雷 | `few_reviews`, `new_merchant` 放入 `avoid`；同时设置 `trust_need: "high"` |
| 不要外卖/只要堂食 | `takeaway_only` 放入 `avoid`；`restaurant_type` 加 `dine_in` |

## 5. `scenario_activities` 规范

`scenario_activities` 是 B 的召回提示，不应该只放中文短语。建议同时放 canonical tags。

推荐：

```json
["parent_child", "indoor", "light_activity", "low_calorie", "nearby"]
```

可以附加中文解释，但不要只给中文：

```json
["parent_child", "indoor", "light_activity", "亲子乐园", "轻食餐厅"]
```

不推荐：

```json
["亲子乐园", "低强度室内活动", "轻食餐厅"]
```

原因：中文可以被 B 做部分映射，但覆盖不如 canonical tags 稳定。

## 6. 路线字段要求

路线现在已经进入 B optimizer。A 不需要调用路线 API，但需要提供路线规划上下文：

```json
{
  "route_origin": "121.508,31.308",
  "route_mode": "driving",
  "city": "上海",
  "max_distance_km": 8,
  "transport_mode": "driving"
}
```

如果 A 没有坐标，只能传文字地点，也可以传：

```json
{
  "location": {
    "origin": "杨浦区大学路附近"
  }
}
```

B 当前最优先消费坐标。文字地点后续需要 C/地图服务补 geocode。

## 7. 示例 handoff

### 7.1 亲子低卡场景

```json
{
  "scene_type": "family",
  "constraints": {
    "scene": "family",
    "people_count": 3,
    "child_age": 5,
    "mom_diet": "low_calorie",
    "max_distance_km": 8,
    "max_queue_time_min": 25,
    "duration_range": [4, 6],
    "budget": 520,
    "start_time": "14:00",
    "route_origin": "121.508,31.308",
    "hard_tags": ["kid_friendly"],
    "soft_tags": ["low_intensity", "low_calorie", "light_food", "nearby"],
    "avoid": ["long_queue", "crowded_mall"],
    "planning_preferences": {
      "activity_type": ["parent_child", "indoor", "light_activity"],
      "food_type": ["low_calorie", "light_food", "low_oil"],
      "restaurant_type": ["dine_in"]
    }
  },
  "user_profile": {
    "food_preference": ["light_food", "japanese"],
    "preference_profile": {
      "pace": "relaxed"
    }
  },
  "scenario_activities": [
    "parent_child",
    "indoor",
    "light_activity",
    "low_calorie",
    "nearby"
  ]
}
```

### 7.2 微度假情侣场景

```json
{
  "scene_type": "couple",
  "constraints": {
    "people_count": 2,
    "max_distance_km": 9,
    "max_queue_time_min": 30,
    "duration_range": [4, 6],
    "budget": 850,
    "start_time": "14:00",
    "ritual_need": true,
    "planning_preferences": {
      "activity_type": ["micro_vacation", "wellness"],
      "food_type": ["light_food"],
      "emotion_type": ["relaxation", "healing"],
      "atmosphere_type": ["quiet", "romantic"]
    }
  },
  "user_profile": {
    "emotion_need": ["relaxation", "healing"],
    "food_preference": ["light_food", "atmosphere"]
  },
  "scenario_activities": [
    "micro_vacation",
    "relaxation",
    "healing",
    "light_food"
  ]
}
```

## 8. 验收标准

A/B 联调时建议用这些标准验收：

- A 输出的 `scene_type` 必须落在 B 支持枚举内。
- 亲子 case 必须能传 `child_age` 或 companions 中的 child age。
- 减脂/低卡 case 必须能传 `mom_diet: "low_calorie"` 或 food tags。
- 如果用户说“附近/别太远”，必须传 `max_distance_km` 和 `nearby`。
- 如果用户说“不排队/少排队”，必须传 `max_queue_time_min` 或 `avoid: ["long_queue"]`。
- 如果用户说“堂食/订座”，必须传 `restaurant_type: ["dine_in"]` 或 `avoid: ["takeaway_only"]`。
- 情绪类需求必须进入 `emotion_need` 或 `planning_preferences.emotion_type`，不要只放在自然语言原文。
- A 输出后跑 `python experiments/run_b_eval.py`，B eval 应保持 18/18。
- 跑 `python experiments/analyze_b_plan_quality.py`，不能出现 `intent_category_miss`、`health_intent_miss`、`missing_execution_target_ids`。

## 9. 当前 B 已做的兼容

B 侧已经兼容以下 A 中间标签：

- `parent_child` -> `kid_friendly`, `family_friendly`
- `light_activity` -> `low_intensity`
- `group_activity` -> `group_friendly`, `social`
- `date_activity` -> `romantic`, `atmosphere`
- `budget_activity` / `budget_restaurant` -> `budget`
- `healthy` -> `low_calorie`, `light_food`
- `relaxed` / `comfortable` -> `low_intensity`

但这只是兜底兼容。正式联调仍建议 A 直接输出 canonical tags，减少歧义。

## 10. 后续待确认

- A 是否能稳定给出 `route_origin` 坐标；如果不能，需要 C 或地图层提供 geocode。
- A 是否区分“总预算”和“人均预算”；B 当前默认总预算。
- A 是否能识别“情绪价值/仪式感/微度假/本地文化”这类研报高频动机，并输出到结构化字段。
- A 是否需要保留 `preference_evidence`，例如每个 tag 来自用户原话哪一段，方便 B explanation 后续引用。
