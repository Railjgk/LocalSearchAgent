# WeekendFlow C Mock API Layer 需求文档

版本：v0.2
日期：2026-05-16
面向对象：C 模块同学
上游：B Planner / Optimizer
下游：C Tool Router / Execution Manager / Mock API

## 1. 核心边界

C 不需要重新做推荐排序，也不需要重新造一套 POI/商家/产品/券数据。

B 负责：
- 维护 normalized local-life supply fixture。
- 生成活动/餐厅候选组合。
- 根据偏好、路线、预算、可用性、体验、风险做排序。
- 输出 `selected_plan`、`action_hints`、`execution_contract`。

C 负责：
- 读取或镜像 B 的 fixture ID。
- 校验指定 `poi_id/product_id/deal_id/time` 当前是否可执行。
- 执行预约、买券、下单、commit 的 mock 状态推进。
- 返回失败原因、替代时段、重试记录和原始 mock/API 结果。

一句话边界：

> B 决定“推荐什么、为什么推荐、是否值得推荐”；C 决定“按 B 给的动作现在能不能执行、执行后状态是什么”。

## 2. 当前 B 数据底座

B 当前默认读取：

```text
experiments/mock_data/
  activities.json
  restaurants.json
  merchants.json
  products.json
  deals.json
  availability.json
  routes.json
```

当前主 mock 数据规模：

| 文件 | 数量 | 说明 |
| --- | ---: | --- |
| `activities.json` | 25 | 活动/体验 POI |
| `restaurants.json` | 23 | 餐厅 POI |
| `merchants.json` | 48 | 商家主体 |
| `products.json` | 57 | 可售卖/可预约产品 |
| `deals.json` | 49 | 券/团购/套餐 |
| `availability.json` | 48 | 可用性、库存、时段 |
| `routes.json` | 10 | 离线路线 overlay |

`experiments/mock_data/gaode_seed_v1/` 是真实上海 POI seed，可用于后续扩充数据；C v0.1 不需要直接依赖它。

## 3. 共享 ID 约定

C 必须复用 B 的 ID，不要生成另一套业务 ID。

| 字段 | 归属 | 要求 |
| --- | --- | --- |
| `poi_id` | B/C 共享 | 活动或餐厅主 ID，所有执行接口必须能按它查询。 |
| `merchant_id` | B/C 共享 | 商家主 ID。 |
| `product_id` | B/C 共享 | 可预约/可购买产品 ID。 |
| `deal_id` | B/C 共享，可为空 | 券/团购/套餐券 ID。若 B 发现当前时间不适用任何券，会传 `null`，C 应按 `product_id` 继续执行。 |
| `amap_id` | 可选 | 高德真实 POI ID，用于路线和真实 POI 溯源。 |
| `plan_id` | B 生成 | B 方案 ID。 |
| `execution_id` | C 生成 | C 执行链路 ID。 |
| `reservation_id` | C 生成 | 预约 mock ID。 |
| `order_id` | C 生成 | 订单 mock ID。 |

## 4. B 给 C 的输入

B 当前会输出 `selected_plan`，C 重点消费下面几个字段：

```json
{
  "plan_id": "plan_001",
  "supply_identity": {
    "activity_id": "act_micro_vacation_spa",
    "activity_name": "近场温泉康养半日体验",
    "activity_category": "micro_vacation",
    "restaurant_id": "res_spa_light_tea",
    "restaurant_name": "云栖温泉轻餐茶室",
    "restaurant_category": "spa_light_food"
  },
  "timeline": [],
  "route": {},
  "availability": {},
  "action_hints": [],
  "execution_contract": {
    "ready": true,
    "blocking_reasons": [],
    "checks": []
  },
  "execution_ready": true
}
```

### 4.1 `action_hints`

B 当前 action types：

- `order_activity_ticket`
- `reserve_restaurant`

示例：

```json
[
  {
    "action_type": "order_activity_ticket",
    "poi_id": "act_micro_vacation_spa",
    "merchant_id": "m_act_micro_vacation_spa",
    "product_id": "prod_micro_spa_two_person",
    "deal_id": "deal_act_micro_spa",
    "time": "14:00",
    "quantity": 2,
    "requires_reservation": true,
    "product_type": "activity_ticket",
    "inventory_model": "slot_capacity",
    "fulfillment_mode": "onsite_verify",
    "deal_type": "activity_ticket",
    "coupon_type": "two_person_relax"
  },
  {
    "action_type": "reserve_restaurant",
    "poi_id": "res_spa_light_tea",
    "merchant_id": "m_res_spa_light_tea",
    "product_id": "prod_spa_light_tea_couple_set",
    "deal_id": "deal_res_spa_light_tea",
    "time": "17:30",
    "people": 2,
    "requires_reservation": true,
    "product_type": "meal_package",
    "inventory_model": "table_slot",
    "fulfillment_mode": "dine_in_reservation",
    "deal_type": "meal_coupon",
    "coupon_type": "spa_light_dinner"
  }
]
```

约定：
- 活动人数用 `quantity`。
- 餐厅人数用 `people`。
- `time` 必须落在对应 POI 的 `available_slots` 或 `reservation_slots`。
- `deal_id` 可以为 `null`。这表示当前排程时间没有适配券，C 不能因此直接失败，应走产品预约/下单。
- C 可把 `action_hints` 转成自己的 `action_sequence`，但必须保留原始 ID。

### 4.2 `execution_contract`

B 已经在 optimizer 内做了一层执行合同校验：

```json
{
  "ready": true,
  "blocking_reasons": [],
  "checks": [
    {
      "name": "activity_product_id_known",
      "status": "pass",
      "message": "activity product_id must exist in selected supply"
    }
  ]
}
```

C 可以把 `execution_contract.ready=true` 作为“B 侧数据结构自洽”的信号，但仍然需要在执行前重新查自己的 mock state，因为库存和时段可能变化。

## 5. C 需要维护的 mock state

建议 C 新增目录：

```text
experiments/mock_data/c_execution/
  availability_state.json
  reservation_state.json
  coupon_state.json
  order_state.json
  route_state.json
  execution_state.json
```

C state 只表达执行状态，不表达推荐分数。

C 不负责维护：
- `route_score`
- `trust_score`
- `experience_score`
- `risk_score`
- `value_for_money_score`
- final ranking
- `selected_plan`

这些属于 B。

## 6. 接口需求

所有接口建议返回统一基础字段：

```json
{
  "success": true,
  "status": "available",
  "failure_reason": null,
  "verified_fields": [],
  "estimated_fields": [],
  "alternatives": [],
  "raw_api_results": {
    "source": "c_mock_api",
    "mock_version": "v0.2"
  }
}
```

### 6.1 `/availability/check`

用途：检查某个 POI/product/deal 在指定时间是否可用。

请求：

```json
{
  "poi_id": "res_spa_light_tea",
  "merchant_id": "m_res_spa_light_tea",
  "product_id": "prod_spa_light_tea_couple_set",
  "deal_id": "deal_res_spa_light_tea",
  "time": "17:30",
  "party_size": 2
}
```

响应：

```json
{
  "success": true,
  "status": "available",
  "available": true,
  "inventory_left": 8,
  "capacity_limit": 24,
  "reservation_required": true,
  "queue_time_min": 8,
  "alternative_slots": ["18:30", "19:30"],
  "verified_fields": ["available", "inventory_left", "reservation_required"],
  "estimated_fields": ["queue_time_min"],
  "failure_reason": null
}
```

必须支持失败原因：
- `unknown_poi`
- `unknown_product`
- `unknown_deal`
- `slot_full`
- `inventory_empty`
- `merchant_closed`
- `holiday_closed`
- `party_size_exceeded`
- `product_unavailable`

### 6.2 `/route/check`

用途：提供路线事实。路线会被 B 在规划阶段使用，不只是 C 执行后检查。

请求：

```json
{
  "from_id": "act_micro_vacation_spa",
  "to_id": "res_spa_light_tea",
  "origin": "121.472,31.356",
  "destination": "121.478,31.350",
  "mode": "walking",
  "max_duration_min": 45,
  "city": "上海"
}
```

响应：

```json
{
  "success": true,
  "status": "feasible",
  "feasible": true,
  "mode": "walking",
  "duration_min": 8,
  "distance_km": 0.8,
  "traffic_status": "low",
  "walking_time_min": 8,
  "failure_reason": null,
  "verified_fields": ["duration_min", "distance_km"],
  "estimated_fields": ["traffic_status"],
  "raw_api_results": {
    "source": "offline_routes_json"
  }
}
```

实现优先级：
1. 先读 `experiments/mock_data/routes.json`。
2. 如果接高德 RoutePlanner，再把高德结果标准化成同一响应结构。
3. API timeout 时返回失败原因，B 可使用离线 fallback。

必须支持失败原因：
- `route_too_far`
- `route_not_found`
- `mode_unavailable`
- `api_timeout`
- `traffic_peak_risk`
- `weather_risk`

### 6.3 `/reservation/check`

用途：检查可预约性，但不创建预约。

请求：

```json
{
  "poi_id": "res_spa_light_tea",
  "merchant_id": "m_res_spa_light_tea",
  "product_id": "prod_spa_light_tea_couple_set",
  "time": "17:30",
  "party_size": 2
}
```

响应：

```json
{
  "success": true,
  "status": "reservable",
  "reservable": true,
  "available_slots": [
    {"time": "17:30", "inventory_left": 8},
    {"time": "18:30", "inventory_left": 6}
  ],
  "failure_reason": null,
  "verified_fields": ["reservable", "available_slots"],
  "estimated_fields": []
}
```

### 6.4 `/reservation/create`

用途：创建 mock 预约。

请求：

```json
{
  "plan_id": "plan_001",
  "step_id": "step_restaurant_1",
  "poi_id": "res_spa_light_tea",
  "merchant_id": "m_res_spa_light_tea",
  "product_id": "prod_spa_light_tea_couple_set",
  "time": "17:30",
  "party_size": 2,
  "user_id": "mock_user_001"
}
```

响应：

```json
{
  "success": true,
  "status": "confirmed",
  "reservation_id": "rsv_mock_001",
  "poi_id": "res_spa_light_tea",
  "time": "17:30",
  "expires_at": "2026-05-16T17:20:00+08:00",
  "retryable": false,
  "failure_reason": null
}
```

### 6.5 `/coupon/check`

用途：检查券是否可购买、可核销、可退。`deal_id=null` 时应返回 `status=skipped_no_deal`，不要报错。

请求：

```json
{
  "deal_id": "deal_res_spa_light_tea",
  "product_id": "prod_spa_light_tea_couple_set",
  "merchant_id": "m_res_spa_light_tea",
  "time": "17:30"
}
```

响应：

```json
{
  "success": true,
  "status": "coupon_available",
  "purchase_available": true,
  "redeem_available": true,
  "sale_price": 190,
  "refund_policy": "before_1h_free",
  "hidden_cost_risk": "low",
  "failure_reason": null
}
```

必须支持失败原因：
- `coupon_sold_out`
- `coupon_expired`
- `not_valid_for_slot`
- `requires_reservation_first`
- `nonrefundable_risk`
- `hidden_addon_risk`

### 6.6 `/coupon/buy`

用途：mock 买券，不需要真实支付。

请求：

```json
{
  "plan_id": "plan_001",
  "step_id": "step_restaurant_1",
  "deal_id": "deal_res_spa_light_tea",
  "product_id": "prod_spa_light_tea_couple_set",
  "merchant_id": "m_res_spa_light_tea",
  "quantity": 1
}
```

响应：

```json
{
  "success": true,
  "status": "paid_mock",
  "order_id": "ord_mock_001",
  "deal_id": "deal_res_spa_light_tea",
  "amount": 190,
  "payment_required": false,
  "redeem_code": "MOCK-839201",
  "refund_policy": "before_1h_free"
}
```

### 6.7 `/order/create`

用途：统一创建订单，可覆盖活动票、餐券、套餐。

请求：

```json
{
  "plan_id": "plan_001",
  "step_id": "step_activity_1",
  "order_type": "activity_ticket",
  "poi_id": "act_micro_vacation_spa",
  "merchant_id": "m_act_micro_vacation_spa",
  "product_id": "prod_micro_spa_two_person",
  "deal_id": "deal_act_micro_spa",
  "quantity": 2,
  "time": "14:00"
}
```

响应：

```json
{
  "success": true,
  "status": "created",
  "order_id": "ord_mock_002",
  "order_type": "activity_ticket",
  "amount": 238,
  "payment_required": false,
  "completion_status": "pending_use",
  "failure_reason": null
}
```

### 6.8 `/execution/commit`

用途：C 根据 B 的 `action_hints` 执行完整 mock 链路。

请求：

```json
{
  "plan_id": "plan_001",
  "user_id": "mock_user_001",
  "action_hints": []
}
```

响应：

```json
{
  "success": true,
  "execution_id": "exec_mock_001",
  "plan_id": "plan_001",
  "overall_status": "completed",
  "partial_success": false,
  "failed_step": null,
  "steps": [
    {
      "step_id": "step_activity_1",
      "action_type": "order_activity_ticket",
      "status": "ordered",
      "order_id": "ord_mock_002",
      "failure_reason": null
    },
    {
      "step_id": "step_restaurant_1",
      "action_type": "reserve_restaurant",
      "status": "reserved",
      "reservation_id": "rsv_mock_001",
      "failure_reason": null
    }
  ],
  "retry_history": [],
  "raw_api_results": {
    "source": "execution_manager_mock",
    "mock_version": "v0.2"
  }
}
```

### 6.9 `/restaurants/search` 可选

这个接口不是 v0.1 必需。

如果 C 实现它，只做“餐厅候选事实查询”，不做排序：

- 可以返回餐厅基本信息、堂食能力、可预约时段、队列、产品/券 ID。
- 不返回 `score`、`rank`、`selected`。
- B 仍然负责餐厅排序、方案组合和最终推荐。

如果 C 暂不实现，B 继续直接读 `experiments/mock_data/restaurants.json`。

## 7. 状态枚举

通用状态：
- `available`
- `unavailable`
- `checked`
- `confirmed`
- `created`
- `paid_mock`
- `reserved`
- `ordered`
- `failed`
- `retrying`
- `partial_success`
- `completed`
- `cancelled`
- `skipped_no_deal`

失败原因：
- `unknown_poi`
- `unknown_product`
- `unknown_deal`
- `slot_full`
- `inventory_empty`
- `capacity_exceeded`
- `merchant_closed`
- `holiday_closed`
- `reservation_not_supported`
- `party_size_exceeded`
- `product_unavailable`
- `route_too_far`
- `route_not_found`
- `traffic_peak_risk`
- `weather_risk`
- `mode_unavailable`
- `api_timeout`
- `coupon_sold_out`
- `coupon_expired`
- `not_valid_for_slot`
- `requires_reservation_first`
- `nonrefundable_risk`
- `hidden_addon_risk`
- `payment_failed_mock`
- `temporary_api_failure`
- `schema_mismatch`

## 8. Merge 方式

推荐流程：

1. B 维护 normalized fixture。
2. C 读取或复制 `merchants/products/deals/availability/routes`。
3. C 在 `c_execution/` 维护动态执行状态，不直接改 B fixture。
4. B 输出 `selected_plan.action_hints`。
5. C 执行 `/execution/commit`，返回执行状态和 retry history。
6. B 或上层 UI 根据 C 结果展示成功、失败、替代时段或需要用户确认的动作。

不要做：

- C 重新生成独立 `poi_id/product_id/deal_id`。
- C 返回推荐排序。
- C 把 B score 写成 mock API 的事实字段。

## 9. 验收标准

v0.1 最小验收：

- C 可以按 B 给的 `poi_id/product_id/deal_id` 查到执行状态。
- `/availability/check` 能返回可用、不可用、失败原因、替代时段。
- `/route/check` 能返回路线可行性、距离、时长，并支持离线 `routes.json`。
- `/reservation/check` 和 `/reservation/create` 能处理餐厅预约。
- `/coupon/check` 能处理 `deal_id=null` 和 `not_valid_for_slot`。
- `/order/create` 能处理活动票或餐券订单。
- `/execution/commit` 能返回多步骤状态和 `retry_history`。
- 所有接口都保留 `raw_api_results`，便于 eval/debug。
- C 不做最终推荐排序。

建议联调用例：

- `micro_vacation_relaxation`：`act_micro_vacation_spa + res_spa_light_tea`
- `execution_slot_alignment`：检查 `execution_contract.ready=true`
- `strict_holiday_like_no_ticket_relaxation`：B 无可行方案，C 不应被调用执行
