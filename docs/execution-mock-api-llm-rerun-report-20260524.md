# C Execution Mock API Fix and LLM Rerun Report

## Summary

This change fixes C-stage execution mock API compatibility with the current B-stage supply fixtures and records the result of a full pipeline rerun with model API switches enabled.

The original failure was that C returned `unknown_poi` for valid B-selected POIs such as `邻里桌游咖啡馆`. The POI existed in `activities.json`, but C's reference resolver still expected older fixture shapes.

## Code Changes

File changed: `src/tools/execution_mock_api.py`

- Added merchant-to-POI matching for both legacy `merchant.poi_id` and current `merchant.poi_ids[]`.
- Added deal validation that accepts current B fixture shape where deals are linked by `poi_id` and `product_id`, while `merchant_id` may be absent.
- Added product fallback from POI `product_ids[]` when `default_product_id` is not present.
- Added availability parsing for current POI-keyed overlays:
  - `availability[poi_id].available_slots[].time`
  - `availability[poi_id].available_slots[].inventory_left`
  - `availability[poi_id].queue_time_min`
- Kept compatibility with legacy product-keyed availability overlays used by tests and mutable C state.
- Added coupon compatibility for:
  - `valid_time` as an alias for `valid_slots`
  - `stock_limit_per_slot` as default remaining coupon inventory
  - `sale_price` as coupon amount
- Added route fixture compatibility for current `routes.json` object shape:
  - reads `overrides[]` instead of assuming the root is a list
  - supports `from`/`to` as aliases for `from_id`/`to_id`
  - normalizes `driving`/`walking` to C's internal route modes
  - ignores route metadata objects so they are not treated as route records

## Verification

Direct checks run:

```bash
python -m py_compile src/tools/execution_mock_api.py
```

```bash
python - <<'PY'
from src.tools.execution_mock_api import availability_check, coupon_check

payload = dict(
    poi_id="act_board_game_cafe",
    merchant_id="m_act_board_game_cafe",
    product_id="prod_board_game_cafe_table",
    deal_id="deal_act_board_game_cafe",
    time="14:00",
    party_size=2,
)

print(availability_check(**payload)["success"])
print(coupon_check(**payload)["amount"])
PY
```

Observed result:

- `availability_check` returned `True`.
- `coupon_check` returned amount `90`.

## Full Pipeline Rerun

Command:

```bash
set -a
source .env
set +a
python run.py
```

User input:

```text
今天下午和老婆孩子出去玩，孩子5岁，老婆最近在减肥
```

Generated trace:

- `docs/run-status-test-20260524-190339.md`
- `docs/run-status-test-20260524-190339.json`

Model API usage observed:

- A-stage intent parsing used LongCat OpenAI-format API successfully.
- B-stage explainability used LongCat API successfully.
- B-stage semantic hints did not run because `WF_B_AI_SEMANTIC_HINTS_ENABLED` was set to an empty string, which the current code treats as explicit false instead of falling back to `WF_B_AI_ENABLED=1`.

Pipeline result:

- Final execution status: `partial`
- Payment status: `not_required`
- Selected plan: `plan_001`
- Optimization score: `93.51`

Selected itinerary:

- `15:30-17:00` 树屋亲子陶艺体验馆
- `17:00-17:30` 附近休息与转场
- `17:30-19:00` 禾间轻食日料

Execution results:

- `order_activity_ticket` for 树屋亲子陶艺体验馆 succeeded.
- `reserve_restaurant` for 禾间轻食日料 succeeded.
- `order_addon_service` for 庆祝蛋糕 failed with `unknown_poi`.

## Remaining Issue

The remaining `partial` status is not the original POI fixture mismatch. It comes from `tool_router` generating an addon action:

```json
{
  "action_type": "order_addon_service",
  "addon_type": "cake",
  "name": "庆祝蛋糕"
}
```

That action has no `poi_id`, `merchant_id`, `product_id`, or `deal_id`, so C correctly cannot resolve it through the POI/product/deal execution contract.

Recommended next fix:

- Either stop generating `order_addon_service` unless an addon product exists in supply fixtures, or
- Add first-class addon fixtures and C execution support for addon services.

