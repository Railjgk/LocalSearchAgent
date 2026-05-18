# WeekendFlow Mock Data Strategy

## Current Position

B uses a hybrid supply model:

- Real POI grounding from Gaode: `amap_id`, name, address, coordinates, district/adcode, phone, and route overlays when explicitly generated.
- B-owned business mock fields: inventory, queue, bookable slots, deals, product packages, refund/cancel policies, child friendliness, health menu options, trust, operation risk, and commercial guardrails.

This keeps planning and eval deterministic while making the demo data feel close to real Shanghai local-life supply.

## Mocking Rules

Use real POIs for place identity whenever possible, but do not pretend Gaode provides Meituan-style supply facts.

Mock data should be treated like a training/eval fixture, not decorative demo
content. Keep these field-source boundaries clear:

- `observed_fixture`: stable facts from Gaode or curated source material, such
  as name, address, coordinates, `amap_id`, phone and source evidence.
- `rule_imputed_fixture`: business fields imputed by WeekendFlow rules, such as
  category, service mode, health menu options, refund policy, activity duration
  and package structure.
- `execution_mock_state`: mutable execution state, such as inventory, slots,
  queue, reservation availability, coupon availability, order status and route
  query result.
- `b_derived_or_prior`: scores or priors used by B, such as `trust_score`,
  `content_heat_score`, `operation_stability_score`, `ritual_score` and future
  `route_score`. These may be materialized for the current baseline, but they
  should not be treated as verified real-world facts.

Category coverage should be deliberately balanced:

- Parent-child: indoor playground, handcraft, science/museum, child-friendly restaurants.
- Friends/social: hotpot, board games, escape room, frisbee/sports, citywalk.
- Couple/relaxation: exhibition, handcraft, wellness/spa, atmospheric local streets.
- Low-budget/health: light food, salad, simple local meals, free/low-price citywalk or museum.
- Local culture: Wukang Road, Anfu Road, Yuyuan Road, Tianzifang, museums, galleries, markets.

Practical realism rules:

- Each POI must have a stable `poi_id`, `name`, `type`, `location`, coordinates when real, and source evidence.
- Each demo-ready POI should have matching `merchant_id`, product, deal, and availability records.
- Inventory and queue should vary by category: hotpot and popular attractions have higher peak risk; museums and citywalk have lower direct inventory dependence; handcraft and parent-child activities require slots.
- Health claims should be conservative. Light food can have `low_calorie`; hotpot/fried food should carry high-calorie risk even if healthy options exist.
- Commercial fields should not dominate ranking. `ad_boost` must not be enough to beat trust, fit, route, and availability.
- Curated real POIs added into the main mock should have moderate trust/heat scores so handwritten eval anchors remain stable.

## Data Locations

- `experiments/mock_data`: default deterministic B mock used by eval and normal local runs.
- `experiments/mock_data/gaode_seed_v1`: larger real Shanghai POI seed for demo and inspection.
- `experiments/mock_data/gaode_seed_smoke*`: small smoke outputs used to verify Gaode key/API behavior.

Runtime source switching:

```powershell
$env:WF_MOCK_DATA_DIR="experiments/mock_data/gaode_seed_v1"
```

Optional source order override:

```powershell
$env:WF_DATA_SOURCE_ORDER="local_supply_mock,c_mock_api"
```

The default remains `experiments/mock_data`, so offline eval is reproducible.

## Local vs Cloud Mock

Use local JSON mock as the source of truth during this stage. It is versioned, reviewable, and deterministic.

Cloud mock can come later when the schema is stable:

- Put the same normalized records into a shared table or object store.
- Keep `poi_id`, `merchant_id`, `product_id`, and `deal_id` stable across local and cloud.
- Export a JSON snapshot for eval, so model/planner tests do not depend on live cloud state.
- Treat cloud as a deployment/distribution layer, not as the first place where schema decisions are made.

## Merge Contract With C Mock API Layer

B should emit normalized planning candidates and selected plan `action_hints`.
C should own execution-style mock APIs: order creation, reservation confirmation, ticket verification, queue check, and final route validation.

Recommended shared IDs:

- `poi_id`: B candidate and C place/order target.
- `merchant_id`: merchant capability and trust lookup.
- `product_id`: ticket/meal/package to reserve or buy.
- `deal_id`: coupon/package applied to the plan.
- `amap_id`: optional real-world POI reference for route/geocode.

Recommended merge path:

1. Keep B's normalized supply JSON as the fixture source.
2. C imports or mirrors `merchants.json`, `products.json`, `deals.json`, and `availability.json`.
3. C mock API responses use these IDs instead of inventing separate IDs.
4. B selected plans include `action_hints` with `poi_id`, `merchant_id`, `product_id`, `deal_id`, preferred slot, and route origin/destination.
5. C returns execution status such as reservation success, inventory changed, queue changed, or route infeasible.
6. B handles C failures by choosing alternatives or explaining why the plan cannot execute.

## When To Use Live Gaode

Do not make B call live Gaode for every eval/dev run.

Use live Gaode for:

- Offline seed generation.
- Demo route overlays from a user origin.
- Optional fallback when local seed has no matching POI and quota/network are available.
- On-demand route checks for a small set of top B candidates.

Avoid live Gaode for:

- Offline eval.
- Unit tests.
- Core planner scoring tests.
- Any path where result fluctuation would make debugging harder.

Route data is different from merchant supply. Merchant, product, deal and
availability fixtures need local enrichment because public POI data does not
contain WeekendFlow business fields. Route planning should be live-first for
demo/user runs because it depends on origin, departure time, transport mode and
traffic. B therefore uses this order by default:

1. `offline_routes_json`
2. `coordinate_estimate`
3. `poi_distance_fallback`

For demo/live route checks, set:

```powershell
$env:WF_ROUTE_SOURCE_ORDER="live_route_api,offline_routes_json,coordinate_estimate,poi_distance_fallback"
```

When live routing is enabled, B should still call it only after initial supply
recall has narrowed the candidate set.

## Quality Check

Run the data quality checker before using a fixture snapshot to tune B:

```powershell
python experiments/check_mock_data_quality.py
```

The checker validates JSON shape, ID uniqueness, POI/product/deal/merchant
links, availability overlays, category coverage, real POI evidence, and common
realism rules such as takeaway-only restaurants not supporting reservations.
It reports structural errors separately from quality warnings. Informational
notes identify fields that are B priors or derived-like scores, so they are not
mistaken for verified source facts.
