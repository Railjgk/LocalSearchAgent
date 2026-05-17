# WeekendFlow B Supply Mock Data

This directory contains B-owned local-life supply data for planner development.

The split is intentional:

- Gaode POI search can provide base place fields such as `id`, `name`, `address`, `location`, and sometimes `rating`.
- B-owned mock data provides Meituan-style business fields that public map POI data does not provide: deals, inventory, queue, bookable slots, child friendliness, low-calorie suitability, refund policy, and commercial/risk tags.
- C-owned execution APIs remain responsible for final order/reservation/ticket/queue mock execution after B emits `selected_plan.action_hints`.

## Files

- `activities.json`: Activity candidates normalized into B POI schema.
- `restaurants.json`: Restaurant candidates normalized into B POI schema.
- `availability.json`: Inventory, queue, and available slot overlays keyed by `poi_id`.
- `deals.json`: Coupon/ticket/meal package mock data keyed by `poi_id`.
- `merchants.json`: Merchant-level trust, operation, and capability records keyed by `merchant_id`.
- `products.json`: Product/package-level supply records for tickets, meal packages, coupons, and takeaway-only products.
- `routes.json`: Offline route overrides for deterministic B eval. Real route checks should use C's Gaode `RoutePlanner` when a key is available.
- `MOCK_DATA_STRATEGY.md`: Mocking rules, real-vs-synthetic field ownership, local/cloud guidance, and C mock API merge contract.

Quality check:

```powershell
python experiments/check_mock_data_quality.py
```

## Required Candidate Fields

Each activity/restaurant should be convertible to:

```json
{
  "poi_id": "string",
  "name": "string",
  "type": "activity | restaurant",
  "tags": ["kid_friendly", "low_calorie"],
  "price": 120,
  "distance_km": 2.4,
  "duration_min": 90,
  "rating": 4.6,
  "queue_time_min": 10,
  "available": true,
  "available_slots": [{"time": "17:30"}],
  "location": "string"
}
```

## B Requirement Coverage

The current mock set is aligned with the project book and the `研发落地` sheet:

- Supply category coverage: `parent_child_activity`, `citywalk`, `sports`, `local_market`, `micro_vacation`, `handcraft`, `museum`, `escape_room`.
- Trust and explainability fields: `trust_score`, `verified_reviews`, `review_count`, `source_channel`, `source_evidence`, `ugc_summary`.
- Experience fields: `emotion_tags`, `atmosphere_tags`, `ritual_score`, `local_character_tags`, `citywalk_score`.
- Fulfillment fields: `available_slots`, `inventory_left`, `capacity_limit`, `reservation_required`, `queue_time_by_period`, `reservation_slots`.
- Restaurant suitability fields: `restaurant_category`, `avg_price_per_person`, `category_price_band`, `health_tags`, `menu_health_options`, `child_menu`, `service_mode`.
- Risk fields: `weather_sensitivity`, `traffic_risk`, `walking_time_min`, `operation_stability_score`, `serving_speed_min`, `dine_in_available`.

B consumes these fields through `src/nodes/mock_api_adapter.py`. The optimizer currently uses trust, ritual, operation stability, health/menu options, restaurant category, and dine-in support as scoring/filtering signals; the remaining fields are available for explainability, C execution checks, and future eval cases.

## Policy Switch

`experiments/planner_policy.yaml` controls data source order:

```yaml
candidate_generation:
  data_source_order:
    - local_supply_mock
    - c_mock_api
```

After setting `GAODE_API_KEY`, `gaode_poi_search` can be inserted between local mock and C mock API for base POI experiments.

For demo runs against the larger real Shanghai seed without editing policy files:

```powershell
$env:WF_MOCK_DATA_DIR="experiments/mock_data/gaode_seed_v1"
```

For temporary source ordering:

```powershell
$env:WF_DATA_SOURCE_ORDER="local_supply_mock,c_mock_api"
```

For offline seed generation, run:

```bash
python experiments/build_supply_from_gaode.py --city 上海 --pages 1 --offset 10
```

The script writes to `experiments/mock_data/gaode_seed/` by default. Inspect that output before merging it into the curated mock files.

To add a bounded real-route overlay from a user origin, pass `--route-origin`
and a small `--route-limit`, for example:

```bash
python experiments/build_supply_from_gaode.py --city 上海 --pages 1 --offset 5 --route-origin "121.4737,31.2304" --route-mode driving --route-limit 15 --output-dir experiments/mock_data/gaode_seed_v1
```

## C Interface Fit

- `POISearcher` can enrich base merchant facts from Gaode: `id`, `name`, `address`, `location`, `tel`, `adcode`, and sometimes `biz_ext.rating/cost`. These fields are useful for discovery and geocoding, but they are not enough for WeekendFlow planning by themselves.
- B-owned mock data supplies the business fields Gaode POI normally does not expose: table/ticket inventory, slot capacity, queue by period, package components, coupon rules, refund/cancel policy, dine-in/takeaway capability, child friendliness, health menu options, and commercial guardrails.
- `RoutePlanner` can validate `duration`, `distance`, `walking_distance`, `segments`, night transit, tolls, and traffic-light counts when `GAODE_API_KEY` is available. `routes.json` remains the deterministic offline fallback for eval and demos.
