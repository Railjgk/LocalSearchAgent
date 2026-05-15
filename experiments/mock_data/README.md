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
- `routes.json`: Offline route overrides for deterministic B eval. Real route checks should use C's Gaode `RoutePlanner` when a key is available.

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

## Policy Switch

`experiments/planner_policy.yaml` controls data source order:

```yaml
candidate_generation:
  data_source_order:
    - local_supply_mock
    - c_mock_api
```

After setting `GAODE_API_KEY`, `gaode_poi_search` can be inserted between local mock and C mock API for base POI experiments.
