# C Execution Mock Alignment Notes

This seed is the current largest B-owned Shanghai merchant POI dataset and
should be the reference supply base for C-side execution mock alignment.

## Counts

- Activities: 563
- Restaurants: 502
- Total POIs: 1065
- Merchants: 1065
- Products: 1065
- Deals / coupons: 1065
- Availability entries: 1065

## Data Source Boundary

Gaode provides the real-world POI base facts:

- `amap_id`
- merchant / POI name
- address and coordinates
- adcode / citycode
- phone when available
- Gaode type and raw POI payload

WeekendFlow local enrichment owns the executable commerce fields:

- ticket / table inventory
- available slots
- package and coupon records
- reservation and refund policies
- queue and capacity overlays
- family / diet / risk tags
- merchant capability and stability fields

## Referential Integrity

C should treat `poi_id` as the stable join key across B supply files:

- Every `merchants[].poi_ids[]` points to an activity or restaurant POI.
- Every `products[].poi_id` points to an activity or restaurant POI.
- Every `deals[].poi_id` points to an activity or restaurant POI.
- Every `availability` key points to an activity or restaurant POI.
- Activity and restaurant POIs have matching merchant records.

## Files C Should Align To

- `activities.json`: activity POIs and B planning fields.
- `restaurants.json`: restaurant POIs and dining fields.
- `merchants.json`: merchant IDs, trust, capability, service risk.
- `products.json`: purchasable tickets / meal packages.
- `deals.json`: coupons / deal metadata.
- `availability.json`: slot / inventory / table mock overlays.

## C-side Action Mapping

For B `selected_plan.action_hints`:

- `order_activity_ticket`
  - use `poi_id`
  - join `merchant_id`
  - join `product_id`
  - join `deal_id`
  - check `availability[poi_id]`

- `reserve_restaurant`
  - use `poi_id`
  - join `merchant_id`
  - join `product_id`
  - join `deal_id`
  - check `availability[poi_id]`

## Notes

This seed intentionally keeps `routes.json` empty. Route, weather and traffic
checks should remain runtime services or deterministic eval overlays rather
than being baked into merchant identity data.

The search run attempted 293 of 310 planned keyword/page calls. The remaining
17 calls were intentionally left unattempted to preserve developer quota.
