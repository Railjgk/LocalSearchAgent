# C Execution Mock Alignment Notes

This seed is currently the largest B-owned Shanghai supply dataset available
locally and should be the reference for C-side execution mock alignment.

## Counts

- Activities: 191
- Restaurants: 183
- Total POIs: 374
- Merchants: 374
- Products: 374
- Deals / coupons: 374
- Availability entries: 374

## Referential Integrity

Checked locally:

- Every `merchants[].poi_ids[]` points to an activity or restaurant POI.
- Every `products[].poi_id` points to an activity or restaurant POI.
- Every `deals[].poi_id` points to an activity or restaurant POI.
- Every `availability` key points to an activity or restaurant POI.
- No activity / restaurant POI is missing a merchant record.

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

## Important Boundary

Gaode provides base POI facts such as name, address, coordinates, `amap_id`,
phone, district and adcode. WeekendFlow-specific fields such as inventory,
reservation slots, packages, coupons, family suitability, dietary suitability
and execution readiness are local enrichment overlays owned by the mock supply
dataset.
