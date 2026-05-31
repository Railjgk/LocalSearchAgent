# LocalSearchBench RAG Handoff

This handoff is for the WeekendFlow B-side data/RAG track. It keeps changes out
of the production B planner and gives the next owner a small local probe to
continue from.

## Files

- `experiments/run_localsearchbench_rag_probe.py`
  - Downloads/reads LocalSearchBench parquet.
  - Filters the Shanghai 100 samples.
  - Extracts multi-node local-life intents from each question.
  - Builds a light local TF-IDF retriever over the existing Gaode/mock supply.
  - Returns top evidence POIs per node intent.
- `experiments/localsearchbench_rag_handoff.md`
  - This handoff note.

## Local Data Used

- LocalSearchBench cache:
  - `experiments/artifacts/localsearchbench/train-00000-of-00001.parquet`
- Existing WeekendFlow Shanghai supply:
  - `experiments/mock_data/gaode_supply_shanghai_v2_20260527_full/deduped_pois.json`
  - `activities.json`
  - `restaurants.json`
  - `products.json`
  - `deals.json`
  - `availability.json`

No API keys are used. The script does not call Gaode. It only downloads the
public benchmark parquet if the local cache is missing.

## How To Run

Compact stdout summary only:

```powershell
python experiments\run_localsearchbench_rag_probe.py --limit 10 --top-k 3
```

Write JSON/Markdown artifacts only when needed:

```powershell
python experiments\run_localsearchbench_rag_probe.py --limit 100 --top-k 5 --write-artifacts
```

## Current Baseline Shape

Intent extraction is deterministic and deliberately simple. Current roles:

- `exhibition`
- `entertainment`
- `restaurant_lunch`
- `restaurant_specific`
- `cafe`
- `souvenir_shopping`
- `shopping`
- `lodging`
- `convenience_store`
- `parking`
- `life_service`
- `general_local_search`

Retrieval is `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))` over a
text bundle built from local POI fields, enriched activity/restaurant fields,
product/deal fields, availability fields, and source queries. A small role and
location boost is applied after TF-IDF scoring.

## Evidence Schema To Keep

Each retrieved evidence item should stay structured, not just text:

- `merchant_evidence`
  - `poi_id`, `amap_id`, `name`, `address`, `coordinates`
  - `merchant_id`, `trust_score`, `review_count`, `operation_stability_score`
  - `gaode_type`, `primary_category`, `business_area`, `source_queries`
- `product_deal_evidence`
  - `product_id`, `product_type`, `package_components`, `price`
  - `deal_id`, `title`, `sale_price`, `valid_time`, `refund_policy`
- `review_ugc_evidence`
  - `review_keywords`, `signature_dishes`, `recommended_dishes` when present
  - rating/review count as structured quality signals
- `route_location_evidence`
  - longitude/latitude, address, district, business area
  - future route graph or travel-time overlay
- `business_hours_price_rating_evidence`
  - business hours/open time, holiday status
  - price/cost, rating, queue time, inventory
- `mock_field_source`
  - Gaode facts: `source_channel=gaode_poi_search`
  - WeekendFlow mock-owned fields: inventory, deal, refund policy, queue,
    reservation slots

## Early Findings From Smoke Runs

- Shanghai samples are multi-hop and often require 3-5 nodes.
- The existing supply snapshot is excellent for `activity` and `restaurant`.
- It is not first-class for `hotel`, `parking`, `convenience_store`, and
  `souvenir_shopping`; those roles can be detected, but evidence quality is
  weak or forced through activity/restaurant-like POIs.
- This is exactly why the legacy pair planner collapses tasks like
  exhibition + lunch + souvenir, lodging + noodle shop + convenience store +
  parking, or shopping + cafe + hotel.

## Integration Direction

Do not replace B with a vector database.

Recommended next B-side seam:

1. `question -> node_intents`
2. `node_intents -> evidence bundles`
3. `evidence bundles -> planner contract`
4. Structured candidate loaders/filtering/ranking remain responsible for hard
   constraints: distance, time, budget, inventory, reservation, price, rating,
   and business hours.
5. RAG evidence is used for semantic grounding, missing-domain detection,
   candidate explanation, and LLM-assisted decomposition.

## Next TODO

- Improve role extractor with LocalSearchBench search path supervision.
- Add a first-class supply adapter for generic POIs beyond activity/restaurant.
- Add per-role coverage metrics that distinguish:
  - intent role detected
  - evidence domain present
  - exact benchmark answer recovered
  - structured fields sufficient for planning
- Connect the output to `b_itinerary_blueprint` or a new B RAG adapter without
  changing C mock APIs.
