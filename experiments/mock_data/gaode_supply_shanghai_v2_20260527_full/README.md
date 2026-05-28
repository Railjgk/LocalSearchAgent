# Gaode Shanghai Supply Snapshot v2 2026-05-27

This snapshot is the current largest B-owned Shanghai local-life supply base. It combines Gaode POI facts with B-owned mock business fields for planning, evaluation, and C mock alignment.

Use it explicitly for large-data B runs:

```powershell
$env:WF_MOCK_DATA_DIR="experiments/mock_data/gaode_supply_shanghai_v2_20260527_full"
python experiments/run_b_eval.py --cases experiments/b_eval_cases_gaode_v2.jsonl
```

Large activity and restaurant files are stored as JSONL shards under `activities_shards/` and `restaurants_shards/` to avoid GitHub single-file size limits. Runtime loaders support both normal JSON files and these shard directories.

Sensitive material is excluded: no API key, secret, or full request URL is stored here.
