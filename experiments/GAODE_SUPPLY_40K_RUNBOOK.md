# WeekendFlow 高德 POI 供给数据提取 Runbook

目标：用高德 Web 服务批量获取真实商家 / 活动 POI 的轻量底座，再由 WeekendFlow 本地规则补齐库存、券、套餐、排队、亲子、健康、解释证据等 mock 字段。

## 当前策略

- 第一阶段只做上海，优先解决 B 推荐中“真实 POI 不够”和“烤肉返回火锅”等供给覆盖问题。
- 高德只负责真实观测字段：`amap_id`、商家名、地址、经纬度、电话、类别、行政区、商圈等。
- WeekendFlow 每次构建都会在真实 POI 上生成详细商家 mock：产品、套餐、优惠券、库存、排队、订座、营业时间、停车、宝宝椅、招牌菜、网友推荐菜、评价拆分、履约动作、替换策略等。
- 不批量下载图片。图片 URL / 详情图后续只对高质量候选少量补充。
- 天气、路线规划、距离矩阵适合实时调用，不进入这次离线 POI 大批量提取。
- 执行顺序必须从小到大：`smoke` -> `pilot_grid` -> `shanghai_5k` -> `gapfill_scene_semantics_v2` / `main_balanced`。

## 文件

- 配置：`experiments/gaode_supply_extraction_config.yaml`
- 脚本：`experiments/build_supply_from_gaode_v2.py`
- 无人值守总控：`experiments/run_gaode_supply_autopilot.py`
- 建议输出目录：`experiments/mock_data/gaode_supply_shanghai_v2_YYYYMMDD`

## 模式

| mode | 用途 | 调用上限 | 特点 |
| --- | --- | ---: | --- |
| `smoke` | 检查 key、接口、字段结构 | 50 | 只用命名商圈，小样本；总计划 128 次 |
| `pilot_grid` | 检查网格搜索质量 | 500 | 启用上海中心城区网格 |
| `shanghai_5k` | 第一版可用上海底座 | 5,000 | 启用网格，适合接 B eval |
| `gapfill_scene_semantics_v2` | 场景缺口补采 | 1,500 | 优先补轻食、亲子、手作、城市漫步、放松、运动等 B 关键场景 |
| `main_balanced` | 大规模均衡扩容 | 40,000 计划调用 | 主类目均衡扩展，适合后续补总量 |
| `main` | 大规模上海底座 | 38,596 计划调用 | 旧版主模式，保留兼容 |

## 无人值守 autopilot

推荐夜间由 Codex cron 调用：

```powershell
python experiments/run_gaode_supply_autopilot.py
```

默认策略：

- 每批最多 500 次调用。
- 每天最多 5,000 次 `quota_attempted_calls`。
- 累计调用到 39,000 自动停止，为 4 万总量保留安全余量。
- 到 2026-06-03 18:00 后自动停止，避免临近 key 到期继续跑。
- 优先跑 `gapfill_scene_semantics_v2`；该模式剩余计划不足一批时，切到 `main_balanced`。
- 如果发现已有 `build_supply_from_gaode_v2.py` 进程，自动跳过，避免并发重复取数。
- 每批跑完自动执行质量检查和 B eval。
- 日报写入 `experiments/artifacts/gaode_runs/autopilot_summary_YYYYMMDD.json` 和 `.md`。

先不消耗额度检查计划：

```powershell
python experiments/run_gaode_supply_autopilot.py --dry-run --daily-call-budget 100 --batch-size 50 --max-batches 1
```

小批量实跑：

```powershell
python experiments/run_gaode_supply_autopilot.py --daily-call-budget 100 --batch-size 50 --max-batches 1
```

日报重点看：

- 当天调用量、累计调用量、剩余额度。
- 去重 POI、活动、餐饮、电话字段增量。
- 详细 mock 覆盖：`review_breakdown`、`package_options`、`decision_profile`、`fulfillment_actions`、`substitution_strategy`、`parking_available` 等。
- 类目增量，尤其轻食、亲子、手作、城市漫步、近场放松。
- 质量检查 warning 和 B eval 结果。

## 推荐命令

先在当前 PowerShell 会话里配置自己的 key，不要写进仓库文件：

```powershell
$env:GAODE_API_KEY="你的高德 Web 服务 key"
```

也可以在仓库根目录创建本地文件 `.env.local`：

```text
GAODE_API_KEY=你的高德 Web 服务 key
```

`.env.local` 已被 `.gitignore` 忽略，不要提交真实 key。

先不消耗额度，只看计划：

```powershell
python experiments/build_supply_from_gaode_v2.py --dry-run-plan --mode pilot_grid --output-dir experiments/mock_data/gaode_supply_shanghai_v2_YYYYMMDD
```

小样本实跑：

```powershell
python experiments/build_supply_from_gaode_v2.py --mode smoke --output-dir experiments/mock_data/gaode_supply_shanghai_v2_YYYYMMDD
```

检查没问题后，同一个输出目录继续跑 500 次网格样本：

```powershell
python experiments/build_supply_from_gaode_v2.py --mode pilot_grid --output-dir experiments/mock_data/gaode_supply_shanghai_v2_YYYYMMDD
```

再跑 5k：

```powershell
python experiments/build_supply_from_gaode_v2.py --mode shanghai_5k --output-dir experiments/mock_data/gaode_supply_shanghai_v2_YYYYMMDD
```

最后再考虑 main：

```powershell
python experiments/build_supply_from_gaode_v2.py --mode main --output-dir experiments/mock_data/gaode_supply_shanghai_v2_YYYYMMDD
```

## 重要注意

- 同一个输出目录会自动 resume，已经成功的 `plan_id` 不会重复调用。
- 不要随便加 `--reset-logs`，它会清空本目录的原始请求记录。
- 如果只想从已有 `raw_pois.jsonl` 重新生成 mock 数据，用：

```powershell
python experiments/build_supply_from_gaode_v2.py --rebuild-from-raw --mode main --output-dir experiments/mock_data/gaode_supply_shanghai_v2_YYYYMMDD
```

## 跑完后检查

```powershell
python experiments/validate_gaode_supply_run.py --data-dir experiments/mock_data/gaode_supply_shanghai_v2_YYYYMMDD
```

重点看：

- `coverage_report.json`：类目覆盖、中文标签覆盖、真实 POI 来源覆盖。
- `dedupe_report.json`：重复率是否异常。
- `build_report.json`：调用数、错误数、最终 activities/restaurants 数量。
- B eval 是否还出现“用户要烤肉却推荐火锅”的错配。
- `炸鸡小吃` 是否单独成类，避免混在 `小吃快餐` 里导致健康/低卡场景误判。
