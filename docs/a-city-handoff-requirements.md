# A 城市槽位交接需求

A 遇到“我人在上海，周末去青岛”这类跨城需求时，必须把当前位置和出游目的地拆开：

```json
{
  "current_city": "上海",
  "trip_city": "青岛",
  "destination_city": "青岛",
  "route_origin": null
}
```

`city` 兼容旧字段，但应等于 `destination_city`。`current_city/current_location` 只用于判断是否需要跨城提醒，不能当成本地 POI、天气或路线起点；只有明确坐标或具体出发点才能写入 `route_origin`。

B 侧消费规则：POI 检索、天气查询、路线城市参数永远优先使用 `trip_city/destination_city`；`current_city/current_location` 不参与本地候选选择，也不作为默认路线起点。
