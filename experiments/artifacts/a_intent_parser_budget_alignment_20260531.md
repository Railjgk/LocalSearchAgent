# WeekendFlow A Intent Parser 对齐需求：预算、人数、时间数字解析

日期：2026-05-31

## 背景

B 在接入更大规模本地 POI 和 LocalSearchBench 风格用例后，发现一些失败并不是候选生成或优化器本身导致，而是 A 的 `intent_parser` 把用户原话里的数字、预算、人数、时间、同行人关系解析错了。B 可以做兜底，但预算和人数属于上游意图理解的核心槽位，建议 A 明确输出结构化字段和证据。

## 当前远端 A 的测试结果

已 fetch `origin/main`，最新远端主要合入 C POI，A 的 `src/nodes/intent_parser.py` 没有新的预算/人数修复。用现有 A 直接跑以下句子，结果如下：

| 用户输入 | 当前 A 输出问题 | 期望 |
| --- | --- | --- |
| 我想参加2025上海国际儿童戏剧艺术节，参加完想在活动举办区找个便宜点的地方过夜，还想找个餐厅吃饭。 | `budget=300, type=total`，来自“便宜点”的默认值；但用户没有给明确数字预算 | 不要给 numeric budget；只标记“价格敏感/平价偏好”为软约束 |
| 公司15个同事周五晚上想团建，先玩桌游再吃饭再唱歌，人均总共300预算。 | `people_count=2`，`budget_type=total` | `people_count=15`，`budget=300`，`budget_type=per_person` |
| 周末下午5点后能预约一个适合女朋友的浪漫餐厅，最好安静一点。 | `people_count=3`，同时出现 partner 和 friends | `people_count=2`，同行人是 partner/couple，不应自动加 friends |
| 一家四口周末想找亲子活动和晚餐，别太远。 | `people_count=1`，没有识别“一家四口” | `people_count=4`，scene/family/亲子成立 |
| 我们4个人周末想吃烤肉，预算人均200左右。 | 这个 case 当前 A 基本正确：`people_count=4`，`budget=200`，`budget_type=per_person` | 保持 |

## A 需要输出的预算合同

建议 A 在 `intent.budget` 里保留当前字段，同时增加证据字段，避免 B 只能凭数字猜：

```json
{
  "budget": {
    "amount": 300,
    "type": "per_person",
    "currency": "CNY",
    "sensitivity": "high",
    "is_explicit": true,
    "softness": "hard",
    "evidence": {
      "raw_span": "人均总共300预算",
      "normalized_amount": 300,
      "confidence": 0.92
    }
  }
}
```

字段含义：

| 字段 | 要求 |
| --- | --- |
| `amount` | 只有用户明确给出金额数字时才填；不要因为“便宜点/平价/别太贵”填默认 300 |
| `type` | `per_person` / `total` / `null` |
| `is_explicit` | 只有金额数字靠近预算触发词时为 `true` |
| `softness` | 明确“不超过/以内/封顶”是 `hard`；“左右/差不多/便宜点”是 `soft` 或 `vague` |
| `evidence.raw_span` | 保留触发预算的原文片段，方便 B 调试和前端解释 |

## 预算解析规则

1. 只有数字附近出现预算/金额触发词时，才输出 numeric budget。
2. 触发词包括：`人均`、`每人`、`单人`、`预算`、`总共`、`一共`、`控制在`、`不超过`、`以内`、`以下`、`封顶`、`左右`、`¥`、`￥`、`元`、`块`。
3. `人均300`、`每人200左右`、`人均总共300预算` 都应是 `type=per_person`。
4. `总预算500`、`总共500`、`一共500`、`预算控制在500以内` 应是 `type=total`。
5. “便宜点/平价/别太贵/省钱/预算有限”没有数字时，不要填 `amount`，只作为软偏好。
6. 不要把年份、时间、人数、年龄识别为预算：
   - `2025上海国际儿童戏剧艺术节` 是年份/活动名，不是预算。
   - `下午5点后` 是时间，不是预算。
   - `15个同事` 是人数，不是预算。
   - `孩子5岁` 是年龄，不是预算。

## 人数与同行人规则

建议 A 输出更稳定的 `people_count` 和 `companions`：

```json
{
  "people_count": 15,
  "companions": [
    {"role": "colleague", "count": 14, "relationship": "同事"}
  ],
  "people_evidence": {
    "raw_span": "公司15个同事",
    "confidence": 0.9
  }
}
```

需要补的规则：

1. `公司15个同事`、`15人团建`、`十五个同事` 应识别为大团体人数。
2. `一家四口`、`三口之家` 应识别家庭人数。
3. `女朋友/男朋友/对象/伴侣` 默认总人数为 2，不要额外加 friends。
4. `我们4个人` 是人数，不是孩子年龄。
5. `孩子5岁` 是孩子年龄；只有出现孩子/小朋友/亲子等词时才写 `child_age=5`。

## 建议增加 numeric_mentions

为了让 B 更稳，建议 A 增加 `numeric_mentions`，明确每个数字的语义：

```json
{
  "numeric_mentions": [
    {"span": "2025", "type": "year", "confidence": 0.95},
    {"span": "5点后", "type": "time", "confidence": 0.95},
    {"span": "15个同事", "type": "people_count", "value": 15, "confidence": 0.9},
    {"span": "300预算", "type": "budget", "value": 300, "budget_type": "per_person", "confidence": 0.92}
  ]
}
```

这能避免 B 在复杂场景里重新做一次不可靠的数字猜测。

## 对 B 的影响

如果 A 只输出错误的硬预算，B 会把本来合理的方案过滤掉，尤其是：

1. 过夜/酒店/两天行程被 300 元总预算误杀。
2. 大团建被当成 2 人，预算、座位、包间、可执行动作都错。
3. 情侣场景被当成多人聚会，餐厅氛围和座位需求会偏。

B 侧已经做了防御：只有检测到明确预算数字证据时才硬过滤；过夜、多节点行程会软化默认预算；人均预算会按人数扩展为内部总预算。但这只是兜底，正确方案仍应由 A 给出清晰语义。

## A 侧验收用例

建议 A 至少加入这些测试：

```jsonl
{"input":"我想参加2025上海国际儿童戏剧艺术节，参加完想在活动举办区找个便宜点的地方过夜，还想找个餐厅吃饭。","expect":{"budget.amount":null,"budget.is_explicit":false,"budget.softness":"vague"}}
{"input":"公司15个同事周五晚上想团建，先玩桌游再吃饭再唱歌，人均总共300预算。","expect":{"people_count":15,"budget.amount":300,"budget.type":"per_person"}}
{"input":"周末下午5点后能预约一个适合女朋友的浪漫餐厅，最好安静一点。","expect":{"people_count":2,"scene":"couple","budget.amount":null}}
{"input":"一家四口周末想找亲子活动和晚餐，别太远。","expect":{"people_count":4,"scene":"family"}}
{"input":"我们4个人周末想吃烤肉，预算人均200左右。","expect":{"people_count":4,"budget.amount":200,"budget.type":"per_person"}}
```

## 优先级

P0：
- 不再把“便宜点/平价”直接默认成 300 总预算。
- 修复 `人均总共300预算` 的 `per_person` 识别。
- 修复 `公司15个同事`、`一家四口`、`女朋友` 的人数识别。

P1：
- 增加 `budget.evidence`、`budget.is_explicit`、`numeric_mentions`。
- 补中文数字：`十五个同事`、`一家三口`、`两个人`。

P2：
- 让 A 输出“预算是硬约束还是软偏好”，B 根据这个决定过滤还是降分。
