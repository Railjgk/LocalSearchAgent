# Value Probe 补充评测与实验完善计划

版本日期：2026-05-26  
适用范围：Qwen3-4B token value probe、fluent-level value probe、后续 A-A / A-C / C-A / C-C 迁移与 steering 实验  
暂不纳入：人类标注一致性、人类标注流程、人类评审质量控制

## 0. 与现有文档的关系

本文件不是替代训练报告，也不是重复 MVP 实施方案，而是把后续研究评测拆成可执行任务。

已有内容：

- `docs/qwen3-4b-token-value-probe-training-report.html` 已记录现有 sentence-level probe 的数据、层扫描、AUC/F1/diagonal margin、限制与复现命令。
- `docs/qwen3-4b-fluent-level-probe-training-report.md` 已记录 fluent-level probe 的 RMSE/MAE/direction accuracy、分数分布、对齐方式和最佳层。
- `docs/intent-value-probe-implementation-plan.html` 已给出高层 A-A / A-C / C-C 改写，以及 Pearson/Spearman、Macro F1/AUC、Evidence span hit rate、Diagonal dominance、Downstream ablation 等粗粒度评估项。
- `docs/intent-value-probe-defense-editable.md` 已把上述设计整理成答辩材料。

本文件新增或细化：

- 把 `opposite` 和 `unrelated` 从同一个负类中拆开评估。
- 增加 control task / selectivity，避免 probe 只学到模板或关键词。
- 增加 PR-AUC、校准、bootstrap CI、layer stability、鲁棒性和迁移矩阵。
- 明确需要新增 Abstract 数据，否则 A-A / A-C / C-A 评测无法成立。
- 给出不依赖人类标注讨论的实验顺序、数据规格和最小 steering 方案。

## 1. 当前评测基线

当前已具备的主要指标如下。

| 层级 | 已有指标 | 现状 |
| --- | --- | --- |
| Sentence-level | per-value precision / recall / F1 / AUC、macro F1、macro AUC、diagonal margin | 已覆盖 related-vs-non-related 检测 |
| Fluent-level | RMSE、MAE、direction accuracy、nonzero direction accuracy、threshold direction accuracy | 已覆盖逐词 signed score 回归 |
| Layer scan | 每层 val/test 指标和最佳层 | sentence probe 已扫 36 层，fluent probe 已扫 36 层 |
| 数据完整性 | split 分布、value 分布、fluent token 分布、alignment summary | 已有基础检查 |

关键缺口：

- sentence-level 当前把 `opposite` 和 `unrelated` 都作为非 `related` 负类，无法区分“明确反向价值”和“未提及价值”。
- 当前高分主要来自同一类合成 concrete query 数据，缺少 Abstract 数据和跨类型迁移。
- 目前证明的是“能读出”，还没有证明“方向有因果作用”。
- test split 曾被用于层选择叙事，后续正式实验需要改为 val 选层，test 只最终汇报一次。

## 2. 补充评测指标

### 2.1 分类与有符号价值

| 指标 | 计算对象 | 目的 |
| --- | --- | --- |
| 3-way confusion matrix | `related` / `opposite` / `unrelated` | 拆开正向、反向、未提及 |
| Macro 3-way F1 | 每个 value 单独算，再 macro | 防止某个 value 或某类关系塌缩 |
| Signed binary F1 | positive-vs-rest、negative-vs-rest | 分别看正向和反向价值命中 |
| Opposite-vs-unrelated AUC | 只在负类内部计算 | 判断 probe 是否真的识别“反向价值” |
| Top-1 matched value accuracy | 每条样本最高分 value 是否为目标 value | 检查 specificity |
| Matched rank / MRR | 目标 value 在所有 value 中的排序 | 比 top-1 更平滑 |

建议输出文件：

```text
experiments/value_probe_runs/qwen3_4b_token_probe/eval_extended/sentence_relation_metrics.json
experiments/value_probe_runs/qwen3_4b_token_probe/eval_extended/relation_confusion_matrix.csv
```

### 2.2 连续分数与校准

| 指标 | 目的 |
| --- | --- |
| Pearson r | 检查线性相关，适合连续强度分 |
| Spearman rho | 检查排序相关，抗非线性缩放 |
| PR-AUC | 适合 fluent token 非零标签稀疏场景 |
| Precision@k / Recall@k | 检查高亮 top tokens 是否集中在非零证据上 |
| Brier score | 检查概率分数是否可当置信度 |
| ECE | 检查校准分箱误差 |
| Threshold sweep | 给产品阈值和实验阈值选择依据 |

注意：fluent-level token 中 0 分占多数，所以 `direction_accuracy` 容易被 0 类影响。后续报告必须同时给 `all-token` 和 `nonzero-token` 指标。

### 2.3 Specificity 与显著性

| 指标 | 目的 |
| --- | --- |
| Diagonal margin bootstrap CI | 给现有 diagonal margin 加置信区间 |
| Matched-vs-mismatched effect size | 判断 matched 分数优势是否足够大 |
| Per-value margin | 避免均值掩盖某个 value 失败 |
| Cross-value leakage matrix | 看 `家庭照护` 是否总误激活 `省心便利` 等邻近 value |

推荐最小报告：

```text
mean_diagonal_margin
mean_diagonal_margin_ci95_low
mean_diagonal_margin_ci95_high
per_value_diagonal_margin
cross_value_leakage_matrix
```

### 2.4 Control Task 与 Selectivity

核心目标是排除“probe 准只是因为模板、关键词或 probe 容量足够大”。

| Control | 做法 | 期望 |
| --- | --- | --- |
| Random label | 保持文本不变，随机打乱 label | 指标明显下降 |
| Shuffled value | 保持 relation 分布，把 value_id 置换 | specificity 明显下降 |
| Keyword baseline | 用关键词规则或 TF-IDF/logistic 做 baseline | probe 应显著优于浅层 baseline |
| Length baseline | 只用长度、标点数、词数等特征 | 接近随机 |
| Template split | 按生成模板或语义模式切分 train/test | 测试泛化仍保持 |

统一指标：

```text
selectivity = real_task_score - control_task_score
```

建议至少报告：

- `selectivity_macro_auc`
- `selectivity_macro_f1`
- `selectivity_diagonal_margin`
- `selectivity_pr_auc`

### 2.5 Layer 稳定性与鲁棒性

| 指标 | 目的 |
| --- | --- |
| Best layer across seeds | 检查最佳层是否稳定 |
| Layer curve AUC | 不只看单层峰值，看中层整体可读性 |
| Probe direction cosine similarity | 检查不同 seed 训练出的方向是否一致 |
| Paraphrase consistency | 同义改写后 value 排序是否稳定 |
| Noise robustness | 加口语噪声、错别字、冗余上下文后是否稳定 |
| Length-controlled score | 避免长文本天然得分更高 |

## 3. Abstract 数据需求

如果要做 A-A / A-C / C-A / C-C 评测，必须新增 Abstract 数据。现有数据主要是 concrete local-life query，无法单独回答“抽象 value 表征是否存在”。

### 3.1 数据类型

| 类型 | 样例 | 用途 |
| --- | --- | --- |
| Definition | `省心便利 表示减少时间成本、排队成本、路线成本和不确定性。` | A-A |
| Boundary | `离家近属于 省心便利，但餐厅装修好不一定属于 省心便利。` | A-A |
| Synonym | `少折腾、省心、稳妥、能提前订` | A-A |
| Opposite | `我不介意排队，也愿意多绕路。` | A-A / signed value |
| Abstract preference | `我更重视确定性，而不是探索新鲜感。` | A-A / C-A |
| Abstract tradeoff | `如果省钱会牺牲家人舒适度，我倾向选择舒适度。` | A-A / C-C |
| Abstract unrelated | `我想找评分高、口碑稳定的地方。` | specificity control |

### 3.2 最小数据规模

value 标签统一使用中文 canonical id，英文名只作为旧代码和旧数据兼容 alias。这样做是为了避免生成脚本一部分输出 `convenience`、一部分输出“便利/方便/省心”，导致训练标签、评测分层和人工审查口径不一致。

建议 10 个 value：

| 中文 value id | 英文兼容 alias | 释义 | 正向典型表达 | 反向/边界说明 |
| --- | --- | --- | --- | --- |
| `家庭照护` | `family_care` | 优先照顾同行家人、孩子、老人、伴侣的舒适、适配和特殊需求。 | 带孩子、老人同行、亲子友好、照顾伴侣状态。 | 只是“和家人一起”但没有照护需求时可弱相关；不要和普通多人社交混淆。 |
| `健康克制` | `health` | 关注饮食健康、身体状态、低卡清淡、少油少糖、避免高负担选择。 | 减脂、清淡、低卡、健康餐、别太油。 | “好吃”“精致”不等于健康；偶尔放纵重口可作为 opposite。 |
| `省心便利` | `convenience` | 希望减少路程、等待、换乘、排队、临时不确定性和决策成本。 | 离家近、不排队、好停车、能预约、少折腾。 | 愿意绕路、排队、慢慢逛是 opposite；不要和舒适安全混成一类。 |
| `价格敏感` | `cost_sensitivity` | 重视预算、性价比、优惠和价格上限。 | 人均不超过、便宜点、性价比、别太贵。 | 愿意为品质/仪式感加钱是 opposite 或 tradeoff；高价不一定代表品质可靠。 |
| `品质可靠` | `quality_reliability` | 重视口碑稳定、评价可信、服务/卫生/出品稳定，避免踩雷。 | 评分高、评价稳定、老店、靠谱、少踩雷。 | 高级、贵、网红不自动等于可靠；新奇探索可能与它冲突。 |
| `体验享受` | `experience_enjoyment` | 重视过程好玩、沉浸、满足感、活动丰富度和主观愉悦。 | 好玩、沉浸、有意思、体验感强、玩得尽兴。 | 体验享受不等于新奇；熟悉但好玩的项目也可相关。 |
| `新奇探索` | `novelty_exploration` | 偏好新鲜、小众、未知、本地探索、尝试没去过的地方。 | 新店、小众、没去过、探索、隐藏宝藏。 | 只要求“靠谱稳定”不算新奇；可能和品质可靠存在 tradeoff。 |
| `社交连接` | `social_connection` | 重视朋友/群体互动、聊天、热闹、共同参与和关系连接。 | 朋友聚会、多人互动、热闹、适合聊天、一起玩。 | 只出现多人同行但强调安静/各自休息时弱相关；不要和家庭照护混淆。 |
| `氛围仪式` | `atmosphere_ritual` | 重视氛围、浪漫、纪念意义、审美场景和仪式感。 | 纪念日、浪漫、氛围好、出片、有仪式感。 | 舒适不一定有仪式感；贵也不必然有仪式感。 |
| `舒适安全` | `comfort_safety` | 重视身体舒适、环境安全、低风险、不过度拥挤或劳累，适合特殊人群。 | 不累、不挤、安全、无烟、适合老人孩子、低强度。 | 省时间属于省心便利；身体/环境风险才归舒适安全。 |

第一阶段训练建议先覆盖 4 个核心 value：`家庭照护`、`健康克制`、`省心便利`、`价格敏感`。schema 和生成脚本从一开始支持 10 个 value，避免后续扩展时再次改格式。

建议每个 value 的最小规模：

| 子集 | 每 value 数量 | 说明 |
| --- | ---: | --- |
| positive abstract | 100 | 定义、原则、抽象偏好 |
| opposite abstract | 100 | 明确反向价值 |
| unrelated abstract | 100 | 同领域但不涉及目标 value |
| tradeoff abstract | 50 | 两个 value 冲突或排序 |

核心 4 个总计约 `4 * 350 = 1400` 条；扩展到 10 个时总计约 `10 * 350 = 3500` 条。数据文件建议落点：

```text
experiments/value_probe_data/abstract_value_samples.jsonl
experiments/value_probe_runs/qwen3_4b_token_probe/abstract_dataset/
```

### 3.3 建议 JSONL 格式

```json
{
  "example_id": "abstract-省心便利-positive-0001",
  "text": "便利意味着减少排队、路线切换和临时不确定性，让用户少花精力完成目标。",
  "text_type": "abstract_definition",
  "target_value": "省心便利",
  "relation": "related",
  "labels": {
    "家庭照护": 0,
    "健康克制": 0,
    "省心便利": 6,
    "价格敏感": 0,
    "品质可靠": 0,
    "体验享受": 0,
    "新奇探索": 0,
    "社交连接": 0,
    "氛围仪式": 0,
    "舒适安全": 0
  },
  "split": "train"
}
```

字段说明：

- `text_type` 用于区分 `abstract_definition`、`abstract_boundary`、`abstract_preference`、`abstract_tradeoff`、`abstract_unrelated`。
- `relation` 继续使用现有 `related`、`opposite`、`unrelated`，便于复用训练脚本。
- `labels` 使用 `-6..6` signed score，便于与 fluent-level 体系对齐。
- 暂不要求 evidence span，因为本计划先忽略人类标注相关问题；如需自动生成 evidence，可后续作为弱监督字段加入。

## 4. A-A / A-C / C-A / C-C 评测矩阵

### 4.1 数据分层

| 简写 | Train | Test | 目标 |
| --- | --- | --- | --- |
| A-A | Abstract value text | Held-out abstract value text | 抽象 value 是否可线性读出 |
| A-C | Abstract value text | Concrete local-life query | 抽象方向能否落到具体请求 |
| C-A | Concrete local-life query | Abstract value text | 具体经验学到的方向能否识别抽象价值 |
| C-C | Concrete local-life query | Concrete decision/rationale/action | value 是否进入具体决策过程 |

### 4.2 最小实验设计

| 实验 | Probe 训练数据 | 测试数据 | 主要指标 |
| --- | --- | --- | --- |
| `AA_probe` | abstract train | abstract test | Pearson/Spearman、3-way F1、specificity |
| `AC_transfer` | abstract train | concrete test | transfer macro AUC/F1、performance drop |
| `CA_transfer` | concrete train | abstract test | transfer macro AUC/F1、specificity |
| `CC_probe` | concrete query train | concrete rationale/action test | value choice accuracy、rationale value AUC |
| `mixed_probe` | abstract + concrete train | abstract + concrete test | 是否优于单域训练 |

统一迁移指标：

```text
transfer_drop = in_domain_score - cross_domain_score
relative_transfer = cross_domain_score / in_domain_score
```

### 4.3 C-C 数据形态

C-C 不应只看用户 query，还要看模型或系统的具体选择、拒绝理由、改写理由。

建议样例类型：

| 类型 | 输入 | 标签 |
| --- | --- | --- |
| Option choice | 用户需求 + A/B 两个方案 | 哪个方案更符合目标 value |
| Rejection rationale | 用户需求 + 未选方案理由 | 理由是否使用目标 value |
| Revision request | 用户说“别太折腾/预算放宽/健康点” | 哪个 value 被增强或削弱 |
| Plan explanation | planner 生成解释 | 解释中的 value 是否和用户需求一致 |

下游指标：

- `choice_accuracy`
- `target_value_logprob_margin`
- `rationale_value_auc`
- `explanation_consistency`
- `plan_quality_delta`

## 5. A 部分分层架构修订

本计划需要和 A 阶段架构一起修订。原因是当前 A 阶段里 `intent parser`、`memory`、`constraints`、`scenario_planner` 都会描述用户、偏好、场景和值，字段之间存在重复表达：

- `intent.scene`、`constraints.scene`、顶层 `scene_type` 都可能表达场景。
- `intent.people`、`constraints.companions`、`memory.companion_profile` 都可能表达同行人。
- `intent.planning_preferences`、`constraints.soft_tags`、`memory.preference_profile` 都可能表达偏好。
- `memory.value_profile`、`constraints.value_weights`、后续 value probe 结果都可能表达 value。

如果继续把 value-probe 结果塞进现有 `intent` 或 `memory`，会进一步放大这些重叠。更合适的方向是把 A 阶段拆成“事实归属明确 + 单一仲裁出口”的结构。

### 5.1 总体链路

建议将 A 阶段从：

```text
user_input
  -> intent_parser_node
  -> memory_manager_node
  -> scenario_planner_node
  -> intent + memory + constraints + scenario_activities
```

修订为：

```text
user_input
  -> current_intent extractor
  -> user_profile_memory retriever
  -> current_value_signal probe
  -> scene_frame resolver
  -> planning_contract compiler
  -> resolution_trace
```

其中：

| 层 | 责任 | B/C 使用方式 |
| --- | --- | --- |
| `current_intent` | 本轮用户当前说了什么 | 可作为 B/C LLM 的解释上下文，不作为机器执行主契约 |
| `user_profile_memory` | 跨轮稳定或半稳定画像 | 可作为解释/审计上下文，必须经仲裁后才能影响规划 |
| `current_value_signal` | 当前输入表达的 value 方向与强度 | 可作为 B/C LLM 的 value evidence，必须经仲裁后进入权重 |
| `scene_frame` | 本轮唯一场景判断 | 经 `planning_contract` 映射到执行字段，也可给 LLM 做场景说明 |
| `planning_contract` | B/C 机器可执行的主规划契约 | 主输入；硬约束、权重和执行字段以此为准 |
| `resolution_trace` | 冲突、覆盖、跳过的审计记录 | 可供 B/C LLM 解释、质检和调试使用 |

兼容旧接口时可以继续输出 `intent`、`constraints`、`scene_type`、`scenario_activities`，但它们应由上述新结构派生，不能各自独立判断。

### 5.2 `current_intent`

`current_intent` 只表示“本轮用户明确说了什么或强隐含说了什么”。它不记录长期画像，不做跨轮偏好判断，也不直接产出最终 `value_weights`。

推荐字段：

```json
{
  "current_intent": {
    "task_type": "local_life_plan",
    "raw_text": "今天下午想和老婆孩子出去玩几个小时，别离家太远，孩子5岁，老婆最近在减肥。",
    "goal": "安排一次本地生活出行计划",
    "time": {
      "window": "today_afternoon",
      "start_time": null,
      "end_time": null,
      "duration_range": [4, 6],
      "evidence": ["今天下午", "几个小时"]
    },
    "participants": [
      {"role": "self", "source": "default"},
      {"role": "spouse", "state": "dieting", "needs": ["low_calorie", "light_food"], "evidence": ["老婆最近在减肥"]},
      {"role": "child", "age": 5, "needs": ["kid_friendly", "low_intensity"], "evidence": ["孩子5岁"]}
    ],
    "location_request": {
      "origin": "home",
      "distance_preference": "nearby",
      "max_distance_km": 8,
      "evidence": ["别离家太远"]
    },
    "budget_request": {
      "amount": null,
      "budget_type": null,
      "sensitivity": "unknown",
      "evidence": []
    },
    "explicit_tags": ["kid_friendly", "low_calorie", "light_food", "nearby"],
    "avoid": ["too_far"],
    "missing_slots": ["budget", "transport_mode"],
    "confidence": {
      "time": 0.85,
      "participants": 0.95,
      "location": 0.9
    }
  }
}
```

关键原则：

- 当前输入优先级最高。
- `current_intent` 可以识别当前偏好，但不写“用户长期喜欢”。
- `current_intent` 可以产生候选 tags，但不直接决定 B 的最终权重。
- 每个重要字段尽量带 `evidence`，方便后续 `resolution_trace` 和 explanation 复用。

### 5.3 `user_profile_memory`

`user_profile_memory` 只表示历史画像、短期状态和跨轮偏好。它不能直接覆盖当前输入，必须通过 `resolution_trace` 仲裁后才能进入 `planning_contract`。

推荐字段：

```json
{
  "user_profile_memory": {
    "stable_profile": [
      {
        "field": "home_area",
        "value": "杨浦区大学路附近",
        "confidence": 0.82,
        "ttl": "long_term",
        "source": "historical_profile",
        "evidence": ["多次从大学路附近出发"]
      }
    ],
    "companion_profile": [
      {
        "role": "child",
        "age": 5,
        "confidence": 0.9,
        "ttl": "medium_term",
        "source": "historical_profile",
        "evidence": ["历史多次亲子出行"]
      }
    ],
    "preference_profile": [
      {
        "tag": "avoid_long_queue",
        "confidence": 0.78,
        "ttl": "long_term",
        "source": "history_feedback",
        "evidence": ["上次拒绝了排队久的餐厅"]
      }
    ],
    "value_profile": [
      {
        "value_id": "省心便利",
        "score": 0.76,
        "confidence": 0.84,
        "ttl": "long_term",
        "source": "history_feedback",
        "evidence": ["多次偏好少排队、少换乘"]
      }
    ]
  }
}
```

关键原则：

- Memory 是“历史事实与偏好”，不是本轮最终决策。
- Memory 只能补缺或弱影响，不直接改写当前明确请求。
- 家庭画像、伴侣状态、孩子年龄等必须做场景隔离。
- 历史 value 和当前 value-probe 结果要分开，避免把长期偏好误当作当前表达。

### 5.4 `current_value_signal`

`current_value_signal` 是 value-probe 在 A 阶段的独立落点。它表示“当前这句话表达了哪些 value，方向是什么，强度是多少”，不表示长期画像。

推荐字段：

```json
{
  "current_value_signal": {
    "source": "qwen3_4b_value_probe",
    "model": "Qwen3-4B",
    "sentence_layer": 11,
    "fluent_layer": 29,
    "values": {
      "家庭照护": {
        "relation": "related",
        "signed_score": 0.83,
        "positive_score": 0.88,
        "negative_score": 0.04,
        "confidence": 0.88,
        "evidence": ["老婆孩子", "孩子5岁"]
      },
      "健康克制": {
        "relation": "related",
        "signed_score": 0.74,
        "positive_score": 0.79,
        "negative_score": 0.03,
        "confidence": 0.79,
        "evidence": ["老婆最近在减肥"]
      },
      "省心便利": {
        "relation": "related",
        "signed_score": 0.69,
        "positive_score": 0.72,
        "negative_score": 0.05,
        "confidence": 0.72,
        "evidence": ["别离家太远"]
      },
      "价格敏感": {
        "relation": "unrelated",
        "signed_score": 0.02,
        "positive_score": 0.05,
        "negative_score": 0.03,
        "confidence": 0.05,
        "evidence": []
      }
    },
    "token_evidence": [
      {"token": "孩子", "value_id": "家庭照护", "score": 4.2},
      {"token": "减肥", "value_id": "健康克制", "score": 4.6},
      {"token": "别离家太远", "value_id": "省心便利", "score": 3.8}
    ]
  }
}
```

与第 2 章指标的对应关系：

| `current_value_signal` 字段 | 评测指标 |
| --- | --- |
| `relation` | 3-way F1、confusion matrix |
| `positive_score` | positive-vs-rest F1/AUC/PR-AUC、Brier/ECE |
| `negative_score` | negative-vs-rest F1/AUC/PR-AUC、opposite-vs-unrelated AUC |
| `signed_score` | Pearson/Spearman、direction accuracy |
| `evidence` / `token_evidence` | fluent PR-AUC、Precision@k、Recall@k |
| value 间分数排序 | Top-1 matched value accuracy、MRR、leakage matrix |

关键原则：

- `values` 字段应包含 10 个中文 canonical value；未命中的 value 也要显式给出 `unrelated` 或接近 0 的分数，便于 leakage、specificity 和 threshold sweep。
- Probe 输出是当前轮信号，不直接等于最终规划权重。
- Probe 只负责“读出 value”，不负责 memory 冲突处理。
- `related`、`opposite`、`unrelated` 必须保留，不再合并成二分类负类。
- `unrelated` 的低分同样重要，它是 specificity 的基础。

### 5.5 `scene_frame`

`scene_frame` 是本轮唯一场景来源。它由 `current_intent`、相关 memory 和 `current_value_signal` 共同派生。

推荐字段：

```json
{
  "scene_frame": {
    "scene_type": "family",
    "scene_subtype": ["parent_child", "spouse_health_sensitive", "nearby_relaxed"],
    "participants": [
      {"role": "self"},
      {"role": "spouse", "state": "dieting"},
      {"role": "child", "age": 5}
    ],
    "scenario_facets": {
      "activity_type": ["parent_child", "indoor", "light_activity"],
      "food_type": ["low_calorie", "light_food", "low_oil"],
      "route_type": ["nearby", "low_transfer"],
      "risk_avoidance": ["long_queue", "crowded_mall"]
    },
    "scene_evidence": [
      {"source": "current_intent", "field": "participants", "evidence": "老婆孩子"},
      {"source": "current_value_signal", "field": "家庭照护", "evidence": "孩子5岁"},
      {"source": "current_value_signal", "field": "健康克制", "evidence": "老婆最近在减肥"}
    ],
    "scene_confidence": 0.92
  }
}
```

关键原则：

- `scene_type` 只有这一处权威来源。
- `scenario_planner` 不再重新从原文猜场景，而是消费 `scene_frame`。
- 旧顶层 `scene_type` 应由 `scene_frame.scene_type` 复制得到。
- 如果场景证据不足，`scene_confidence` 要下降，并在 `missing_slots` 或 `resolution_trace` 中记录。

### 5.6 `planning_contract`

`planning_contract` 是 B/C 阶段机器可执行决策的主输入，而不是 B/C LLM 唯一能看的上下文。它是 `current_intent`、`user_profile_memory`、`current_value_signal`、`scene_frame` 仲裁后的结果。

当前项目里 B 不是纯 LLM：候选生成和排序主要由规则、供给数据和 optimizer 完成，LLM 只在 semantic hints 和 explanation 两处可选启用；C 当前主要是 tool routing、mock API 和 execution manager。因此本计划把 `planning_contract` 设计成稳定的机器执行契约，同时允许 B/C 的 LLM 读取 `current_intent`、`current_value_signal`、`resolution_trace` 做解释、语义补充和质检。

推荐字段：

```json
{
  "planning_contract": {
    "version": "a_contract_v2",
    "scene_type": "family",
    "city": "上海",
    "time_window": "today_afternoon",
    "duration_range": [4, 6],
    "start_time": "14:00",
    "people_count": 3,
    "companions": [
      {"role": "spouse", "state": "dieting", "needs": ["low_calorie", "light_food"]},
      {"role": "child", "age": 5, "needs": ["kid_friendly", "low_intensity"]}
    ],
    "route_origin": "home",
    "route_mode": "driving",
    "max_distance_km": 8,
    "max_queue_time_min": 20,
    "budget": null,
    "budget_type": null,
    "hard_tags": ["kid_friendly"],
    "soft_tags": ["parent_child", "low_intensity", "low_calorie", "light_food", "nearby"],
    "avoid": ["long_queue", "too_far", "crowded_mall"],
    "planning_preferences": {
      "activity_type": ["parent_child", "indoor", "light_activity"],
      "food_type": ["low_calorie", "light_food", "low_oil"],
      "restaurant_type": ["dine_in"]
    },
    "scenario_activities": ["parent_child", "indoor", "light_activity", "low_calorie", "nearby"],
    "scenario_template": {
      "poi_mix": ["activity", "restaurant"],
      "route_pattern": ["start", "kid_friendly_activity", "restaurant"],
      "pace": "relaxed"
    },
    "route_pattern_hints": {
      "should_search": true,
      "search_terms": ["parent_child", "indoor", "light_activity", "low_calorie", "nearby"]
    },
    "value_weights": {
      "家庭照护": 0.83,
      "健康克制": 0.74,
      "省心便利": 0.69,
      "价格敏感": 0.0
    },
    "value_confidence": {
      "家庭照护": 0.88,
      "健康克制": 0.79,
      "省心便利": 0.72,
      "价格敏感": 0.05
    },
    "score_weights": {
      "group_fit": 0.38,
      "health": 0.24,
      "route": 0.23,
      "availability": 0.2,
      "budget": 0.1,
      "experience": 0.1
    },
    "missing_slots": ["budget"],
    "memory_policy": "current_intent_first"
  }
}
```

B/C 读取约定：

- B/C 的过滤、排序、执行和工具调用以 `planning_contract` 为主输入。
- B/C LLM 可以读取 `current_intent`、`current_value_signal`、`scene_frame`、`resolution_trace` 作为补充上下文。
- B/C LLM 不能静默覆盖 `planning_contract` 中的硬约束、已仲裁 value 权重和执行字段。
- 如果 B/C LLM 判断 contract 不充分或冲突，应输出显式 `revision_request`、`contract_warning` 或 `needs_clarification`，而不是直接改写执行字段。
- 旧字段 `constraints` 可由 `planning_contract` 降级生成，保证现有 B eval 不断。
- `scenario_activities` 建议优先使用 canonical tags，可附带中文，但不应只给中文。
- `value_weights` 来源可以是当前 probe、memory、规则或混合，但必须在 `resolution_trace` 中说明。

### 5.7 `resolution_trace`

`resolution_trace` 记录当前输入、probe、memory 和默认值之间如何仲裁。

推荐字段：

```json
{
  "resolution_trace": {
    "priority": [
      "current_intent.explicit_constraints",
      "current_value_signal",
      "short_term_memory",
      "long_term_memory",
      "derived_defaults"
    ],
    "applied": [
      {
        "target": "planning_contract.max_distance_km",
        "value": 8,
        "source": "current_intent",
        "evidence": "别离家太远"
      },
      {
        "target": "planning_contract.value_weights.健康克制",
        "value": 0.74,
        "source": "current_value_signal",
        "evidence": "老婆最近在减肥"
      }
    ],
    "skipped": [
      {
        "target": "planning_contract.value_weights.价格敏感",
        "source": "long_term_memory",
        "reason": "current request has no budget signal and confidence is low"
      }
    ],
    "conflicts": [
      {
        "field": "food_type",
        "current": "hotpot",
        "memory": "light_food",
        "resolution": "use_current",
        "reason": "current explicit request overrides long-term preference"
      }
    ]
  }
}
```

评测指标：

| 指标 | 含义 |
| --- | --- |
| `scene_consistency_rate` | `scene_frame`、旧 `scene_type`、`planning_contract.scene_type` 是否一致 |
| `profile_leakage_rate` | 朋友/solo/couple 场景是否误带入孩子、伴侣减脂等画像 |
| `current_override_accuracy` | 当前明确请求和历史画像冲突时是否尊重当前请求 |
| `slot_completeness` | 本轮人、时间、地点、预算、饮食、风险字段是否完整 |
| `trace_coverage` | 应用/跳过/冲突的 memory 是否都有 reason |
| `contract_validity_rate` | B/C 所需字段是否都能从 `planning_contract` 获得 |
| `downstream_stability` | B/C 以 `planning_contract` 为主输入时是否仍稳定 |
| `llm_contract_adherence` | B/C LLM 是否遵守 contract，不改写硬约束和执行字段 |
| `llm_context_usefulness` | B/C LLM 读取 intent/value/trace 后 explanation 或 hint 是否更一致 |

### 5.8 A 阶段消融实验

建议后续实验从“probe 本身好不好”扩展为“A 架构是否更稳定”。

| 版本 | 设计 | 目的 |
| --- | --- | --- |
| `A0_entangled` | 当前混合式 A 输出 | 作为对照 |
| `A1_split_intent_memory` | 拆出 `current_intent` / `user_profile_memory` | 看画像污染是否下降 |
| `A2_scene_frame` | 增加唯一 `scene_frame` | 看场景冲突是否下降 |
| `A3_contract_trace` | 增加 `planning_contract` + `resolution_trace` | 看 B/C 稳定性和可解释性 |
| `A4_probe_value_signal` | `A3` + `current_value_signal` | 看 value-probe 是否带来规划增益 |
| `A5_probe_memory_hybrid` | 当前 probe + 历史 value memory 仲裁 | 看当前 value 和长期 value 如何互补 |

推荐冲突测试：

| 测试 | 预期 |
| --- | --- |
| 历史偏轻食，但当前想吃火锅 | 当前请求优先，`健康克制` memory 被跳过或降权 |
| 历史有孩子画像，但当前和朋友聚餐 | 不激活 `家庭照护` 和 child constraints |
| 历史低预算，但当前纪念日想有仪式感 | 当前 `experience/ritual` 优先，budget 降权 |
| 当前说“不介意排队”，历史偏少排队 | `省心便利` 可变为 opposite 或降权 |
| 当前预算严格，人均不超过 80 | `价格敏感` 强激活，体验权重让位 |

### 5.9 暂不实现的 1/3 优化方案

本节对应前面关于 `task_type / domain / constraints` 和 `memory / scene_frame` 的疑问，只作为后续优化方案记录，当前阶段不改代码实现。

#### 5.9.1 `task_type`、`domain`、`constraints` 优化

当前实现中 `task_type` 主要是 `local_life_plan` / `clarify_request`，B/C 实际更依赖 `scene_type`、`constraints`、`scenario_activities` 和 `score_weights`。后续优化方向是把 `task_type` 从“泛化意图名”改成“流程路由字段”。

建议 `task_type` 闭集：

| `task_type` | 含义 | B 阶段预期 | C 阶段预期 |
| --- | --- | --- | --- |
| `local_life_plan` | 活动 + 餐饮 + 路线组合规划 | 召回活动和餐厅，组合排序 | 可产生订票、订座、路线 action |
| `restaurant_only` | 只找餐饮 | 只召回餐厅 | 订座、排队、到店 |
| `activity_only` | 只找活动 | 只召回活动/展览/玩乐 | 购票、预约 |
| `route_or_area_search` | 找附近、商圈、路线或区域 | 以位置、距离、路线为主排序 | 导航或路线建议 |
| `reservation_or_execution` | 用户已有明确目标，要求执行 | 校验目标可用性，不做大规模召回 | 直接生成执行动作 |
| `revision_request` | 修改上一轮计划 | 复用候选或重新优化 | 更新未执行动作 |
| `explanation_request` | 问为什么这样推荐 | 读取已有计划和 trace | 解释，不执行 |

建议新增 `domain` 闭集并允许多选：

```json
{
  "domain": ["activity", "restaurant", "route"]
}
```

推荐枚举：`activity`、`restaurant`、`route`、`execution`、`explanation`。`task_type` 决定流程，`domain` 决定召回范围；例如 `local_life_plan` 通常对应 `["activity", "restaurant", "route"]`，`restaurant_only` 对应 `["restaurant"]`。

`constraints` 后续拆成 4 层，减少现在 `hard_tags`、`soft_tags`、`planning_preferences` 混用：

| 层 | 字段 | 作用 |
| --- | --- | --- |
| 硬约束 | `hard_constraints` | 违反后原则上不能入选，例如儿童友好、低卡选项、距离上限 |
| 软偏好 | `soft_preferences` | 影响排序但可权衡，例如氛围好、少排队、体验感强 |
| 规避项 | `avoid` | 负向过滤或惩罚，例如太远、排队久、拥挤 |
| 执行参数 | `execution_requirements` | 是否需要订座、购票、库存、时间段 |

兼容旧接口时，`hard_constraints` 派生旧 `hard_tags`，`soft_preferences` 派生旧 `soft_tags`，`execution_requirements` 派生 C 阶段 action hints 所需字段。

#### 5.9.2 `memory` 与 `scene_frame` 优化

当前 memory 检索已经有 sparse、semantic、graph、short-term、episodic 多来源，但后续应把检索结果分成“可直接应用”“仅解释可见”“本轮跳过”三类，防止历史画像污染当前请求。

建议 memory 输出增加：

```json
{
  "memory_retrieval": {
    "applied": [],
    "visible_for_explanation": [],
    "skipped": [],
    "query_tags": ["family", "kid_friendly", "省心便利"],
    "policy": "current_intent_first"
  }
}
```

判定规则：

| 类别 | 进入条件 | 是否影响 `planning_contract` |
| --- | --- | --- |
| `applied` | 和当前 intent/scene/value 一致，且置信度足够 | 可以，但必须写入 `resolution_trace` |
| `visible_for_explanation` | 相关但不应影响本轮决策 | 不影响，只给解释或审计 |
| `skipped` | 与当前场景不匹配或被当前输入覆盖 | 不影响，并记录跳过原因 |

`scene_frame` 后续保持两层结构：

```json
{
  "scene_type": "family",
  "scene_subtype": ["family_health_food", "nearby_low_wait"],
  "scenario_facets": {
    "participants": ["child", "spouse"],
    "food": ["low_calorie", "light_food"],
    "route": ["nearby"],
    "risk": ["long_queue"]
  }
}
```

`scene_type` 继续使用闭集：`family`、`friends`、`couple`、`low_budget`、`solo`，因为当前 B 的模板和默认权重已经对齐这些类别。`scene_subtype` 和 `scenario_facets` 用来承载细粒度差异，不再新增过多顶层 scene，避免 B 的权重表和模板爆炸。

后续验收指标：

| 指标 | 目的 |
| --- | --- |
| `task_type_route_accuracy` | 检查 A 是否把请求路由到正确 B/C 流程 |
| `domain_recall_accuracy` | 检查召回范围是否包含必要 domain 且不过宽 |
| `constraint_projection_accuracy` | 检查 hard/soft/avoid/execution 是否正确投影到旧字段 |
| `memory_application_precision` | 检查被应用的 memory 是否真的该影响本轮 |
| `scene_downstream_alignment` | 检查 `scene_frame` 是否能稳定驱动 B 的模板和权重 |

## 6. 数据格式与脚本改造计划

本章说明为了支持第 2 章补充评测、第 4 章迁移矩阵和第 5 章 A 架构修订，需要新增哪些数据字段、这些字段是什么意思、生成脚本如何修改，以及哪些改动需要重新训练。

### 6.1 数据格式总原则

数据需要同时支持三类任务：

1. value-probe 训练和评测：判断文本是否表达某个 value，方向是什么，证据在哪里。
2. A 阶段架构评测：判断 intent、memory、scene、contract 是否职责分明且不冲突。
3. C-C / downstream 评测：判断 value 是否进入方案选择、解释、拒绝和改写。

因此 JSONL 需要从当前的最小字段：

```json
{
  "example_id": "省心便利-related-0001",
  "text": "想找附近不用排队的地方。",
  "target_value": "省心便利",
  "relation": "related",
  "split": "train"
}
```

扩展为带元数据、分组、证据和 A 架构字段的格式。

### 6.2 推荐统一 JSONL schema

推荐字段：

```json
{
  "example_id": "concrete-query-省心便利-related-0001",
  "text": "周末想找个离家近、不用排队的亲子餐厅。",
  "language": "zh",
  "domain": "local_life",
  "text_type": "concrete_query",
  "target_value": "省心便利",
  "relation": "related",
  "labels": {
    "家庭照护": 2,
    "健康克制": 0,
    "省心便利": 5,
    "价格敏感": 0,
    "品质可靠": 0,
    "体验享受": 0,
    "新奇探索": 0,
    "社交连接": 0,
    "氛围仪式": 0,
    "舒适安全": 0
  },
  "value_relations": {
    "家庭照护": "related",
    "健康克制": "unrelated",
    "省心便利": "related",
    "价格敏感": "unrelated",
    "品质可靠": "unrelated",
    "体验享受": "unrelated",
    "新奇探索": "unrelated",
    "社交连接": "unrelated",
    "氛围仪式": "unrelated",
    "舒适安全": "unrelated"
  },
  "evidence_spans": [
    {"value_id": "省心便利", "start": 6, "end": 15, "text": "离家近、不用排队", "score": 5},
    {"value_id": "家庭照护", "start": 15, "end": 17, "text": "亲子", "score": 2}
  ],
  "template_id": "nearby_low_wait_family_restaurant",
  "semantic_pattern": "reduce_friction",
  "paraphrase_group_id": "conv_family_nearby_0001",
  "noise_group_id": "",
  "pair_id": "",
  "split": "train",
  "source": {
    "generator": "llm_synthetic",
    "generator_version": "value_probe_v2",
    "created_at": "2026-05-26",
    "seed": 20260526
  },
  "a_stage_expected": {
    "current_intent": {
      "participants": [{"role": "child"}],
      "explicit_tags": ["nearby", "kid_friendly"],
      "avoid": ["long_queue"]
    },
    "current_value_signal": {
      "省心便利": {"relation": "related", "signed_score": 5},
      "家庭照护": {"relation": "related", "signed_score": 2}
    },
    "scene_frame": {
      "scene_type": "family",
      "scene_subtype": ["parent_child", "nearby_low_wait"]
    },
    "planning_contract": {
      "hard_tags": ["kid_friendly"],
      "soft_tags": ["nearby"],
      "avoid": ["long_queue"],
      "value_weights_nonzero": ["家庭照护", "省心便利"]
    }
  }
}
```

字段分组说明：

| 字段 | 含义 | 是否训练必需 | 是否评测必需 |
| --- | --- | --- | --- |
| `example_id` | 稳定样本 ID | 是 | 是 |
| `text` | 输入文本 | 是 | 是 |
| `text_type` | 抽象/具体/query/rationale/choice 等类型 | A-A/A-C 必需 | 是 |
| `target_value` | 当前单目标训练/评测 value | 是 | 是 |
| `relation` | 目标 value 的 `related/opposite/unrelated` | 是 | 是 |
| `labels` | 所有 value 的 signed score，范围 `-6..6` | fluent/multi-value 必需 | 是 |
| `value_relations` | 所有 value 的三分类关系 | 多 value 评测需要 | 是 |
| `evidence_spans` | 字符级证据片段 | fluent/evidence 训练可选 | 强烈建议 |
| `template_id` | 生成模板 ID | 否 | template split 必需 |
| `semantic_pattern` | 语义模式 ID | 否 | semantic split 必需 |
| `paraphrase_group_id` | 同义改写组 ID | 否 | paraphrase consistency 必需 |
| `noise_group_id` | 原句/加噪组 ID | 否 | noise robustness 必需 |
| `pair_id` | 方案/解释/改写成对比较 ID | C-C 需要 | C-C 必需 |
| `a_stage_expected` | A 架构期望输出 | 否 | A 架构评测必需 |

### 6.3 `text_type`

`text_type` 区分文本属于哪一层或哪类任务。

建议枚举：

| `text_type` | 含义 | 用途 |
| --- | --- | --- |
| `concrete_query` | 具体本地生活请求 | C-C、C-A、现有 concrete probe |
| `abstract_definition` | value 定义 | A-A |
| `abstract_boundary` | value 边界说明 | A-A specificity |
| `abstract_preference` | 抽象偏好表达 | A-A / A-C |
| `abstract_opposite` | 抽象反向价值 | signed value |
| `abstract_tradeoff` | 抽象价值权衡 | tradeoff / C-C |
| `abstract_unrelated` | 同领域但不涉及目标 value | specificity control |
| `option_choice` | 用户需求 + A/B 方案 | C-C choice |
| `rejection_rationale` | 拒绝某方案的理由 | C-C rationale |
| `revision_request` | 用户改写/加强/削弱需求 | C-C revision |
| `plan_explanation` | planner 生成的解释 | explanation consistency |
| `memory_atom` | 历史画像/反馈文本 | memory-value 对齐 |

为什么需要：

- A-A / A-C / C-A / C-C 不能只靠 `split` 区分，必须知道文本类型。
- `current_intent` 应主要来自 `concrete_query`。
- `user_profile_memory` 应主要来自 `memory_atom`。
- `current_value_signal` 可在所有文本上评测，但汇报时必须按 `text_type` 分层。

### 6.4 `template_id` 与 `semantic_pattern`

`template_id` 表示样本由哪个生成模板产生；`semantic_pattern` 表示更抽象的语义套路。

示例：

```json
{
  "template_id": "nearby_low_wait_query",
  "semantic_pattern": "reduce_friction"
}
```

建议 `semantic_pattern` 枚举：

| `semantic_pattern` | 含义 | 常见 value |
| --- | --- | --- |
| `reduce_friction` | 减少排队、路程、换乘、不确定性 | `省心便利` |
| `accept_extra_effort` | 愿意绕路、排队、花时间 | `省心便利` opposite |
| `protect_family_comfort` | 照顾孩子/老人/伴侣舒适 | `家庭照护` |
| `ignore_family_constraints` | 不考虑家庭成员约束 | `家庭照护` opposite |
| `avoid_unhealthy_food` | 避免高油高糖重辣 | `健康克制` |
| `indulge_taste` | 追求重口、热闹、放纵 | `健康克制` opposite |
| `save_money` | 预算敏感、找优惠 | `价格敏感` |
| `pay_for_quality` | 预算放宽、愿意贵一点 | `价格敏感` opposite |
| `trust_and_certainty` | 要靠谱、少踩雷、评价稳定 | `品质可靠` |
| `seek_enjoyment` | 好玩、沉浸、玩得尽兴 | `体验享受` |
| `novelty_exploration` | 探索新鲜、网红、小众 | `新奇探索` |
| `social_connection` | 多人互动、聊天、热闹、共同参与 | `社交连接` |
| `ritual_atmosphere` | 仪式感、氛围、纪念日 | `氛围仪式` |
| `comfort_and_safety` | 不累、不挤、安全、低强度 | `舒适安全` |

为什么需要：

- `template_id` 用于检测 probe 是否记住生成句式。
- `semantic_pattern` 用于检测 probe 是否只学了少数语义套路。
- 这两个字段能支持 template split 和 semantic split，不一定要参与训练。

### 6.5 `paraphrase_group_id`、`noise_group_id`、`pair_id`

这三个字段用于鲁棒性和 C-C 成对评测。

`paraphrase_group_id`：同义改写组。

```json
{"example_id": "p1", "text": "我想找离家近、不用排队的地方。", "paraphrase_group_id": "conv_easy_001"}
{"example_id": "p2", "text": "最好路上少折腾，到了也别等太久。", "paraphrase_group_id": "conv_easy_001"}
```

`noise_group_id`：原句和加噪版本。

```json
{"example_id": "n1_clean", "text": "想找一家健康清淡的餐厅。", "noise_group_id": "健康克制_noise_001"}
{"example_id": "n1_noisy", "text": "想找个健康点的餐厅吧，别太油，嗯最好清淡些。", "noise_group_id": "健康克制_noise_001"}
```

`pair_id`：成对或成组比较。

```json
{
  "example_id": "choice-001-a",
  "text_type": "option_choice",
  "pair_id": "choice-001",
  "option_id": "A",
  "target_value": "省心便利"
}
```

为什么需要：

- 没有 `paraphrase_group_id`，无法知道哪些样本应该语义等价。
- 没有 `noise_group_id`，无法计算加噪前后稳定性。
- 没有 `pair_id`，无法做 option choice、rejection rationale、steering 前后对比。

### 6.6 `a_stage_expected`

`a_stage_expected` 用来评测 A 架构，而不是训练 value probe。它是“这条样本经过 A 阶段后应该产生哪些关键结构”的弱标注。

推荐只标关键字段，不要求完整复制 A 输出：

```json
{
  "a_stage_expected": {
    "current_intent": {
      "participants": [{"role": "friend"}],
      "explicit_tags": ["social", "nearby"],
      "avoid": ["long_queue"]
    },
    "current_value_signal": {
      "省心便利": {"relation": "related", "signed_score_min": 3}
    },
    "scene_frame": {
      "scene_type": "friends",
      "scene_subtype_any": ["social", "nearby"]
    },
    "planning_contract": {
      "must_include_soft_tags": ["social", "nearby"],
      "must_not_include_hard_tags": ["kid_friendly"],
      "must_not_activate_values": ["家庭照护"]
    },
    "resolution_trace": {
      "must_skip_memory": ["child_profile"],
      "expected_reason_contains": ["current scene is friends"]
    }
  }
}
```

用途：

- 评测 `profile_leakage_rate`。
- 评测 `scene_consistency_rate`。
- 评测 current-first 冲突处理。
- 确保 B/C 以 `planning_contract` 为主输入时仍能运行。
- 检查 B/C LLM 是否正确使用补充上下文且不覆盖 contract。

### 6.7 脚本改造清单

建议新增或修改以下脚本。

| 脚本 | 改造内容 | 是否需要重新训练 |
| --- | --- | --- |
| `experiments/generate_value_probe_paragraphs.py` | 生成 `text_type`、`template_id`、`semantic_pattern`、`labels`、`value_relations`、可选 `evidence_spans` | 如果只做评测，不需要；如果扩充训练集，需要 |
| `experiments/value_probe/prepare_dataset.py` | 校验新 schema，保留元数据到 split 文件和 cache metadata | 如果 activation cache 需要这些字段，需要重跑 cache；probe 本身不一定要重训 |
| `experiments/value_probe/cache_qwen_activations.py` | metadata 中保留 `text_type/template_id/semantic_pattern/group_id` 等字段 | 不改变 hidden states，但需重跑 cache 才能让 metadata 完整 |
| `experiments/value_probe/evaluate_extended.py` | 按 `text_type/template_id/semantic_pattern/group_id` 分层汇报 | 不需要重训 |
| `experiments/value_probe/control_tasks.py` | 增加 keyword、length、lexical baseline、template/semantic split | 不需要重训 |
| `experiments/value_probe/generate_abstract_value_samples.py` | 生成 abstract value 数据 | 新数据训练 A-A 时需要 |
| `experiments/value_probe/generate_robustness_samples.py` | 生成 paraphrase/noise 数据并写 group id | 只评测现有 probe 不需要重训；纳入训练则需要 |
| `experiments/value_probe/generate_cc_value_cases.py` | 生成 option choice/rationale/revision/explanation 数据 | C-C probe 或 downstream 模型训练时需要 |
| `experiments/a_stage/evaluate_a_contract.py` | 新增 A 架构评测：scene consistency、profile leakage、override、trace coverage | 不需要 probe 重训 |
| `src/state.py` | 增加 `current_intent/current_value_signal/scene_frame/planning_contract/resolution_trace` TypedDict 字段 | 不涉及训练 |
| `src/nodes/intent_parser.py` | 输出或包装 `current_intent`，旧 `intent` 兼容 | 不涉及训练 |
| `src/nodes/memory_manager.py` | 输出 `user_profile_memory`，不要直接覆盖当前 intent | 不涉及训练 |
| `src/nodes/scenario_planner.py` | 消费 `scene_frame`，不重复猜场景 | 不涉及训练 |
| `src/nodes/value_signal.py` | 新增当前 value-probe 推理节点，可先规则/文件 fallback | 使用已有 probe 不需要；换训练数据才需要 |
| `src/nodes/planning_contract.py` | 新增仲裁和 contract 编译节点 | 不涉及训练 |

### 6.8 生成脚本输出规范

生成脚本应避免只生成文本，还要保留生成意图。

建议每个模板定义为：

```python
{
    "template_id": "nearby_low_wait_query",
    "text_type": "concrete_query",
    "semantic_pattern": "reduce_friction",
    "target_value": "省心便利",
    "relation": "related",
    "label_score": 5,
    "slots": {
        "companion": ["self", "child"],
        "activity": ["restaurant", "activity"],
        "constraint": ["nearby", "low_wait"]
    }
}
```

每条样本生成后要写：

- `template_id`
- `semantic_pattern`
- `text_type`
- `target_value`
- `relation`
- `labels`
- `value_relations`
- `split`
- 可选 `evidence_spans`
- 可选 `a_stage_expected`

去重建议：

```text
dedupe_key = (text, target_value, relation, text_type)
```

分层 split 建议：

```text
stratify_key = (text_type, target_value, relation, semantic_pattern)
```

如果要做 template split，则必须保证某些 `template_id` 完全不进入 train。

### 6.9 哪些需要重新训练

不需要重新训练，只需重新评测：

- 3-way confusion matrix。
- Macro 3-way F1。
- Positive/negative F1、AUC、PR-AUC。
- Opposite-vs-unrelated AUC。
- Top-1 / MRR。
- Pearson/Spearman、Brier、ECE、threshold sweep。
- Bootstrap CI。
- Cross-value leakage。
- Random label、shuffled value、keyword baseline、length baseline、lexical baseline。
- Length bias。
- 如果已有 paraphrase/noise/pair metadata，也可直接做鲁棒性评测。

需要重新缓存 activations，但不一定重新训练：

- 旧 dataset 缺少 `text_type/template_id/semantic_pattern/group_id`，而评测希望从 score 文件追溯这些字段。
- 新增 abstract / robustness / C-C 样本后，希望用已有 probe 对这些新样本打分。
- 此时需要跑 activation cache 和 score 脚本，但可以加载已有 probe checkpoint 做 inference。

需要重新训练 probe：

- 训练集加入 Abstract 数据，做 `AA_probe` 或 `mixed_probe`。
- 训练集加入更强 hard negatives，修正 keyword baseline 过强问题。
- 训练目标从单目标 relation 扩展到全 value `labels` 多任务。
- fluent-level evidence span 标注大幅变化。
- 做多 seed layer stability。

需要改 A 阶段代码，但不需要训练：

- 新增 `current_intent` 包装。
- 新增 `scene_frame`。
- 新增 `planning_contract`。
- 新增 `resolution_trace`。
- 让 B/C 的机器可执行链路优先读 `planning_contract`。
- 让 B/C LLM 可读取 intent/value/trace 作为解释和质检上下文。

需要下游重新评测：

- B optimizer 开始显式消费 `value_weights/score_weights`。
- C 执行解释开始引用 `resolution_trace` 或 value evidence。
- C-C choice/rationale/revision 数据加入后，需要重新跑 `run_b_eval.py` 和 plan-quality analysis。

## 7. Steering 最小实验

Steering 是后续阶段，不阻塞当前 probe 评测。目标是从“能读出 value”推进到“value direction 能影响行为”。

### 7.1 最小设置

| 项 | 建议 |
| --- | --- |
| 模型 | 先用当前 Qwen3-4B |
| 层 | sentence probe 默认 layer 11；fluent probe 可对比 layer 29 |
| 方向 | 使用 value-specific linear head 或 matched-minus-mismatched activation mean |
| 强度 | `[-2, -1, 0, 1, 2]` |
| 任务 | 二选一方案选择、推荐理由排序 |
| 场景 | 每个 value 至少 50 个 concrete pair |

### 7.2 指标

| 指标 | 含义 |
| --- | --- |
| `target_choice_shift` | 正向 steering 后目标 value 方案选择率提升 |
| `logprob_margin_shift` | 目标选项 logprob margin 是否增加 |
| `dose_response_slope` | steering 强度和行为变化是否单调相关 |
| `monotonicity_rate` | 每个样本在强度递增时是否大体单调 |
| `off_target_shift` | 非目标 value 是否被过度影响 |
| `fluency_degradation` | 输出困惑度、重复率、格式错误是否变差 |

### 7.3 示例场景

```text
value = 省心便利
user = 周末想和朋友吃饭，最好省心点。
option_a = 附近商场，能预约，排队短，但口味普通。
option_b = 远一点的网红店，排队久，但更有新鲜感。
positive steering 预期 = 更偏向 option_a
negative steering 预期 = 更能接受 option_b
```

## 8. 实施顺序

### Phase 1：评测闭环修正

- 用 val 选层，test 只最终汇报一次。
- 增加 `related` / `opposite` / `unrelated` 三分类评测。
- 增加 PR-AUC、Pearson/Spearman、ECE/Brier、threshold sweep。
- 给 diagonal margin 加 bootstrap CI。

完成标准：

- 新增 `eval_extended/` 产物。
- 报告中明确区分 layer selection split 和 final test split。

### Phase 2：Control Task 与 Selectivity

- 实现 random label、shuffled value、keyword baseline、length baseline。
- 输出 `selectivity = real_task_score - control_task_score`。
- 增加 template split 或 semantic pattern split。

完成标准：

- 真实任务显著优于 control。
- 如果 keyword baseline 接近 probe，需要回到数据生成侧增加 hard negatives。

### Phase 3：数据 schema 与 A 架构契约

- 给 JSONL 增加 `text_type`、`template_id`、`semantic_pattern`、group id、`a_stage_expected`。
- 在 `prepare_dataset.py` 和 cache metadata 中保留这些字段。
- 在 A 阶段引入 `current_intent`、`current_value_signal`、`scene_frame`、`planning_contract`、`resolution_trace` 的兼容输出。
- 新增 A contract 评测脚本。

完成标准：

- 旧 B/C eval 不因字段迁移中断。
- `planning_contract` 可单独支撑 B/C 的机器可执行链路运行。
- B/C LLM 在读取补充上下文时满足 contract adherence。
- 能汇报 scene consistency、profile leakage、override accuracy、trace coverage。

### Phase 4：Abstract 数据与 A-A

- 生成 `abstract_value_samples.jsonl`。
- 复用当前数据准备和 activation cache 流程。
- 训练 `AA_probe` 并评估 abstract held-out。

完成标准：

- A-A 在 abstract held-out 上高于 baseline。
- per-value 结果无明显塌缩。

### Phase 5：A-C / C-A / Mixed Transfer

- 用 abstract train 测 concrete test。
- 用 concrete train 测 abstract test。
- 用 abstract + concrete 混合训练，比较迁移收益。

完成标准：

- 汇报 `transfer_drop` 和 `relative_transfer`。
- 能说明抽象 value 表征是否真正落到本地生活请求。

### Phase 6：C-C 与下游消融

- 构造 option choice、rationale、revision、plan explanation 数据。
- 对比 `intent only`、`rule probe`、`Qwen3-4B probe`、`probe + memory`。

完成标准：

- probe 能带来可解释的 plan quality 或 explanation consistency 提升。
- memory 不覆盖当前明确请求的 current-first 约束仍成立。

### Phase 7：Steering

- 固定 probe direction 和 layer。
- 做 strength sweep。
- 汇报 target shift、dose-response、off-target 和 fluency degradation。

完成标准：

- 正向 steering 在目标 value 上有单调或近似单调行为变化。
- 非目标 value 和输出质量没有不可接受退化。

## 9. 近期可落地文件清单

建议新增或更新：

| 文件 | 用途 |
| --- | --- |
| `experiments/value_probe/evaluate_extended.py` | 统一计算新增指标 |
| `experiments/value_probe/control_tasks.py` | 构造 random label、shuffled value、baseline 特征 |
| `experiments/value_probe/generate_abstract_value_samples.py` | 生成 Abstract 数据 |
| `experiments/value_probe/generate_robustness_samples.py` | 生成 paraphrase/noise 鲁棒性数据 |
| `experiments/value_probe/generate_cc_value_cases.py` | 生成 C-C option/rationale/revision/explanation 数据 |
| `experiments/a_stage/evaluate_a_contract.py` | 评测 A 架构拆分后的 contract、trace 和泄漏问题 |
| `src/nodes/value_signal.py` | 当前轮 value-probe 推理节点 |
| `src/nodes/planning_contract.py` | A 阶段仲裁和 B/C 合约编译节点 |
| `experiments/value_probe_runs/qwen3_4b_token_probe/eval_extended/` | 存放扩展评测结果 |
| `docs/value-probe-evaluation-extension-plan.md` | 本计划 |

暂不建议直接修改训练报告 HTML。训练报告应继续记录已完成实验，本文件负责规划未完成实验。

## 10. 优先级摘要

最高优先级：

1. 修正 val/test 使用边界。
2. 拆分 `opposite` 和 `unrelated`。
3. 加 control task / selectivity。
4. 补数据 schema 元字段：`text_type`、`template_id`、`semantic_pattern`、group id。
5. 明确 A 阶段 `planning_contract` 是 B/C 机器执行主入口，intent/value/trace 是 B/C LLM 的补充上下文。

第二优先级：

1. 补 Abstract 数据。
2. A-A / A-C / C-A transfer matrix。
3. A 架构消融：profile leakage、current override、trace coverage。
4. C-C decision/rationale 数据。
5. 下游消融。

第三优先级：

1. Steering dose-response。
2. 多模型复现。
3. 更大 value taxonomy。

一句话目标：先把当前 probe 从“分类结果很好”升级为“评测可信”，再用 Abstract 数据证明“抽象价值可迁移到具体本地生活决策”，最后用 steering 证明 value direction 具备因果影响。
