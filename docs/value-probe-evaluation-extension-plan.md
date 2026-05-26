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
| Cross-value leakage matrix | 看 `family_care` 是否总误激活 `convenience` 等邻近 value |

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
| Definition | `convenience 表示减少时间成本、排队成本、路线成本和不确定性。` | A-A |
| Boundary | `离家近属于 convenience，但餐厅装修好不一定属于 convenience。` | A-A |
| Synonym | `少折腾、省心、稳妥、能提前订` | A-A |
| Opposite | `我不介意排队，也愿意多绕路。` | A-A / signed value |
| Abstract preference | `我更重视确定性，而不是探索新鲜感。` | A-A / C-A |
| Abstract tradeoff | `如果省钱会牺牲家人舒适度，我倾向选择舒适度。` | A-A / C-C |
| Abstract unrelated | `我想找评分高、口碑稳定的地方。` | specificity control |

### 3.2 最小数据规模

先覆盖 4 个现有 value：

- `family_care`
- `health`
- `convenience`
- `cost_sensitivity`

建议每个 value 的最小规模：

| 子集 | 每 value 数量 | 说明 |
| --- | ---: | --- |
| positive abstract | 100 | 定义、原则、抽象偏好 |
| opposite abstract | 100 | 明确反向价值 |
| unrelated abstract | 100 | 同领域但不涉及目标 value |
| tradeoff abstract | 50 | 两个 value 冲突或排序 |

总计约 `4 * 350 = 1400` 条。数据文件建议落点：

```text
experiments/value_probe_data/abstract_value_samples.jsonl
experiments/value_probe_runs/qwen3_4b_token_probe/abstract_dataset/
```

### 3.3 建议 JSONL 格式

```json
{
  "example_id": "abstract-convenience-positive-0001",
  "text": "便利意味着减少排队、路线切换和临时不确定性，让用户少花精力完成目标。",
  "text_type": "abstract_definition",
  "target_value": "convenience",
  "relation": "related",
  "labels": {
    "family_care": 0,
    "health": 0,
    "convenience": 6,
    "cost_sensitivity": 0
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

## 5. Steering 最小实验

Steering 是后续阶段，不阻塞当前 probe 评测。目标是从“能读出 value”推进到“value direction 能影响行为”。

### 5.1 最小设置

| 项 | 建议 |
| --- | --- |
| 模型 | 先用当前 Qwen3-4B |
| 层 | sentence probe 默认 layer 11；fluent probe 可对比 layer 29 |
| 方向 | 使用 value-specific linear head 或 matched-minus-mismatched activation mean |
| 强度 | `[-2, -1, 0, 1, 2]` |
| 任务 | 二选一方案选择、推荐理由排序 |
| 场景 | 每个 value 至少 50 个 concrete pair |

### 5.2 指标

| 指标 | 含义 |
| --- | --- |
| `target_choice_shift` | 正向 steering 后目标 value 方案选择率提升 |
| `logprob_margin_shift` | 目标选项 logprob margin 是否增加 |
| `dose_response_slope` | steering 强度和行为变化是否单调相关 |
| `monotonicity_rate` | 每个样本在强度递增时是否大体单调 |
| `off_target_shift` | 非目标 value 是否被过度影响 |
| `fluency_degradation` | 输出困惑度、重复率、格式错误是否变差 |

### 5.3 示例场景

```text
value = convenience
user = 周末想和朋友吃饭，最好省心点。
option_a = 附近商场，能预约，排队短，但口味普通。
option_b = 远一点的网红店，排队久，但更有新鲜感。
positive steering 预期 = 更偏向 option_a
negative steering 预期 = 更能接受 option_b
```

## 6. 实施顺序

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

### Phase 3：Abstract 数据与 A-A

- 生成 `abstract_value_samples.jsonl`。
- 复用当前数据准备和 activation cache 流程。
- 训练 `AA_probe` 并评估 abstract held-out。

完成标准：

- A-A 在 abstract held-out 上高于 baseline。
- per-value 结果无明显塌缩。

### Phase 4：A-C / C-A / Mixed Transfer

- 用 abstract train 测 concrete test。
- 用 concrete train 测 abstract test。
- 用 abstract + concrete 混合训练，比较迁移收益。

完成标准：

- 汇报 `transfer_drop` 和 `relative_transfer`。
- 能说明抽象 value 表征是否真正落到本地生活请求。

### Phase 5：C-C 与下游消融

- 构造 option choice、rationale、revision、plan explanation 数据。
- 对比 `intent only`、`rule probe`、`Qwen3-4B probe`、`probe + memory`。

完成标准：

- probe 能带来可解释的 plan quality 或 explanation consistency 提升。
- memory 不覆盖当前明确请求的 current-first 约束仍成立。

### Phase 6：Steering

- 固定 probe direction 和 layer。
- 做 strength sweep。
- 汇报 target shift、dose-response、off-target 和 fluency degradation。

完成标准：

- 正向 steering 在目标 value 上有单调或近似单调行为变化。
- 非目标 value 和输出质量没有不可接受退化。

## 7. 近期可落地文件清单

建议新增或更新：

| 文件 | 用途 |
| --- | --- |
| `experiments/value_probe/evaluate_extended.py` | 统一计算新增指标 |
| `experiments/value_probe/control_tasks.py` | 构造 random label、shuffled value、baseline 特征 |
| `experiments/value_probe/generate_abstract_value_samples.py` | 生成 Abstract 数据 |
| `experiments/value_probe_runs/qwen3_4b_token_probe/eval_extended/` | 存放扩展评测结果 |
| `docs/value-probe-evaluation-extension-plan.md` | 本计划 |

暂不建议直接修改训练报告 HTML。训练报告应继续记录已完成实验，本文件负责规划未完成实验。

## 8. 优先级摘要

最高优先级：

1. 修正 val/test 使用边界。
2. 拆分 `opposite` 和 `unrelated`。
3. 加 control task / selectivity。
4. 补 Abstract 数据。

第二优先级：

1. A-A / A-C / C-A transfer matrix。
2. C-C decision/rationale 数据。
3. 下游消融。

第三优先级：

1. Steering dose-response。
2. 多模型复现。
3. 更大 value taxonomy。

一句话目标：先把当前 probe 从“分类结果很好”升级为“评测可信”，再用 Abstract 数据证明“抽象价值可迁移到具体本地生活决策”，最后用 steering 证明 value direction 具备因果影响。
