# Qwen3-4B Fluent-Level Value Probe 训练记录

版本日期：2026-05-25  
模型：`Qwen/Qwen3-4B`  
目标：在同一段用户需求上同时支持 sentence-level value 预测和 fluent-level 逐词 `-6..6` 相关性预测。

## 结论

当前 fluent-level probe 已完成 36 层扫描。按验证集 RMSE 选择最佳层为 `layer_29`。

| 指标 | 数值 |
|---|---:|
| best layer | 29 |
| best epoch | 11 |
| val RMSE | 1.1437 |
| val MAE | 0.7961 |
| test RMSE | 1.1609 |
| test MAE | 0.8152 |
| test nonzero direction accuracy | 0.8967 |
| test threshold nonzero direction accuracy | 0.8130 |

sentence-level probe 继续使用原最佳层 `layer_11`：

| 指标 | 数值 |
|---|---:|
| test macro AUC | 1.0000 |
| test macro F1 | 0.9737 |
| mean diagonal margin | 0.7936 |

## 训练数据

fluent-level 标注来自：

`experiments/value_probe_runs/qwen3_4b_token_probe/fluent_level/all.jsonl`

| 项 | 数值 |
|---|---:|
| source examples | 1200 |
| train / val / test | 960 / 120 / 120 |
| values | 4 |
| examples per value | 300 |
| fluent tokens | 42401 |
| nonzero fluent tokens | 8073 |
| invalid examples | 0 |
| missing examples | 0 |

value 分布：

| value | examples |
|---|---:|
| `family_care` | 300 |
| `health` | 300 |
| `convenience` | 300 |
| `cost_sensitivity` | 300 |

分数分布：

| score | tokens |
|---:|---:|
| -6 | 13 |
| -5 | 174 |
| -4 | 595 |
| -3 | 1444 |
| -2 | 1219 |
| -1 | 506 |
| 0 | 34328 |
| 1 | 891 |
| 2 | 742 |
| 3 | 1027 |
| 4 | 717 |
| 5 | 703 |
| 6 | 42 |

## 对齐校验

训练时没有重新缓存 Qwen hidden states，而是复用：

`experiments/value_probe_runs/qwen3_4b_token_probe/activations`

fluent 分词和 Qwen tokenizer token 之间采用字符覆盖对齐：

1. fluent token 去掉标点、符号、空格后展开为字符跨度。
2. Qwen token decode 后也去掉标点、符号、空格。
3. 每个 Qwen token 的监督分数为其覆盖 fluent token 分数的字符加权平均。
4. 推理时再把 model-token 分数按字符覆盖聚合回 fluent token。

对齐结果：

| 项 | 数值 |
|---|---:|
| aligned examples | 1200 |
| supervised model tokens | 36571 |
| truncated examples | 0 |
| alignment errors | 0 |

## 层扫描结果

按 `best_val_metric = val_rmse` 排序的前 8 层：

| layer | best epoch | val RMSE | test RMSE | test MAE | test nonzero direction accuracy |
|---:|---:|---:|---:|---:|---:|
| 29 | 11 | 1.1437 | 1.1609 | 0.8152 | 0.8967 |
| 33 | 10 | 1.2057 | 1.1738 | 0.8489 | 0.9130 |
| 21 | 7 | 1.2080 | 1.1727 | 0.8614 | 0.8989 |
| 32 | 10 | 1.2138 | 1.1881 | 0.8727 | 0.9141 |
| 20 | 7 | 1.2187 | 1.1731 | 0.8503 | 0.9054 |
| 28 | 7 | 1.2291 | 1.1838 | 0.8583 | 0.9185 |
| 23 | 10 | 1.2309 | 1.2172 | 0.9051 | 0.9043 |
| 16 | 8 | 1.2335 | 1.1723 | 0.8603 | 0.8804 |

## 测试用例

### 用例 1：便利正向

输入：

`周末想在家附近找个少排队的餐厅，最好能提前订座。`

预期：

- sentence-level: `convenience` 为正向主信号。
- fluent-level: `附近`、`少`、`提前`、`订座` 为便利正向证据。

smoke 输出摘录：

| token | convenience predicted score |
|---|---:|
| 附近 | 2.91 |
| 少 | 2.19 |
| 提前 | 3.83 |
| 订座 | 4.64 |

### 用例 2：便利反向

输入：

`我想找一家藏在老城区巷子深处的小店，路远一点没关系，排队也可以。`

预期：

- sentence-level: `convenience` negative score 高。
- fluent-level: `藏`、`巷子`、`深处`、`远`、`排队` 偏负向。

### 用例 3：家庭照顾 + 健康 + 便利混合

输入：

`今天带爸妈和孩子出去吃饭，想找个离家近、清淡一点、能订座的地方。`

预期：

- sentence-level: `family_care`、`health`、`convenience` 均有正向信号。
- fluent-level: `爸妈`、`孩子` 对家庭照顾正向；`清淡` 对健康正向；`近`、`订座` 对便利正向。

### 用例 4：健康反向 + 预算不敏感

输入：

`今晚就想吃重口味火锅，排队久一点也没关系，预算不用卡太死。`

预期：

- sentence-level: `health` 负向，`convenience` 负向，`cost_sensitivity` 负向。
- fluent-level: `重口味`、`火锅` 对健康负向；`排队`、`久` 对便利负向；`预算`、`不用`、`卡` 对预算敏感负向。

## 产物

| 产物 | 路径 |
|---|---|
| fluent-level 标注数据 | `experiments/value_probe_runs/qwen3_4b_token_probe/fluent_level/all.jsonl` |
| fluent-level probe | `experiments/value_probe_runs/qwen3_4b_token_probe/fluent_level_probes/layer_29/probe.pt` |
| fluent-level best layer | `experiments/value_probe_runs/qwen3_4b_token_probe/fluent_level_probes/best_layer.json` |
| sentence-level probe | `experiments/value_probe_runs/qwen3_4b_token_probe/probes/layer_11/probe.pt` |
| HTML 应用 | `docs/value-probe-predictor-app.html` |
| 本地预测服务 | `experiments/value_probe/serve_probe_app.py` |

## 复现命令

对齐检查：

```bash
.venv-probe/bin/python experiments/value_probe/train_fluent_level_probe.py \
  --layers 11 \
  --align-only \
  --local-files-only
```

训练全层 fluent-level probe：

```bash
.venv-probe/bin/python -u experiments/value_probe/train_fluent_level_probe.py \
  --layers all \
  --epochs 12 \
  --local-files-only \
  --write-word-scores
```

启动 HTML 应用服务：

```bash
.venv-probe/bin/python experiments/value_probe/serve_probe_app.py \
  --local-files-only
```

打开：

`http://127.0.0.1:8765`

API：

```bash
curl -X POST http://127.0.0.1:8765/api/predict \
  -H 'content-type: application/json' \
  -d '{"text":"周末想在家附近找个少排队的餐厅，最好能提前订座。"}'
```

## 注意

- HTML 页面本身不直接运行模型；它通过本地 Python API 获取预测。
- fluent-level 输入分词在服务端用训练集 fluent token 词表做最长匹配；如需完全复用人工分词，可在 API 请求中传入 `tokens` 数组。
- `direction_accuracy` 对大量 0 分词较苛刻；实际查看非零证据时更应关注 `nonzero_direction_accuracy` 和逐词分数。
