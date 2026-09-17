# 可视化工作区契约

本目录是团队可视化应用的开发区。当前刻意保持为空：在这里开发的前端代码只能消费
canonical 预测输出，不能包含任何算法实现。

## 静态网页入口

交付入口是 `dist/visual/index.html`。执行 `npm run build` 后，Vite 会把纯静态页面和资源直接生成在本目录；页面不需要 Node 或 Python 后端。可用任意静态服务器托管仓库根目录，例如：

```powershell
cd D:\BME\BMEcontest-2
py -m http.server 4173
```

然后打开 `http://127.0.0.1:4173/dist/visual/`。直接双击页面可打开演示数据；浏览器本地文件策略可能阻止 Model 页读取发布指标，此时使用静态服务器即可。

`npm run dev` 只用于开发预览；生产交付使用 `npm run build`。

## 前端可以拿到什么

只支持以下三种接口：

1. `dist/schema/prediction.schema.json` —— 稳定的 JSON 契约；
2. `dist/examples/example_prediction.json` —— 符合 schema 的合成安全示例
   （`input.source == "synthetic-example"`，不含真实传感器数据）；
3. `dist/inference` 产出的预测 JSON（`python predict.py ...`）或 canonical
   `Predictor` API 的输出。

前端**不得重实现**阈值判定、候选准入、事件融合、解码或任何特征提取。预测 JSON 里没有
给出的决策，就不是前端该计算的东西。

## 预测 JSON 结构

始终存在的必需字段：

| 字段 | 含义 |
|---|---|
| `schema_version` | 契约版本，当前为 `"1.0"`。 |
| `model.name` / `model.run_key` | 固定为 `"event-stack"` 与冻结的 release key。 |
| `input.source` / `input.duration_seconds` | 输入描述与覆盖时长。 |
| `events[]` | 最终解码出的进食事件。 |
| `diagnostics.coverage` / `diagnostics.warnings` / `diagnostics.resolved_device` | 覆盖率、告警、执行设备。 |

`events[]` 记录：

| 键 | 含义 |
|---|---|
| `id` | 从 0 开始的稠密序号；数组顺序即 canonical 顺序。 |
| `session_id`、`start_ms`、`end_ms` | 事件区间（原始会话时间戳）。 |
| `duration_s` | `(end_ms - start_ms) / 1000`，精确值。 |
| `confidence` | 冻结解码器的最终分数，`[0, 1]`。 |

可选调试块（用 `--include-timeline` / `--include-candidates` 请求，永远不是必需字段）：

- `candidates[]`：`session_id`、`start_ms`、`end_ms`、`score`、`admitted`——并集候选
  全量及准入是否通过。被拒绝的候选（`admitted: false`）也属于展示契约，界面可以据此
  说明某段为什么被/未被输出为事件。
- `gaps[]`：`session_id`、`start_ms`、`end_ms`——有效段之间的未记录区间。按"无数据"
  渲染，绝不在缺口上做插值。
- `timeline`：`session_ids`、`macro_windows`、`micro_windows` 计数，外加可选的
  逐时间点 `series[]`：`timestamp_ms`、`macro_probability`、`micro_probability`、
  `valid`、`gap`。`gap: true` 的点携带 `valid: false` 与零概率；请渲染为"无数据"，
  而不是实测的零值。

## 渲染建议

建议的展示层（全部由预测 JSON 驱动，不做任何重算）：

```text
raw activity（仅作参考，来自会话文件本身）
macro probability          timeline.series[].macro_probability
micro probability          timeline.series[].micro_probability
candidate regions          candidates[]（admitted 与 rejected 区分）
verifier decisions         candidates[].admitted
final eating episodes      events[]
```

若 `timeline.series` 缺失（默认输出不包含），退化为事件层与候选层展示；不要自行合成
概率曲线。

## 开发说明

- 前端代码放在本目录（`dist/visual/`）；本目录**不允许**出现 Python 算法文件——
  发布测试会强制检查。
- 用 schema 校验 fixture：
  `python -m jsonschema -i dist/examples/example_prediction.json dist/schema/prediction.schema.json`
- 生产真实预测 JSON 可用自包含推理包：

```bash
cd dist/inference
python predict.py path/to/collect_data1_2_3.txt --output prediction.json --include-timeline --include-candidates
```
