# 竞赛提交包（event-stack）

本目录由 `python scripts/build_submission.py` 从仓库唯一真源确定性生成。
请勿手工修改任何生成文件；需要变更时重建整个包。

## 包结构

```text
dist/submission/
├── main.py                        # 竞赛入口：--raw 模式 + 官方 adapter 边界
├── event_stack/                   # 推理运行时（canonical 源机械 vendored；main.py 使用）
├── models/
│   └── event_stack/<run_key>/
│       ├── deployment/            # 冻结推理模型（macro/micro/verifier + policy）
│       ├── outer-fold-0..4/       # 五折 outer-fold evidence bundle
│       ├── promotion_summary.json
│       └── promotion_attestation.json
├── src/                           # canonical 算法源码（复现与审阅用）
├── scripts/                       # 训练/评估/发布/构建全链脚本
├── tests/                         # 单元/集成/parity/release 测试
├── visual/                        # 可视化应用（dist/visual 工作区原样打包）
├── schema/                        # prediction.schema.json（预测契约 v1.0）
├── examples/                      # example_prediction.json（合成安全示例）
├── feature_schema.json            # 特征 schema v2（macro 63 / micro 47 / verifier 116）
├── requirements.txt               # 精确依赖 pin（来自 deployment manifest）
├── manifest.json                  # 逐文件 SHA-256 与 source/model 闭包
└── README.md
```

## 快速开始（推理）

```bash
python -m pip install -r requirements.txt
python main.py --raw path/to/collect_data1_2_3.txt --output result.json
python main.py --raw path/to/subject-folder --output result.json --include-timeline --include-candidates
```

## 可用接口

1. **命令行（本包）**：`main.py --raw INPUT --output OUTPUT [--include-timeline]
   [--include-candidates] [--device cpu]`；输出为 `schema/prediction.schema.json`
   （v1.0）定义的预测文档（events / 可选 timeline、candidates、gaps）。
2. **Python API（本包）**：
   ```python
   from event_stack.inference import Predictor
   predictor = Predictor.from_bundle("models/event_stack/<run_key>/deployment",
                                     run_key="<run_key>")
   result = predictor.predict_file("path/to/collect_data1_2_3.txt")
   ```
3. **可视化**：`visual/` 是团队可视化应用；它只消费预测 JSON（见
   `visual/README.md` 与 `schema/`、`examples/`），不重实现任何算法决策。

## 官方竞赛模式（已知约束）

已审计的发布材料中未定义官方竞赛输入/输出契约。因此官方模式在注册具体 adapter 前
会显式拒绝：

```bash
python main.py --official-input INPUT --output OUTPUT
# 退出码 2；stderr: competition submission refused: official competition
# input/output adapter is not registered
```

如需启用：在 `src/pipeline/inference/competition_adapter.py` 中实现 adapter，
通过 `register_adapter(...)` 注册，然后重建本包；其余组件无需改动。

## 复现与证据

- `scripts/reproduce_release.py --run-key <run_key>`：免训练验证发布的
  registry、attestation、五折 evidence 与 deployment bundle（需要完整仓库与
  缓存；本包内为源码副本，不在包内直接运行）。
- `scripts/evaluate_event_stack.py`：严格 subject-disjoint nested 五折评估；
  `scripts/train_event_stack.py`：full-target 训练与晋级。
- 完整运行需要原始数据集与 `cache/`（不在提交包内）；数据清洗与评估协议的说明
  随仓库 `docs/` 维护，`tests/` 为对应回归。
- `models/event_stack/<run_key>/promotion_attestation.json` 绑定 canonical
  summary、严格五折与每个 manifest 的 SHA-256。

## 运行时 ABI

Python 3.11.x，配合 `requirements.txt` 中来自 deployment manifest 的精确 pin：
numpy、joblib、scikit-learn、lightgbm。支持 `--device cpu`；`--device gpu/cuda`
会被拒绝，因为当前发布没有经过审计的 CUDA 适配器。
