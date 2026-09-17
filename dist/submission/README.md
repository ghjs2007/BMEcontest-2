# 竞赛提交包

本目录由 `python scripts/build_submission.py` 从 promoted canonical release 自动
生成。请勿手工修改 `event_stack/`、`models/`、`main.py` 或 `manifest.json`；
需要变更时重建整个包。

## 包结构

```text
dist/submission/
├── main.py               # 竞赛入口：--raw 模式 + 官方 adapter 边界
├── event_stack/          # canonical 源码 vendored 副本（含 competition_adapter）
├── models/               # 冻结 deployment bundle（macro/micro/verifier 模型 + policy）
├── manifest.json         # 逐文件 SHA-256 与 source/model 闭包
├── feature_schema.json   # schema v2（macro 63 / micro 47 / verifier 116）
├── requirements.txt      # 精确依赖 pin（来自 deployment manifest）
└── README.md
```

## 从零验证

```bash
python -m pip install -r requirements.txt

# raw 模式冒烟测试（canonical 原始输入 → canonical 预测文档）：
python main.py --raw path/to/collect_data1_2_3.txt --output result.json
```

`result.json` 必须与同输入下 canonical `Predictor.predict_file()` 的文档完全一致。
本包只包含源码与模型：`manifest.json` 记录每个文件的 SHA-256 以及精确的
source/model 闭包，构建流程会在隔离导入探测前后各校验一次。

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

## 运行时 ABI

Python 3.11.x，配合 `requirements.txt` 中来自 deployment manifest 的精确 pin：
numpy、joblib、scikit-learn、lightgbm。支持 `--device cpu`；`--device gpu/cuda`
会被拒绝，因为当前发布没有经过审计的 CUDA 适配器。
