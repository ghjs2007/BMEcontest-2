# Distribution Workspace

`dist/` 是发布与交付工作区。算法唯一真源在仓库 `src/pipeline/`；`dist/` 下的
**生成物不得手工修改**，一律由仓库脚本重建。

```text
dist/
├── inference/      # canonical 独立推理包（生成物）：原始 collect_data*.txt → 预测 JSON
├── visual/         # 团队可视化开发区（只依赖 prediction schema 与 inference 输出）
├── submission/     # 竞赛最终提交包（生成物）：raw 模式 + 官方 adapter 边界
├── examples/       # 合成安全示例 example_prediction.json（符合 schema，无真实数据）
├── schema/         # 稳定契约 prediction.schema.json（v1.0）
└── event_stack/    # 旧 serialized-payload 运行时（发布证据，保留，勿手改）
```

发布线：`160afaf81debf1ee`（严格 subject-disjoint nested 五折开发 F1 = 0.6514285714，
TP/eligible/pred = 114/153/197）。发布证据与模型在 `models/event_stack/<run_key>/`，
指针在 `release/event_stack_incumbent.json`；`dist/` 内的任何包都必须与它们一致。

## 重建命令（均需仓库根目录运行）

```bash
python scripts/build_inference_distribution.py     # 重建 dist/inference/
python scripts/build_submission.py                 # 重建 dist/submission/
```

## 快速推理（inference 包）

```bash
cd dist/inference
python -m pip install -r requirements.txt
python predict.py path/to/collect_data1_2_3.txt --output prediction.json
python predict.py path/to/subject-folder --output prediction.json --include-timeline
```

`--device cpu` 支持；当前发布没有经过审计的 CUDA 适配器，`--device gpu/cuda`
会被明确拒绝（退出码 2），不会伪装成 CPU 成功。输出 JSON 遵循
`dist/schema/prediction.schema.json`。

## legacy 运行时（保留审计，勿用于新集成）

`dist/event_stack/` 是旧的 **serialized-payload** 运行时（输入为预计算的
63/47/116 维候选特征 JSON，不读取原始传感器文件），由
`scripts/package_event_stack.py` 生成，文件集与哈希由
`dist/event_stack/runtime_manifest.json` 锁定（**不得手工添加/删除其中任何文件**）。
它是当前 release 的部署 bundle 的可独立运行副本，也是发布证据链的一部分；
新集成一律使用 `dist/inference/`。

- 输入契约：`feature_schema`（schema v2：macro 63 / micro 47 / verifier 116，
  含 Context-v1 60 列）、`schema_hash`、`sessions[].candidates[]` 有限值数组；
  未知字段、NaN/Inf、宽度或哈希不匹配均拒绝。
- 输出：`{"events": [{"sid","start_ms","end_ms","score"}], "resolved_device": "cpu"}`，
  稳定排序；`subject_id` 只影响冻结的准入/预算，不出现在输出中。
- 运行时 ABI：Python 3.11.x + `event_stack/requirements.txt` 的精确 pin
  （numpy 2.4.6 / joblib 1.5.3 / scikit-learn 1.9.0 / lightgbm 4.7.0），
  加载任何模型前校验，不匹配即拒绝。

历史滑窗/检测即排序对照产物（旧 predict 脚本、滑窗模型权重与只读 src 副本）
已按交付清理移除，可从 git 历史检出；删除台账见
`tests/fixtures/deletion_manifest.json`，审计记录见 `docs/repository_cleanup_audit.md`。
