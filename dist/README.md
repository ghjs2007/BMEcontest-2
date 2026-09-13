# 进食事件检测发布包：event-stack

状态（2026-09-13）：`event_stack/` 为实验 key `a7396a9aa7c38f42` 的 CPU 发布包。该版本
来自严格 subject-disjoint nested 五折开发证据，聚合 F1=`0.5432098765`
（TP/eligible/pred=`110/153/252`，PPV=`0.4365079365`，recall=`0.7189542484`，
FP=`142`），最终短餐 recall=`18/39=0.4615384615`（candidate short recall=`0.5384615385`）。它优于旧 `0.4786324786`，但短餐尚未
达到推荐门 `0.65`，F1 也未达到项目目标 `0.65`，所以这是已固化但暂不推荐默认的开发版本，
不是 untouched 测试集泛化承诺。

## 输入契约

当前包只接受已生成、可审计的候选特征 JSON，不读取原始 `collect_data*.txt` 会话。每个
候选必须携带有限值数组：macro 63 维、micro 47 维、verifier 56 维；同时必须有稳定
`subject_id` 和全局唯一 `sid`。`sid` 只用于同会话 NMS 与事件几何，准入阈值、candidate
cap 和 event budget 按 `subject_id` 执行。

```json
{
  "feature_schema": {"macro": 63, "micro": 47, "verifier": 56},
  "schema_hash": "SHA-256 of canonical feature_schema JSON",
  "sessions": [{
    "subject_id": "stable-subject-id",
    "sid": "session-id",
    "candidates": [{
      "start_ms": 0, "end_ms": 1000,
      "macro": ["63 finite values"],
      "micro": ["47 finite values"],
      "verifier": ["56 finite values"]
    }]
  }]
}
```

未知字段、缺失 ID、重复 `sid`、schema/hash 不匹配、`NaN`/`Inf` 或错误列宽均拒绝。输出
为稳定排序的 canonical JSON：
`{"events":[{"sid":...,"start_ms":...,"end_ms":...,"score":...}],"resolved_device":"cpu"}`。

## 运行

### 运行时 ABI（必须满足）

该 deployment manifest 记录的构建环境为 Python `3.11.15`；运行时必须为 Python
`3.11.x`。Python 本身不写入 requirements，但 `event_stack/requirements.txt` 会从该 manifest
生成 numpy `2.4.6`、joblib `1.5.3`、scikit-learn `1.9.0`、lightgbm `4.7.0` 的精确 `==` pin。
推理程序会在加载任何 joblib 模型之前再次核验 Python 主/次版本和全部四个包的精确版本；不匹配会
明确拒绝，不能依赖 sklearn 的兼容性 warning 继续运行。

```bash
pip install -r event_stack/requirements.txt
python event_stack/predict_event_stack.py \
  --bundle event_stack/bundle \
  --input-features candidates.json \
  --output predictions.json \
  --device auto
```

`--device` 支持 `auto|cpu|gpu|cuda`，其中 `gpu` 是 `cuda` 别名。当前 sklearn/LightGBM
组件均为 CPU，`auto` 明确解析为 CPU；强制 `gpu/cuda` 在 CUDA 不可用或 bundle 没有显式
CUDA-capable component 时必须失败，不能伪报 GPU。当前 CUDA adapter registry 为空。

## 真实架构

```
240s/15s macro（63 维） ∪ 15s/7.5s ACC+GYRO micro（47 维）
  → 同 sid 稳定 NMS
  → subject admission（阈值、IoU、cap 由 train OOF 冻结）
  → LogisticRegression + 受限 LightGBM（56 维事件复核）概率 blend
  → 冻结 event policy → canonical Episode JSON
```

当前五折 raw union 为 3,413，micro candidates 为 2,773，admission 后候选为 310。模型和
追溯信息在 `event_stack/bundle/`：manifest 保存模型 SHA-256、输入 provenance、依赖版本
和 deployment role；根项目还保留五个 outer-fold evidence、`promotion_summary.json` 和
`promotion_attestation.json`。发布包只能由注册的 `scripts/package_event_stack.py` 从合法
deployment bundle 原子生成，不能手工替换。

## 限制与后续

raw-session adapter 尚未完成，不能把本包描述为原始会话端到端推理。下一步先补齐该适配器、
短餐/餐时 hard-negative 与候选簇去重，再启动 FD-I/FD-II 的独立迁移门控；必须保留随机
初始化和 `external_weight=0` 对照，并遵守 FD 的 CC BY-NC-ND 4.0 许可及署名限制。

仓库完整协议、逐折指标和开发时间线见根目录 `README.md` 与 `docs/三阶段重构设计.md`。
遗留的 `predict.py`/`predict_legacy.py` 仅作旧滑窗/检测即排序对照，不是当前 event-stack
输入或指标的替代。
