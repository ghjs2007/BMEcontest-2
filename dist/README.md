# 进食事件检测发布包：event-stack

状态（2026-09-15）：`event_stack/` 为 Context-v1 实验 key `160afaf81debf1ee` 的 CPU
发布包。严格 subject-disjoint nested 五折开发证据为 F1=`0.6514285714`
（TP/eligible/pred=`114/153/197`，PPV=`0.5786802030`，recall=`0.7450980392`，FP=`83`），
短餐最终 recall=`20/39=0.5128205128`。它比前一 release `035644cf0889a5dd` 高
`+0.0924542125` F1，满足技术与推荐晋级门；仍是开发 CV 证据，不是 untouched 测试集泛化承诺。

## 输入契约

当前包只接受已生成、可审计的候选特征 JSON，不读取原始 `collect_data*.txt` 会话。每个
候选必须携带有限值数组：macro 63 维、micro 47 维、verifier 116 维；同时必须有稳定
`subject_id` 和全局唯一 `sid`。`sid` 只用于同会话 NMS 与事件几何，准入阈值、candidate
cap 和 event budget 按 `subject_id` 执行。

```json
{
  "feature_schema": {"schema_version": 2, "widths": {"macro": 63, "micro": 47, "verifier": 116}, "context": {"version": "v1", "columns": ["60 fixed columns"], "schema_hash": "1ee5f35cb5e97623039d95aba84ad300be0e226849d53cebb288a886a9d62f53"}},
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

`feature_schema` 必须与 `event_stack/bundle/feature_schema.json` **逐 JSON 值相等**，
`schema_hash` 是该 schema 以 UTF-8、键排序、紧凑 JSON（`,` 与 `:` 无空格）序列化后的
SHA-256。前端不应手写 60 个 Context-v1 列；从 bundle 中读取 schema 原样随 payload 发送。

## 前端集成 CLI

把整个 `event_stack/` 目录（而非单个脚本）部署到前端后端可访问的位置：其中 `bundle/` 是模型与
策略，`requirements.txt` 是严格运行时依赖，`runtime_manifest.json` 是文件完整性清单。默认命令
已指向同目录的 `bundle/`，不依赖仓库、训练数据或任何相对项目路径：

```bash
python event_stack/predict_event_stack.py \
  --input-features request.json \
  --output response.json \
  --device auto
```

也可显式传入 `--bundle event_stack/bundle`。成功时进程退出码为 `0`、标准输出为
`resolved_device=cpu`，并原子/规范化地写出 `response.json`；拒绝输入、损坏 bundle、版本不匹配
或 GPU 强制请求时退出码为 `2`，错误原因写入 stderr 前缀 `event-stack inference refused:`，不产生
可信结果。`--device auto` 与 `--device cpu` 都使用 CPU；`--device gpu` 是 `cuda` 别名，而当前包
没有经过审计的 CUDA 组件，因此 `gpu/cuda` 必定以退出码 2 拒绝，绝不把 CPU 冒充成 GPU。
某些 Windows 环境的 joblib 会把物理核心探测失败作为 warning 写到 stderr；这不影响退出码或
结果。若前端要求成功请求的 stderr 为空，请在启动进程前设置 `LOKY_MAX_CPU_COUNT` 为该部署主机
允许使用的正整数线程数，例如 PowerShell 的 `$env:LOKY_MAX_CPU_COUNT='16'`。

最小成功请求的结构如下；数组中的数字必须全部有限，长度必须严格为 63、47、116：

```json
{
  "feature_schema": "copy event_stack/bundle/feature_schema.json verbatim",
  "schema_hash": "SHA-256 of the canonical copied schema",
  "sessions": [{
    "subject_id": "user-42",
    "sid": "session-20260915-a",
    "candidates": [{
      "start_ms": 0,
      "end_ms": 1000,
      "macro": ["63 finite numbers"],
      "micro": ["47 finite numbers"],
      "verifier": ["116 finite numbers"]
    }]
  }]
}
```

响应只有 `events` 与 `resolved_device` 两个字段。每个 event 含 `sid`、`start_ms`、`end_ms`、`score`；
events 按 `(sid,start_ms,end_ms,score)` 稳定排序。空数组是合法的“未检出事件”响应。`subject_id`
只影响冻结的准入/预算，不会出现在输出中。

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
  → LogisticRegression + 受限 LightGBM（56 基础 + 60 Context-v1 = 116 维事件复核）概率 blend
  → 冻结 event policy → canonical Episode JSON
```

当前五折 raw union 为 3,413，micro candidates 为 2,773，admission 后候选为 230。模型和
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
