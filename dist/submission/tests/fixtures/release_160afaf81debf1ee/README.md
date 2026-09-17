# Release 160afaf81debf1ee parity fixtures

本目录只存放安全的派生 fixture 元数据，不包含原始传感器数据、标签、受试者标识或
竞赛输入。`fixture_manifest.json` 中的五个哈希是显式的**聚合证据别名**，指向已有的
promoted evidence（diagnostics、schema、summary、attestation）——它们不是直接的
raw-session span / 特征 / 候选 / 准入 / 事件 fixture，不得如此解读。这些别名已由
Task 3 用直接的真实会话分层哈希替代/补充（见下节）。

未来任何 fixture 都必须来自可合法分发的来源，并在 `fixture_manifest.json` 中记录其
schema 与 SHA-256。不要将原始数据复制进本仓库。

## Task 3 macro parity 记录

manifest 现存一组合法的 raw/session-cache/slide-cache 配对。其原始解析数组与配对的
session cache 一致；legacy `scripts/slide_features.py:_process_session` 的输出在
canonical 窗口行序列化后与所选 slide 行**逐字节一致**。证据为 278 行 × **62** 列
`float32`。

已批准的 ABI 明确拆成两层：canonical raw macro 生产器返回历史 62 维矩阵，
`add_time_prior` 追加独立的冻结单列模型适配器。manifest 记录两个矩阵、两份 schema
以及适配器源码的哈希。三路 parity 必须保持精确：legacy 62 维 = canonical 62 维；
适配器 = legacy runner 边界；适配后的 63 维 = 冻结的模型输入矩阵。

生成命令（仅元数据）：

```text
python scripts/bootstrap_event_stack_diagnostics.py --run-key 160afaf81debf1ee
```
