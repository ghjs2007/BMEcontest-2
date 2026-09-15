# BMEcontest-2 Competition Delivery Refactor Specification

## 0. 任务背景

当前项目已经达到预定开发目标：严格 subject-independent nested-CV 开发结果已经达到约 F1 >= 0.65。

现在停止以“继续堆实验脚本”为主要目标，进入：

1. 当前最佳算法冻结与可复现；
2. 仓库历史实验垃圾清理；
3. 正式 raw-data inference pipeline 建设；
4. `dist/` 重构为竞赛交付工作区；
5. 为小组成员的可视化开发提供稳定接口；
6. 为最终竞赛源代码/模型/可执行程序提交准备独立 submission bundle。

本次工作首先是工程重构，不允许在重构过程中修改算法行为、重新调参或偷偷改变模型输出。

---

# 1. 第一原则：先审计，后修改，最后删除

禁止一开始批量删除 `scripts/`、`cache/`、`outputs/` 或旧模型。

第一阶段只读检查当前仓库，建立实际 dependency graph。

必须首先确认：

- 当前 promoted run key；
- 当前五折 aggregate metrics；
- 当前 canonical model bundle；
- 当前 feature widths/schema；
- macro feature producer；
- micro ACC+GYRO feature producer；
- Context-v1 feature producer；
- candidate generation；
- candidate admission；
- verifier；
- final decoder；
- model release/build；
- attestation；
- dist inference；
- 当前 tests；
- README 中正式 reproduction commands。

以仓库当前代码和 canonical evidence 为准，不要因为本 specification 中出现了旧文件名就假定它仍然存在。

输出一个临时审计报告：

`docs/repository_cleanup_audit.md`

至少包含：

| Path | Current role | Used by promoted pipeline? | Needed for training reproduction? | Needed for inference? | Action |
|---|---|---:|---:|---:|---|
| ... | ... | yes/no | yes/no | yes/no | KEEP / MOVE / REFACTOR / DELETE |

任何准备 DELETE 的文件都必须先证明：

- 不被 current promoted pipeline import；
- 不被 release/build import；
- 不被 tests import；
- 不属于 current reproduction path；
- 不属于竞赛需要提交的核心训练/算法源码；
- 删除后完整测试仍能通过。

Git history 作为历史实验归档，不创建 `old/`、`deprecated/`、`backup/` 等垃圾目录。

---

# 2. 冻结当前算法

本次重构必须首先建立一个 immutable baseline。

记录：

- run key；
- aggregate F1；
- TP；
- FP；
- predictions；
- recall；
- PPV；
- candidate recall；
- short-meal recall；
- five outer-fold records；
- feature schema version；
- model hashes；
- source fingerprints。

将其定义为：

`CURRENT_PROMOTED_RELEASE`

重构期间禁止：

- 修改模型超参数；
- 修改 threshold；
- 修改 admission policy；
- 修改 candidate policy；
- 修改 feature mathematical definition；
- 修改 imputation semantics；
- 修改 event matching/evaluation；
- 重新选择 outer-CV 最佳配置。

如果为了抽取公共函数而移动代码，新旧输出必须通过 parity tests。

本任务的成功标准不是 F1 提升，而是：

> 重构后的 canonical pipeline 与重构前 promoted pipeline 输出一致。

---

# 3. 最终仓库职责

目标结构原则：

```text
BMEcontest-2/
├── src/
│   └── pipeline/
│       ├── io/
│       ├── preprocessing/
│       ├── features/
│       ├── candidates/
│       ├── inference/
│       ├── evaluation/
│       └── training/
│
├── scripts/
│   ├── train_event_stack.py
│   ├── evaluate_event_stack.py
│   ├── release_event_stack.py
│   ├── build_submission.py
│   └── reproduce_release.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── parity/
│   └── release/
│
├── models/
│   └── event_stack/
│       └── <promoted-run-key>/
│
├── outputs/
│   └── release/
│       └── <promoted-run-key>/
│
├── docs/
│
├── dist/
│   ├── inference/
│   ├── visual/
│   ├── submission/
│   ├── examples/
│   ├── schema/
│   └── README.md
│
└── README.md
```

这只是目标职责结构，不要求为了匹配目录而机械移动所有文件。

如果当前代码结构已经有更合理的模块，优先保留兼容性。

---

# 4. `src/` 是算法唯一真源

正式算法只能存在一份 canonical implementation。

禁止长期存在：

```text
src/.../macro_features.py
dist/inference/.../macro_features.py
dist/submission/.../macro_features.py
```

三份人工维护的实现。

必须形成：

```text
                    src/pipeline
                         |
                  canonical code
                         |
                 release/build
                  /            \
                 v              v
        dist/inference    dist/submission
```

如果 `dist` 必须 self-contained，可以由 release script COPY/vendor canonical source，但不得人工维护另一套逻辑。

所有 generated files 必须可由 build script 重建。

---

# 5. 将 scripts 中的算法实现抽回 src

逐个检查当前脚本。

如果某个 script 同时承担：

- CLI；
- feature extraction；
- preprocessing；
- candidate generation；
- model inference；

则进行 dependency inversion。

目标：

```python
def main():
    args = parse_args()
    pipeline = ...
    pipeline.run(...)
```

script 只做参数解析和 orchestration。

例如逻辑上应形成：

```text
raw input
   |
SessionReader
   |
Timeline / preprocessing
   |
MacroFeatureExtractor
   |
MicroFeatureExtractor
   |
window models
   |
CandidateGenerator
   |
ContextFeatureExtractor
   |
Verifier
   |
EventDecoder
```

具体类名可按现有代码风格调整，不要为了面向对象而强制重写纯函数。

---

# 6. 建立统一 Predictor API

在 canonical source 中建立稳定推理入口。

建议接口：

```python
from pipeline.inference import Predictor

predictor = Predictor.from_bundle(path)

result = predictor.predict_file(path)
result = predictor.predict_folder(path)
```

如果当前数据格式要求一个 session 由多个文件组成，则 `predict_folder()` 负责发现和组合。

不要要求最终调用者：

- 手动 build cache；
- 手动生成 macro features；
- 手动生成 micro features；
- 手动构造 63/47/116 dimensional JSON；
- 指定训练 fold；
- 知道内部 candidate policy。

输入应该尽可能接近比赛提供的原始数据格式。

---

# 7. Raw-session inference 是本次核心新增工程能力

当前训练 pipeline 中已有的 raw preprocessing / feature extraction 必须成为 inference 的 canonical implementation。

绝对禁止照着旧脚本重新手写一个“差不多”的 inference feature extractor。

要求：

```text
TRAINING RAW
      |
      +------ same function ------+
                                  |
INFERENCE RAW --------------------+
```

目标：

`training-serving skew = 0`

必须保留真实 timestamp semantics：

- session boundaries；
- data gaps；
- valid coverage；
- no cross-gap windows；
- no cross-session aggregation；
- gravity alignment；
- macro window definition；
- micro window definition；
- Context-v1 definition；
- NaN handling；
- Inf rejection。

---

# 8. Predictor 输出统一 Prediction Schema

建立：

`dist/schema/prediction.schema.json`

至少支持以下逻辑结构：

```json
{
  "schema_version": "1.0",
  "model": {
    "name": "event-stack",
    "run_key": "..."
  },
  "input": {
    "source": "...",
    "duration_seconds": 0
  },
  "events": [
    {
      "id": 0,
      "session_id": "...",
      "start_ms": 0,
      "end_ms": 0,
      "duration_s": 0,
      "confidence": 0
    }
  ],
  "diagnostics": {
    "coverage": 0,
    "warnings": []
  }
}
```

同时支持可选：

```json
"timeline": {},
"candidates": [],
"gaps": []
```

不要要求普通竞赛 inference 默认输出所有 debug 数据。

---

# 9. 为 visual/ 设计稳定接口

`dist/visual/` 是小组成员负责的可视化开发区。

算法代码不得直接写在这里。

visual 只依赖：

1. Prediction Schema；
2. inference API 或 inference-generated JSON。

提供：

`dist/examples/example_prediction.json`

必须是一个符合 schema 的、可公开提交的示例，不包含敏感或不应进入 Git 的原始数据。

建议 timeline schema 支持：

- timestamp；
- macro probability；
- micro probability；
- valid/gap indicator；
- candidate interval；
- candidate score；
- admitted/rejected；
- final event；
- final confidence。

这样 visual 可以展示：

```text
raw/activity
macro probability
micro probability
candidate regions
verifier decisions
final eating episodes
```

visual 不得重新实现：

- threshold；
- candidate admission；
- event fusion；
- final decoder。

前端只展示 canonical inference 的结果。

---

# 10. dist/inference/

目标：

```text
dist/inference/
├── predict.py
├── event_stack/
├── models/
├── manifest.json
├── feature_schema.json
├── requirements.txt
└── README.md
```

具体目录可根据 Python packaging 调整。

必须能够在没有以下目录的情况下工作：

```text
cache/
outputs/
training splits/
training data/
research scripts/
```

至少支持：

```bash
python predict.py path/to/file
python predict.py path/to/folder
```

建议支持：

```bash
python predict.py INPUT --output result.json
python predict.py INPUT --include-timeline
python predict.py INPUT --include-candidates
```

如果批处理很重要，再加入：

```bash
python predict.py DATASET_DIR --recursive --output-dir predictions/
```

但不要为了 CLI feature creep 延误核心 inference。

---

# 11. dist/submission/

这是比赛最终提交包，不等于 inference 开发包。

必须由：

```bash
python scripts/build_submission.py
```

自动生成。

不要人工复制维护。

逻辑结构：

```text
dist/submission/
├── main.py
├── src/ or package/
├── models/
├── requirements.txt
├── manifest.json
└── README.md
```

`main.py` 只做比赛 input/output adapter：

```text
official competition input
          |
          v
      Predictor
          |
          v
official competition output
```

如果当前比赛的最终官方输入输出规范尚未完全确定，则：

- 将 competition adapter 单独隔离；
- 不猜测官方格式；
- Predictor 保持稳定；
- README 明确 TODO/known constraint。

---

# 12. inference / submission / canonical parity

这是本次最重要的测试。

选择若干现有合法 session，建立：

```text
A = current promoted canonical pipeline
B = new src Predictor
C = dist/inference
D = dist/submission
```

逐层比较。

要求：

### preprocessing

- session segmentation identical；
- timestamps identical；
- gap decisions identical。

### features

- macro feature schema identical；
- micro feature schema identical；
- Context-v1 schema identical；
- NaN masks identical；
- finite values within numerical tolerance。

### model

- macro probabilities allclose；
- micro probabilities allclose；
- verifier scores allclose。

### decoder

必须优先要求：

```text
candidate intervals identical
admission decisions identical
final event intervals identical
```

如果浮点差异位于 decision threshold 附近，记录明确 tolerance/tie policy，不能偷偷忽略。

最终：

```python
assert canonical_events == predictor_events
assert canonical_events == dist_inference_events
assert canonical_events == submission_events
```

---

# 13. dist 必须做 clean-room smoke test

创建临时目录，只复制：

```text
dist/inference/
```

不得访问仓库父目录。

安装 requirements 后：

```bash
python predict.py <fixture>
```

必须成功。

再对：

```text
dist/submission/
```

执行相同 clean-room test。

测试应主动检测 accidental imports，例如：

```text
../../src
../../cache
../../models
../../scripts
```

均不得成为 dist runtime dependency。

---

# 14. scripts 清理规则

完成 Predictor + parity 后才开始删除。

分类为：

## KEEP

仍属于：

- current training；
- current strict CV；
- current evaluation；
- current release；
- current bundle verification；
- current attestation；
- current submission build；
- release reproduction。

## REFACTOR

仍有价值，但算法实现埋在 script 中。

将算法移入 `src/`，script 变薄。

## DELETE

满足：

- historical experiment only；
- current pipeline 不使用；
- reproduction 不使用；
- tests 不使用；
- release 不使用；
- submission 不使用。

优先检查并清理：

- old proposal pipeline；
- old ranker pipeline；
- obsolete rank_events variants；
- superseded density experiments；
- leak-era scripts；
- one-off fix scripts；
- one-off diagnostic launchers；
- superseded raw-summary experiments；
- superseded model experiments；
- temporary migration scripts；
- duplicate release scripts；
- `*_old.py`；
- `*_backup.py`；
- `*_final2.py` 类文件；
- Python bytecode；
- pytest caches。

但不要仅凭文件名删除。

必须做 import/search/reproduction audit。

---

# 15. outputs 清理

版本控制中的实验 outputs 收缩到 current promoted evidence。

建议：

```text
outputs/
└── release/
    └── <promoted-run-key>/
        ├── summary.json
        ├── fold_0.json
        ├── fold_1.json
        ├── fold_2.json
        ├── fold_3.json
        ├── fold_4.json
        └── promotion_attestation.json
```

如果当前 evidence chain 还需要其他文件，保留。

历史结果只需：

- README history table；
- Git history。

不要为了目录漂亮破坏已有 attestation hash chain。

如果移动 canonical evidence 会破坏 fingerprint，优先保留原位置并记录原因，而不是强制迁移。

---

# 16. models 清理

Git 中仅保留：

- current promoted model；
- current release 必需 bundle；
- submission 必需 model artifacts。

研究 checkpoint：

```text
SSL checkpoints
temporary fold models
failed experiment models
```

应该进入 ignored cache/checkpoint directory，除非当前 promoted release 依赖。

不要删除 current 5+1 fold bundle 或 attestation 所引用 artifact。

---

# 17. cache 清理

以下不是第一轮 cleanup target：

- canonical session cache；
- current macro feature cache；
- current micro feature cache；
- current split definitions；
- current promoted reproduction 所需 cache。

只有当 raw Predictor parity 完成并确认某 cache 完全可重建、且不是 reproduction contract 的一部分时，才考虑从 Git/工作区清除。

`.gitignore` 应覆盖：

```text
__pycache__/
.pytest_cache/
*.pyc
temporary experiment outputs
SSL checkpoints
rebuildable embeddings
temporary release directories
```

不要忽略 canonical release evidence。

---

# 18. 保留竞赛可解释源码

不要把“submission 最小化”误解为“把训练源码全部删掉”。

仓库需要让评委/老师能够回答：

> 模型是怎么训练出来的？

至少应保留清晰入口：

```text
scripts/train_event_stack.py
scripts/evaluate_event_stack.py
scripts/reproduce_release.py
```

以及：

```text
src/pipeline/training/
src/pipeline/features/
src/pipeline/candidates/
src/pipeline/evaluation/
```

最终算法代码应该比历史实验时期更容易阅读。

---

# 19. README 重构

根 README 重点说明：

1. competition task；
2. current promoted model；
3. current development F1；
4. leakage-safe evaluation；
5. architecture；
6. quick inference；
7. reproduce training/evaluation；
8. repository structure；
9. competition submission；
10. limitations。

历史实验压缩成简洁表格，不再让 README 成为实验日志全文。

`dist/README.md`：

```text
# Distribution Workspace

inference/
    Canonical standalone model inference.

visual/
    Team visualization application.
    Depends only on prediction schema/API.

submission/
    Generated competition submission bundle.
    Do not edit generated algorithm files manually.

examples/
    Safe example inputs/outputs.

schema/
    Stable interface between inference and visualization.
```

`dist/inference/README.md` 必须给出从安装到预测的完整命令。

`dist/submission/README.md` 必须描述如何从零验证提交包。

---

# 20. Manifest

为 inference 和 submission 生成 machine-readable manifest：

```json
{
  "release_run_key": "...",
  "model_version": "...",
  "prediction_schema_version": "1.0",
  "feature_schema_version": "...",
  "python_version": "...",
  "dependencies": {},
  "model_files": [],
  "source_files": [],
  "hashes": {}
}
```

build 后验证所有 hashes。

如果 bundle 当前已经存在 manifest/attestation 机制，复用现有机制，不另造不兼容体系。

---

# 21. 不要在本任务中做的事情

禁止顺便：

- 调 admission；
- 调 verifier；
- 加新 feature；
- 加 TCN/Transformer；
- 重做 Context-v2；
- 修改 evaluation metric；
- 修改 CV folds；
- 修改 GT eligibility；
- 优化 F1；
- 使用 outer folds 做新模型选择；
- 重写整个 pipeline；
- 引入大型 Web framework；
- 实现完整前端。

本任务只做：

> architecture cleanup + canonical raw inference + distribution/submission infrastructure.

---

# 22. 推荐执行顺序

严格按以下顺序：

### Phase 1 — Audit

- inspect repository；
- locate promoted run；
- dependency graph；
- cleanup audit；
- run current full tests；
- save baseline hashes/results。

### Phase 2 — Canonicalize

- move reusable preprocessing/features from scripts into src；
- scripts become thin wrappers；
- legacy behavior unchanged；
- tests pass after every extraction。

### Phase 3 — Predictor

- implement raw file/folder loading；
- implement canonical end-to-end inference；
- prediction schema；
- CLI。

### Phase 4 — Parity

- current pipeline vs Predictor；
- feature parity；
- probability parity；
- event parity。

Do not continue until this passes.

### Phase 5 — dist/inference

- build standalone inference；
- clean-room test；
- manifest；
- documentation。

### Phase 6 — dist/visual contract

- schema；
- example_prediction.json；
- README for frontend developers；
- no need to implement full visual UI.

### Phase 7 — dist/submission

- implement build_submission.py；
- competition adapter；
- standalone package；
- clean-room test；
- parity with Predictor.

### Phase 8 — Garbage collection

Only now:

- delete obsolete scripts；
- delete obsolete models；
- delete obsolete outputs；
- update gitignore；
- remove bytecode/temp files。

### Phase 9 — Final verification

Run complete test suite with cache/bytecode generation disabled where appropriate.

Verify:

```text
current promoted metrics unchanged
canonical event predictions unchanged
dist/inference works standalone
dist/submission works standalone
visual example validates against schema
manifest/attestation valid
README commands execute successfully
git status contains no accidental caches/temp files
```

---

# 23. Commit strategy

Do not perform the whole refactor as one giant commit.

Recommended commits:

```text
refactor: extract canonical preprocessing and feature modules

feat: add raw event-stack predictor API

test: add canonical inference parity coverage

build: create standalone inference distribution

docs: define visualization prediction schema

build: add competition submission bundle

chore: remove obsolete experiment scripts and artifacts

docs: update competition release and reproduction guide
```

After each structural commit run the relevant tests.

The cleanup commit must contain deletions only after the replacement implementation is already tested.

---

# 24. Failure policy

If moving a module causes parity failure:

STOP.

Do not compensate by changing thresholds or model parameters.

Find the training-serving difference.

If a historical script turns out to contain canonical behavior:

do not delete it until that behavior has been extracted and parity-tested.

If current dist depends on undocumented behavior:

document it first, add regression test, then refactor.

If official competition input/output requirements are unclear:

do not invent them. Keep `Predictor` complete and isolate the unresolved portion to the submission adapter.

---

# 25. Definition of Done

This refactor is complete only when all of the following are true.

### Algorithm

Current promoted development metrics remain unchanged.

### Source

There is one canonical implementation of preprocessing/features/candidates/inference.

### Raw inference

A user can provide a supported raw competition data file or folder and receive eating-event predictions without manually generating intermediate features.

### Inference distribution

In a clean directory containing only `dist/inference/`, documented installation and prediction commands work.

### Visualization contract

A teammate can build `dist/visual/` using only documented prediction schema and example output without importing training code.

### Submission

`build_submission.py` deterministically creates `dist/submission/`, which can run independently and produces the same final events as canonical Predictor.

### Parity

For regression fixtures:

```text
legacy promoted pipeline
=
canonical Predictor
=
dist/inference
=
dist/submission
```

for final decoded events.

### Repository hygiene

Historical experimental scripts/artifacts that no longer serve training reproduction, evaluation, inference, submission or documentation have been removed.

### Evidence

Current promoted run evidence, bundle validation and attestation remain valid.

### Documentation

A new team member can understand from README:

```text
how to predict
how to reproduce
how to build submission
where to develop visualization
what not to modify
```

without reading historical scripts.

---

# 26. 最重要的工程约束

Throughout this task, optimize for correctness and traceability rather than minimum file count.

The repository does not need to be tiny.

It needs to have obvious ownership:

```text
src/       algorithm truth
scripts/   training/build entrypoints
tests/     correctness
models/    promoted model artifacts
outputs/   promoted scientific evidence
dist/
  inference/   standalone prediction
  visual/      teammate visualization work
  submission/  competition delivery
```

Do not delete useful source code merely because it is not required at inference time.

Do delete obsolete experimental implementations once Git history is sufficient and the promoted pipeline no longer depends on them.

Most importantly:

**Never allow repository cleanup to alter the model that achieved the current promoted F1.**


Post-refactor research policy: Refactor completion unfreezes algorithm development. New methods are implemented as versioned components in src/, registered through experiment configurations, and evaluated using the locked leakage-safe framework. dist/inference and dist/submission always represent only CURRENT_PROMOTED_RELEASE and are rebuilt only after a candidate passes promotion. Experimental algorithms must never be developed directly inside dist/.