"""Build the competition submission bundle from canonical source.

``dist/submission`` 是竞赛最终交付物，由本脚本从仓库唯一真源确定性生成：

- 推理接口：``main.py``（raw 模式 + 隔离的官方 adapter 边界）与机械 vendored 的
  canonical 运行时 ``event_stack/``；
- 完整模型资产：整个冻结模型根 ``models/event_stack/<run_key>/``（deployment
  bundle + 五个 outer-fold evidence bundle + promotion summary 与 attestation）；
- 复现代码：``src/``（canonical 算法源码）、``scripts/``（训练/评估/发布/构建全链）
  与 ``tests/``；
- 可视化：``visual/``（dist/visual 工作区，队友产出原样打包）以及 ``schema/``、
  ``examples/``（预测契约与安全示例）。

文档不进入提交包（文档随仓库维护）。

The official competition input/output contract is not defined by the audited
release materials, so the official mode refuses explicitly until a concrete
adapter is registered in ``src/pipeline/inference/competition_adapter.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_inference_distribution import build_distribution  # noqa: E402
from src.pipeline.artifacts import load_current_promoted_release  # noqa: E402


_SUBMISSION_CLOSURE_ROOTS = (
    "src.pipeline.inference.predictor",
    "src.pipeline.inference.competition_adapter",
)

_SUBMISSION_REQUIRED_ROOTS = (
    "event_stack", "models", "src", "scripts", "tests", "visual", "schema", "examples",
)

_MAIN_ENTRYPOINT = '''"""竞赛提交入口：冻结 event-stack 发布的完整推理接口。

raw 模式完整可用，输出与 canonical Predictor 文档逐值一致。官方竞赛模式隔离在
注册 adapter 之后；在官方输入/输出契约注册之前，本入口会显式拒绝（退出码 2），
绝不猜测线格式。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from event_stack.inference import Predictor
from event_stack.inference.competition_adapter import registered_adapter


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--raw", type=Path, help="collect_data*.txt 文件或目录（canonical raw 模式）")
    mode.add_argument("--official-input", type=Path, help="官方竞赛输入（需要已注册 adapter）")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSON 路径")
    parser.add_argument("--adapter", default="official", help="官方模式的 adapter 注册名")
    parser.add_argument("--include-timeline", action="store_true")
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def _predict(predictor, path, *, subject_id=None, options):
    path = Path(path)
    if path.is_dir():
        return predictor.predict_folder(path, subject_id=subject_id, options=options)
    return predictor.predict_file(path, subject_id=subject_id, options=options)


def main(argv=None):
    args = _args(argv)
    try:
        manifest = json.loads((_ROOT / "manifest.json").read_text(encoding="utf-8"))
        run_key = str(manifest["release_run_key"])
        predictor = Predictor.from_bundle(
            _ROOT / "models" / "event_stack" / run_key / "deployment",
            device=args.device, run_key=run_key,
        )
        options = predictor.options(include_timeline=args.include_timeline,
                                    include_candidates=args.include_candidates,
                                    device=args.device)
        if args.raw is not None:
            result = _predict(predictor, args.raw, options=options)
            args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
            return 0
        adapter = registered_adapter(args.adapter)
        raw_input, subject_id = adapter.load(args.official_input)
        result = _predict(predictor, raw_input, subject_id=subject_id, options=options)
        adapter.dump(result, args.output)
        return 0
    except NotImplementedError as exc:
        print(f"competition submission refused: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"competition submission refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''

_SUBMISSION_README = """# 竞赛提交包（event-stack）

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
"""


def _extra_trees(root: Path) -> tuple[tuple[Path, str], ...]:
    trees: list[tuple[Path, str]] = [
        (root / "src", "src"),
        (root / "scripts", "scripts"),
        (root / "tests", "tests"),
        (root / "dist" / "visual", "visual"),
        (root / "dist" / "schema", "schema"),
        (root / "dist" / "examples", "examples"),
    ]
    missing = [str(source) for source, _ in trees if not (root / source).exists()]
    if missing:
        raise ValueError("submission source tree is missing: " + ", ".join(missing))
    return tuple(trees)


def build_submission(*, repository_root: Path, destination: Path,
                     bundle_path: Path | None = None) -> Path:
    """Atomically construct the verified competition submission package."""
    root = Path(repository_root).resolve()
    release = load_current_promoted_release(root)
    run_key = str(release["run_key"])
    if bundle_path is None:
        bundle_path = root / "models" / "event_stack" / run_key / "deployment"
    return build_distribution(
        repository_root=root, bundle_path=Path(bundle_path), destination=destination,
        entrypoint="main.py", entrypoint_text=_MAIN_ENTRYPOINT, readme_text=_SUBMISSION_README,
        closure_roots=_SUBMISSION_CLOSURE_ROOTS,
        models_source=root / "models" / "event_stack" / run_key,
        models_destination=f"models/event_stack/{run_key}",
        extra_trees=_extra_trees(root),
        required_roots=_SUBMISSION_REQUIRED_ROOTS,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--destination", type=Path, default=ROOT / "dist/submission")
    args = parser.parse_args(argv)
    try:
        built = build_submission(repository_root=ROOT, destination=args.destination,
                                 bundle_path=args.bundle)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"submission build refused: {exc}", file=sys.stderr)
        return 2
    print(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
