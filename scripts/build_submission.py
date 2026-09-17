"""Build the competition submission bundle from canonical source.

``dist/submission`` is a generated package: ``main.py`` is a thin competition
input/output adapter around the canonical ``Predictor``, and the ``event_stack``
package is the same mechanically vendored canonical closure used by
``dist/inference`` (plus the isolated official-contract adapter boundary).

The official competition input/output contract is not defined by the audited
release materials, so the official mode refuses with an explicit error until a
concrete adapter is registered in
``src/pipeline/inference/competition_adapter.py``.
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

_MAIN_ENTRYPOINT = '''"""Competition submission entry point for the frozen event-stack release.

Raw mode is fully supported and reproduces the canonical Predictor document.
Official competition mode is isolated behind a registered adapter; until the
official input/output contract is registered, this entry point refuses
(exit code 2) rather than guessing the wire format.
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
    mode.add_argument("--raw", type=Path, help="collect_data*.txt file or folder (canonical raw mode)")
    mode.add_argument("--official-input", type=Path, help="official competition input (requires a registered adapter)")
    parser.add_argument("--output", type=Path, required=True, help="destination JSON path")
    parser.add_argument("--adapter", default="official", help="registered adapter name for official mode")
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
        predictor = Predictor.from_bundle(_ROOT / "models", device=args.device,
                                          run_key=str(manifest["release_run_key"]))
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

_SUBMISSION_README = """# 竞赛提交包

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
"""


def build_submission(*, repository_root: Path, destination: Path,
                     bundle_path: Path | None = None) -> Path:
    """Atomically construct the verified competition submission package."""
    root = Path(repository_root).resolve()
    if bundle_path is None:
        release = load_current_promoted_release(root)
        bundle_path = root / "models" / "event_stack" / str(release["run_key"]) / "deployment"
    return build_distribution(
        repository_root=root, bundle_path=bundle_path, destination=destination,
        entrypoint="main.py", entrypoint_text=_MAIN_ENTRYPOINT, readme_text=_SUBMISSION_README,
        closure_roots=_SUBMISSION_CLOSURE_ROOTS,
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
