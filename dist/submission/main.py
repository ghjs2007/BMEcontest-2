"""竞赛提交入口：冻结 event-stack 发布的完整推理接口。

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
            args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
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
