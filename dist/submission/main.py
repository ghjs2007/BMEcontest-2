"""Competition submission entry point for the frozen event-stack release.

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
