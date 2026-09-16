"""Standalone raw event-stack inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from event_stack.inference import Predictor

def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="collect_data*.txt file or folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-timeline", action="store_true")
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "cuda"), default="auto")
    return parser.parse_args(argv)

def main(argv=None):
    args = _args(argv)
    try:
        manifest = json.loads((_ROOT / "manifest.json").read_text(encoding="utf-8"))
        predictor = Predictor.from_bundle(_ROOT / "models", device=args.device,
                                          run_key=str(manifest["release_run_key"]))
        options = predictor.options(include_timeline=args.include_timeline, include_candidates=args.include_candidates, device=args.device)
        result = predictor.predict_folder(args.input, options=options) if args.input.is_dir() else predictor.predict_file(args.input, options=options)
        args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"event-stack inference refused: {exc}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
