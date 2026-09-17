"""Local inference bridge entry: canonical inference API + static visual application.

Serves the visualization at http://127.0.0.1:PORT/ (default 4173) and the bridge API
under /api/*. Raw TXT selections from the browser are analyzed by the canonical
Predictor shipped in this package; numbers, events, and telemetry are never computed
in the browser.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from event_stack.inference import local_server


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--visual-dir", type=Path, default=_ROOT.parent / "visual")
    parser.add_argument("--open", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _args(argv)
    try:
        manifest = json.loads((_ROOT / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"local inference bridge refused: {exc}", file=sys.stderr)
        return 2
    forwarded = ["--bundle", str(_ROOT / "models"), "--run-key", str(manifest["release_run_key"]),
                 "--host", args.host, "--port", str(args.port)]
    if args.visual_dir:
        forwarded += ["--visual-dir", str(args.visual_dir)]
    if args.open:
        forwarded.append("--open")
    return local_server.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
