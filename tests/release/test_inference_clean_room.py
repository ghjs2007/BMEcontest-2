"""Release-only proof that the raw inference distribution has no repo imports."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

from src.pipeline.inference.schema import validate_prediction


ROOT = Path(__file__).resolve().parents[2]


def _raw_file(path: Path) -> Path:
    header = "ACC_TIME\tPPG_TIME\tGYRO_TIME\t" + "\t".join(f"v{i}" for i in range(50)) + "\n"
    values = ["1"] * 44 + ["1", "2", "3", "4", "5", "6"]
    path.write_text(header + "100\t100\t100\t" + "\t".join(values) + "\n" + "150\t150\t150\t" + "\t".join(values) + "\n", encoding="utf-8")
    return path


def test_inference_runs_when_only_package_is_copied(tmp_path: Path):
    """Replacing vendored runtime with repository imports must fail this release gate."""
    from scripts.build_inference_distribution import build_inference_distribution

    raw = _raw_file(tmp_path / "collect_data1_2_3.txt")
    package = build_inference_distribution(
        repository_root=ROOT,
        bundle_path=ROOT / "models/event_stack/160afaf81debf1ee/deployment",
        destination=tmp_path / "built" / "inference",
    )
    assert not list(package.rglob("__pycache__"))
    former_parent = tmp_path / "former-parent"
    former_parent.mkdir()
    for name in ("src", "cache", "models", "scripts"):
        (former_parent / name).write_text("sentinel", encoding="utf-8")
    clean = former_parent / "clean"
    shutil.copytree(package, clean)
    done = subprocess.run(
        [sys.executable, "-I", "predict.py", str(raw), "--output", "result.json"],
        cwd=clean,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    result = json.loads((clean / "result.json").read_text(encoding="utf-8"))
    validate_prediction(result)
    assert str(ROOT) not in done.stdout
    assert str(ROOT.parent) not in done.stdout


def test_inference_manifest_rejects_runtime_file_drift(tmp_path: Path):
    """A copied source/model file may not evade the release hash manifest."""
    from scripts.build_inference_distribution import (
        build_inference_distribution,
        verify_distribution_manifest,
    )

    package = build_inference_distribution(
        repository_root=ROOT,
        bundle_path=ROOT / "models/event_stack/160afaf81debf1ee/deployment",
        destination=tmp_path / "inference",
    )
    (package / "event_stack" / "unexpected.py").write_text("x = 1\n", encoding="utf-8")
    try:
        verify_distribution_manifest(package)
    except ValueError as exc:
        assert "checksums" in str(exc)
    else:
        raise AssertionError("manifest must reject unrecorded runtime source")
