"""Release-only proof that the competition submission bundle is standalone and honest.

The submission must (a) reproduce the canonical Predictor document from raw
input in a clean room, and (b) refuse official-contract modes explicitly until a
registered adapter exists.  It may never guess the official wire format.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import src.config as config
from src.pipeline.inference import PredictionOptions, Predictor
from src.pipeline.inference.competition_adapter import UnsupportedCompetitionAdapter


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "models/event_stack/160afaf81debf1ee/deployment"


def _fixture_raw() -> Path:
    manifest = json.loads(
        (ROOT / "tests/fixtures/release_160afaf81debf1ee/fixture_manifest.json").read_text(encoding="utf-8")
    )
    source_session_id = str(manifest["macro_golden"]["session_id"])
    return next((config.SENSOR_DIR / source_session_id).glob("collect_data*.txt"))


def _run_clean_room(package: Path, tmp_path: Path, args: list[str]) -> subprocess.CompletedProcess:
    """Copy only the package into a directory whose parent holds sentinel repo names."""
    former_parent = tmp_path / "former-parent"
    former_parent.mkdir(exist_ok=True)
    for name in ("src", "cache", "models", "scripts"):
        (former_parent / name).write_text("sentinel", encoding="utf-8")
    clean = former_parent / "clean-room"
    shutil.copytree(package, clean)
    return subprocess.run(
        [sys.executable, "-I", "main.py", *args], cwd=clean, capture_output=True, text=True
    )


def test_submission_packages_the_full_delivery(tmp_path: Path):
    """The submission must ship visual, reproduction source and the evidence chain."""
    from scripts.build_submission import build_submission

    package = build_submission(repository_root=ROOT, destination=tmp_path / "built" / "submission")
    for component in ("main.py", "event_stack", "models", "src", "scripts", "tests",
                      "visual", "schema", "examples", "manifest.json", "requirements.txt", "README.md"):
        assert (package / component).exists(), f"submission is missing {component}"
    assert not (package / "docs").exists(), "documentation must not ship inside the submission"
    model_root = package / "models" / "event_stack" / "160afaf81debf1ee"
    assert (model_root / "deployment" / "manifest.json").is_file()
    for fold in range(5):
        assert (model_root / f"outer-fold-{fold}" / "manifest.json").is_file()
    assert (model_root / "promotion_summary.json").is_file()
    assert (model_root / "promotion_attestation.json").is_file()
    assert (package / "schema" / "prediction.schema.json").is_file()
    assert (package / "examples" / "example_prediction.json").is_file()
    assert (package / "src" / "pipeline" / "inference" / "predictor.py").is_file()
    assert (package / "scripts" / "reproduce_release.py").is_file()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert any(path.endswith("promotion_attestation.json") for path in manifest["model_files"])
    assert "visual/README.md" in manifest["source_files"]
    assert "src/pipeline/inference/predictor.py" in manifest["source_files"]


def test_distribution_text_bytes_are_lf_normalized(tmp_path: Path):
    """Manifest hashes must survive a fresh clone: shipped text files are LF-only.

    Git checks text files out with LF (``* text=auto eol=lf``); a package that
    embedded CRLF worktree bytes would fail its own manifest verification after
    checkout, so the builders normalize on copy.
    """
    from scripts.build_submission import build_submission

    text_suffixes = {".py", ".md", ".json", ".txt", ".html", ".css", ".js", ".ts", ".tsx", ".jsx", ".mjs"}
    packages = [ROOT / "dist" / "inference",
                build_submission(repository_root=ROOT, destination=tmp_path / "submission")]
    for package in packages:
        for path in package.rglob("*"):
            if path.is_file() and path.suffix.lower() in text_suffixes:
                try:
                    assert b"\r" not in path.read_bytes(), f"CR byte in shipped text file: {path}"
                except OSError:
                    continue


def test_submission_raw_mode_matches_predictor(tmp_path: Path):
    """The shipped submission must reproduce the canonical prediction document."""
    from scripts.build_submission import build_submission

    raw = _fixture_raw()
    package = build_submission(repository_root=ROOT, destination=tmp_path / "built" / "submission")
    output = tmp_path / "result.json"
    done = _run_clean_room(
        package, tmp_path,
        ["--raw", str(raw), "--output", str(output), "--include-timeline", "--include-candidates"],
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    expected = Predictor.from_bundle(BUNDLE).predict_file(
        raw,
        options=PredictionOptions(include_timeline=True, include_candidates=True),
    )
    assert result == expected


def test_submission_official_mode_refuses_without_registered_adapter(tmp_path: Path):
    """The unregistered official mode must fail loudly, never invent a format."""
    from scripts.build_submission import build_submission

    package = build_submission(repository_root=ROOT, destination=tmp_path / "submission")
    done = _run_clean_room(
        package, tmp_path,
        ["--official-input", str(tmp_path / "official-input"), "--output", str(tmp_path / "out.json")],
    )
    assert done.returncode == 2
    assert "not registered" in done.stderr


def test_official_adapter_refuses_unknown_contract(tmp_path: Path):
    with pytest.raises(NotImplementedError, match="not registered"):
        UnsupportedCompetitionAdapter().load(tmp_path / "input")
    with pytest.raises(NotImplementedError, match="not registered"):
        UnsupportedCompetitionAdapter().dump({"events": []}, tmp_path / "output")


def test_submission_manifest_rejects_runtime_file_drift(tmp_path: Path):
    """A copied source file may not evade the submission hash manifest."""
    from scripts.build_submission import build_submission
    from scripts.build_inference_distribution import verify_distribution_manifest

    package = build_submission(repository_root=ROOT, destination=tmp_path / "submission")
    (package / "event_stack" / "unexpected.py").write_text("x = 1\n", encoding="utf-8")
    try:
        verify_distribution_manifest(package, entrypoint="main.py")
    except ValueError as exc:
        assert "checksums" in str(exc)
    else:
        raise AssertionError("manifest must reject unrecorded runtime source")
