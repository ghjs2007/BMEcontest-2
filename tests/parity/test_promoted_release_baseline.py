from pathlib import Path
import shutil

import pytest

from src.pipeline.artifacts import load_current_promoted_release, verify_current_promoted_release


def test_current_promoted_release_is_attested():
    root = Path(__file__).parents[2]
    release = load_current_promoted_release(root)
    assert release["run_key"] == "160afaf81debf1ee"
    metrics = release["aggregate"]["outer_metrics"]
    assert metrics["n_tp"] == 114
    assert metrics["n_pred"] == 197
    assert metrics["n_true"] == 153
    assert metrics["f1"] == pytest.approx(0.6514285714285715)
    assert verify_current_promoted_release(root) == ()


@pytest.mark.parametrize(
    "relative",
    [
        "release/event_stack_incumbent.json",
        "models/event_stack/160afaf81debf1ee/promotion_attestation.json",
        "models/event_stack/160afaf81debf1ee/promotion_summary.json",
        "models/event_stack/160afaf81debf1ee/outer-fold-0/diagnostics.json",
        "models/event_stack/160afaf81debf1ee/deployment/manifest.json",
    ],
)
def test_release_lock_rejects_tampered_evidence(tmp_path: Path, relative: str):
    source = Path(__file__).parents[2]
    for directory in ("release", "models"):
        shutil.copytree(source / directory, tmp_path / directory)
    target = tmp_path / relative
    target.write_bytes(target.read_bytes() + b"\n tampered")
    assert verify_current_promoted_release(tmp_path)
