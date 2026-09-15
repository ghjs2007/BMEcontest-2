from pathlib import Path

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
