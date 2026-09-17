"""Boundary parity for the frozen raw inference graph.

The regression caught here is any training-serving skew introduced while
extracting a raw-data Predictor.  Expected fingerprints are stored separately
from the code under test in the release fixture manifest.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.pipeline.event_stack import EventRef, apply_event_policy
from src.pipeline.inference import Predictor
from src.pipeline.inference.legacy_payload import canonical_trace, legacy_trace


ROOT = Path(__file__).resolve().parents[2]


def _matrix_hash(values: np.ndarray) -> str:
    matrix = np.ascontiguousarray(values)
    return hashlib.sha256(matrix.dtype.str.encode() + str(matrix.shape).encode() + matrix.tobytes()).hexdigest()


def _serialized_hash(rows: tuple[dict[str, object], ...]) -> str:
    """Hash the public, sorted decoder records rather than their Python identities."""
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_hashes(trace, hashes: dict[str, str]) -> None:
    for name in (
        "macro_features_62", "macro_features_63", "micro_features",
        "context_features", "macro_probabilities", "micro_probabilities",
        "verifier_scores",
    ):
        assert _matrix_hash(getattr(trace, name)) == hashes[name]
    for name in ("candidates", "admitted", "events"):
        assert _serialized_hash(getattr(trace, name)) == hashes[name]


def test_legacy_canonical_predictor_trace_is_identical(inference_traces):
    """A raw producer/candidate/policy change must surface at its first layer."""
    old, direct, predictor = inference_traces
    assert old.spans == direct.spans == predictor.spans
    for old_values, direct_values, predictor_values in (
        (old.macro_features_62, direct.macro_features_62, predictor.macro_features_62),
        (old.macro_features_63, direct.macro_features_63, predictor.macro_features_63),
        (old.micro_features, direct.micro_features, predictor.micro_features),
        (old.context_features, direct.context_features, predictor.context_features),
        (old.macro_probabilities, direct.macro_probabilities, predictor.macro_probabilities),
        (old.micro_probabilities, direct.micro_probabilities, predictor.micro_probabilities),
        (old.verifier_scores, direct.verifier_scores, predictor.verifier_scores),
    ):
        np.testing.assert_allclose(old_values, direct_values, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(old_values, predictor_values, rtol=1e-12, atol=1e-12, equal_nan=True)
        assert np.array_equal(np.isnan(old_values), np.isnan(direct_values))
        assert np.array_equal(np.isnan(old_values), np.isnan(predictor_values))
    assert old.candidates == direct.candidates == predictor.candidates
    assert old.admitted == direct.admitted == predictor.admitted
    assert old.events == direct.events == predictor.events


def test_all_trace_boundaries_match_release_fixture_hashes(inference_fixture, inference_traces):
    """Every boundary is anchored to immutable, release-owned layer hashes."""
    for trace in inference_traces:
        _assert_hashes(trace, inference_fixture.manifest["inference_trace_hashes"])


def test_trace_declares_and_decoder_honours_exact_threshold_ties(inference_traces):
    """Scores at a frozen threshold are accepted; near-boundary values stay auditable."""
    for trace in inference_traces:
        assert trace.threshold_tie_rule == "score >= threshold"
        assert trace.decision_boundary_epsilon == 1e-12
        assert trace.admission_threshold >= 0.0
        assert trace.event_threshold >= 0.0
    event = EventRef("fixture", 0, 1)
    threshold = 0.5
    assert apply_event_policy([event], np.array([threshold]), ["subject"], threshold) == [event]
    assert apply_event_policy([event], np.array([np.nextafter(threshold, 0.0)]), ["subject"], threshold) == []


def test_legacy_and_direct_traces_do_not_delegate_to_predictor_orchestration(
    monkeypatch, inference_fixture, deployment_bundle,
):
    """A Predictor orchestration regression cannot make its own parity anchor pass."""
    def forbidden(*_args, **_kwargs):
        raise AssertionError("parity anchor delegated to Predictor._trace_sources")

    monkeypatch.setattr(Predictor, "_trace_sources", forbidden)
    kwargs = {"session_id": inference_fixture.session_id, "subject_id": inference_fixture.subject_id}
    assert canonical_trace(inference_fixture.raw, deployment_bundle, **kwargs).macro_features_62.shape[1] == 62
    assert legacy_trace(inference_fixture.raw, deployment_bundle, **kwargs).macro_features_62.shape[1] == 62


def test_final_events_match_all_four_boundaries(tmp_path, inference_fixture, inference_traces):
    """legacy = canonical Predictor = dist/inference = dist/submission for final events."""
    from scripts.build_inference_distribution import build_inference_distribution
    from scripts.build_submission import build_submission
    from src.pipeline.artifacts import load_current_promoted_release

    expected = [dict(row) for row in inference_traces[2].events]
    assert expected == [dict(row) for row in inference_traces[0].events]
    assert expected == [dict(row) for row in inference_traces[1].events]

    release = load_current_promoted_release(ROOT)
    deployment_bundle = ROOT / "models" / "event_stack" / str(release["run_key"]) / "deployment"
    inference_package = build_inference_distribution(
        repository_root=ROOT, bundle_path=deployment_bundle, destination=tmp_path / "built" / "inference",
    )
    submission_package = build_submission(repository_root=ROOT, destination=tmp_path / "built" / "submission")

    inference_output = tmp_path / "inference-prediction.json"
    done = subprocess.run(
        [sys.executable, "-I", "predict.py", str(inference_fixture.raw),
         "--output", str(inference_output), "--include-candidates"],
        cwd=inference_package, capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    submission_output = tmp_path / "submission-prediction.json"
    done = subprocess.run(
        [sys.executable, "-I", "main.py", "--raw", str(inference_fixture.raw),
         "--output", str(submission_output), "--include-candidates"],
        cwd=submission_package, capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr

    for path in (inference_output, submission_output):
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["events"] == expected
        assert document["model"]["run_key"] == "160afaf81debf1ee"
        for row in document["candidates"]:
            assert set(row) == {"session_id", "start_ms", "end_ms", "score", "admitted"}
    assert json.loads(inference_output.read_text(encoding="utf-8")) == json.loads(submission_output.read_text(encoding="utf-8"))
