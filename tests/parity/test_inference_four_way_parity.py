"""Boundary parity for the frozen raw inference graph.

The regression caught here is any training-serving skew introduced while
extracting a raw-data Predictor.  Expected fingerprints are stored separately
from the code under test in the release fixture manifest.
"""

import hashlib

import numpy as np

from src.pipeline.event_stack import EventRef, apply_event_policy


def _matrix_hash(values: np.ndarray) -> str:
    matrix = np.ascontiguousarray(values)
    return hashlib.sha256(matrix.dtype.str.encode() + str(matrix.shape).encode() + matrix.tobytes()).hexdigest()


def _assert_hashes(trace, hashes: dict[str, str]) -> None:
    for name in (
        "macro_features_62", "macro_features_63", "micro_features",
        "context_features", "macro_probabilities", "micro_probabilities",
        "verifier_scores",
    ):
        assert _matrix_hash(getattr(trace, name)) == hashes[name]


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
