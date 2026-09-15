from src.pipeline.inference.schema import make_prediction_result, validate_prediction


def test_minimum_prediction_result_has_the_public_contract():
    """Removing a required public field must make validation fail."""
    result = make_prediction_result(
        run_key="160afaf81debf1ee",
        source="fixture.txt",
        duration_seconds=12.5,
        events=[],
    )
    validate_prediction(result)
    assert set(result) == {"schema_version", "model", "input", "events", "diagnostics"}


def test_prediction_schema_rejects_nonfinite_confidence_and_invalid_diagnostics():
    """Permissive schema validation would leak unusable JSON into visualization."""
    import math
    import pytest

    result = make_prediction_result(
        run_key="run", source="x", duration_seconds=1.0,
        events=[{"id": 0, "session_id": "s", "start_ms": 0, "end_ms": 1000,
                 "duration_s": 1.0, "confidence": math.nan}],
    )
    result["diagnostics"] = {"coverage": 1.2, "warnings": ["ok"]}
    with pytest.raises(ValueError, match="event 0|diagnostics"):
        validate_prediction(result)


def test_prediction_schema_rejects_malformed_optional_debug_blocks():
    """A frontend must not receive ambiguous timeline/candidate/gap records."""
    import pytest

    result = make_prediction_result(run_key="run", source="x", duration_seconds=1.0, events=[])
    result["timeline"] = {"session_ids": ["s"], "macro_windows": -1, "micro_windows": 0}
    with pytest.raises(ValueError, match="timeline"):
        validate_prediction(result)

    result = make_prediction_result(run_key="run", source="x", duration_seconds=1.0, events=[])
    result["candidates"] = [{"session_id": "s", "start_ms": 0, "end_ms": 1, "score": float("nan"), "admitted": False}]
    with pytest.raises(ValueError, match="candidates"):
        validate_prediction(result)

    result = make_prediction_result(run_key="run", source="x", duration_seconds=1.0, events=[])
    result["gaps"] = [{"session_id": "s", "start_ms": 2, "end_ms": 1, "unexpected": True}]
    with pytest.raises(ValueError, match="gaps"):
        validate_prediction(result)
