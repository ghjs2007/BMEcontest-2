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
