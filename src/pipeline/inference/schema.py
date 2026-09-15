"""Small, dependency-free contract shared by inference and visualization."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

PREDICTION_SCHEMA_VERSION = "1.0"


def make_prediction_result(*, run_key: str, source: str, duration_seconds: float,
                           events: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Create the minimum public prediction document."""
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "model": {"name": "event-stack", "run_key": str(run_key)},
        "input": {"source": str(source), "duration_seconds": float(duration_seconds)},
        "events": [dict(event) for event in events],
        "diagnostics": {"coverage": None, "warnings": []},
    }


def validate_prediction(value: Mapping[str, object]) -> None:
    """Validate the stable public envelope without requiring jsonschema at runtime."""
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "model", "input", "events", "diagnostics"}:
        raise ValueError("prediction must contain exactly the public schema fields")
    if value["schema_version"] != PREDICTION_SCHEMA_VERSION:
        raise ValueError("prediction schema version is unsupported")
    model, input_value, events, diagnostics = value["model"], value["input"], value["events"], value["diagnostics"]
    if not isinstance(model, Mapping) or model.get("name") != "event-stack" or not isinstance(model.get("run_key"), str) or not model["run_key"]:
        raise ValueError("prediction model is invalid")
    if not isinstance(input_value, Mapping) or not isinstance(input_value.get("source"), str):
        raise ValueError("prediction input is invalid")
    try:
        duration = float(input_value.get("duration_seconds"))
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction duration_seconds is invalid") from exc
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("prediction duration_seconds is invalid")
    if not isinstance(events, list) or not isinstance(diagnostics, Mapping):
        raise ValueError("prediction events or diagnostics is invalid")
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or set(event) != {"id", "session_id", "start_ms", "end_ms", "duration_s", "confidence"}:
            raise ValueError(f"prediction event {index} is invalid")
        if not isinstance(event["id"], int) or not isinstance(event["session_id"], str):
            raise ValueError(f"prediction event {index} is invalid")
        if not isinstance(event["start_ms"], int) or not isinstance(event["end_ms"], int) or event["end_ms"] <= event["start_ms"]:
            raise ValueError(f"prediction event {index} is invalid")
