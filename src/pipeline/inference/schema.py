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
        "diagnostics": {"coverage": 0.0, "warnings": []},
    }


def validate_prediction(value: Mapping[str, object]) -> None:
    """Validate the stable public envelope without requiring jsonschema at runtime."""
    required = {"schema_version", "model", "input", "events", "diagnostics"}
    allowed = required | {"timeline", "candidates", "gaps"}
    if not isinstance(value, Mapping) or not required <= set(value) or not set(value) <= allowed:
        raise ValueError("prediction must contain only public schema fields")
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
    if set(diagnostics) - {"coverage", "warnings", "resolved_device"} or set(diagnostics) < {"coverage", "warnings"}:
        raise ValueError("prediction diagnostics is invalid")
    try:
        coverage = float(diagnostics["coverage"])
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction diagnostics coverage is invalid") from exc
    warnings = diagnostics["warnings"]
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0 or not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise ValueError("prediction diagnostics is invalid")
    if "resolved_device" in diagnostics and diagnostics["resolved_device"] not in {"cpu", "cuda"}:
        raise ValueError("prediction diagnostics is invalid")
    for optional in ("timeline", "candidates", "gaps"):
        if optional in value and not isinstance(value[optional], (Mapping, list)):
            raise ValueError(f"prediction {optional} is invalid")
    seen_ids: set[int] = set()
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or set(event) != {"id", "session_id", "start_ms", "end_ms", "duration_s", "confidence"}:
            raise ValueError(f"prediction event {index} is invalid")
        if not isinstance(event["id"], int) or event["id"] != index or event["id"] in seen_ids or not isinstance(event["session_id"], str) or not event["session_id"]:
            raise ValueError(f"prediction event {index} is invalid")
        seen_ids.add(event["id"])
        if not isinstance(event["start_ms"], int) or not isinstance(event["end_ms"], int) or event["end_ms"] <= event["start_ms"]:
            raise ValueError(f"prediction event {index} is invalid")
        try:
            duration_s = float(event["duration_s"])
            confidence = float(event["confidence"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prediction event {index} is invalid") from exc
        if (not math.isfinite(duration_s) or duration_s != (event["end_ms"] - event["start_ms"]) / 1000.0
                or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0):
            raise ValueError(f"prediction event {index} is invalid")
