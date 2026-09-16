"""Release gate: the published visualization contract stays valid and algorithm-free.

The frontend workspace consumes canonical prediction JSON only.  This gate
pins three properties: the shipped example is a legal instance of the shipped
schema, the JSON schema and the dependency-free runtime validator agree, and no
algorithm implementation may appear under ``dist/visual``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from src.pipeline.inference.schema import validate_prediction


ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
EXAMPLE = DIST / "examples" / "example_prediction.json"
SCHEMA = DIST / "schema" / "prediction.schema.json"
VISUAL = DIST / "visual"


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_visual_example_validates_against_published_schema():
    """The safe example must be a legal instance of the published schema."""
    example = _load(EXAMPLE)
    jsonschema.validate(example, _load(SCHEMA))
    assert example["input"]["source"] == "synthetic-example"


def test_visual_example_matches_runtime_validator():
    """The JSON schema and the dependency-free runtime validator must not drift."""
    validate_prediction(_load(EXAMPLE))


def test_published_schema_rejects_malformed_series_point():
    """An out-of-range timeline point must fail schema validation, not slip through."""
    example = _load(EXAMPLE)
    assert "series" in example["timeline"], "example must demonstrate the optional timeline series"
    broken = copy.deepcopy(example)
    broken["timeline"]["series"][0]["macro_probability"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, _load(SCHEMA))


def test_visual_workspace_has_no_algorithm_python():
    """Visualization displays canonical output; it may not reimplement decisions."""
    assert VISUAL.is_dir()
    assert not list(VISUAL.rglob("*.py"))


def test_visual_readme_documents_contract_boundaries():
    """Frontend developers need the schema path and the non-reimplementation rule."""
    readme = (VISUAL / "README.md").read_text(encoding="utf-8")
    assert "schema/prediction.schema.json" in readme
    assert "must not reimplement" in readme.lower()
