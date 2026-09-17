"""Legal, repository-local inputs for inference-boundary parity checks."""

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

import src.config as config
from src.pipeline.inference import Predictor
from src.pipeline.inference.legacy_payload import canonical_trace, legacy_trace


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RawParityFixture:
    raw: Path
    session_id: str
    subject_id: str
    manifest: dict[str, object]


@pytest.fixture(scope="session")
def inference_fixture() -> RawParityFixture:
    """Return the audited raw session already used by producer parity tests."""
    manifest = json.loads(
        (ROOT / "tests/fixtures/release_160afaf81debf1ee/fixture_manifest.json").read_text(encoding="utf-8")
    )
    golden = manifest["macro_golden"]
    source_session_id = str(golden["session_id"])
    raw = next((config.SENSOR_DIR / source_session_id).glob("collect_data*.txt"))
    # Predictor.predict_file deliberately uses the file stem as its one-file
    # session ID; the legacy compatibility trace uses that same public contract.
    return RawParityFixture(raw=raw, session_id=raw.stem, subject_id=source_session_id, manifest=manifest)


@pytest.fixture(scope="session")
def deployment_bundle() -> Path:
    return ROOT / "dist/event_stack/bundle"


@pytest.fixture(scope="session")
def inference_traces(inference_fixture: RawParityFixture, deployment_bundle: Path):
    """Compute the expensive three boundary traces only once per parity run."""
    kwargs = {
        "session_id": inference_fixture.session_id,
        "subject_id": inference_fixture.subject_id,
    }
    return (
        legacy_trace(inference_fixture.raw, deployment_bundle, **kwargs),
        canonical_trace(inference_fixture.raw, deployment_bundle, **kwargs),
        Predictor.from_bundle(deployment_bundle).trace_file(
            inference_fixture.raw, subject_id=inference_fixture.subject_id
        ),
    )
