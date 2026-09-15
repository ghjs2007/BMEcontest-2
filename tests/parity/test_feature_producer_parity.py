import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import src.config as config
from src.data import manifests
from src.data.loader import load_session_tsv
from src.pipeline.io.raw_session import RawSessionSource, load_raw_session


ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((ROOT / "tests/fixtures/release_160afaf81debf1ee/fixture_manifest.json").read_text(encoding="utf-8"))["macro_golden"]


def _matrix_sha256(values: np.ndarray) -> str:
    matrix = np.ascontiguousarray(values)
    return hashlib.sha256(matrix.dtype.str.encode() + str(matrix.shape).encode() + matrix.tobytes()).hexdigest()


def _window_sha256(windows) -> str:
    encoded = [json.dumps(tuple(window), separators=(",", ":")).encode("utf-8") for window in windows]
    return hashlib.sha256(b"\n".join(encoded)).hexdigest()


def _golden_inputs():
    sid = GOLDEN["session_id"]
    raw = next((config.SENSOR_DIR / sid).glob("collect_data*.txt"))
    index = manifests.load_sensor_index()
    meal_meta, _ = manifests.load_meal_meta()
    row = index.loc[index["session_id"].astype(str) == sid].iloc[0]
    meals = [
        meal for meal in meal_meta.get(row["externalid"], [])
        if meal["before"] >= int(row["timeStamp.startTime"])
        and meal["after"] <= int(row["timeStamp.endTime"])
    ]
    return sid, raw, meals


def test_legal_raw_session_matches_recorded_session_layer_fingerprint():
    sid, raw, _ = _golden_inputs()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == GOLDEN["raw_source_sha256"]
    direct = load_raw_session(RawSessionSource(raw, sid))
    legacy = load_session_tsv(raw)
    for field in ("acc", "gyro", "ppg", "t_acc", "t_ppg", "imu_valid", "ppg_valid"):
        assert _matrix_sha256(getattr(direct, field)) == _matrix_sha256(getattr(legacy, field))


def test_legacy_62_adapter_63_and_canonical_features_match_promoted_fixture():
    from src.pipeline.features.macro import MacroFeatureConfig, add_time_prior, extract_macro_windows
    from src.pipeline.runner import _with_time_prior

    sid, raw, meals = _golden_inputs()
    sys.path.insert(0, str(ROOT / "scripts"))
    import slide_features

    legacy_features, _, legacy_windows = slide_features._process_session((sid, json.dumps(meals), "val"))
    legacy_62 = np.asarray(legacy_features, dtype=np.float32)
    canonical = extract_macro_windows(load_raw_session(RawSessionSource(raw, sid)), session_id=sid, config=MacroFeatureConfig())
    assert _window_sha256(legacy_windows) == GOLDEN["window_rows_sha256"]
    assert _window_sha256(((window.sid, window.start_ms, window.end_ms) for window in canonical.windows)) == GOLDEN["window_rows_sha256"]
    assert _matrix_sha256(legacy_62) == GOLDEN["macro_62_matrix_sha256"]
    assert _matrix_sha256(canonical.features) == GOLDEN["macro_62_matrix_sha256"]
    np.testing.assert_allclose(canonical.features, legacy_62, rtol=0, atol=0, equal_nan=True)
    adapted = add_time_prior(canonical.features, canonical.windows)
    runner_adapted = _with_time_prior(canonical.features, canonical.windows)
    np.testing.assert_allclose(adapted, runner_adapted, rtol=0, atol=0, equal_nan=True)
    assert _matrix_sha256(adapted) == GOLDEN["macro_63_matrix_sha256"]
    assert adapted.shape == (len(canonical.windows), 63)


def test_canonical_micro_windows_match_current_cache_rows_for_golden_session():
    from src.pipeline.features.micro import extract_micro_windows
    from src.pipeline.imu_features import MicroFeatureConfig

    sid, raw, _ = _golden_inputs()
    canonical = extract_micro_windows(
        load_raw_session(RawSessionSource(raw, sid)), session_id=sid, config=MicroFeatureConfig()
    )
    with np.load(ROOT / "cache/micro15/fold0_val.npz", allow_pickle=False) as cache:
        selected = np.asarray([json.loads(str(value))[0] == sid for value in cache["wid"]], dtype=bool)
        legacy_features = np.asarray(cache["feat"])[selected]
        legacy_windows = [json.loads(str(value)) for value in cache["wid"][selected]]
    assert _window_sha256(legacy_windows) == _window_sha256(
        (window.sid, window.start_ms, window.end_ms) for window in canonical.windows
    )
    np.testing.assert_allclose(canonical.features, legacy_features, rtol=0, atol=0, equal_nan=True)
