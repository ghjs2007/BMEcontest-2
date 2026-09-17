import numpy as np

from src.pipeline.io.raw_session import SessionData


def _session_with_gap() -> SessionData:
    left = np.arange(0, 12_500, 10, dtype=np.int64)
    right = np.arange(30_000, 42_500, 10, dtype=np.int64)
    timestamps = np.concatenate((left, right))
    acc = np.vstack((np.zeros(len(timestamps)), np.zeros(len(timestamps)), np.ones(len(timestamps)))).astype(np.float32)
    return SessionData(
        acc=acc,
        gyro=np.zeros_like(acc),
        ppg=np.empty((44, len(timestamps)), dtype=np.float32),
        t_acc=timestamps,
        t_ppg=np.full(len(timestamps), -1, dtype=np.int64),
        imu_valid=np.ones(len(timestamps), dtype=bool),
        ppg_valid=np.zeros(len(timestamps), dtype=bool),
        meta={},
    )


def test_raw_macro_is_62d_and_never_crosses_gap():
    from src.pipeline.features.macro import MacroFeatureConfig, extract_macro_windows

    batch = extract_macro_windows(
        _session_with_gap(),
        session_id="s1",
        config=MacroFeatureConfig(window_ms=10_000, stride_ms=2_500, coverage_min=0.80),
    )
    assert batch.features.shape[1] == 62
    assert all(window.end_ms <= 12_500 or window.start_ms >= 30_000 for window in batch.windows)


def test_frozen_time_prior_adapts_exactly_to_runner_boundary():
    from src.pipeline.event_stack import EventRef
    from src.pipeline.features.macro import add_time_prior
    from src.pipeline.runner import _with_time_prior

    raw_62 = np.arange(124, dtype=np.float32).reshape(2, 62)
    windows = (EventRef("s1", 0, 240_000), EventRef("s1", 3_600_000, 3_840_000))
    adapted = add_time_prior(raw_62, windows)
    assert adapted.shape == (2, 63)
    np.testing.assert_allclose(adapted, _with_time_prior(raw_62, windows), rtol=0, atol=0, equal_nan=True)


def test_runner_time_prior_preserves_legacy_arbitrary_width_features():
    from src.pipeline.event_stack import EventRef
    from src.pipeline.runner import _with_time_prior

    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    windows = (EventRef("s1", 0, 240_000), EventRef("s1", 3_600_000, 3_840_000))

    adapted = _with_time_prior(raw, windows)

    assert adapted.shape == (2, 4)
    np.testing.assert_allclose(adapted[:, :3], raw)


def test_micro_delegates_to_frozen_47d_primitive(monkeypatch):
    import src.pipeline.features.micro as features_micro
    from src.pipeline.imu_features import MicroFeatureConfig

    timestamps = np.arange(0, 30_000, 10, dtype=np.int64)
    acc = np.vstack((np.zeros(len(timestamps)), np.zeros(len(timestamps)), np.ones(len(timestamps)))).astype(np.float32)
    session = SessionData(
        acc=acc, gyro=np.zeros_like(acc), ppg=np.empty((44, len(timestamps)), dtype=np.float32),
        t_acc=timestamps, t_ppg=np.full(len(timestamps), -1, dtype=np.int64),
        imu_valid=np.ones(len(timestamps), dtype=bool), ppg_valid=np.zeros(len(timestamps), dtype=bool), meta={},
    )
    monkeypatch.setattr(features_micro, "extract_micro_features", lambda a, g, hz: np.zeros(47, np.float32))
    batch = features_micro.extract_micro_windows(session, session_id="s1", config=MicroFeatureConfig())
    assert batch.features.shape[1] == 47
