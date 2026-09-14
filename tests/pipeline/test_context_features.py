import hashlib
import json

import numpy as np
import pytest

from src.pipeline.context_features import (
    CONTEXT_V1_COLUMNS,
    CONTEXT_V1_CONTEXT_MS,
    CONTEXT_V1_RUN_GAP_MS,
    CONTEXT_V1_MIN_RUN_WINDOWS,
    CONTEXT_V1_SCHEMA_HASH,
    MACRO_STRIDE_MS,
    MACRO_WINDOW_MS,
    MICRO_STRIDE_MS,
    MICRO_WINDOW_MS,
    context_v1_features,
)
from src.pipeline.event_stack import EventRef


def _idx(name):
    return CONTEXT_V1_COLUMNS.index(name)


def test_context_v1_schema_is_fixed_and_hashed():
    assert len(CONTEXT_V1_COLUMNS) == 60
    assert CONTEXT_V1_COLUMNS[:4] == (
        "macro_pre_mean", "macro_pre_max", "macro_pre_std",
        "macro_pre_above_fraction",
    )
    assert CONTEXT_V1_COLUMNS[30] == "micro_pre_mean"
    expected = hashlib.sha256(
        json.dumps(CONTEXT_V1_COLUMNS, separators=(",", ":")).encode()
    ).hexdigest()
    assert CONTEXT_V1_SCHEMA_HASH == expected


def test_context_v1_uses_same_session_clipped_real_time_regions():
    candidate = EventRef("s1", 1_200_000, 1_800_000)
    macro = {
        "s1": [(0, 30_000, .1), (1_300_000, 1_330_000, .8), (1_900_000, 1_930_000, .2)],
        "s2": [(1_300_000, 1_330_000, 1.0)],
    }
    row = context_v1_features([candidate], macro, {}, {"s1": (0, 3_000_000)})[0]
    assert row[_idx("macro_candidate_mean")] == pytest.approx(.8)
    assert row[_idx("macro_post_mean")] == pytest.approx(.2)


def test_context_v1_empty_scale_has_nan_statistics_and_zero_coverage():
    event = EventRef("s", 1_000_000, 1_100_000)
    row = context_v1_features([event], {}, {}, {"s": (0, 3_000_000)})[0]
    assert np.isnan(row[:4]).all()
    assert np.isnan(row[12:20]).all()
    assert row[_idx("macro_pre_coverage")] == 0
    assert row[_idx("macro_candidate_coverage")] == 0
    assert row[_idx("macro_post_coverage")] == 0
    assert row[_idx("macro_neighbor_run_count")] == 0
    assert row[_idx("macro_neighbor_run_total_duration_s")] == 0
    assert np.isnan(row[_idx("macro_neighbor_run_max_duration_s")])
    assert np.isnan(row[_idx("macro_pre_coverage") + 9])


def test_context_v1_clips_regions_to_explicit_session_bounds():
    event = EventRef("s", 100_000, 200_000)
    macro = {"s": [(60_000, 70_000, .4), (210_000, 220_000, .6)]}
    row = context_v1_features([event], macro, {}, {"s": (50_000, 220_000)})[0]
    assert row[_idx("macro_pre_mean")] == pytest.approx(.4)
    assert row[_idx("macro_post_mean")] == pytest.approx(.6)


def test_context_v1_score_gap_does_not_redefine_explicit_session_boundary():
    event = EventRef("s", 100_000, 200_000)
    macro = {"s": [(0, 20_000, .4), (500_000, 520_000, .6)]}
    row = context_v1_features([event], macro, {}, {"s": (0, 1_000_000)})[0]
    assert row[_idx("macro_pre_mean")] == pytest.approx(.4)
    assert row[_idx("macro_post_mean")] == pytest.approx(.6)


def test_context_v1_uses_fixed_thresholds_for_above_fraction():
    event = EventRef("s", 1_000_000, 1_100_000)
    macro = {"s": [(1_000_000, 1_010_000, .28838), (1_020_000, 1_030_000, .288379)]}
    micro = {"s": [(1_000_000, 1_005_000, .2), (1_010_000, 1_015_000, .19999)]}
    row = context_v1_features([event], macro, micro, {"s": (0, 2_000_000)})[0]
    assert row[_idx("macro_candidate_above_fraction")] == pytest.approx(.5)
    assert row[_idx("micro_candidate_above_fraction")] == pytest.approx(.5)


def test_context_v1_joins_two_window_run_at_sixty_second_gap():
    event = EventRef("s", 1_000_000, 1_100_000)
    macro = {"s": [(810_000, 820_000, .5), (870_000, 880_000, .5), (1_300_000, 1_310_000, .5)]}
    row = context_v1_features([event], macro, {}, {"s": (0, 2_000_000)})[0]
    assert row[_idx("macro_neighbor_run_count")] == 1
    assert row[_idx("macro_neighbor_run_total_duration_s")] == pytest.approx(300)


def test_context_v1_excludes_candidate_own_intersecting_run_from_neighbors():
    event = EventRef("s", 1_000_000, 1_100_000)
    macro = {"s": [(900_000, 910_000, .5), (950_000, 960_000, .5),
                     (1_300_000, 1_310_000, .5), (1_350_000, 1_360_000, .5)]}
    row = context_v1_features([event], macro, {}, {"s": (0, 2_000_000)})[0]
    assert row[_idx("macro_neighbor_run_count")] == 1
    assert row[_idx("macro_pre_mean")] == pytest.approx(.5)


def test_context_v1_no_neighbor_has_nan_distances_and_max_duration():
    event = EventRef("s", 1_000_000, 1_100_000)
    macro = {"s": [(900_000, 910_000, .5), (1_050_000, 1_060_000, .5)]}
    row = context_v1_features([event], macro, {}, {"s": (0, 2_000_000)})[0]
    assert row[_idx("macro_neighbor_run_count")] == 0
    assert row[_idx("macro_neighbor_run_total_duration_s")] == 0
    assert np.isnan(row[_idx("macro_neighbor_run_max_duration_s")])
    assert np.isnan(row[_idx("macro_preceding_run_distance_s")])
    assert np.isnan(row[_idx("macro_following_run_distance_s")])


def test_context_v1_center_half_mass_and_first_minus_second_mean():
    event = EventRef("s", 1_000_000, 1_100_000)
    macro = {"s": [(1_000_000, 1_025_000, 1.0), (1_025_000, 1_050_000, 3.0),
                     (1_050_000, 1_075_000, 5.0), (1_075_000, 1_100_000, 7.0)]}
    row = context_v1_features([event], macro, {}, {"s": (0, 2_000_000)})[0]
    assert row[_idx("macro_candidate_center_half_mass_fraction")] == pytest.approx(0.5)
    assert row[_idx("macro_candidate_first_minus_second_mean")] == pytest.approx(-4.0)


def test_context_v1_is_stable_under_input_ordering_and_accepts_wrapped_candidates():
    events = [EventRef("s", 1_000_000, 1_100_000), EventRef("s", 1_200_000, 1_300_000)]
    windows = [(1_000_000, 1_010_000, .5), (1_050_000, 1_060_000, .7), (1_200_000, 1_210_000, .4)]
    reverse = list(reversed(windows))
    bounds = {"s": (0, 2_000_000)}
    first = context_v1_features(events, {"s": windows}, {}, bounds)
    second = context_v1_features([type("C", (), {"event": e})() for e in events], {"s": reverse}, {}, bounds)
    np.testing.assert_equal(first, second)


def test_context_v1_exposes_named_window_and_stride_constants():
    assert (MACRO_STRIDE_MS, MACRO_WINDOW_MS) == (15_000, 240_000)
    assert (MICRO_STRIDE_MS, MICRO_WINDOW_MS) == (7_500, 15_000)
    assert CONTEXT_V1_CONTEXT_MS == 1_200_000
    assert CONTEXT_V1_RUN_GAP_MS == 60_000
    assert CONTEXT_V1_MIN_RUN_WINDOWS == 2


def test_context_v1_rejects_infinite_probability():
    event = EventRef("s", 1_000_000, 1_100_000)
    with pytest.raises(ValueError, match="infinity"):
        context_v1_features([event], {"s": [(1_000_000, 1_010_000, np.inf)]}, {}, {"s": (0, 2_000_000)})


def test_context_v1_rejects_nonfinite_window_timestamp():
    event = EventRef("s", 1_000_000, 1_100_000)
    with pytest.raises(ValueError, match="finite integer-like"):
        context_v1_features(
            [event],
            {"s": [(1_000_000, np.inf, .5)]},
            {},
            {"s": (0, 2_000_000)},
        )


def test_context_v1_rejects_nonfinite_event_boundary():
    event = EventRef("s", 1_000_000, np.inf)
    with pytest.raises(ValueError, match="finite integer-like"):
        context_v1_features([event], {}, {}, {"s": (0, 2_000_000)})


def test_context_v1_rejects_nonfinite_session_boundary():
    event = EventRef("s", 1_000_000, 1_100_000)
    with pytest.raises(ValueError, match="finite integer-like"):
        context_v1_features([event], {}, {}, {"s": (0, np.inf)})
