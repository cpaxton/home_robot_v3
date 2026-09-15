# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import pytest

from emet.core.navigation_result import ArrivalMonitor


def monitor():
    return ArrivalMonitor({"resolved_goal": [0, 0, 0]}, "precision", now=0)


def update(m, t, pose=(0, 0, 0), stopped=True):
    return m.update(pose, sample_time=t, now=t, stopped=stopped)


def test_requires_three_distinct_samples_and_settling():
    m = monitor()
    assert update(m, 0) is None
    assert m.update([0, 0, 0], sample_time=0, now=0.3, stopped=True) is None
    assert update(m, 0.4) is None
    assert update(m, 0.6)[0] == "succeeded"


def test_boundary_noise_resets_settling_without_failure():
    m = monitor()
    assert update(m, 0) is None
    assert update(m, 0.2, (0.021, 0, 0), stopped=False) is None
    assert update(m, 0.4) is None
    assert update(m, 0.6) is None
    assert update(m, 1.0)[0] == "succeeded"


def test_moving_inside_tolerance_is_not_success():
    m = monitor()
    assert update(m, 0) is None
    assert update(m, 0.3, (0.01, 0, 0)) is None
    assert update(m, 0.6, (0.019, 0, 0)) is None


@pytest.mark.parametrize("pose", [(float("nan"), 0, 0), (0,), None])
def test_invalid_pose_does_not_refresh_freshness(pose):
    m = monitor()
    assert update(m, 0.5, pose) is None
    assert update(m, 1.1)[1]["reason"] == "stale navigation pose"


def test_duplicate_poses_go_stale():
    m = monitor()
    assert update(m, 0) is None
    assert m.update([0, 0, 0], sample_time=0, now=1.1, stopped=True)[0] == "failed"


def test_corrections_bounded_and_only_requested_when_stopped():
    m = monitor()
    assert update(m, 0, (1, 0, 0), stopped=False) is None
    for start in (0.25, 1.25):
        assert update(m, start, (1, 0, 0)) is None
        assert update(m, start + 0.25, (1, 0, 0)) is None
        assert update(m, start + 0.5, (1, 0, 0))[0] == "correct"
    assert update(m, 2.25, (1, 0, 0)) is None
    assert update(m, 2.5, (1, 0, 0)) is None
    assert update(m, 2.75, (1, 0, 0))[1]["reason"] == "navigation correction budget exhausted"


def test_stalled_motion_has_fresh_telemetry_but_no_progress():
    m = monitor()
    for i in range(10):
        assert update(m, i * 0.5, (1, 0, 0), stopped=False) is None
    assert update(m, 5, (1, 0, 0), stopped=False)[1]["reason"] == "navigation stalled"


def test_approach_inside_acceptance_radius_counts_before_final_heading():
    m = ArrivalMonitor({"resolved_goal": [0, 0, 0]}, "exploration", now=0)
    # Approach toward the controller's inner XY threshold while holding the
    # travel heading, which need not improve final-yaw error yet.
    for i in range(11):
        assert update(m, i * 0.5, (0.065 - i * 0.004, 0, 0.7), stopped=False) is None
    for i in range(1, 11):
        assert update(m, 5 + i * 0.5, (0.025, 0, 0.7 - i * 0.06), stopped=False) is None
    for t in (10.1, 10.4):
        assert update(m, t, (0.025, 0, 0.1)) is None
    assert update(m, 10.7, (0.025, 0, 0.1))[0] == "succeeded"


def test_stationary_inside_xy_radius_with_wrong_heading_still_stalls():
    m = ArrivalMonitor({"resolved_goal": [0, 0, 0]}, "exploration", now=0)
    for i in range(10):
        assert update(m, i * 0.5, (0.025, 0, 0.7), stopped=False) is None
    assert update(m, 5, (0.025, 0, 0.7), stopped=False)[1]["reason"] == "navigation stalled"
