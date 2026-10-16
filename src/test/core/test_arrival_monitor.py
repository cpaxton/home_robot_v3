# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import runpy
from pathlib import Path

import pytest

from emet.core.navigation_result import ArrivalMonitor

BridgeArrivalMonitor = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "emet_core/emet/core/navigation_result.py")
)["ArrivalMonitor"]


def monitor(monitor_type=ArrivalMonitor):
    return monitor_type({"resolved_goal": [0, 0, 0]}, "precision", now=0)


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


@pytest.mark.parametrize("monitor_type", [ArrivalMonitor, BridgeArrivalMonitor])
def test_stalled_motion_has_fresh_telemetry_but_no_progress(monitor_type):
    m = monitor(monitor_type)
    for i in range(10):
        assert update(m, i * 0.5, (1, 0, 0), stopped=False) is None
    assert update(m, 5, (1, 0, 0), stopped=False)[1]["reason"] == "navigation stalled"


@pytest.mark.parametrize("monitor_type", [ArrivalMonitor, BridgeArrivalMonitor])
def test_turn_toward_waypoint_is_progress_even_away_from_final_heading(monitor_type):
    m = monitor(monitor_type)
    # The waypoint is behind the robot, but its requested final yaw is zero.
    # Turning toward the translation direction legitimately increases final-yaw
    # error while XY distance stays unchanged.
    for tick in range(11):
        assert update(m, tick * 0.5, (1, 0, tick * 0.1), stopped=False) is None


@pytest.mark.parametrize("monitor_type", [ArrivalMonitor, BridgeArrivalMonitor])
def test_approach_heading_does_not_mask_stall_inside_xy_tolerance(monitor_type):
    m = monitor(monitor_type)
    for tick in range(10):
        assert update(m, tick * 0.5, (0.005, 0, tick * 0.1), stopped=False) is None
    assert update(m, 5, (0.005, 0, 1), stopped=False)[1]["reason"] == "navigation stalled"


@pytest.mark.parametrize("monitor_type", [ArrivalMonitor, BridgeArrivalMonitor])
def test_slow_simulation_gets_a_measured_time_progress_window(monitor_type):
    m = monitor(monitor_type)
    for tick in range(51):
        assert m.update([1 - 0.0003 * tick, 0, 0], sample_time=0.1 * tick, now=tick, stopped=False) is None


@pytest.mark.parametrize("monitor_type", [ArrivalMonitor, BridgeArrivalMonitor])
def test_slow_simulation_still_rejects_a_real_stall(monitor_type):
    m = monitor(monitor_type)
    for tick in range(50):
        assert m.update([1, 0, 0], sample_time=0.1 * tick, now=tick, stopped=False) is None
    assert m.update([1, 0, 0], sample_time=5, now=50, stopped=False)[1]["reason"] == "navigation stalled"


def test_manipulation_policy_requires_tighter_measured_arrival():
    m = ArrivalMonitor({"resolved_goal": [0, 0, 0]}, "manipulation", now=0)
    for t in [0, 0.3, 0.6]:
        result = update(m, t, pose=(0.015, 0, 0.02))
        assert result is None or result[0] == "correct"
    for t in [0.8, 1.1]:
        assert update(m, t, pose=(0.008, 0, 0.01)) is None
    assert update(m, 1.4, pose=(0.008, 0, 0.01))[0] == "succeeded"
