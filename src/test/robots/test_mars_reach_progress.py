import math

import pytest

from emet.robots.innate_mars.reach_progress import (
    ElbowLoadGuard,
    ElbowProgressGuard,
    check_reach_envelope,
    pace_elbow_target,
)


def test_progress_guard_admits_continued_tracking():
    guard = ElbowProgressGuard()
    for i in range(21):
        measured = math.radians(90 - i * 0.2)
        guard.check(now=i / 10, measured_rad=measured, target_rad=measured - math.radians(0.8))


def test_progress_guard_stops_when_one_count_of_motion_is_all_the_elbow_makes():
    guard = ElbowProgressGuard()
    q = math.pi / 2
    target = math.radians(88)
    guard.check(now=0.0, measured_rad=q, target_rad=target)
    guard.check(now=0.4, measured_rad=q - math.tau / 4096, target_rad=target)
    with pytest.raises(RuntimeError, match="stopped progressing"):
        guard.check(now=1.01, measured_rad=q - math.tau / 4096, target_rad=target)


def test_progress_guard_does_not_count_time_spent_at_hold():
    guard = ElbowProgressGuard()
    q = math.pi / 2
    target = math.radians(88)
    guard.check(now=0.0, measured_rad=q, target_rad=target)
    guard.check(now=0.6, measured_rad=q, target_rad=q)
    guard.check(now=2.0, measured_rad=q, target_rad=target)
    with pytest.raises(RuntimeError, match="stopped progressing"):
        guard.check(now=3.01, measured_rad=q, target_rad=target)


def test_load_guard_accepts_brief_excursion_and_stops_on_sustained_load():
    guard = ElbowLoadGuard()
    guard.check(now=0.0, load_percent=35.2)
    guard.check(now=0.2, load_percent=34.8)
    guard.check(now=0.3, load_percent=35.2)
    guard.check(now=0.5, load_percent=36.0)
    with pytest.raises(RuntimeError, match="sustained"):
        guard.check(now=0.56, load_percent=36.0)


def test_load_guard_stops_immediately_at_upper_ceiling():
    with pytest.raises(RuntimeError, match="immediate"):
        ElbowLoadGuard().check(now=0.0, load_percent=-40.0)


def test_target_pacing_limits_lead_instead_of_increasing_load_ceiling():
    measured = math.radians(74)
    requested = math.radians(71)
    assert pace_elbow_target(measured_rad=measured, requested_rad=requested, load_percent=34) == requested
    eased = pace_elbow_target(measured_rad=measured, requested_rad=requested, load_percent=37.5)
    assert math.degrees(measured - eased) == pytest.approx(0.25)
    assert pace_elbow_target(measured_rad=measured, requested_rad=requested, load_percent=40) == measured


def test_reach_envelope_admits_small_overshoot_only_with_clearance_and_progress():
    args = {
        "measured_rad": [0, math.radians(7.6), 0, 0, 0, 0],
        "start_rad": [0] * 6,
        "goal_rad": [0, math.radians(7), 0, 0, 0, 0],
        "hand_error_m": 0.25,
        "previous_hand_error_m": 0.26,
        "clearance_verified": True,
    }
    check_reach_envelope(**args)
    with pytest.raises(RuntimeError, match="checked reach"):
        check_reach_envelope(**(args | {"clearance_verified": False}))
    with pytest.raises(RuntimeError, match="checked reach"):
        check_reach_envelope(**(args | {"hand_error_m": 0.27}))
