# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import pytest

from emet.motion.guarded_placement_probe import GuardedPlacementProbe, ProbeCalibration


@pytest.fixture
def probe():
    return GuardedPlacementProbe(
        ProbeCalibration("lab", 1, 0.5, 1.5, 0.2, 0.08),
        start_xyz=[0, 0, 1], baseline=[0, 0, 2, 0, 0, 0], started_at=0.0, contact_band_m=(.00035, .0008),
    )


def sample(probe, t, z, fz=2.0, *, fx=0.0, torque=0.0, retention=True, now=None):
    return probe.update(
        now=t if now is None else now, sampled_at=t, xyz=[0., 0., z],
        wrench=[fx, 0., fz, torque, 0., 0.], retention_ok=retention,
    )


def test_probe_latches_only_three_fresh_axial_contact_samples(probe):
    assert sample(probe, .05, .9998) == "descending"
    assert sample(probe, .10, .9996, 2.6) == "holding"
    assert sample(probe, .10, .9996, 2.6) == "holding"
    assert sample(probe, .15, .9996, 2.6) == "holding"
    assert sample(probe, .20, .9996, 2.6) == "contact"
    assert probe.reason == "support_contact"
    assert sample(probe, .25, .9996) == "contact"


@pytest.mark.parametrize(("values", "reason"), [
    ({"fx": .3}, "probe_unexpected_force"),
    ({"torque": .2}, "probe_unexpected_force"),
    ({"fz": 4.0}, "probe_unexpected_force"),
    ({"fz": 2.6}, "probe_contact_outside_band"),
    ({"retention": False}, "probe_payload_tracking_lost"),
    ({"now": .3}, "probe_feedback_stale"),
    ({"now": 3.1}, "probe_timeout"),
])
def test_probe_latches_a_stop_on_bad_feedback(probe, values, reason):
    assert sample(probe, .05, .9998, **values) == "stopped"
    assert probe.reason == reason


def test_probe_stops_after_traversing_band_without_contact(probe):
    for t, z in ((.05, .9998), (.10, .9996), (.15, .9994)):
        assert sample(probe, t, z) == "descending"
    assert sample(probe, .20, .99919) == "stopped"
    assert probe.reason == "probe_no_contact"
