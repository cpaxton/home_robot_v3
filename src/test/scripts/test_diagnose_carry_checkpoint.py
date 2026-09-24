# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Check command profiles without loading a simulator or privileged scene."""

import runpy
from pathlib import Path

import numpy as np
import pytest

helpers = runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/diagnose_carry_checkpoint.py"))


@pytest.mark.parametrize("t,expected", [(0, 0), (1, 0.5), (2, 1), (5, 1), (7, 0.5), (8, 0), (35, 0)])
def test_bounded_drive_profile(t, expected):
    assert helpers["drive_envelope"](t) == expected


@pytest.mark.parametrize("mode", ["hold", "straight", "turn", "brake"])
def test_wheel_profiles_stop_and_apply_transmission_gearing(mode):
    args = {"linear_speed": 0.05, "angular_speed": 0.2, "radius": 0.0508, "separation": 0.3153}
    control = helpers["wheel_controls"]
    np.testing.assert_allclose(control(mode, 35, **args, gears=np.array([3.0, 3.0])), 0)
    peak = control(mode, 3, **args, gears=np.array([3.0, 3.0]))
    np.testing.assert_allclose(peak, 3 * control(mode, 3, **args, gears=np.ones(2)))
    if mode == "turn":
        assert peak[0] < 0 < peak[1]
    if mode in {"straight", "brake"}:
        assert peak[0] == peak[1] > 0


def test_braking_ablation_changes_only_deceleration_phase():
    assert helpers["drive_envelope"](6.25, brake_s=0.25) == 0
    assert helpers["drive_envelope"](6.25, brake_s=2) == 0.875
