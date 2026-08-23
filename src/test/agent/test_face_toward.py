# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for CHAT face_toward yaw math."""

from __future__ import annotations

import math

import pytest

from emet.agent.face_toward import signed_yaw_delta_rad, yaw_to_face_xy


def test_signed_yaw_delta_wraps_shortest():
    assert abs(signed_yaw_delta_rad(0.0, math.pi / 2) - math.pi / 2) < 1e-6
    assert abs(signed_yaw_delta_rad(0.0, -math.pi / 2) + math.pi / 2) < 1e-6
    # +270° target should go -90° (CW), not +270°
    assert abs(signed_yaw_delta_rad(0.0, 3 * math.pi / 2) + math.pi / 2) < 1e-6


@pytest.mark.parametrize(
    "base,target,expect_deg_sign",
    [
        # Facing +x; target to the left (+y) → +90°
        ([0.0, 0.0, 0.0], [0.0, 1.0], 1),
        # Facing +x; target to the right (−y) → −90°
        ([0.0, 0.0, 0.0], [0.0, -1.0], -1),
        # Already facing target → ~0
        ([0.0, 0.0, math.pi / 2], [0.0, 2.0], 0),
    ],
)
def test_yaw_to_face_xy(base, target, expect_deg_sign):
    delta, _bearing = yaw_to_face_xy(base, target)
    deg = math.degrees(delta)
    if expect_deg_sign == 0:
        assert abs(deg) < 5.0
    else:
        assert deg * expect_deg_sign > 0
        assert abs(abs(deg) - 90.0) < 1.0


def test_yaw_to_face_xy_offset_pose():
    # At (1,1) facing +x; target at (1,3) → face +y → +90°
    delta, _ = yaw_to_face_xy([1.0, 1.0, 0.0], [1.0, 3.0])
    assert abs(math.degrees(delta) - 90.0) < 1.0
