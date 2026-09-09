# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for CHAT face_toward yaw math."""

from __future__ import annotations

import math

import numpy as np
import pytest

from emet.agent.face_toward import resolve_object_xy, signed_yaw_delta_rad, yaw_to_face_xy


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


def test_resolve_object_xy_uses_voxel_pin():
    class _Voxel:
        def __init__(self) -> None:
            self.hits = {"red cylinder": [0.08, -0.55, 0.6]}
            self._last_localize_stats: dict[str, object] = {}
            self.n_live = 0

        def localize_text(self, text, debug=False, return_debug=False):
            self.n_live += 1
            q = str(text or "").strip().lower()
            pt = self.hits.get(q)
            return None if pt is None else np.asarray(pt, dtype=np.float64)

    class _Agent:
        def __init__(self) -> None:
            self.voxel_map = _Voxel()

        def get_voxel_map(self):
            return self.voxel_map

    agent = _Agent()
    xy, source = resolve_object_xy(agent, "red cylinder")
    assert source == "voxel"
    assert xy is not None
    assert abs(xy[0] - 0.08) < 1e-6
    agent.voxel_map.hits.clear()
    xy2, source2 = resolve_object_xy(agent, "red cylinder")
    assert source2 == "voxel"
    assert xy2 is not None
    assert abs(xy2[0] - 0.08) < 1e-6
    assert agent.voxel_map.n_live == 1
