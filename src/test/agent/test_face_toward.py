# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for CHAT face_toward yaw math."""

from __future__ import annotations

import math
from types import SimpleNamespace

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


@pytest.mark.parametrize("blocked", [True, False])
@pytest.mark.parametrize("controller_transform", [True, False])
def test_face_uses_world_pose_and_canonical_rotation(monkeypatch, blocked, controller_transform):
    from emet.agent.loop import _dispatch_tool_calls
    from emet.agent.tools import get_tools
    from emet.controller.dynamem import relative_motion
    from emet.utils.geometry import nav_xyt_to_world_xyt

    session = {"navigation_origin_xyt": [10, 20, math.pi / 2]}
    robot = SimpleNamespace(get_base_pose=lambda: np.zeros(3), get_emet_session=lambda: session)
    agent = SimpleNamespace(planner=None)
    if controller_transform:
        agent._planning_base_xyt = lambda pose: nav_xyt_to_world_xyt(pose, session)
        robot.get_emet_session = lambda: pytest.fail("must use controller's cached pose transform")

    def executor(cmds):
        pytest.fail("face_toward must not bypass validated rotation via the legacy executor")

    executor.agent = agent
    monkeypatch.setattr("emet.agent.face_toward.resolve_object_xy", lambda agent, label: (np.array([11, 20]), "voxel"))
    monkeypatch.setattr(relative_motion, "supports_relative_motion", lambda agent: True)
    calls = []

    def rotate(owner, name, degrees):
        assert owner is agent and name == "rotate_base"
        calls.append(degrees)
        return {
            "schema_version": 1,
            "tool": "rotate_base",
            "status": "error" if blocked else "ok",
            "code": "motion_rejected" if blocked else "ok",
            "message": "checked rotation",
            "data": {"execution_started": not blocked},
        }

    monkeypatch.setattr(relative_motion, "execute_relative_motion", rotate)
    context = {"robot": robot, "executor": executor}
    tools = {tool.name: tool for tool in get_tools(context)}
    ok, results, has_info = _dispatch_tool_calls(
        [{"name": "face_toward", "arguments": {"object_label": "microwave"}}], tools, executor
    )
    assert ok and has_info
    assert calls == pytest.approx([-90])
    if blocked:
        assert results[0].startswith("[face_toward] failed status=motion_rejected")
        assert "Measured turn reached" not in results[0]
        assert "last_rotate_degrees" not in context
    else:
        assert results[0].startswith("[face_toward] ok status=ok")
        assert context["last_rotate_degrees"] == pytest.approx(-90)
