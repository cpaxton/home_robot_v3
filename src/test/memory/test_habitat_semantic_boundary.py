# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("expose", [False, True])
def test_robot_observations_and_labeler_share_semantic_boundary(monkeypatch, expose):
    from emet_habitat import robot_client

    robot = object.__new__(robot_client.HabitatRobotClient)
    labels = object()
    semantic = object()
    frame = SimpleNamespace(rgb=None, depth=None, agent_state=None, intrinsics=None, semantic=semantic)
    robot._sim = SimpleNamespace(
        semantic_labeler=labels,
        uses_hm3d_semantics=True,
        get_frame=lambda: frame,
        floor_y=0,
        sensor_height=1,
        camera_tilt_deg=0,
    )
    robot._expose_semantics = expose
    monkeypatch.setattr(robot_client, "habitat_rgb_depth_to_observations", lambda **kwargs: kwargs)
    assert robot.hm3d_semantic_labeler is (labels if expose else None)
    assert robot.uses_hm3d_semantics is expose
    assert robot.get_observation()["semantic"] is (semantic if expose else None)
