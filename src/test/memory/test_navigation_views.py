# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Moving sensor evidence survives independently of map/graph updates."""

from types import SimpleNamespace

import numpy as np
import pytest

from emet.memory.navigation_views import NavigationViewBuffer, bind_navigation_views, retain_navigation_observation


def observation(x, stamp, value=80, clock="sim"):
    pose = np.eye(4)
    pose[0, 3] = x
    return SimpleNamespace(
        rgb=np.full((8, 8, 3), value, dtype=np.uint8),
        camera_pose=pose,
        image_timing={"head_cam_left": {"timestamp_ns": stamp, "clock_domain": clock}},
        semantic="must not be retained",
        depth="must not be retained",
    )


def test_sampling_is_pose_based_bounded_and_owns_pixels_without_geometry():
    buffer = NavigationViewBuffer("episode", max_frames=3)
    robot = SimpleNamespace(navigation_view_buffer=buffer)
    source = observation(0, 10**9)
    retain_navigation_observation(robot, source)
    source.rgb[:] = 0
    assert np.all(buffer.frames()[0].rgb == 80)
    assert not hasattr(buffer.frames()[0], "depth") and not hasattr(buffer.frames()[0], "semantic")
    retain_navigation_observation(robot, observation(0.1, 2 * 10**9))
    assert len(buffer.frames()) == 1
    for index in range(1, 7):
        retain_navigation_observation(robot, observation(index, (index + 2) * 10**9, value=index))
    frames = buffer.frames()
    assert len(frames) == 3 and [frame.rgb[0, 0, 0] for frame in frames] == [4, 5, 6]
    assert all(frame.scope_id == "episode:sim" for frame in frames)
    with pytest.raises(ValueError):
        frames[-1].rgb[:] = 9


def test_decoded_wire_and_native_observations_share_the_producer_contract():
    source = observation(1, 10**9)
    first = SimpleNamespace(navigation_view_buffer=NavigationViewBuffer("episode"))
    second = SimpleNamespace(navigation_view_buffer=NavigationViewBuffer("episode"))
    retain_navigation_observation(first, vars(source))
    retain_navigation_observation(second, source)
    assert np.array_equal(first.navigation_view_buffer.frames()[0].rgb, second.navigation_view_buffer.frames()[0].rgb)
    assert first.navigation_view_buffer.frames()[0].timestamp_s == second.navigation_view_buffer.frames()[0].timestamp_s


@pytest.mark.parametrize("timing", [None, {}, {"head_cam_left": {"timestamp_ns": True, "clock_domain": "sim"}}])
def test_unknown_acquisition_is_not_replaced_with_receive_or_wall_time(timing):
    buffer = NavigationViewBuffer("episode")
    source = observation(0, 1)
    source.image_timing = timing
    retain_navigation_observation(SimpleNamespace(navigation_view_buffer=buffer), source)
    assert not buffer.frames() and buffer.summary()["skipped"] == {"acquisition_unavailable": 1}


def test_map_reset_and_explicit_ablation_drop_old_buffer():
    agent = SimpleNamespace(
        robot=SimpleNamespace(), voxel_map=SimpleNamespace(observation_scope_id="old"), parameters={"eqa": {}}
    )
    old = bind_navigation_views(agent)
    retain_navigation_observation(agent.robot, observation(0, 1))
    assert len(old.frames()) == 1
    assert bind_navigation_views(agent) is old
    agent.voxel_map.observation_scope_id = "new"
    assert not bind_navigation_views(agent).frames()
    agent.parameters["eqa"]["navigation_views"] = {"enabled": False}
    assert bind_navigation_views(agent) is None and agent.robot.navigation_view_buffer is None


def test_habitat_retains_the_already_acquired_step_frame_without_extra_render(monkeypatch):
    import emet_habitat.robot_client as module

    frame = SimpleNamespace(rgb=np.ones((8, 8, 3), dtype=np.uint8))
    robot = object.__new__(module.HabitatRobotClient)
    robot._sim = SimpleNamespace(step=lambda action: frame)
    robot._sync_pose_from_sim = lambda: None
    robot._post_step_hooks = []
    robot.navigation_view_buffer = NavigationViewBuffer("episode")
    source = observation(0, 1)
    captures = []

    def from_frame(actual):
        captures.append(actual)
        return source

    robot._observation_from_frame = from_frame
    assert robot._sim_step("move_forward") is frame
    assert captures == [frame] and len(robot.navigation_view_buffer.frames()) == 1
