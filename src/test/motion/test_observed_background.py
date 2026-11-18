# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Robot endpoints cannot manufacture free space or renew hidden evidence."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_collision import ObservedNativeCollisionChecker
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.motion.arm_rrt import plan_arm_joint_path
from emet.motion.observed_background import ObservedBackground, StaticBackgroundAssumption
from emet.motion.observed_workspace import ObservedWorkspace


def partition(states, points=(), labels=()):
    grid = ObservedWorkspace(np.zeros(3), 0.1, np.array(states, np.uint8).reshape(-1, 1, 1))
    return grid.partition(np.array(points, float).reshape(-1, 3), np.array(labels, dtype=str))


def initial():
    return ObservedBackground.from_capture(
        partition([1, 1, 0, 2], [[0.35, 0.05, 0.05]], ["wall"]),
        excluded_refs=("robot",),
        scene_epoch="scene",
        capture=stamp("scan", 10.0),
        now=10.1,
    )


def stamp(name, t):
    return CaptureInterval(name, t, t + 0.001, t + 0.01, "clock")


def test_self_occlusion_preserves_only_real_historical_free_space_and_original_age():
    background = initial()
    current = partition([2, 1, 2, 0], [[0.05, 0.05, 0.05], [0.25, 0.05, 0.05]], ["robot", "robot"])
    updated = background.update(
        current, excluded_refs=("robot",), scene_epoch="scene", capture=stamp("next", 11.0), now=11.1
    )
    np.testing.assert_array_equal(updated.workspace.cells[:, 0, 0], [1, 1, 0, 2])
    np.testing.assert_allclose(updated.observed_monotonic[[0, 1, 3], 0, 0], [10.0, 11.0, 10.0])
    box = [[0.01, 0.01, 0.01], [0.09, 0.09, 0.09]]
    assert updated.known_free(box, now=11.5)
    assert not updated.known_free(box, now=12.5)
    assumption = StaticBackgroundAssumption("scene", 10.0, 20.0, "explicit-static-workspace")
    assert updated.known_free(box, now=12.5, static_assumption=assumption)
    assert not updated.known_free(box, now=20.1, static_assumption=assumption)
    assert not updated.known_free([[0.21, 0.01, 0.01], [0.29, 0.09, 0.09]], now=12.0, static_assumption=assumption)
    # Visibility refresh changes age but not geometry; robot masking never frees
    # its previously unknown cell or removes the cached background wall.
    assert updated.geometry_revision == background.geometry_revision
    with pytest.raises(ValueError):
        updated.observed_monotonic.setflags(write=True)


def test_mixed_robot_obstacle_cell_invalidates_static_epoch_instead_of_erasing_wall():
    points = [[0.05, 0.05, 0.05]] * 2
    current = partition([2, 0, 0, 0], points, ["robot", "obstacle"])
    with pytest.raises(ObservationUnavailable, match="scene_changed"):
        initial().update(current, excluded_refs=("robot",), scene_epoch="scene", capture=stamp("next", 11.0), now=11.1)


@pytest.mark.parametrize("change", ["replay", "epoch", "reverse_time", "cleared_obstacle"])
def test_stale_or_contradictory_background_cannot_be_published(change):
    current = partition([0, 0, 0, 1 if change == "cleared_obstacle" else 0])
    args = {"excluded_refs": ("robot",), "scene_epoch": "scene", "capture": stamp("next", 11.0), "now": 11.1}
    if change == "replay":
        args["capture"] = stamp("scan", 11.0)
    if change == "epoch":
        args["scene_epoch"] = "other"
    if change == "reverse_time":
        args["capture"] = stamp("next", 9.0)
    with pytest.raises(ObservationUnavailable):
        initial().update(current, **args)


def test_native_path_uses_previously_measured_free_space_without_freeing_unseen_space():
    # A workspace scan before the robot entered supplies actual depth evidence.
    # It is not inferred from the robot's present AABB or from missing endpoints.
    xml = """<mujoco><worldbody><body name="robot" pos="0 0 .5">
      <joint name="x" type="slide" axis="1 0 0" range="0 .4"/>
      <geom type="box" size=".08 .08 .06"/>
      <body name="ee" pos=".1 0 0"><geom size=".015"/></body></body>
      </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    sensor = mujoco.MjModel.from_xml_string(
        xml.replace("</worldbody>", '<geom name="wall" type="box" pos="0 0 1.4" size="1 1 .02"/></worldbody>')
    )
    measured = mujoco.MjData(sensor)
    intrinsic = np.array([[20.0, 0, 10], [0, 20.0, 10], [0, 0, 1]])
    bounds = [[-0.4, -0.4, 0.1], [0.6, 0.4, 1.5]]

    def capture(t, x):
        measured.qpos[0] = x  # Sensor fixture only; no controller execution.
        mujoco.mj_forward(sensor, measured)
        depth, points, labels = np.zeros((21, 21)), [], []
        for y, u in np.ndindex(depth.shape):
            ray = np.array([(u - 10) / 20, (y - 10) / 20, 1.0])
            ray /= np.linalg.norm(ray)
            hit = np.full(1, -1, np.int32)
            distance = mujoco.mj_ray(sensor, measured, np.zeros(3), ray, None, True, -1, hit)
            assert distance > 0
            depth[y, u] = distance * ray[2]
            points.append(distance * ray)
            labels.append("background" if hit[0] == sensor.geom("wall").id else "robot")
        grid = ObservedWorkspace.from_depth(
            depth, intrinsic, np.eye(4), workspace=bounds, resolution=0.05, depth_margin_m=0.005
        )
        return grid.partition(np.array(points), np.array(labels)), stamp(str(t), t)

    prior, ticket = capture(10.0, 2.0)
    background = ObservedBackground.from_capture(
        prior, excluded_refs=("robot",), scene_epoch="static", capture=ticket, now=10.1
    )
    current, ticket = capture(11.0, 0.0)
    background = background.update(current, excluded_refs=("robot",), scene_epoch="static", capture=ticket, now=11.1)
    clock = [11.1]
    checker = ObservedNativeCollisionChecker(
        model,
        robot_body="robot",
        ee_body="ee",
        scene=background.placement_scene(clock=lambda: clock[0]),
        payload_provider=SimpleNamespace(input_mode="observed", observe_payload=lambda _: None),
        scene_revision=background.geometry_revision,
        current_revision=lambda: background.geometry_revision,
        received_monotonic=11.01,
        clock=lambda: clock[0],
        margin_m=0.0,
    )
    mujoco.mj_forward(model, data)

    def plan(goal):
        return plan_arm_joint_path(
            model,
            data,
            joint_names=["x"],
            q_start=[0],
            q_goal=[goal],
            collision=checker,
            planner="linear",
            step_size=0.01,
        )

    assert plan(0.05).success
    clock[0] = 12.1
    assert plan(0.05).reason == "invalid_start"  # Hidden scan evidence has expired.
    assumption = StaticBackgroundAssumption("static", 10.0, 20.0, "declared-static-scan")
    checker.scene = background.placement_scene(static_assumption=assumption, clock=lambda: clock[0])
    assert plan(0.05).success
    assert not plan(0.35).success  # Beyond the old camera frustum remains unknown.
