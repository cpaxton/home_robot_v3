# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""KinematicPickPlace must sync the MJCF freejoint in MuJoCo world, not episode GPS."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np


def test_world_base_xyt_composes_navigation_origin():
    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
    from emet.utils.geometry import xyt_base_to_global

    origin = np.array([1.5, 1.6, 0.1], dtype=np.float64)
    episode = np.array([-1.2, -1.8, -np.pi / 2], dtype=np.float64)
    expected = xyt_base_to_global(episode, origin)

    robot = MagicMock()
    robot.get_base_pose.return_value = episode
    robot.get_emet_session.return_value = {"navigation_origin_xyt": origin.tolist()}
    robot._state = {"base_xyz": [float(expected[0]), float(expected[1]), 0.42]}

    exe = object.__new__(KinematicPickPlaceExecutor)
    exe.robot = robot
    world = exe._world_base_xyt()
    assert world is not None
    np.testing.assert_allclose(world[:2], expected[:2], atol=1e-6)
    np.testing.assert_allclose(world[2], expected[2], atol=1e-6)


def test_world_base_xyt_prefers_base_xyz_xy():
    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

    robot = MagicMock()
    robot.get_base_pose.return_value = np.array([0.0, 0.0, -1.57])
    robot.get_emet_session.return_value = {
        "navigation_origin_xyt": [1.0, 2.0, 0.0],
    }
    # World XY from MuJoCo body (authoritative); yaw still from composed GPS.
    robot._state = {"base_xyz": [0.273, -0.203, 0.55]}

    exe = object.__new__(KinematicPickPlaceExecutor)
    exe.robot = robot
    world = exe._world_base_xyt()
    assert world is not None
    np.testing.assert_allclose(world[:2], [0.273, -0.203], atol=1e-6)


def test_write_offline_base_xyt_sourccey_planar():
    import mujoco

    from emet.controller.manipulation.kinematic_pick_place import write_offline_mjcf_base_xyt
    from emet.robots.sourccey import SourcceyBackend

    spec = SourcceyBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    xyt = np.array([0.4, -0.25, 0.7], dtype=np.float64)
    assert write_offline_mjcf_base_xyt(
        model, data, xyt, planar_joint_names=spec.planar_base_joint_names, freejoint_name="base_freejoint"
    )
    mujoco.mj_forward(model, data)
    for jn, val in zip(spec.planar_base_joint_names, xyt, strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert abs(float(data.qpos[int(model.jnt_qposadr[jid])]) - float(val)) < 1e-9


def test_write_offline_base_xyt_xlerobot_spec_names():
    import mujoco

    from emet.controller.manipulation.kinematic_pick_place import write_offline_mjcf_base_xyt
    from emet.robots.xlerobot import XLeRobotBackend

    spec = XLeRobotBackend().get_spec()
    assert spec.planar_base_joint_names == ("slide_joint_x", "slide_joint_y", "hinge_joint_z")
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    xyt = np.array([0.3, 0.15, -0.4], dtype=np.float64)
    assert write_offline_mjcf_base_xyt(
        model, data, xyt, planar_joint_names=spec.planar_base_joint_names, freejoint_name="base_freejoint"
    )
    mujoco.mj_forward(model, data)
    for jn, val in zip(spec.planar_base_joint_names, xyt, strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert abs(float(data.qpos[int(model.jnt_qposadr[jid])]) - float(val)) < 1e-9


def test_sync_executor_base_to_xyt_uses_spec_planar_names():
    import mujoco

    from emet.controller.task.tamp.task_search import _sync_executor_base_to_xyt
    from emet.motion.arm_manip_profile import ArmManipProfile
    from emet.robots.sourccey import SourcceyBackend

    spec = SourcceyBackend().get_spec()
    arm_profile = ArmManipProfile.for_robot("sourccey", arm="left")
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)

    class _Robot:
        _spec = spec

    class _Exe:
        pass

    exe = _Exe()
    exe._model = model
    exe._data = data
    exe.profile = arm_profile
    exe.robot = _Robot()
    exe.joint_names = arm_profile.joint_names

    xyt = np.array([-0.8, 1.1, 0.25], dtype=np.float64)
    _sync_executor_base_to_xyt(exe, xyt)
    for jn, val in zip(spec.planar_base_joint_names, xyt, strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert abs(float(data.qpos[int(model.jnt_qposadr[jid])]) - float(val)) < 1e-6


def test_sync_executor_home_seed_matches_sourccey_act_suffix():
    from types import SimpleNamespace

    import mujoco

    from emet.controller.task.tamp.task_search import _sync_executor_base_to_xyt
    from emet.motion.arm_manip_profile import ArmManipProfile
    from emet.robots.sourccey import SourcceyBackend

    spec = SourcceyBackend().get_spec()
    arm_profile = ArmManipProfile.for_robot("sourccey", arm="left")
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    home = tuple(0.11 + 0.01 * i for i in range(len(arm_profile.actuator_names)))

    class _Robot:
        _spec = spec

    exe = SimpleNamespace(
        _model=model,
        _data=data,
        profile=SimpleNamespace(
            home_cmd=home,
            actuator_names=arm_profile.actuator_names,
            base_freejoint_name="base_freejoint",
        ),
        robot=_Robot(),
        joint_names=arm_profile.joint_names,
    )
    _sync_executor_base_to_xyt(exe, np.zeros(3))
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pan")
    assert abs(float(data.qpos[int(model.jnt_qposadr[jid])]) - home[0]) < 1e-9


def test_sync_base_freejoint_writes_sourccey_planar():
    import mujoco

    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
    from emet.motion.arm_manip_profile import ArmManipProfile
    from emet.robots.sourccey import SourcceyBackend

    spec = SourcceyBackend().get_spec()
    profile = ArmManipProfile.for_robot("sourccey", arm="right")
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    robot = MagicMock()
    robot._spec = spec
    robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    robot.get_emet_session.return_value = {}
    robot._state = {"base_xyz": [1.25, -0.5, 0.1]}

    exe = object.__new__(KinematicPickPlaceExecutor)
    exe.robot = robot
    exe.profile = profile
    exe._model = model
    exe._data = data
    exe._sync_base_freejoint()
    for jn, val in zip(spec.planar_base_joint_names, (1.25, -0.5, 0.0), strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert abs(float(data.qpos[int(model.jnt_qposadr[jid])]) - float(val)) < 1e-6


def test_approach_xy_side_matches_plan_standoff():
    """Place re-approach must use +Y standoff, not radial-from-current, for Sourccey."""
    from types import SimpleNamespace

    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
    from emet.controller.task.tamp.task_search import approach_pose_for_object_xy

    moved: list[np.ndarray] = []
    robot = SimpleNamespace(
        _spec=SimpleNamespace(tamp_approach="side"),
        move_base_to=lambda xyt, blocking=True, world_frame=True: moved.append(
            np.asarray(xyt, dtype=np.float64).copy()
        ),
    )
    exe = object.__new__(KinematicPickPlaceExecutor)
    exe.robot = robot
    exe.arm = "left"
    exe._sleep = lambda _s: None
    recep = np.array([-0.02, -0.55], dtype=np.float64)
    exe._approach_xy(recep, standoff_m=0.55)
    expected = approach_pose_for_object_xy(recep, standoff=0.55, mode="side", arm="left")
    assert len(moved) == 1
    np.testing.assert_allclose(moved[0], expected, atol=1e-9)
    # Radial-from-[0.08,0] would land near [0.079, -0.009], not the canonical side pose.
    assert abs(moved[0][1] - 0.0) < 1e-9


def test_approach_xy_front_keeps_radial_standoff():
    from types import SimpleNamespace

    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

    moved: list[np.ndarray] = []
    robot = SimpleNamespace(
        _spec=SimpleNamespace(tamp_approach="front"),
        move_base_to=lambda xyt, blocking=True, world_frame=True: moved.append(
            np.asarray(xyt, dtype=np.float64).copy()
        ),
    )
    exe = object.__new__(KinematicPickPlaceExecutor)
    exe.robot = robot
    exe.arm = "left"
    exe._sleep = lambda _s: None
    exe._world_base_xyt = lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64)
    exe._approach_xy(np.array([0.0, 0.0], dtype=np.float64), standoff_m=0.55)
    assert len(moved) == 1
    np.testing.assert_allclose(moved[0][:2], [0.55, 0.0], atol=1e-9)
