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


def test_write_offline_base_xyt_merged_base_uses_fk():
    import mujoco

    from emet.controller.manipulation.kinematic_pick_place import write_offline_mjcf_base_xyt

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="scene" pos="10 0 0" euler="0 0 1.0">
              <body name="base">
                <joint name="base_x" type="slide" axis="1 0 0"/>
                <joint name="base_y" type="slide" axis="0 1 0"/>
                <joint name="base_yaw" type="hinge" axis="0 0 1"/>
                <geom name="base_geom" type="box" size="0.1 0.1 0.1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    target = np.array([1.0, 2.0, 0.3], dtype=np.float64)
    assert write_offline_mjcf_base_xyt(
        model,
        data,
        target,
        planar_joint_names=("base_x", "base_y", "base_yaw"),
        base_body_name="base",
    )
    mujoco.mj_forward(model, data)
    body = model.body("base").id
    np.testing.assert_allclose(data.xpos[body][:2], target[:2], atol=1e-6)
    yaw = np.arctan2(data.xmat[body].reshape(3, 3)[1, 0], data.xmat[body].reshape(3, 3)[0, 0])
    assert abs(yaw - target[2]) < 1e-6


def test_write_offline_base_xyt_planar_failure_falls_back_to_freejoint():
    import mujoco

    from emet.controller.manipulation.kinematic_pick_place import write_offline_mjcf_base_xyt

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="planar">
              <joint name="base_x" type="slide" axis="1 0 0"/>
              <joint name="base_y" type="slide" axis="0 1 0"/>
              <joint name="base_yaw" type="hinge" axis="0 0 1"/>
              <geom name="planar_geom" type="box" size="0.1 0.1 0.1"/>
            </body>
            <body name="free" pos="0 5 0">
              <freejoint name="base_freejoint"/>
              <geom name="free_geom" type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    target = np.array([0.4, -0.2, 0.1], dtype=np.float64)
    # The missing base body makes write_planar_base_xyt fail; the freejoint
    # fallback must still be attempted.
    assert write_offline_mjcf_base_xyt(
        model,
        data,
        target,
        planar_joint_names=("base_x", "base_y", "base_yaw"),
        freejoint_name="base_freejoint",
        base_body_name="missing_base",
    )
    qadr = int(model.joint("base_freejoint").qposadr[0])
    np.testing.assert_allclose(data.qpos[qadr : qadr + 2], target[:2], atol=1e-9)


def test_sync_base_freejoint_standalone_uses_raw_qpos():
    import mujoco

    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
    from emet.robots.sourccey import SourcceyBackend

    spec = SourcceyBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)

    class _Robot:
        _spec = spec
        _state = {}

    executor = object.__new__(KinematicPickPlaceExecutor)
    executor.robot = _Robot()
    executor._model = model
    executor._data = data
    executor._scene_model_loaded = False
    executor.profile = type("Profile", (), {"base_freejoint_name": "base_freejoint"})()
    executor._world_base_xyt = lambda: np.array([0.4, -0.25, 0.7])
    executor._sync_base_freejoint()
    for name, value in zip(spec.planar_base_joint_names, (0.4, -0.25, 0.7), strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert abs(float(data.qpos[int(model.jnt_qposadr[jid])]) - value) < 1e-9


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


def test_placement_ring_side_yaw():
    from emet.motion.placement import placement_base_candidates

    poses = placement_base_candidates([0, 0], current_xyt=[2, 0, 0], radii=(.55,), count=4, yaw_offset=np.pi / 2)
    above = next(p for p in poses[1:] if np.allclose(p[:2], [0, .55]))
    assert abs(above[2]) < 1e-9


def test_placement_ring_front_yaw():
    from emet.motion.placement import placement_base_candidates

    poses = placement_base_candidates([0, 0], current_xyt=[2, 0, 0], radii=(.55,), count=4)
    for pose in poses[1:]:
        np.testing.assert_allclose([np.cos(pose[2]), np.sin(pose[2])], -pose[:2] / .55, atol=1e-9)
