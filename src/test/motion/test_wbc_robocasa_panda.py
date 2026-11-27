# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""RoboCasa Panda task-target plumbing tests (no scene construction)."""

from __future__ import annotations

import numpy as np
import pytest

from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend


def _backend() -> RoboCasaPandaBackend:
    backend = object.__new__(RoboCasaPandaBackend)
    backend._origin_xy = np.array([1.0, 2.0], dtype=np.float64)
    backend._origin_yaw = np.pi / 2
    backend._target_base = None
    backend._teacher_phase = "arm"
    backend._gripper_target = np.array([0.04, -0.04])
    backend._force_gripper = False
    return backend


def test_base_pose_uses_the_reset_anchored_yaw():
    backend = _backend()
    backend._world_base_pose = lambda: (1.0, 3.0, np.pi)
    pose = backend._base_pose()
    np.testing.assert_allclose(pose[:2], [1.0, 0.0], atol=1e-9)
    assert pose[2] == pytest.approx(np.pi / 2)


def test_world_base_target_is_converted_to_reset_frame():
    backend = _backend()
    backend.set_world_base_target((1.0, 4.0, np.pi))
    assert backend._target_base is not None
    np.testing.assert_allclose(backend._target_base[:2], [2.0, 0.0], atol=1e-9)
    assert backend._target_base[2] == pytest.approx(np.pi / 2)
    assert backend._teacher_phase == "base"


def test_local_base_target_round_trips_and_clears():
    backend = _backend()
    backend.set_base_target((0.5, -0.2, 0.3))
    assert backend._target_base == (0.5, -0.2, 0.3)
    backend.set_base_target(None)
    assert backend._target_base is None
    assert backend._teacher_phase == "arm"


def test_scripted_gripper_target_is_forced():
    backend = _backend()
    backend.set_gripper(False)
    assert backend._force_gripper is True
    np.testing.assert_allclose(backend._gripper_target, [0.0, 0.0])
    backend.set_gripper(True)
    np.testing.assert_allclose(backend._gripper_target, [0.04, -0.04])


def test_body_twist_joint_velocity_round_trip_with_rotated_yaw_and_lever():
    from pathlib import Path

    import mujoco
    import robosuite

    xml_path = Path(robosuite.__file__).resolve().parent / "models/assets/bases/omron_mobile_base.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    backend = _backend()
    backend.model = model
    backend.data = data
    backend._base_jnt = np.asarray(
        [model.joint(n).id for n in ("joint_mobile_forward", "joint_mobile_side", "joint_mobile_yaw")], dtype=int
    )
    backend._base_q = np.asarray([model.jnt_qposadr[j] for j in backend._base_jnt], dtype=int)
    backend._base_v = np.asarray([model.jnt_dofadr[j] for j in backend._base_jnt], dtype=int)
    backend._base_body = int(model.body("base").id)
    data.qpos[backend._base_q] = [0.4, -0.3, 2.1]
    mujoco.mj_forward(model, data)

    desired = (0.18, -0.07, 0.35)
    qdot = backend._body_twist_to_joint_vel(*desired)
    data.qvel[backend._base_v] = qdot
    recovered = backend._joint_vel_to_body_twist()
    np.testing.assert_allclose(recovered, desired, atol=1e-9)


def test_reanchor_origin_preserves_world_ee_target():
    backend = _backend()
    backend.model = object()
    backend.data = object()
    backend._ee_in_base_T = None
    backend._target_q = None
    backend.set_ee_target((2.0, 3.5, 1.1, 0.0, 0.0, 0.0, 1.0))
    before_pos, before_rot = backend._world_pose(backend._target_ee)
    backend._world_base_pose = lambda: (1.8, 2.4, -0.7)
    backend.reanchor_origin()
    np.testing.assert_allclose(backend._base_pose(), [0.0, 0.0, 0.0], atol=1e-9)
    after_pos, after_rot = backend._world_pose(backend._target_ee)
    np.testing.assert_allclose(after_pos, before_pos, atol=1e-9)
    np.testing.assert_allclose(after_rot.as_matrix(), before_rot.as_matrix(), atol=1e-9)


