# Copyright (c) Hello Robot, Inc. All rights reserved.

import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco

from emet.motion import constants as motion_constants
from emet.motion.kinematics import HelloStretchIdx
from emet.robots.stretch import STRETCH_ROBOCASA_MJCF_JOINT_NAMES, StretchBackend
from emet.simulation.head_look_action import apply_head_to_robosuite, apply_stretch_posture_to_robosuite


def test_apply_stretch_navigation_posture_sets_lift():
    spec = StretchBackend().get_robosuite_robocasa_spec()
    mjcf = spec.mjcf_path
    if mjcf is None:
        pytest.skip("stretch MJCF missing")
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    apply_stretch_posture_to_robosuite(spec, model, data, "navigation")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_lift")
    qadr = int(model.jnt_qposadr[jid])
    assert abs(float(data.qpos[qadr]) - float(motion_constants.STRETCH_NAVIGATION_Q[HelloStretchIdx.LIFT])) < 0.05


def test_stretch_head_to_holds_tilt_after_physics_step():
    """head_to must set actuator ctrl, not only qpos (Robocasa spec has actuator_names=[])."""
    spec = StretchBackend().get_robosuite_robocasa_spec()
    mjcf = spec.mjcf_path
    if mjcf is None:
        pytest.skip("stretch MJCF missing")
    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    tilt = float(motion_constants.look_front[1])
    apply_head_to_robosuite(spec, model, data, 0.0, tilt)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_tilt")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_head_tilt")
    qadr = int(model.jnt_qposadr[jid])
    for _ in range(20):
        mujoco.mj_step(model, data)
    assert abs(float(data.qpos[qadr]) - tilt) < 0.08
    assert abs(float(data.ctrl[aid]) - tilt) < 0.08


def test_update_servo_base_pose_no_numpy_truthiness():
    from unittest.mock import MagicMock, patch

    from emet.controller.zmq_client import StretchZmqClient

    client = StretchZmqClient.__new__(StretchZmqClient)
    client._state = {"base_pose": np.array([1.0, 2.0, 0.5])}
    client._servo_lock = __import__("threading").Lock()
    client._state_lock = client._servo_lock
    client._servo = None
    msg = {
        "head_color_image": b"\xff\xd8\xff\xd9",  # minimal jpeg-ish; patched below
        "head_depth_image": None,
        "joint_positions": np.zeros(len(STRETCH_ROBOCASA_MJCF_JOINT_NAMES)),
        "base_pose": np.array([3.0, 4.0, 1.0]),
    }
    with (
        patch("emet.controller.zmq_client.compression.from_jpg", return_value=np.zeros((4, 4, 3), dtype=np.uint8)),
        patch.object(client, "_note_emet_session_from_zmq_dict"),
        patch.object(
            client,
            "_coerce_stretch_joint_vector",
            side_effect=lambda jp, base_xyt=None: np.zeros(HelloStretchIdx.HEAD_TILT + 1),
        ) as coerce,
    ):
        client.update_servo(msg)
    assert coerce.called
    assert float(coerce.call_args.kwargs["base_xyt"][0]) == 3.0
