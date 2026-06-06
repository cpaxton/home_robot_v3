# Copyright (c) Hello Robot, Inc.
# All rights reserved.

import numpy as np

from emet.motion.kinematics import HelloStretchIdx
from emet.robots.stretch.joint_layout import hello_stretch_config_from_joint_positions


def test_robocasa_mjcf_to_hello_stretch_config():
    q = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    base = np.array([1.0, 2.0, 0.5])
    out = hello_stretch_config_from_joint_positions(q, base_xyt=base)

    assert out.shape == (HelloStretchIdx.HEAD_TILT + 1,)
    assert out[HelloStretchIdx.BASE_X] == 1.0
    assert out[HelloStretchIdx.BASE_Y] == 2.0
    assert out[HelloStretchIdx.BASE_THETA] == 0.5
    assert out[HelloStretchIdx.LIFT] == 0.5
    assert out[HelloStretchIdx.ARM] == 0.4
    assert out[HelloStretchIdx.WRIST_YAW] == 0.2
    assert out[HelloStretchIdx.HEAD_TILT] == 0.6


def test_hello_stretch_config_passthrough():
    q = np.arange(11, dtype=np.float64)
    out = hello_stretch_config_from_joint_positions(q)
    np.testing.assert_array_equal(out, q)


def test_robocasa_mjcf_roundtrip():
    from emet.robots.stretch.joint_layout import robocasa_mjcf_joint_positions_from_hello_stretch

    q10 = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    hello = hello_stretch_config_from_joint_positions(q10, base_xyt=np.array([1.0, 2.0, 0.5]))
    back = robocasa_mjcf_joint_positions_from_hello_stretch(hello)
    np.testing.assert_allclose(back, q10, rtol=1e-9)
