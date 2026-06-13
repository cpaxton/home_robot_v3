# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.

import numpy as np

from emet.utils.geometry import nav_xyt_to_world_xyt, xyt_global_to_base


def test_nav_xyt_identity_at_origin_with_session():
    origin = np.array([3.0, -1.5, 0.4], dtype=np.float64)
    session = {"navigation_origin_xyt": origin.tolist()}
    local = np.zeros(3, dtype=np.float64)
    world = nav_xyt_to_world_xyt(local, session)
    np.testing.assert_allclose(world, origin, atol=1e-6)


def test_nav_xyt_roundtrip_with_global_to_base():
    origin = np.array([2.0, 4.0, 0.1], dtype=np.float64)
    current_world = np.array([2.5, 4.25, 0.2], dtype=np.float64)
    local = xyt_global_to_base(current_world, origin)
    session = {"navigation_origin_xyt": origin.tolist()}
    back = nav_xyt_to_world_xyt(local, session)
    np.testing.assert_allclose(back, current_world, atol=1e-5)


def test_nav_xyt_no_session_returns_input():
    xyt = np.array([1.0, 2.0, 0.3], dtype=np.float64)
    np.testing.assert_allclose(nav_xyt_to_world_xyt(xyt, None), xyt)


def test_nav_xyt_session_without_origin_returns_input():
    xyt = np.array([1.0, 2.0, 0.3], dtype=np.float64)
    np.testing.assert_allclose(nav_xyt_to_world_xyt(xyt, {"capabilities": {}}), xyt)


def test_zmq_frame_contract_accepts_nav_compose_obs():
    from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
    from emet.simulation.zmq_frame_contract import assert_zmq_observation_frames_consistent

    origin = np.array([2.5, -1.0, 0.1])
    cam = np.eye(4)
    cam[:3, 3] = [2.55, -0.95, 1.05]
    msg = {
        "gps": np.array([0.0, 0.0]),
        "compass": np.array([0.0]),
        "camera_pose": cam,
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": origin.tolist()},
    }
    assert_zmq_observation_frames_consistent(msg)
