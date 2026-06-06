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
