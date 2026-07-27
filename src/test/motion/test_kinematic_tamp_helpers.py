# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import numpy as np

from emet.core.zmq_protocol import (
    EMET_ACTION_SIM_ATTACH_BODY_KEY,
    build_sim_attach_body_action,
    build_sim_detach_body_action,
)
from emet.motion.mujoco_arm_ik import (
    RBY1_LEFT_ARM_JOINTS,
    actuator_vector_from_dict,
    interpolate_arm_waypoints,
    pack_arm_into_actuator_dict,
)
from emet.motion.voxel_arm_collision import link_samples_collide_2d
from emet.robots.galaxea_r1 import R1_ACTUATOR_NAMES


def test_pack_arm_into_actuator_dict_rby1():
    q = np.linspace(0.1, 0.6, 6)
    d = pack_arm_into_actuator_dict(R1_ACTUATOR_NAMES, RBY1_LEFT_ARM_JOINTS, q)
    assert d["left_arm1"] == 0.1
    assert d["left_arm6"] == 0.6
    vec = actuator_vector_from_dict(R1_ACTUATOR_NAMES, d, fill=0.0)
    assert len(vec) == len(R1_ACTUATOR_NAMES)
    assert vec[R1_ACTUATOR_NAMES.index("left_arm1")] == 0.1


def test_interpolate_arm_waypoints():
    path = interpolate_arm_waypoints(np.zeros(6), np.ones(6), n_steps=4)
    assert len(path) == 5
    assert np.allclose(path[0], 0.0)
    assert np.allclose(path[-1], 1.0)


def test_link_samples_collide_2d():
    obs = np.zeros((10, 10), dtype=bool)
    obs[5, 5] = True
    origin = np.array([0.0, 0.0])
    # cell (5,5) covers world [0.5, 0.6) if resolution=0.1
    assert link_samples_collide_2d(obs, grid_origin=origin, resolution=0.1, sample_xy=[(0.55, 0.55)], inflate_cells=0)
    assert not link_samples_collide_2d(obs, grid_origin=origin, resolution=0.1, sample_xy=[(0.1, 0.1)], inflate_cells=0)


def test_sim_attach_detach_zmq_actions():
    act = build_sim_attach_body_action(2, "bowl_x", "left_arm_link6")
    assert act[EMET_ACTION_SIM_ATTACH_BODY_KEY]["body"] == "bowl_x"
    assert act[EMET_ACTION_SIM_ATTACH_BODY_KEY]["ee_body"] == "left_arm_link6"
    det = build_sim_detach_body_action(3, "bowl_x")
    assert det["sim_detach_body"]["body"] == "bowl_x"
