# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Live checks for Habitat agent yaw vs sensor pitch (requires habitat-sim)."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HABITAT_TESTS") != "1",
    reason="Set RUN_HABITAT_TESTS=1 and use .venv-habitat",
)


def _view_forward(sensor_state) -> np.ndarray:
    from emet.habitat.coordinates import _rotation_matrix_from_agent_rotation

    rot = _rotation_matrix_from_agent_rotation(sensor_state.rotation)
    return rot @ np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _planar_yaw(forward: np.ndarray) -> float:
    f = np.asarray(forward, dtype=np.float64).reshape(3)
    return float(math.atan2(-f[2], -f[0]))


def test_body_up_stays_world_vertical_with_sensor_pitch() -> None:
    from emet.habitat.config import default_hm3d_scene_dir
    from emet.habitat.coordinates import _rotation_matrix_from_agent_rotation
    from emet.habitat.datasets import get_question, load_hmeqa_questions, load_scene_init_poses
    from emet_habitat.simulator import HabitatEQASimulator

    q = get_question(load_hmeqa_questions(None), question_id=17)
    init_pose = load_scene_init_poses(None)[(q.scene, q.floor)]
    sim = HabitatEQASimulator.from_scene_id(
        q.scene,
        hm3d_root=default_hm3d_scene_dir(),
        use_hm3d_semantics=False,
    )
    sim.set_init_pose(init_pose)
    rot = _rotation_matrix_from_agent_rotation(sim._agent.get_state().rotation)
    up = rot @ np.array([0.0, 1.0, 0.0])
    assert up[1] == pytest.approx(1.0, abs=1e-4)
    assert np.linalg.norm(up[[0, 2]]) < 1e-4


def test_turn_left_changes_planar_yaw_by_action_amount() -> None:
    from emet.habitat.config import default_hm3d_scene_dir
    from emet.habitat.datasets import get_question, load_hmeqa_questions, load_scene_init_poses
    from emet_habitat.simulator import HabitatEQASimulator

    q = get_question(load_hmeqa_questions(None), question_id=17)
    init_pose = load_scene_init_poses(None)[(q.scene, q.floor)]
    sim = HabitatEQASimulator.from_scene_id(
        q.scene,
        hm3d_root=default_hm3d_scene_dir(),
        use_hm3d_semantics=False,
    )
    sim.set_init_pose(init_pose)
    before = sim._agent.get_state().sensor_states["color_sensor"]
    f0 = _view_forward(before)
    y0 = _planar_yaw(f0)
    sim.step("turn_left")
    after = sim._agent.get_state().sensor_states["color_sensor"]
    f1 = _view_forward(after)
    y1 = _planar_yaw(f1)
    delta = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
    assert math.degrees(abs(delta)) == pytest.approx(10.0, abs=0.2)
