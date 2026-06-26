# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Live Habitat compass ↔ move_forward alignment (requires habitat-sim)."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HABITAT_TESTS") != "1",
    reason="Set RUN_HABITAT_TESTS=1 and use .venv-habitat",
)


def test_compass_heading_matches_move_forward_live() -> None:
    from emet.habitat.config import default_hm3d_scene_dir
    from emet.habitat.datasets import get_question, load_hmeqa_questions, load_scene_init_poses
    from emet_habitat.robot_client import HabitatRobotClient
    from emet_habitat.simulator import HabitatEQASimulator

    questions = load_hmeqa_questions(None)
    q = get_question(questions, question_id=14)
    poses = load_scene_init_poses(None)
    init_pose = poses[(q.scene, q.floor)]
    sim = HabitatEQASimulator.from_scene_id(
        q.scene,
        hm3d_root=default_hm3d_scene_dir(),
        use_hm3d_semantics=True,
    )
    sim.set_init_pose(init_pose)
    robot = HabitatRobotClient(sim)
    for _ in range(4):
        before = np.asarray(robot.get_base_pose(), dtype=np.float64)
        sim.step("move_forward")
        after = np.asarray(robot.get_base_pose(), dtype=np.float64)
        motion = math.atan2(after[1] - before[1], after[0] - before[0])
        err = abs((before[2] - motion + math.pi) % (2 * math.pi) - math.pi)
        assert err < 0.05, f"compass={before[2]:.3f} motion={motion:.3f} err={err:.3f}"
