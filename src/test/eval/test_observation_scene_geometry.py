# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU MuJoCo geometry controls, not a robot controller or VLM qualification."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.memory.vlm_region_grounding import region_depth_mask
from emet.motion.navigation_sweep import validate_navigation_sweep


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).parents[1] / "fixtures/agent_observation_scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def target_ray(scene, camera_x, *, include_target=True):
    model, data = scene
    groups = np.zeros(6, dtype=np.uint8)
    groups[1] = include_target
    hit = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, np.array([camera_x, 0, 0.8]), np.array([1.0, 0, 0]), groups, True, -1, hit)
    return distance, hit[0]


def test_visible_far_target_requires_closer_observation_geometry(scene):
    distance, hit = target_ray(scene, 0)
    assert hit == scene[0].geom("target").id
    assert distance == pytest.approx(3.4)
    # Uniform support represents this ray's measured planar depth, not a
    # rendered image or evidence fed to an actual model.
    with pytest.raises(ValueError):
        region_depth_mask(np.full((40, 40), distance), [0, 0, 1000, 1000], [500, 500], min_depth=0.25, max_depth=2.5)
    closer, hit = target_ray(scene, 1.2)
    assert closer == pytest.approx(2.2)
    assert np.any(
        region_depth_mask(np.full((40, 40), closer), [0, 0, 1000, 1000], [500, 500], min_depth=0.25, max_depth=2.5) == 0
    )


def test_target_absent_sensor_control(scene):
    # Excluding the target from the sensor group isolates the no-return case.
    # It is a sensor control, not a test of language-model absence judgment.
    distance, hit = target_ray(scene, 0, include_target=False)
    assert distance < 0 and hit == -1


@pytest.mark.parametrize("lane_y,reachable", [(0.0, False), (1.0, True)])
def test_obstructed_and_clear_routes_have_distinct_collision_truth(scene, lane_y, reachable):
    model, data = scene
    obstacle, footprint = model.geom("obstacle").id, model.geom("footprint").id
    space = SimpleNamespace(last_validity={})

    def valid(pose):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        collision = any({contact.geom1, contact.geom2} == {obstacle, footprint} for contact in data.contact)
        space.last_validity = {"reason": "occupied_footprint" if collision else "ok"}
        return not collision

    space.is_valid = valid
    assert validate_navigation_sweep(space, [0, lane_y, 0], [[2.5, lane_y, 0]])[0] is reachable
