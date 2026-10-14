# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Independent goal scoring must reject initial, unsupported, and partial-footprint states."""
import importlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest


@pytest.fixture
def scoring(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    module = importlib.import_module('eval_native_table_agent')
    model = mujoco.MjModel.from_xml_path(module.default_scene_xml_path)
    robot = module.ContactRobot(model, 'object1', separation=.15, separation_axis='y')
    regions = json.loads((root / 'configs/eval/native_table_two_object_regions.json').read_text())
    return module, robot, dict(zip(('object2', 'object1'), regions, strict=True))


def test_both_frozen_goals_initially_unmet(scoring):
    module, robot, goals = scoring
    rows = module.goal_predicates(robot, goals)
    assert all(not row['in_region'] and row['supported'] for row in rows.values())


def test_center_inside_is_not_enough(scoring):
    module, robot, _ = scoring
    center = robot.data.body('object1').xpos[:2]
    region = [center - [.005, .1], center + [.005, .1]]
    row = module.goal_predicates(robot, {'object1': region})['object1']
    assert row['supported'] and not row['in_region'] and not row['satisfied']


def test_region_scoring_preserves_live_state_and_requires_support(scoring):
    module, robot, _ = scoring
    q = robot.data.qpos.copy()
    contacts = robot.data.contact.dist.copy()
    center = robot.data.body('object1').xpos[:2]
    goal = {'object1': [center - [.1, .1], center + [.1, .1]]}
    assert module.goal_predicates(robot, goal)['object1']['satisfied']
    np.testing.assert_array_equal(robot.data.qpos, q)
    np.testing.assert_array_equal(robot.data.contact.dist, contacts)
    # Test fixture perturbation only; not an execution primitive.
    address = robot.model.jnt_qposadr[robot.model.body('object1').jntadr[0]]
    robot.data.qpos[address+2] += .1
    assert not module.goal_predicates(robot, goal)['object1']['supported']
