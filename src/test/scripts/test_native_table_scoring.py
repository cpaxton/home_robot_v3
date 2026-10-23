# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Independent table predicates retain footprint/support/stillness semantics."""
from types import SimpleNamespace

import mujoco
import numpy as np

from emet.eval.native_table_scoring import goal_predicates


def test_scoring_uses_whole_footprint_and_velocity_without_mutating_live_data():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="table" pos="0 0 .5"><geom type="box" size=".5 .5 .05"/></body>
      <body name="rubber_tip_left" pos="2 0 0"><geom size=".01"/></body>
      <body name="rubber_tip_right" pos="3 0 0"><geom size=".01"/></body>
      <body name="object1" pos="-.2 0 .5999"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
      <body name="object2" pos=".2 0 .5999"><freejoint/><geom type="cylinder" size=".05 .05"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = SimpleNamespace(model=model, data=data)
    goals = {'object1': [[-.26, -.06], [-.14, .06]], 'object2': [[.14, -.06], [.26, .06]]}
    original = {key: getattr(data, key).copy() for key in ('qpos', 'qvel', 'qacc')}
    distances = data.contact.dist.copy()
    rows = goal_predicates(robot, goals)
    assert all(row['satisfied'] for row in rows.values())
    for key, value in original.items():
        np.testing.assert_array_equal(getattr(data, key), value)
    np.testing.assert_array_equal(data.contact.dist, distances)
    narrow = {**goals, 'object1': [[-.21, -.01], [-.19, .01]]}
    assert not goal_predicates(robot, narrow)['object1']['in_region']
    joint = model.body('object1').jntadr[0]
    data.qvel[model.jnt_dofadr[joint]] = .02
    row = goal_predicates(robot, goals)['object1']
    assert row['in_region'] and row['supported']
    assert not row['settled'] and not row['satisfied']
