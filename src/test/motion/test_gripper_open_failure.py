# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Measured release preflight must preserve collision versus bounds attribution."""
import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor
from emet.controller.task.tamp.api import failure_code
from emet.motion.mujoco_collision import MujocoSceneCollisionChecker


@pytest.mark.parametrize('goal,wall,expected', ((.2, True, 'collision'), (.4, False, 'joint_bounds')))
def test_real_opening_preflight_preserves_detail_and_does_not_actuate(goal, wall, expected):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/>
      <worldbody><body name="robot"><joint name="arm" type="slide" axis="0 0 1" range="-1 1"/>
      <geom type="sphere" size=".01" mass="1"/>
      <body name="finger" pos=".1 0 0"><joint name="opening" type="slide" axis="1 0 0" range="-.1 .3"/>
      <geom type="sphere" size=".02" mass=".1"/></body></body>
      {'<geom name="wall" type="box" pos=".3 0 0" size=".02 .2 .2"/>' if wall else ''}
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    executor = PhysicalPickPlaceExecutor(None, model=model, data=data, ee_body='robot', joint_names=('arm',),
        collision=MujocoSceneCollisionChecker(model, robot_body='robot'), synchronize=lambda _: None,
        command_joints=lambda _: pytest.fail('rejected opening must not command motion'), input_mode='privileged',
        gripper_open_configuration=(('opening', goal),))
    path, error = executor.plan_gripper_open()
    assert path is None
    assert error.startswith(f'gripper_open_path_failed:invalid_goal:{expected}')
    assert failure_code(error) == expected
    np.testing.assert_array_equal(data.qpos, before)
