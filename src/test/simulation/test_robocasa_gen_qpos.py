# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Regression: preserve opened fixture joints across the RoboCasa rebuild."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest


def _door_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="cabinet">
              <joint name="cabinet_door_joint" type="hinge" range="-1.6 0"/>
              <geom name="door" type="box" size="0.1 0.02 0.3"/>
            </body>
            <body name="robot">
              <joint name="robot_joint" type="hinge" range="-1 1"/>
              <geom name="robot_geom" type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def test_open_door_qpos_survives_model_rebuild():
    pytest.importorskip("robocasa")
    from emet.simulation.stretch_mujoco.robocasa_gen import _apply_scene_qpos0, _scene_qpos_by_name

    source = _door_model()
    data = mujoco.MjData(source)
    door = int(source.joint("cabinet_door_joint").qposadr[0])
    robot = int(source.joint("robot_joint").qposadr[0])
    data.qpos[door] = -1.2
    data.qpos[robot] = 0.3

    scene_qpos = _scene_qpos_by_name(source, data, robot_prefix="robot")

    assert "cabinet_door_joint" in scene_qpos
    assert "robot_joint" not in scene_qpos
    target = _door_model()
    applied = _apply_scene_qpos0(target, scene_qpos)
    assert applied == 1
    assert target.qpos0[int(target.joint("cabinet_door_joint").qposadr[0])] == np.float64(-1.2)
