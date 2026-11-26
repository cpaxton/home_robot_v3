# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Measured wrist-wrench sensor tests with a fixed, minimal model."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco

from emet.rl.manipulation.wbc_contact import MuJoCoContactSensor

_XML = """
<mujoco>
  <worldbody>
    <body name="base">
      <body name="ee" pos="0 0 1">
        <body name="gripper" pos="0 0 -0.1">
          <geom name="gripper_geom" type="box" size="0.02 0.02 0.02" mass="0.1"/>
          <site name="ft" pos="0 0 0"/>
          <body name="leftfinger" pos="0.04 0 0">
            <geom name="left_pad" type="box" size="0.01 0.01 0.01" mass="0.01"/>
          </body>
          <body name="rightfinger" pos="-0.04 0 0">
            <geom name="right_pad" type="box" size="0.01 0.01 0.01" mass="0.01"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <sensor>
    <force name="f" site="ft"/>
    <torque name="t" site="ft"/>
  </sensor>
</mujoco>
"""


def _sensor() -> tuple[mujoco.MjModel, mujoco.MjData, MuJoCoContactSensor]:
    model = mujoco.MjModel.from_xml_string(_XML)
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    sensor = MuJoCoContactSensor(
        model,
        base_body="base",
        ee_body="ee",
        finger_bodies=("leftfinger", "rightfinger"),
        force_sensor="f",
        torque_sensor="t",
    )
    mujoco.mj_forward(model, data)
    return model, data, sensor


def test_wrist_sensor_reads_zero_without_load():
    _, data, sensor = _sensor()
    reading = sensor.read(data)
    assert reading.contact.in_contact is False
    assert reading.contact.source == "ft_sensor"
    assert np.allclose(reading.contact.wrench, 0.0)
    assert reading.finger_force == (0.0, 0.0)


def test_wrist_sensor_reports_external_force_in_ee_frame():
    model, data, sensor = _sensor()
    gripper = model.body("gripper").id
    data.xfrc_applied[gripper, :3] = [10.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    reading = sensor.read(data)
    assert np.allclose(reading.contact.wrench[:3], [10.0, 0.0, 0.0], atol=1e-6)
    # The gripper origin is 0.1 m below the EE origin, so the same force has a
    # lever-arm torque about the EE: (0, 0, -0.1) x (10, 0, 0) = (0, -1, 0).
    assert np.allclose(reading.contact.wrench[3:], [0.0, -1.0, 0.0], atol=1e-6)


def test_wrist_sensor_reports_external_torque_in_ee_frame():
    model, data, sensor = _sensor()
    gripper = model.body("gripper").id
    data.xfrc_applied[gripper, 3:] = [0.0, 0.0, 2.0]
    mujoco.mj_forward(model, data)
    reading = sensor.read(data)
    assert np.allclose(reading.contact.wrench[:3], 0.0, atol=1e-6)
    assert np.allclose(reading.contact.wrench[3:], [0.0, 0.0, 2.0], atol=1e-6)
