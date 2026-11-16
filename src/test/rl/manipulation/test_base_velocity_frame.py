"""The inertia frame is not the regular robot frame used by wheel commands."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.rl.manipulation.mastery import measured_base_command_velocity


@pytest.mark.parametrize("yaw", [0, np.pi / 2, -np.pi / 2])
@pytest.mark.parametrize("direction", [-1, 1])
def test_forward_yaw_measurement_ignores_rotated_principal_inertia_axes(yaw, direction):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="base">
      <freejoint/><inertial pos="0 0 0" quat="0.7071067811865476 0 0 0.7071067811865476"
        mass="1" diaginertia="0.01 0.02 0.03"/><geom type="sphere" size="0.1"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    data.qvel[:3] = direction * 0.01 * np.array([np.cos(yaw), np.sin(yaw), 0])
    data.qvel[5] = direction * 0.02
    mujoco.mj_forward(model, data)
    before = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    control = SimpleNamespace(model=model, data=data, base=model.body("base").id)
    np.testing.assert_allclose(measured_base_command_velocity(control), direction * np.array([0.01, 0.02]), atol=1e-12)
    for expected, actual in zip(before, (data.qpos, data.qvel, data.ctrl), strict=True):
        np.testing.assert_array_equal(actual, expected)
