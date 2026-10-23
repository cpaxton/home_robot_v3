"""Force decoding must agree with MuJoCo, particularly for pyramidal facets."""

import mujoco
import numpy as np
import pytest

from emet.rl.contact import solved_contact_force

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("dimension", [1, 3, 4, 6])
def test_contact_force_decodes_normal_and_linear_force_in_newtons(cone, dimension):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".002" cone="{cone}"/>
      <default><geom condim="{dimension}" friction=".8 .01 .001"/></default>
      <worldbody><geom type="plane" size="2 2 .1"/>
        <body pos="0 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.xfrc_applied[1, 0] = 1 if dimension > 1 else 0
    for _ in range(500):
        mujoco.mj_step(model, data)
    normal_total = 0
    for i in range(data.ncon):
        reference = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, reference)
        normal, bound = solved_contact_force(model, data, i)
        assert normal == pytest.approx(reference[0], abs=1e-8)
        assert bound == pytest.approx(np.abs(reference[:3]).sum(), abs=1e-8)
        normal_total += normal
    assert normal_total == pytest.approx(9.81, abs=0.01)
