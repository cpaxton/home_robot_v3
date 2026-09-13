# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Private simulator-state control, not an oracle available to the agent."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from emet.simulation.stretch_mujoco.config import wrist_position_rates
from emet.simulation.stretch_mujoco.position_targets import PositionTargets


@pytest.mark.parametrize("insertion", [0.0, 0.015, 0.025])
def test_contact_depth_changes_retention_without_changing_physics(insertion):
    root = Path(__file__).resolve().parent
    captured = json.loads((root / "fixtures/stretch_preclosure_cylinder.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(root.parents[1] / "emet/assets/robot" / captured["scene"]))
    data = mujoco.MjData(model)
    data.qpos[:] = captured["qpos"]
    data.ctrl[:] = captured["ctrl"]
    mujoco.mj_forward(model, data)
    targets = PositionTargets(model, data, wrist_position_rates)

    def step(seconds):
        for _ in range(round(seconds / model.opt.timestep)):
            targets.step()
            mujoco.mj_step(model, data)

    targets.set("arm", float(data.actuator("arm").ctrl[0]) + insertion)
    step(2)
    targets.set("gripper", -0.016)
    step(3)
    targets.set("lift", float(data.actuator("lift").ctrl[0]) + 0.3)
    step(3)
    targets.set("arm", 0.01)
    step(35)
    object_id = model.body("object2").id
    fingers = {model.body("rubber_tip_left").id, model.body("rubber_tip_right").id}
    contact = any(
        object_id in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
        and bool(fingers & {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]})
        for c in data.contact
    )
    distance = np.linalg.norm(data.body("object2").xpos - data.body("link_grasp_center").xpos)
    if insertion == 0:
        assert not contact
        assert distance > 0.5
    else:
        assert contact
        assert distance < 0.03
        assert data.body("object2").xpos[2] > 0.7
