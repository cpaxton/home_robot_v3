# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from emet.simulation.stretch_mujoco.config import joint_position_rates
from emet.simulation.stretch_mujoco.position_targets import PositionTargets


def fixture():
    root = Path(__file__).resolve().parent
    captured = json.loads((root / "fixtures/stretch_held_cylinder.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(root.parents[1] / "emet/assets/robot" / captured["scene"]))
    data = mujoco.MjData(model)
    data.qpos[:] = captured["qpos"]
    data.ctrl[:] = captured["ctrl"]
    mujoco.mj_forward(model, data)
    return model, data


def test_retarget_and_resend_do_not_jump_or_restart_reference():
    model, data = fixture()
    targets = PositionTargets(model, data, joint_position_rates)
    before = data.actuator("wrist_pitch").ctrl[0]
    for _ in range(10):
        targets.set("wrist_pitch", -1.5)
        targets.step()
    assert data.actuator("wrist_pitch").ctrl[0] == pytest.approx(before - 10 * 0.8 * model.opt.timestep)
    before = data.actuator("wrist_pitch").ctrl[0]
    targets.set("wrist_pitch", 0)
    assert data.actuator("wrist_pitch").ctrl[0] == before
    targets.step()
    assert data.actuator("wrist_pitch").ctrl[0] == pytest.approx(before + 0.8 * model.opt.timestep)
    targets.reset()
    before = data.ctrl.copy()
    targets.step()
    np.testing.assert_array_equal(data.ctrl, before)


def test_unprofiled_axes_and_joint_limits_are_preserved():
    model, data = fixture()
    targets = PositionTargets(model, data, joint_position_rates)
    targets.set("head_pan", 0.3)
    assert data.actuator("head_pan").ctrl[0] == 0.3
    targets.set("wrist_pitch", -100)
    assert targets.pending["wrist_pitch"] == model.actuator("wrist_pitch").ctrlrange[0]
    with pytest.raises(ValueError, match="finite"):
        targets.set("wrist_pitch", float("nan"))


@pytest.mark.parametrize("name,rate", [("arm", 0.1), ("lift", 0.15)])
def test_payload_translation_references_are_profiled(name, rate):
    model, data = fixture()
    targets = PositionTargets(model, data, joint_position_rates)
    before = float(data.actuator(name).ctrl[0])
    targets.set(name, before + 0.2)
    assert data.actuator(name).ctrl[0] == before
    for _ in range(100):
        targets.step()
    assert data.actuator(name).ctrl[0] == pytest.approx(before + 100 * rate * model.opt.timestep)


@pytest.mark.parametrize("profiled", [False, True])
def test_saved_physical_hold_survives_wrist_fold_only_with_profile(profiled):
    # Private evaluator fixture; no ground-truth state enters an agent policy.
    # Keep the old controller as a matched negative control for this failure.
    model, data = fixture()
    for _ in range(round(1 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    data.actuator("arm").ctrl = 0.01
    for _ in range(round(1 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    targets = PositionTargets(model, data, joint_position_rates if profiled else {})
    targets.set("wrist_pitch", -1.5)
    peak_velocity = 0
    for _ in range(round(5 / model.opt.timestep)):
        targets.step()
        mujoco.mj_step(model, data)
        peak_velocity = max(peak_velocity, abs(data.actuator("wrist_pitch").velocity[0]))
    obj = data.body("object2").xpos
    if profiled:
        assert obj[2] > 0.5
        assert np.linalg.norm(obj - data.body("link_grasp_center").xpos) < 0.06
        assert peak_velocity < 1.5
        bodies = [(model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]) for c in data.contact]
        assert any(model.body("object2").id in pair and model.body("rubber_tip_left").id in pair for pair in bodies)
    else:
        assert obj[2] < 0.1
        assert peak_velocity > 10
