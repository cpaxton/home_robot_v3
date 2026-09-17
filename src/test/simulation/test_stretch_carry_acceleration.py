# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Private failed-run replay: same contact physics, different wheel references.

This is a causal control, not independent learned pick/place acceptance.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.simulation.stretch_mujoco import utils
from emet.simulation.stretch_mujoco.mujoco_server import BaseController


@pytest.mark.parametrize("profiled", [False, True])
def test_recorded_turn_retains_payload_only_with_profiled_wheel_targets(profiled):
    root = Path(__file__).resolve().parent
    fixture = json.loads((root / "fixtures/stretch_carry_turn.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(root.parents[1] / "emet/assets/robot" / fixture["scene"]))
    data = mujoco.MjData(model)
    initial = fixture["initial"]
    for field in ("qpos", "qvel", "act", "ctrl"):
        getattr(data, field)[:] = initial[field]
    data.time = initial["sim_time"]
    mujoco.mj_forward(model, data)
    data.qacc_warmstart[:] = initial["qacc_warmstart"]
    controller = BaseController(SimpleNamespace(mjmodel=model, mjdata=data))
    wheels = [model.actuator(name).id for name in ("left_wheel_vel", "right_wheel_vel")]
    gears = model.actuator_gear[wheels, 0]
    controls = fixture["controls"]
    index = 0
    peak_distance = 0.0
    while data.time < initial["sim_time"] + fixture["duration"]:
        while index + 1 < len(controls) and controls[index + 1]["sim_time"] <= data.time:
            index += 1
        desired = np.asarray(controls[index]["ctrl"])
        previous_wheels = data.ctrl[wheels].copy()
        data.ctrl[:] = desired
        if profiled:
            data.ctrl[wheels] = previous_wheels
            # Convert recorded transmission targets back to a twist, then use
            # the production control path, including gearing and acceleration.
            controller._set_base_velocity(*utils.diff_drive_fwd_kinematics(*(desired[wheels] / gears)))
        mujoco.mj_step(model, data)
        distance = np.linalg.norm(data.body("object2").xpos - data.body("link_grasp_center").xpos)
        peak_distance = max(peak_distance, distance)
    if profiled:
        assert peak_distance < 0.03
    else:
        assert peak_distance > 0.5, "Negative control must reproduce the recorded ejection"
