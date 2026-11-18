# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Development scene provisioning keeps environment geometry out of runtime FK."""

import json

import mujoco
import numpy as np
from prepare_observed_stretch_fixture import prepare

from emet.config.sim_launch_config import load_sim_launch_config_from_path
from emet.simulation.synchronized_capture import robot_kinematic_signature


def test_prepared_scene_and_robot_only_model_are_separate_and_json_loads(tmp_path):
    prepare(tmp_path, port_offset=3100)
    config = load_sim_launch_config_from_path(str(tmp_path / "sim.json"))
    assert config.robot == "stretch" and config.port_offset == 3100
    scene = mujoco.MjModel.from_xml_path(config.scene_path)
    robot = mujoco.MjModel.from_xml_path(str(tmp_path / "stretch.xml"))
    assert scene.nq > robot.nq
    assert mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, "object2") == -1
    assert mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, "table") == -1
    np.testing.assert_allclose(scene.body("object2").pos, [-0.02, -0.70, 0.6])
    assert robot_kinematic_signature(scene) == robot_kinematic_signature(robot)
    manifest = json.loads((tmp_path / "fixture.json").read_text())
    assert manifest["tier"] == "development_fixture_not_held_out"
    assert len(manifest["source_sha256"]) == 4
