# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import importlib.util
from pathlib import Path

import pytest


def test_real_habitat_config_excludes_semantic_mesh_and_sensors():
    habitat_sim = pytest.importorskip("habitat_sim")
    script = Path(__file__).resolve().parents[3] / "scripts/probe_grapheqa_sensor_only.py"
    spec = importlib.util.spec_from_file_location("grapheqa_probe", script)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    config = probe.habitat_sensor_config(Path("visual.basis.glb"))
    assert not config.sim_cfg.load_semantic_mesh
    assert not config.sim_cfg.use_semantic_textures
    assert config.sim_cfg.scene_dataset_config_file == "default"
    agent = config.agents[0]
    assert {sensor.sensor_type for sensor in agent.sensor_specifications} == {
        habitat_sim.SensorType.COLOR,
        habitat_sim.SensorType.DEPTH,
    }
    assert set(agent.action_space) == {"turn_left"}
