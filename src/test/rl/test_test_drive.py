import json

import numpy as np
import pytest

from emet.rl.test_drive import Simulation, checkpoints, training_status

pytestmark = pytest.mark.sim


def test_only_simulation_modes_are_accepted():
    for mode in ("herman", "hardware", "innate_mars", "tcp://192.168.1.43"):
        with pytest.raises(ValueError, match="local simulation"):
            Simulation(mode)


def test_test_drive_changes_reference_without_teleporting_and_steps_physics():
    sim = Simulation("mars-base", seed=700001)
    try:
        before = sim.env.data.qpos.copy()
        target = sim.env.control.base_pose().copy()
        target[0] += 0.1
        sim.goal(target)
        np.testing.assert_array_equal(before, sim.env.data.qpos)
        before_time = sim.env.data.time
        sim.step()
        assert sim.env.data.time > before_time
        assert sim.state()["simulation_only"]
        assert "unsupported" in sim.state()["wheel_support"]
        with pytest.raises(ValueError):
            sim.goal([np.nan, 0, 0])
        with pytest.raises(ValueError, match="configured local roots"):
            sim.load_policy("/arbitrary/path", [])
    finally:
        sim.close()


def test_checkpoint_discovery_filters_incomplete_and_wrong_robot(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format": "emet-manipulation-v1", "robot": "stretch", "steps": 1024}))
    assert not checkpoints([tmp_path], "stretch-arm")
    (tmp_path / "policy.zip").write_bytes(b"placeholder")
    assert len(checkpoints([tmp_path], "stretch-arm")) == 1
    assert not checkpoints([tmp_path], "mars-base")


def test_training_status_separates_initialization_from_ppo_and_partial_writes(tmp_path):
    seed = tmp_path / "seed-0"
    seed.mkdir()
    (seed / "run.json").write_text(json.dumps({"seed": 0, "initialization_steps": 32768, "steps": 262144}))
    (seed / "metrics.jsonl").write_text('{"timesteps": 40000}\n{"timesteps":')
    row = training_status([tmp_path])[0]
    assert row["steps"] == 40000 and row["ppo_steps"] == 7232 and row["phase"] == "PPO"
