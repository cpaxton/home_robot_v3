# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""WBC foundations: frozen contract, case library, and synthetic generator."""

from __future__ import annotations

import numpy as np

from emet.controller.manipulation.case_library import CaseLibrary, ScenarioRecord
from emet.controller.manipulation.data_generators import synthetic_obstacle_target
from emet.controller.manipulation.wbc_contract import ReachSpec, WbcAction, WbcObservation, reach_reward


def _obs(ee_pose, ee_target, failure_mode=""):
    return WbcObservation(
        q=(0.0,) * 10,
        qd=(0.0,) * 10,
        base_xyt=(0.0, 0.0, 0.0),
        ee_pose=tuple(ee_pose),
        ee_target=tuple(ee_target),
        occupancy=np.zeros((8, 8, 8), dtype=np.uint8),
        failure_mode=failure_mode,
    )


# --- contract ---------------------------------------------------------------


def test_wbc_observation_round_trip():
    o = _obs([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    o2 = WbcObservation.from_dict(o.to_dict())
    assert np.array_equal(o2.occupancy, o.occupancy)
    assert o2.ee_target == o.ee_target
    assert o2.base_xyt == (0.0, 0.0, 0.0)


def test_reach_reward_success_and_collision():
    target = [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    action = WbcAction(base_velocity=(0.0, 0.0, 0.0), joint_torques=(0.0,) * 10)
    # Exact match -> success bonus dominates.
    r, c = reach_reward(action, _obs(target, target), spec=ReachSpec())
    assert c["success"] == 1.0
    assert c["pos_err"] < 1e-9
    assert r > 0
    # Collision -> penalty, no success.
    r2, c2 = reach_reward(action, _obs([0.1, 0.0, 0.5, 1, 0, 0, 0], target), collision=True, spec=ReachSpec())
    assert c2["collision"] == 1.0
    assert r2 < r
    assert c2["success"] == 0.0


# --- case library -----------------------------------------------------------


def _record(voxel_value=1, failure_mode="tracking_lag"):
    return ScenarioRecord(
        initial_voxel=np.full((4, 4, 4), voxel_value, dtype=np.uint8),
        goal={"kind": "ee_target", "ee_target": [0.0, 0.0, 0.5, 1, 0, 0, 0]},
        robot_state={"q": [0.0] * 4, "base_xyt": [0.0, 0.0, 0.0]},
        meta={"failure_mode": failure_mode, "source": "test"},
    )


def test_case_library_add_dedupes_and_fingerprints():
    lib = CaseLibrary()
    r = _record()
    assert lib.add(r) is True
    assert lib.add(r) is False  # duplicate
    assert len(lib) == 1
    # Same content, different object -> same fingerprint -> deduped.
    assert lib.add(_record()) is False
    # Different voxel -> new fingerprint -> added.
    assert lib.add(_record(voxel_value=0)) is True
    assert len(lib) == 2


def test_case_library_by_failure_mode_and_sample():
    lib = CaseLibrary()
    for i, mode in enumerate(("tracking_lag", "ik_failed", "collision")):
        lib.add(_record(voxel_value=i + 1, failure_mode=mode))
    buckets = lib.by_failure_mode()
    assert set(buckets) == {"tracking_lag", "ik_failed", "collision"}
    rng = np.random.default_rng(0)
    assert len(lib.sample(5, rng=rng)) == 5
    assert len(lib.sample(2, rng=rng, failure_mode="ik_failed")) == 2


def test_case_library_save_load_round_trip(tmp_path):
    lib = CaseLibrary()
    lib.add(_record(failure_mode="tracking_lag"))
    lib.add(_record(voxel_value=0, failure_mode="ik_failed"))
    path = lib.save(tmp_path / "cases.jsonl")
    loaded = CaseLibrary.load(path)
    assert len(loaded) == 2
    assert {r.failure_mode for r in loaded} == {"tracking_lag", "ik_failed"}


# --- data generator ---------------------------------------------------------


def test_synthetic_obstacle_target_produces_valid_scenario():
    rng = np.random.default_rng(42)
    rec = synthetic_obstacle_target(rng, grid_shape=(16, 16, 16), n_obstacles=(2, 4))
    assert rec.initial_voxel.shape == (16, 16, 16)
    assert rec.initial_voxel.sum() > 0  # some obstacles present
    # Target position is not inside an occupied voxel (it was rejection-sampled free).
    target_pos = np.asarray(rec.goal["ee_target"][:3], dtype=np.float64)
    lo = np.array([-0.6, -0.6, 0.0])
    hi = np.array([0.6, 0.6, 1.0])
    cell = (hi - lo) / 15
    idx = np.clip(((target_pos - lo) / cell).astype(int), 0, 15)
    assert rec.initial_voxel[tuple(idx)] == 0


def test_synthetic_generator_is_reproducible_with_seed():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    r1 = synthetic_obstacle_target(rng1)
    r2 = synthetic_obstacle_target(rng2)
    assert np.array_equal(r1.initial_voxel, r2.initial_voxel)
    assert r1.goal == r2.goal


def test_scenario_from_attempt_and_record_scenario(tmp_path, monkeypatch):
    from emet.controller.manipulation.failure_record import record_scenario, scenario_from_attempt

    attempt = {
        "failure_mode": "tracking_failed",
        "target_xyz": [0.0, 0.0, 0.9],
        "joint_names": ["torso_joint1", "arm_joint1"],
        "trace": [{"qpos": [0.1, 0.2], "ee_xyz": [0.0, 0.0, 0.5]}],
    }
    rec = scenario_from_attempt(attempt, context={"object_body": "obj_main"})
    assert rec.goal["object"] == "obj_main"
    assert rec.goal["ee_target_pos"] == [0.0, 0.0, 0.9]
    assert rec.robot_state["q"] == [0.1, 0.2]
    assert rec.failure_mode == "tracking_failed"
    # The initial map defaults to an explicit 'unavailable' marker, never a silent blank.
    assert rec.initial_voxel["kind"] == "voxel_map_2d_unavailable"

    monkeypatch.setenv("EMET_CASE_LIBRARY_DIR", str(tmp_path))
    assert record_scenario(rec) is True
    assert record_scenario(rec) is False  # duplicate, deduped by fingerprint


def test_export_initial_voxel_map_from_mock():
    from emet.controller.manipulation.failure_record import export_initial_voxel_map

    class MockVoxelMap:
        grid_origin = np.array([-2.0, -3.0])
        grid_resolution = 0.1

        def get_2d_map(self):
            return (
                np.array([[0, 1], [1, 0]], dtype=np.float32),
                np.array([[1, 1], [1, 0]], dtype=np.float32),
            )

    out = export_initial_voxel_map(MockVoxelMap())
    assert out["kind"] == "voxel_map_2d"
    assert out["grid_origin"] == [-2.0, -3.0]
    assert out["grid_resolution"] == 0.1
    assert out["obstacles"] == [[0, 1], [1, 0]]
    assert out["explored"] == [[1, 1], [1, 0]]


def test_export_initial_voxel_map_never_raises():
    from emet.controller.manipulation.failure_record import export_initial_voxel_map

    class Boom:
        grid_origin = np.array([0.0, 0.0])

        def get_2d_map(self):
            raise RuntimeError("no map")

    out = export_initial_voxel_map(Boom())
    assert out["kind"] == "voxel_map_2d_unavailable"
    assert out["obstacles"] == []


