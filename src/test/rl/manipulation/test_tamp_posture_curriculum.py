"""Reachable reset coverage must include the public executor's elbow regime."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from emet.rl.manipulation.mastery import MasteryEnv, load_suite, select_cases
from emet.rl.manipulation.observation_layout import history_layout

ROOT = Path(__file__).resolve().parents[4]


def suite():
    return load_suite(ROOT / "configs/benchmarks/mars_tracking_mastery_tamp_postures_v1.json")


def test_posture_and_coupled_amplitude_coverage_preserves_frozen_goals():
    spec = suite()
    frozen = load_suite()
    for key in ("cases", "reference", "gate"):
        assert spec[key] == frozen[key]
    stages = ["base", "arm", "coupled"]
    assert select_cases(spec, stages, "all") == select_cases(frozen, stages, "all")
    cases = select_cases(spec, stages, "all", population="admission")
    assert len(cases) == len({c["id"] for c in cases}) == 122
    assert {stage: sum(c["stage"] == stage for c in cases) for stage in stages} == {"base": 50, "arm": 64, "coupled": 8}
    val_seeds = {c["reset_seed"] for c in cases if c["split"] == "validation"}
    train = select_cases(spec, stages, "train", population="training")
    assert {c["reset_seed"] for c in train}.isdisjoint(val_seeds)
    straight = [c for c in train if "arm_start" in c]
    assert len(straight) == 24
    assert min(c["arm_start"][2] for c in straight) < 0
    assert max(c["arm_start"][3] for c in straight) > 0.35
    full = next(c for c in train if c["id"] == "train:coupled:forward_turn")
    for scale in (0.25, 0.5):
        small = next(c for c in train if c["id"] == f"train:coupled:forward_turn@amplitude:{scale:g}")
        np.testing.assert_array_equal(small["arm_delta"], np.asarray(full["arm_delta"]) * scale)
        assert small["travel_m"] == full["travel_m"] * scale
        assert small["turn_rad"] == full["turn_rad"] * scale
    assert np.isclose(sum(spec["training_primitive_weights"]["arm"].values()), 1)


@pytest.mark.parametrize("start", [[0] * 4, [0] * 6, [0, 0, float("nan"), 0, 0], [0, 0, 1.01, 0, 0], None, "bad"])
def test_malformed_posture_recipes_rejected(tmp_path, start):
    spec = suite()
    spec["extra_training_cases"][-1]["arm_start"] = start
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="extra arm start"):
        load_suite(path)


@pytest.mark.parametrize("scales", [[0, 1], [0.5], [1, 0.5], [True, 1], [float("nan"), 1]])
def test_malformed_coupled_amplitude_recipes_rejected(tmp_path, scales):
    spec = suite()
    spec["coupled_training_scales"] = scales
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="coupled training scales"):
        load_suite(path)


@pytest.mark.sim
@pytest.mark.parametrize("label,direction", [("A", "translate_up"), ("B", "translate_down")])
def test_posture_is_initialized_before_settling_and_never_injected_by_runtime(label, direction):
    spec = suite()
    env = MasteryEnv(spec, stages=["arm"], split="train", population="training", training_stage_weights={"arm": 1})
    try:
        cid = f"train:arm:near_straight_{label}_{direction}"
        case = next(c for c in env.cases if c["id"] == cid)
        obs, info = env.reset(options={"case_id": cid})
        assert not info["initialization_failed"]
        assert env.mastery_admission["accepted"]
        np.testing.assert_allclose(env.control.positions(), case["arm_start"], atol=0.005, rtol=0)
        fields, _ = history_layout(env.profile, retention_context=True)
        np.testing.assert_array_equal(obs["history"][:, fields["proprio/reference_error"]], 0)
        before = env.data.qpos.copy()
        env.set_task_trajectory(env.mastery_reference)
        np.testing.assert_array_equal(env.data.qpos, before)
        with pytest.raises(ValueError, match="selected mastery case"):
            env.reset(options={"case_id": cid, "arm_start": case["arm_start"]})
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("stage,fixture", [("loaded", False), ("arm", True)])
def test_posture_override_cannot_alter_loaded_or_pregrasp_fixtures(stage, fixture):
    spec = deepcopy(suite())
    spec["config"]["hold_pregrasp_fixture"] = fixture
    env = MasteryEnv(spec, stages=[stage], split="train")
    try:
        # Deliberately malformed internal case: constructor is not a substitute
        # for the runtime reset-hook guard after a suite object is modified.
        env.cases[0] = {**env.cases[0], "arm_start": [0, -0.38, -0.04, 0.42, 0]}
        with pytest.raises(ValueError, match="restricted to free-space"):
            env.reset(options={"case_id": env.cases[0]["id"]})
    finally:
        env.close()
