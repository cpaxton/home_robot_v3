"""Supported-contact resets extend tracking without adding a grasp task manager."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from emet.rl.manipulation.mastery import MasteryEnv, load_suite, select_cases

ROOT = Path(__file__).resolve().parents[4]
SUPPORTED = ROOT / "configs/benchmarks/mars_tracking_mastery_supported_v1.json"


def test_supported_mixture_preserves_prior_cases_and_bounds_contact_sampling():
    old = load_suite(ROOT / "configs/benchmarks/mars_tracking_mastery_tamp_translation_v1.json")
    new = load_suite(SUPPORTED)
    for key in ("config", "cases", "gate", "reference", "training_stage_weights"):
        assert new[key] == old[key]
    old_cases = select_cases(old, ["base", "arm", "coupled"], "all", population="admission")
    cases = select_cases(new, ["base", "arm", "coupled"], "all", population="admission")
    assert len(old_cases) == 146 and len(cases) == 164
    assert [c for c in cases if not c.get("supported_fixture")] == old_cases
    contact = [c for c in cases if c.get("supported_fixture")]
    assert len(contact) == 18 and all(c["stage"] == "arm" and c["split"] == "train" for c in contact)
    weights = new["training_primitive_weights"]["arm"]
    mass = sum(v for k, v in weights.items() if k.startswith("supported_"))
    assert mass == pytest.approx(0.25)
    assert mass * new["training_stage_weights"]["arm"] == pytest.approx(0.125)
    assert {c["reset_seed"] for c in contact}.isdisjoint(c["reset_seed"] for c in cases if c["split"] == "validation")


@pytest.mark.parametrize("change", ["non_boolean", "arm_start", "random_fixture", "unbalanced"])
def test_supported_reset_rejects_ambiguous_or_incompatible_configuration(tmp_path, change):
    suite = deepcopy(load_suite(SUPPORTED))
    case = next(c for c in suite["extra_training_cases"] if c.get("supported_fixture"))
    if change == "non_boolean":
        case["supported_fixture"] = 1
    elif change == "arm_start":
        case["arm_start"] = [0, -0.4, 0.2, 0.2, 0]
    elif change == "random_fixture":
        suite["config"]["reach_hold_probability"] = 0.5
    else:
        suite["config"]["balanced_grasp_setup"] = False
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="supported fixture"):
        load_suite(path)


@pytest.mark.sim
def test_supported_fixture_is_open_and_does_not_change_subsequent_ordinary_reset():
    suite = load_suite(SUPPORTED)
    env = MasteryEnv(suite, stages=["arm"], split="train", population="admission")
    try:
        ordinary = "train:arm:pitch_up"
        before, info = env.reset(options={"case_id": ordinary})
        assert not info["initialization_failed"]
        identity = {k: env.mastery_admission[k] for k in ("initial_state_sha256", "reference_sha256")}
        _, info = env.reset(options={"case_id": "train:arm:supported_A_down_short"})
        assert not info["initialization_failed"], env.mastery_admission
        assert env.scenario == "reach" and env.config.hold_pregrasp_fixture
        assert env.control.phase == "pregrasp" and env.control.gripper_is_open()
        assert env.control.grip_reference == env.profile.gripper_open
        assert not env.control.retain_object and not env.control.release_authorized
        for _ in range(3):
            env.step(np.zeros(env.profile.action_size))
        assert env.control.phase == "pregrasp"
        assert env.control.grip_reference == env.profile.gripper_open
        after, info = env.reset(options={"case_id": ordinary})
        assert not info["initialization_failed"] and not env.config.hold_pregrasp_fixture
        assert env.control.phase == "reach"
        assert {k: env.mastery_admission[k] for k in identity} == identity
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
    finally:
        env.close()
