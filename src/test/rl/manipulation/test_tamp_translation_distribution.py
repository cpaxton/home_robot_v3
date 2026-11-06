"""Add meaningful fixed-orientation translations without dropping prior gates."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from emet.rl.manipulation.mastery import TrainingCaseSampler, load_suite, select_cases

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / "configs/benchmarks/mars_tracking_mastery_tamp_postures_v1.json"
NEW = ROOT / "configs/benchmarks/mars_tracking_mastery_tamp_translation_v1.json"
FRESH = ROOT / "paper/data/whole_body_control/mars_tamp_translation_audit_v1/cases.json"


def test_translation_expansion_preserves_all_prior_cases_physics_and_thresholds():
    old, new = load_suite(OLD), load_suite(NEW)
    for key in ("cases", "config", "gate", "reference", "stages", "training_stage_weights"):
        assert new[key] == old[key]
    prior = select_cases(old, ["base", "arm", "coupled"], "all", population="admission")
    expanded = {c["id"]: c for c in select_cases(new, ["base", "arm", "coupled"], "all", population="admission")}
    assert len(prior) == 122 and len(expanded) == 146
    assert all(expanded[c["id"]] == c for c in prior)
    added = [c for c in expanded.values() if "tamp_translation_" in c["id"]]
    assert len(added) == 24 and all(c["stage"] == "arm" and c["split"] == "train" for c in added)
    for c in added:
        assert c["travel_m"] == c["turn_rad"] == 0
        np.testing.assert_array_equal(np.asarray(c["arm_delta"])[[0, 4]], 0)
        assert sum(c["arm_delta"][1:4]) == pytest.approx(0, abs=1e-12)
        assert max(abs(x) for x in c["arm_delta"]) <= 0.2


def test_half_of_arm_sampling_mass_rehearses_prior_families():
    suite = load_suite(NEW)
    cases = select_cases(suite, ["arm"], "train", population="training")
    sampler = TrainingCaseSampler(cases, {"arm": 1}, sampling="control_steps", primitive_weights=suite["training_primitive_weights"])
    # The sampler's primitive mass is independent of episode length corrections.
    weights = suite["training_primitive_weights"]["arm"]
    assert sum(v for k, v in weights.items() if k.startswith("tamp_translation_")) == pytest.approx(0.5)
    assert sum(v for k, v in weights.items() if not k.startswith("tamp_translation_")) == pytest.approx(0.5)
    assert len(sampler.cases) == len(cases)


def test_new_suite_and_disjoint_fresh_inputs_are_bound_to_regression_sources():
    from emet.rl.regression_gate import file_digest, source_state

    inputs = json.loads(FRESH.read_text())
    assert inputs["base_suite_sha256"] == hashlib.sha256(NEW.read_bytes()).hexdigest()
    suite = load_suite(NEW)
    train_seeds = {c["reset_seed"] for c in select_cases(suite, ["arm"], "train", population="training")}
    assert len(inputs["cases"]) == 12
    assert {c["reset_seed"] for c in inputs["cases"]} == set(range(34400000, 34400012))
    assert not train_seeds & {c["reset_seed"] for c in inputs["cases"]}
    sources = source_state(ROOT)
    for p in (NEW, FRESH):
        assert sources[str(p.relative_to(ROOT))] == file_digest(p)


@pytest.mark.parametrize("duration", [True, None, 0, -1, float("nan"), 11])
def test_extra_case_duration_rejects_invalid_values_without_changing_canonical_timing(tmp_path, duration):
    suite = json.loads(NEW.read_text())
    suite["extra_training_cases"][-1]["minimum_duration_s"] = duration
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="minimum duration"):
        load_suite(path)


@pytest.mark.sim
def test_larger_translation_uses_private_bounded_witness_with_faster_extra_case_timing():
    from emet.rl.manipulation.mastery import MasteryEnv

    suite = load_suite(NEW)
    env = MasteryEnv(suite, stages=["arm"], split="train", population="training")
    try:
        _, info = env.reset(options={"case_id": "train:arm:tamp_translation_A_elbow_translate_positive"})
        assert env.mastery_admission["accepted"], info
        path = env.mastery_reference.path
        assert 2.5 <= path.end < suite["reference"]["minimum_duration_s"]
        values = [path.sample(float(t)) for t in np.linspace(0, path.end, 501)]
        velocity = np.array([v[1][2:] for v in values])
        acceleration = np.array([v[2][2:] for v in values])
        jerk = np.array([v[3][2:] for v in values])
        assert 0.13 < np.max(abs(velocity)) < 0.16
        assert np.all(abs(velocity) <= np.asarray(env.profile.speed) * 0.35 + 1e-8)
        assert np.all(abs(acceleration) <= np.asarray(env.profile.acceleration) * 0.35 + 1e-8)
        assert np.all(abs(jerk) <= np.asarray(env.profile.acceleration) * 2 + 1e-8)
        reference = env.mastery_reference.record()
        assert reference["joint_reference_actor_input"] is False
        hand = np.array(reference["hand"])
        assert np.linalg.norm(hand[-1, :3] - hand[0, :3]) > 0.018
    finally:
        env.close()
