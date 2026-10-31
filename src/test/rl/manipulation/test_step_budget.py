"""Actual control-step exposure must survive unequal, early-ending episodes."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from emet.rl.manipulation.mastery import MasteryEnv, TrainingCaseSampler, load_suite, primitive_name, select_cases


def suite():
    return load_suite(
        Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_step_budget_v1.json"
    )


def sampler(spec, seed):
    return TrainingCaseSampler(
        select_cases(spec, ["base", "arm"], "train", population="training"),
        spec["training_stage_weights"],
        seed=seed,
        sampling=spec["training_sampling"],
        primitive_weights=spec["training_primitive_weights"],
    )


def test_step_budget_preserves_all_canonical_cases_and_physical_contracts():
    current = suite()
    previous = load_suite(
        Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_arm_to_whole_body_v1.json"
    )
    for key in (
        "config",
        "cases",
        "reference",
        "gate",
        "training_population",
        "arm_training_scales",
        "training_stage_weights",
    ):
        assert current[key] == previous[key]
    assert select_cases(current, ["base", "arm"], "all", population="admission") == select_cases(
        previous, ["base", "arm"], "all", population="admission"
    )


def test_unequal_episode_lengths_preserve_transition_budgets_and_case_cycles():
    spec = suite()
    chosen = sampler(spec, 90)
    visits, steps, primitive_steps = defaultdict(list), Counter(), Counter()
    total = 0
    while total < 100000:
        case = chosen.next()
        stage, family = case["stage"], primitive_name(case)
        length = {"forward": 39, "reverse": 35}.get(family, 240) if stage == "base" else 240
        visits[(stage, family if stage == "base" else "all")].append(case["id"])
        for _ in range(length):
            chosen.record_step(case)
        steps[stage] += length
        if stage == "base":
            primitive_steps[family] += length
        total += length
    assert abs(steps["arm"] / total - 0.75) < 0.003
    assert abs((primitive_steps["forward"] + primitive_steps["reverse"]) / total - 0.20) < 0.007
    for family, weight in spec["training_primitive_weights"]["base"].items():
        assert abs(primitive_steps[family] / steps["base"] - weight) < 0.02
    assert chosen.control_steps == steps
    assert chosen.primitive_steps["base"] == primitive_steps
    for (stage, family), ids in visits.items():
        count = sum(c["stage"] == stage and (family == "all" or primitive_name(c) == family) for c in chosen.cases)
        assert len(set(ids[:count])) == min(count, len(ids))
    assert all(c["split"] == "train" for c in chosen.cases)


def test_step_budget_reseeding_resets_counts_and_reproduces_selection():
    chosen = sampler(suite(), 17)

    def take():
        result = []
        for _ in range(100):
            case = chosen.next()
            result.append(case["id"])
            for _ in range(39 if primitive_name(case) == "forward" else 240):
                chosen.record_step(case)
        return result

    expected = take()
    chosen.reset(17)
    assert not chosen.control_steps and all(not c for c in chosen.primitive_steps.values())
    assert take() == expected


@pytest.mark.parametrize(
    "key,value",
    [
        ("training_sampling", "episodes"),
        ("training_sampling", "unknown"),
        ("training_sampling", []),
        ("training_stage_weights", None),
        ("training_primitive_weights", {"base": {"forward": 1}}),
        ("training_primitive_weights", {"unknown": {"forward": 1}}),
        (
            "training_primitive_weights",
            {"base": {"hold": 0.05, "forward": float("nan"), "reverse": 0.4, "yaw_left": 0.075, "yaw_right": 0.075}},
        ),
    ],
)
def test_step_budget_rejects_invalid_recipe(tmp_path, key, value):
    changed = suite()
    changed[key] = value
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        load_suite(path)


@pytest.mark.sim
def test_native_steps_are_counted_and_episode_reset_keeps_budget():
    env = MasteryEnv(suite(), stages=["arm", "base"], split="train", population="training")
    try:
        _, info = env.reset(seed=19, options={"case_id": "train:arm:pitch_up"})
        assert not info["initialization_failed"]
        for _ in range(3):
            env.step(np.zeros(8))
        assert env.sampler.control_steps == {"arm": 3}
        env.reset(options={"case_id": "train:base:hold"})
        env.step(np.zeros(8))
        assert env.sampler.control_steps == {"arm": 3, "base": 1}
        assert env.sampler.primitive_steps["base"] == {"hold": 1}
        env.reset(seed=19)
        assert not env.sampler.control_steps
    finally:
        env.close()
