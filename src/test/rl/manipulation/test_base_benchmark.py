from dataclasses import asdict

import pytest

from emet.rl.manipulation.base_benchmark import make_suite, suite_sha, wilson
from emet.rl.manipulation.env import ManipulationConfig


def test_suite_freezes_disjoint_cases_without_relaxing_contract():
    config = ManipulationConfig(
        robot="innate_mars", base_xy_tolerance=0.05, base_heading_tolerance=0.25, hand_position_tolerance=0.005
    )
    suite = make_suite(config)
    assert suite_sha(suite) == suite_sha(make_suite(config))
    assert suite_sha(suite) != suite_sha(make_suite(config, seed=9400000))
    seeds = []
    for item in suite["strata"]:
        cfg = ManipulationConfig(**item["config"])
        seeds.extend(range(item["seed"], item["seed"] + suite["cases_per_stratum"]))
        assert cfg.reach_target_mode == "base"
        assert cfg.base_only_min_distance > cfg.base_xy_tolerance
        for key in (
            "base_xy_tolerance",
            "base_heading_tolerance",
            "hand_position_tolerance",
            "hand_orientation_tolerance",
        ):
            assert asdict(cfg)[key] == asdict(config)[key]
    assert len(seeds) == len(set(seeds)) == 140
    variants = {x["name"]: x["config"] for x in suite["strata"]}
    assert variants["goal_change"]["retarget_probability"] == 1
    assert variants["finite_push"]["reach_push_probability"] == 1
    assert variants["long_hold"]["horizon"] == 40
    assert variants["longer_translation"]["target_scale"] == 1
    with pytest.raises(ValueError):
        make_suite(config, cases=19)


def test_success_interval_does_not_call_small_samples_certain():
    low, high = wilson(20, 20)
    assert 0.83 < low < 0.85
    assert high == pytest.approx(1)
    low, high = wilson(0, 20)
    assert low == pytest.approx(0)
    assert 0.15 < high < 0.17
    with pytest.raises(ValueError):
        wilson(1, 0)
