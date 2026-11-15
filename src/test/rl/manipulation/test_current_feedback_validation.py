# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Reset option propagation and honest frozen-budget diagnostic scoring."""

import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def runner():
    return runpy.run_path(str(ROOT / "scripts/validate_mars_current_feedback.py"))


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_primitive_rejects_non_boolean_feedback_before_spawning(value):
    from emet.rl.manipulation.tamp_primitive_env import TampPrimitiveEnv

    with pytest.raises(ValueError, match="must be boolean"):
        TampPrimitiveEnv([{"id": "stub"}], wheel_feedback_current_velocity=value)


@pytest.mark.sim
@pytest.mark.parametrize("enabled", [False, True])
def test_actual_tamp_fixture_carries_feedback_option_through_reset(enabled):
    from emet.eval.mars_tamp_control import fixture
    from emet.rl.manipulation.whole_body_training import load_development

    development, _ = load_development()
    env, admission = fixture(development["train_cases"][0], wheel_feedback_current_velocity=enabled)
    try:
        assert admission["accepted"]
        assert env.config.wheel_feedback_current_velocity is enabled
        assert env.control.base_velocity_feedback_current is enabled
    finally:
        env.close()


def test_loop_manifest_has_exact_declared_conditions_and_preserves_sources(runner):
    import copy

    from emet.rl.manipulation.whole_body_training import load_development

    _, general = load_development()
    original = copy.deepcopy(general)
    suite = runner["loop_suite"](general, True)
    assert general == original
    assert suite["config"]["wheel_feedback_current_velocity"]
    assert len(suite["cases"]) == len({c["id"] for c in suite["cases"]}) == 12
    assert sum(c["probe_push"] for c in suite["cases"]) == 5
    assert sum("cancel_tick" in c for c in suite["cases"]) == 2
    assert {c["source_case_id"].rsplit(":", 1)[-1] for c in suite["cases"] if c["probe_push"]} == set(runner["BASES"])


@pytest.mark.parametrize("kind", ["emet-manipulation-v1", "emet-whole-body-v2"])
def test_primitive_bundle_adapts_only_sensor_fields(runner, kind):
    class Bundle:
        robot, action_mode = "innate_mars", "direct_velocity"
        manifest = {"format": kind, "fingerprint": "test-model"}
        normalization = {"history": None, "preview": None}

        def __call__(self, observation):
            self.seen = observation
            return np.arange(8, dtype=np.float32) / 8

    bundle = Bundle()
    observation = {
        "history": torch.zeros(1, 20, 101),
        "preview": torch.zeros(1, 7, 18),
        "motion": torch.zeros(1, 9),
        "critic": torch.ones(1, 34),
    }
    action = runner["PrimitiveBundle"](bundle)(observation)
    assert set(bundle.seen) == ({"history", "preview", "motion"} if kind.endswith("v2") else {"history", "preview"})
    assert action.shape == (1, 8)
    np.testing.assert_array_equal(action[0].numpy(), np.arange(8) / 8)


def test_archived_budget_score_does_not_inherit_fresh_budget_success(runner, monkeypatch):
    score = runner["archived_scores"]
    monkeypatch.setitem(score.__globals__, "smoothness_budgets", lambda *a: {"base_linear_jerk": 0.1})
    monkeypatch.setitem(score.__globals__, "tracking_contract_passes", lambda *a, **kw: True)
    case = {
        "id": "a",
        "mastered": True,
        "admitted": True,
        "tracking_and_settling_pass": True,
        "metrics": {"base_linear_jerk": 0.2},
        "final_info": {"mastery": {"path_failed": False}},
    }
    rows = score({"cases": [case]}, {"cases": [{"id": "a", "metrics": {}}]}, {"gate": {}})
    assert not rows[0]["comparison_pass"]
    assert rows[0]["smoothness_failures"] == ["base_linear_jerk"]


def test_progress_keeps_completed_cases_without_marking_partial_run_complete(runner, tmp_path):
    import gzip
    import json

    path = tmp_path / "progress.json.gz"
    observer = runner["ProgressObserver"](path)
    report = {"complete": False, "cases": [{"id": str(i)} for i in range(10)]}
    observer.case_complete(report, report["cases"][-1])
    saved = json.loads(gzip.decompress(path.read_bytes()))
    assert saved == report and not saved["complete"]
    assert not path.with_suffix(".partial").exists()
