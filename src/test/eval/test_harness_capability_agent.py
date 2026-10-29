# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Live fixture admission, independent of the language model's task answers."""

from pathlib import Path

import mujoco
import pytest

from emet.eval.harness_capability_agent import agent_turns, build_stretch_case, initial_gaze
from emet.eval.harness_capability_fixtures import load_development_cases

CASES = load_development_cases(Path(__file__).resolve().parents[3] / "configs/eval/harness_capability_dev.json")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_full_stretch_fixture_compiles_with_declared_interventions(tmp_path, case):
    scene, model = build_stretch_case(case, tmp_path / case["id"])
    assert scene.is_file()
    assert model.body("base_link").id >= 0
    present = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target") >= 0
    assert present is case["frames"][0]["target_present"]
    obstacle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle") >= 0
    assert obstacle is case.get("obstacle", False)
    if case["family"] in {"recall", "search_act"}:
        assert case["expected"] not in " ".join(agent_turns(case)).lower()
    if case["id"] == "cap-dev-01":
        pan, tilt = initial_gaze(model)
        assert model.actuator("head_pan").ctrlrange[0] <= pan <= model.actuator("head_pan").ctrlrange[1]
        assert model.actuator("head_tilt").ctrlrange[0] <= tilt <= model.actuator("head_tilt").ctrlrange[1]
