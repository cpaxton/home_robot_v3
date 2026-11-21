# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Live fixture admission, independent of the language model's task answers."""

from pathlib import Path

import mujoco
import pytest

from emet.eval.harness_capability_agent import agent_turns, build_stretch_case, initial_gaze
from emet.eval.harness_capability_fixtures import load_development_cases

CASES = load_development_cases(Path(__file__).resolve().parents[3] / "configs/eval/harness_capability_dev.json")


def test_live_agent_config_survives_chat_merge_in_observed_mode(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    from emet.agent import loop
    from run_harness_capability_agent import run_agent_session

    effective = {}

    def capture_config(**kwargs):
        loop._merge_chat_agent_manip_parameters(
            kwargs["parameters"],
            agent_config="dynav_config.yaml",
            robot=kwargs["robot"],
            agent_section=kwargs.get("agent_section"),
        )
        effective.update(kwargs["parameters"]["agent"])

    monkeypatch.setattr(loop, "run_agent_with_robot", capture_config)
    run_agent_session(object(), lambda _: None, "http://127.0.0.1:1/v1", "test-model")
    assert effective["manip_mode"] == "physical"
    assert effective["tamp_inputs"] == "observed"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_full_stretch_fixture_compiles_with_declared_interventions(tmp_path, case):
    scene, model = build_stretch_case(case, tmp_path / case["id"])
    assert scene.is_file()
    assert model.body("base_link").id >= 0
    present = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target") >= 0
    assert present is case["frames"][0]["target_present"]
    obstacle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle") >= 0
    assert obstacle is case.get("obstacle", False)
    if present:
        # A box-shaped support made the nominal one-box referent ambiguous.
        assert model.geom("target_support").type == mujoco.mjtGeom.mjGEOM_CYLINDER
    if case["family"] in {"recall", "search_act"}:
        assert case["expected"] not in " ".join(agent_turns(case)).lower()
    if case["id"] == "cap-dev-01":
        pan, tilt = initial_gaze(model)
        assert model.actuator("head_pan").ctrlrange[0] <= pan <= model.actuator("head_pan").ctrlrange[1]
        assert model.actuator("head_tilt").ctrlrange[0] <= tilt <= model.actuator("head_tilt").ctrlrange[1]
