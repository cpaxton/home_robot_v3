# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Reject invalid fixture overrides before native physics sees them."""

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "case",
    [
        [],
        {"feature": True},
        {"mobile": 1},
        {"initial_base_x": float("nan")},
        {"approach_m": float("inf")},
        {"initial_object_xy": [0, float("nan")]},
        {"initial_distractor_xy": [0]},
        {"placement_region": [[0, 0], [-1, 1]]},
    ],
)
def test_invalid_initial_conditions_reject_before_model_creation(monkeypatch, tmp_path, case):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5] / "scripts"))
    from tamp_mobile_cases import load_case

    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="invalid_mobile_case_config"):
        load_case(path)


def test_declared_cases_parse_and_scaled_fixture_keeps_density(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[5]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from run_mobile_map_regression import prepare_case
    from tamp_mobile_cases import load_case

    suite = json.loads((root / "configs/eval/mobile_map_regression_20261117.json").read_text())
    for row in suite["cases"]:
        path = tmp_path / "case.json"
        path.write_text(json.dumps(row.get("config", {})))
        assert load_case(path) == row.get("config", {})
    fixture = tmp_path / "source"
    fixture.mkdir()
    (fixture / "scene_environment.xml").write_text(
        '<mujoco><worldbody><body name="object2"><geom size=".02 .04 .04" mass=".5"/></body></worldbody></mujoco>'
    )
    import xml.etree.ElementTree as ET

    prepare_case(fixture, tmp_path / "scaled", {"object": "object2", "object_scale": 1.1})
    geom = ET.parse(tmp_path / "scaled/scene_environment.xml").getroot().find("worldbody/body/geom")
    assert float(geom.get("size").split()[0]) == pytest.approx(0.022)
    assert float(geom.get("mass")) == pytest.approx(0.5 * 1.1**3)
    assert ".02 .04 .04" in (fixture / "scene_environment.xml").read_text()


def test_clutter_layout_cases_parse_within_declared_geometry(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[5]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from tamp_mobile_cases import load_case

    suite = json.loads((root / "configs/eval/mobile_clutter_layout_20261118.json").read_text())
    ids = [row["id"] for row in suite["cases"]]
    assert suite["schema_version"] == 1 and len(ids) == len(set(ids))
    for row in suite["cases"]:
        assert row["object"] in {"object1", "object2"}
        path = tmp_path / "case.json"
        path.write_text(json.dumps(row.get("config", {})))
        assert load_case(path) == row.get("config", {})
        # Distractors are declared on the 0.6 x 0.5 m table and clear the
        # unchanged placement region, so a lateral release correction can move
        # the held object away from any of the four surrounding sides.
        distractor = row["config"]["initial_distractor_xy"]
        assert -0.58 <= distractor[0] <= 0.58 and -1.48 <= distractor[1] <= -0.52
