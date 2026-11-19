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
    import xml.etree.ElementTree as ET

    root = Path(__file__).resolve().parents[5]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from tamp_mobile_cases import load_case

    def horizontal_half_extents(name):
        geom = world.find(f"body[@name='{name}']/geom")
        size = [float(value) for value in geom.get("size").split()]
        # Vertical cylinder: size[0] is the radius in the horizontal plane.
        return (size[0], size[0]) if geom.get("type") == "cylinder" else (size[0], size[1])

    world = ET.parse(root / "src/emet/assets/robot/scene_environment.xml").getroot().find("worldbody")
    half = {name: horizontal_half_extents(name) for name in ("object1", "object2")}
    table_pos = [float(value) for value in world.find("body[@name='table']").get("pos").split()[:2]]
    table_size = [float(value) for value in world.find("body[@name='table']/geom").get("size").split()[:2]]
    table = ((table_pos[0] - table_size[0], table_pos[1] - table_size[1]),
             (table_pos[0] + table_size[0], table_pos[1] + table_size[1]))
    region = ((0.06, -0.73), (0.22, -0.57))

    suite = json.loads((root / "configs/eval/mobile_clutter_layout_20261118.json").read_text())
    ids = [row["id"] for row in suite["cases"]]
    assert suite["schema_version"] == 1 and len(ids) == len(set(ids))
    for row in suite["cases"]:
        assert row["object"] in {"object1", "object2"}
        path = tmp_path / "case.json"
        path.write_text(json.dumps(row.get("config", {})))
        assert load_case(path) == row.get("config", {})
        # Fixture-derived distractor footprint. Touching the region boundary is
        # an intentional stress case for the lateral release search.
        distractor = "object2" if row["object"] == "object1" else "object1"
        hx, hy = half[distractor]
        center = row["config"]["initial_distractor_xy"]
        low = (center[0] - hx, center[1] - hy)
        high = (center[0] + hx, center[1] + hy)
        assert all(table[0][i] <= low[i] <= high[i] <= table[1][i] for i in range(2))
        assert any(high[i] <= region[0][i] + 1e-9 or low[i] >= region[1][i] - 1e-9 for i in range(2))


def test_hard_case_index_covers_declared_layouts():
    root = Path(__file__).resolve().parents[5]
    index = json.loads((root / "configs/eval/mobile_transfer_hard_cases_20261118.json").read_text())
    suite = json.loads((root / index["suite"]).read_text())
    ids = [case["id"] for case in index["cases"]] + index["passing_controls"]
    assert index["schema_version"] == 1
    assert len(ids) == len(set(ids))
    assert {case["id"] for case in suite["cases"]} <= set(ids)
    for case in index["cases"]:
        assert case["baseline"]["code"] != "ok"
        assert case["target"] and case["inputs"]
        assert all(not Path(path).is_absolute() and ".." not in Path(path).parts for path in case["inputs"])
