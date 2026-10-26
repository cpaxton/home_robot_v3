# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import json
import runpy
from pathlib import Path

import mujoco
import pytest


@pytest.fixture
def driver(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    return runpy.run_path(str(root / "scripts/run_shared_harness_integration.py"))


def test_integration_eqa_requires_every_requested_result(driver, tmp_path):
    with pytest.raises(FileNotFoundError):
        driver["eqa_score"](tmp_path)
    (tmp_path / "eqa").mkdir()
    for qid in driver["QUESTIONS"]:
        (tmp_path / "eqa" / f"q{qid}.jsonl").write_text(json.dumps({"correct": True}))
    assert driver["eqa_score"](tmp_path) == len(driver["QUESTIONS"])
    (tmp_path / "eqa" / "q2.jsonl").write_text(json.dumps({"correct": True, "error": "CUDA failure"}))
    with pytest.raises(ValueError, match="Invalid EQA"):
        driver["eqa_score"](tmp_path)


def test_tamp_fixture_compiles_with_preserved_geometry_and_current_assets(driver, tmp_path):
    root = Path(__file__).resolve().parents[3]
    config = driver["tamp_fixture"](root, tmp_path, 1860)
    assert "no_cameras: true" in config.read_text()
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "fixture/scene.xml"))
    assert model.body("object2").pos.tolist() == pytest.approx([-0.02, -0.70, 0.6])
    assert model.body("object1").pos.tolist() == pytest.approx([-0.02, -0.55, 0.6])
