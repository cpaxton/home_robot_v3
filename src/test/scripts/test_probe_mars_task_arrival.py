# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""An experiment's grasp region cannot leak into later stages or faults."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def contract(monkeypatch):
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "probe_mars_task_arrival.py"))["execute_stage_contract"]


@pytest.mark.parametrize("phase", ["pregrasp", "grasp", "lift", "place"])
def test_region_is_scoped_to_the_named_grasp_stage(contract, phase):
    executor = SimpleNamespace(position_tolerance_m=0.003, orientation_tolerance_rad=0.02)

    def execute(stage, path):
        assert stage == phase
        expected = (0.004, 0.04) if phase == "grasp" else (0.003, 0.02)
        assert (executor.position_tolerance_m, executor.orientation_tolerance_rad) == expected
        return "result"

    assert contract(executor, execute, phase, [], grasp_region=True) == "result"
    assert vars(executor) == {"position_tolerance_m": 0.003, "orientation_tolerance_rad": 0.02}


def test_grasp_exception_restores_original_contract(contract):
    executor = SimpleNamespace(position_tolerance_m=0.003, orientation_tolerance_rad=0.02)

    def rejected(*args):
        raise RuntimeError("measured collision")

    with pytest.raises(RuntimeError, match="measured collision"):
        contract(executor, rejected, "grasp", [], grasp_region=True)
    assert vars(executor) == {"position_tolerance_m": 0.003, "orientation_tolerance_rad": 0.02}
