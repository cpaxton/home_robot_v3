# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI coverage for scripted_tamp_pick_place (Sourccey table / counter GT body)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "scripted_tamp_pick_place.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("scripted_tamp_pick_place_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scripted_tamp_pick_place_help_lists_object_gt_body():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--object-gt-body" in result.stdout
    assert "--rerun" in result.stdout
    assert "--rerun-hold-s" in result.stdout
    assert "sourccey" in result.stdout.lower()


def test_connect_benchmark_robot_accepts_rerun_kwargs():
    import inspect

    from emet.eval.sim_eval_session import connect_benchmark_robot

    sig = inspect.signature(connect_benchmark_robot)
    assert "enable_rerun_server" in sig.parameters
    assert "rerun_headless" in sig.parameters


def test_find_graspable_body_pins_object_gt_body(monkeypatch):
    module = _load_script_module()
    pl = {
        "cab_main": {"cat": "cabinet", "pos": [1.0, 0.0, 0.9], "quat": [1.0, 0.0, 0.0, 0.0]},
        "obj_main": {"cat": "sugar cube", "pos": [0.2, -0.5, 0.9], "quat": [1.0, 0.0, 0.0, 0.0]},
    }
    robot = SimpleNamespace(get_emet_session=lambda: {"sim_object_placements": pl})
    monkeypatch.setattr(
        "emet.memory.graph_eqa.sim_ground_truth_graph.read_sim_object_placements",
        lambda _sess: pl,
    )
    monkeypatch.setattr(
        "emet.perception.grasps.asset_id.resolve_asset_id_against_grasps_dir",
        lambda *a, **k: None,
    )

    body, info, poses = module._find_graspable_body(
        robot,
        object_query="obj",
        asset_id=None,
        oracle_client=SimpleNamespace(predict=lambda **k: []),
        object_gt_body="obj_main",
    )
    assert body == "obj_main"
    assert info["cat"] == "sugar cube"
    assert poses
    T = poses[0].T_world
    assert abs(float(T[2, 2]) + 1.0) < 1e-9  # top-down: grasp +Z into the object
