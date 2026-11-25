# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Calibration harness: dry-run + pure aggregation/recommendation."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "calibrate_clearance_floor.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("calibrate_clearance_floor", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_dry_run_validates_manifest_and_prints_matrix():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "7 floors x 6 tasks = 42 runs" in combined
    assert "ithor_nav_goal_00_rby1_n8_table_2" in combined
    assert "robocasa_rby1_floor_to_counter_mcts" in combined


def test_recommend_floor_picks_the_knee():
    mod = _load()
    agg = [
        {"floor_m": 0.10, "success_rate": 0.5},
        {"floor_m": 0.14, "success_rate": 1.0},
        {"floor_m": 0.18, "success_rate": 1.0},
        {"floor_m": 0.22, "success_rate": 1.0},
    ]
    # Smallest floor achieving the best (1.0) is the knee: 0.14.
    assert mod.recommend_floor(agg) == 0.14


def test_recommend_floor_empty_is_none():
    mod = _load()
    assert mod.recommend_floor([]) is None


def test_aggregate_rates_and_margins():
    mod = _load()
    floor_summary = {
        0.22: [
            {"task_success": True, "refused": False, "margin_m": None},
            {"task_success": False, "refused": True, "margin_m": -0.02},
        ]
    }
    agg = mod.aggregate(floor_summary)
    assert agg[0]["n_tasks"] == 2
    assert agg[0]["success_rate"] == 0.5
    assert agg[0]["refusal_rate"] == 0.5
    assert agg[0]["min_margin_m"] == -0.02
