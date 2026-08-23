# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI smoke for eval_tamp_floor --smoke (no sim)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "eval_tamp_floor.py"


def test_eval_tamp_floor_smoke_dry_run_selects_rby1_mcts():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--smoke", "--dry-run"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "robocasa_rby1_floor_to_counter_mcts" in combined
    assert "dynagraph" in combined.lower()
