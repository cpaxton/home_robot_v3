# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI smoke for eval_tamp_clutter --dry-run (no sim)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "eval_tamp_clutter.py"


def test_eval_tamp_clutter_dry_run_lists_stretch_on_stretch_sim():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "ithor_cleanup_s1_bin_n3" in combined
    assert "ithor_cleanup_s1_bin_n3_stretch" in combined
    assert "stretch" in combined


def test_eval_tamp_clutter_test_battery_dry_run():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--test-battery",
            "--dry-run",
            "--battery-robots",
            "nori",
            "--battery-scenes",
            "0",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "test_pickplace_nori_s0" in combined
    assert "test_declutter_nori_s0" in combined
    assert "test_navblocked_nori_s0" in combined
    assert "test_navclear_nori_s0" in combined
