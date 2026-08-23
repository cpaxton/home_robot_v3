# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shell hygiene for the managed TAMP agent-tools gate (no sim)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GATE = REPO / "scripts" / "run_tamp_agent_tools_gate.sh"


def test_gate_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(GATE)], check=True, cwd=REPO)


def test_gate_dry_run_preserves_tool_calls_json(tmp_path):
    out = tmp_path / "gate_out"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "OUT_DIR": str(out),
        "ITEMS": "chat kinematic stretch",
    }
    proc = subprocess.run(
        ["bash", str(GATE)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    log = (out / "gate.log").read_text(encoding="utf-8")
    combined = proc.stdout + proc.stderr + log
    assert proc.returncode == 0, combined
    assert '{"name":"scene_tasks"' in combined
    assert '"object_filter":"bowl"' in combined
    assert '"robot":"rby1"' in combined
    assert '{"name":"plan_pick_place"' in combined
    assert '"object_name":"red cylinder"' in combined
    assert '"receptacle_name":"blue cube"' in combined
    assert "DRY_RUN: skip execution" in combined
    assert (out / "gate_summary.txt").is_file()


def test_gate_smoke_profile_dry_run_is_kinematic_agent_only(tmp_path):
    out = tmp_path / "gate_smoke"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "OUT_DIR": str(out),
        "PROFILE": "smoke",
    }
    proc = subprocess.run(
        ["bash", str(GATE)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    log = (out / "gate.log").read_text(encoding="utf-8")
    combined = proc.stdout + proc.stderr + log
    assert proc.returncode == 0, combined
    meta = (out / "META.txt").read_text(encoding="utf-8")
    assert "profile=smoke" in meta
    assert "items=kinematic" in meta
    assert "configs/sim/molmospaces_ithor_train_0.yaml --manip-mode kinematic" in combined
    assert '{"name":"scene_tasks"' in combined
    assert (out / "molmospaces_kinematic_chat.result").read_text(encoding="utf-8").strip() == "DRY_RUN"
    assert not (out / "molmospaces_chat_teleport.result").exists()
    assert not (out / "stretch_teleport_control.result").exists()
    assert not (out / "robocasa_floor_suite.result").exists()
    assert not (out / "robocasa_floor_smoke.result").exists()


def test_gate_full_profile_dry_run_includes_floor_not_smoke(tmp_path):
    out = tmp_path / "gate_full"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "OUT_DIR": str(out),
        "PROFILE": "full",
    }
    proc = subprocess.run(
        ["bash", str(GATE)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    log = (out / "gate.log").read_text(encoding="utf-8")
    combined = proc.stdout + proc.stderr + log
    assert proc.returncode == 0, combined
    assert "eval_tamp_floor.py --output-dir" in combined
    assert "eval_tamp_floor.py --smoke" not in combined
    assert (out / "robocasa_floor_suite.result").is_file()
    assert not (out / "robocasa_floor_smoke.result").exists()
