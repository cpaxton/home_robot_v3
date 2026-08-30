# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shell hygiene for OVMM eval orchestrators (no sim, no GPU)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
H2H = REPO / "scripts" / "run_ovmm_agentic_h2h.sh"
PAPER = REPO / "scripts" / "run_paper_matrix.sh"
JOINT = REPO / "scripts" / "run_habitat_ovmm_joint_gate.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


@pytest.mark.parametrize("script", [H2H, PAPER, JOINT], ids=["h2h", "paper", "joint"])
def test_ovmm_eval_scripts_are_valid_bash(script: Path):
    subprocess.run(["bash", "-n", str(script)], check=True, cwd=REPO)


def test_h2h_does_not_collapse_arm_args_into_one_token():
    text = H2H.read_text(encoding="utf-8")
    code = "\n".join(ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#"))
    assert 'args=("$(arm_args' not in code
    assert "local -n _argv" in text


def test_h2h_arm_args_expand_to_separate_cli_tokens():
    """Quoted ``$(arm_args)`` used to pass one argv blob to ``emet ovmm find``."""
    text = H2H.read_text(encoding="utf-8")
    start = text.index("arm_args() {")
    end = text.index("\n}", start)
    func = text[start : end + 2]
    probe = f"""
set -euo pipefail
BACKEND=dynagraph
ROUNDS=6
{func}
args=()
arm_args args "default_table_rby1_s0_distinct_recep"
printf 'N=%s\\n' "${{#args[@]}}"
i=0
for a in "${{args[@]}}"; do
  printf 'A%s=%s\\n' "$i" "$a"
  i=$((i + 1))
done
"""
    proc = subprocess.run(
        ["bash", "-c", probe],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    lines = dict(line.split("=", 1) for line in proc.stdout.strip().splitlines() if "=" in line)
    n = int(lines["N"])
    assert n >= 8, out
    tokens = [lines[f"A{i}"] for i in range(n)]
    assert "--episodes" in tokens
    assert "--backend" in tokens
    assert "dynagraph" in tokens
    assert "--episode-id" in tokens
    assert "default_table_rby1_s0_distinct_recep" in tokens
    assert not any(" --" in t for t in tokens), tokens


def test_h2h_rotate_is_mapping_budget_zero_unified_explores():
    text = H2H.read_text(encoding="utf-8")
    assert 'args+=(--mapping-max-nav-steps "$EXPLORE_STEPS" --no-scene-cache)' in text
    assert "args+=(--mapping-max-nav-steps 0)" in text
    assert "--explore-steps" not in text
    assert 'exit "$gate_rc"' in text
    assert "status_close fail" in text


def test_joint_gate_writes_phase_out_dirs():
    text = JOINT.read_text(encoding="utf-8")
    assert 'OUT_DIR="$OUT_BASE/habitat"' in text
    assert 'OUT_DIR="$OUT_BASE/ovmm"' in text


def test_paper_matrix_s0_uses_smoke_and_fails_closed():
    text = PAPER.read_text(encoding="utf-8")
    assert 'env PROFILE=smoke OUT_DIR="$OUT_BASE/ovmm_s0"' in text
    assert 'env PROFILE=slice OUT_DIR="$OUT_BASE/ovmm_s0"' not in text
    assert 'exit "$matrix_rc"' in text


def test_joint_gate_fails_closed_after_both_phases():
    text = JOINT.read_text(encoding="utf-8")
    assert 'exit "$gate_rc"' in text
    assert not text.rstrip().endswith("exit 0")
