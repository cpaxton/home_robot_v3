# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""VLM-free multi-env nav/explore smoke (Robocasa, OVMM kitchen, MolmoSpaces).

Run:
  uv run emet test -v src/test/simulation/test_multi_env_nav_explore_smoke.py
  uv run emet test -k robocasa_l1 src/test/simulation/test_multi_env_nav_explore_smoke.py
"""

from __future__ import annotations

import os

import pytest

from emet.eval.nav_explore_smoke import (
    default_nav_explore_cases,
    molmospaces_wrapper_available,
    run_nav_explore_smoke,
)

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")

_CASES = {c.name: c for c in default_nav_explore_cases()}


@pytest.mark.sim
@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(900)
@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_multi_env_nav_explore_smoke(case_name: str):
    case = _CASES[case_name]
    if case.kind == "molmospaces" and not molmospaces_wrapper_available():
        pytest.skip("MolmoSpaces wrapper not installed (.venv-molmospaces)")

    summary = run_nav_explore_smoke(case)
    if summary.get("skipped"):
        pytest.skip(str(summary.get("skip_reason") or "skipped"))

    assert summary.get("ok"), (
        f"nav/explore smoke failed for {case_name}: "
        f"delta_xy={summary.get('delta_xy_m')} "
        f"explored_delta={summary.get('explored_delta')} "
        f"explore_ok={summary.get('explore_ok')} "
        f"error={summary.get('error')!r} "
        f"summary={summary}"
    )
