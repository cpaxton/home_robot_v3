# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Dynagraph GT navigation + frontier exploration benchmarks (sim)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_run = os.environ.get("RUN_DYNAGRAPH_NAV_BENCHMARK", "0").strip().lower()
RUN_NAV_BENCH = _run in ("1", "true", "yes", "on")


@pytest.mark.skipif(not RUN_NAV_BENCH, reason="RUN_DYNAGRAPH_NAV_BENCHMARK=0")
@pytest.mark.timeout(3600)
def test_dynagraph_nav_benchmark_all_tiers():
    script = REPO / "src/test/app/run_dynagraph_nav_benchmark.py"
    r = subprocess.run(
        [sys.executable, str(script), "--all"],
        cwd=REPO,
        timeout=3300,
        env={**os.environ, "EMET_SIM_NAV_TELEPORT": "1"},
    )
    report_path = (
        Path(os.environ.get("DYNAGRAPH_NAV_BENCH_BASE", "/tmp/dynagraph_nav_bench")) / "nav_benchmark_report.json"
    )
    assert report_path.is_file(), f"missing report at {report_path}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("all_pass") is True, report
    assert r.returncode == 0
