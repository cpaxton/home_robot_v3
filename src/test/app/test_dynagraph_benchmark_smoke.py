# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Unit tests for Dynagraph benchmark infrastructure (no sim by default)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from emet.memory.graph_eqa.dynagraph_eval import (
    graph_stats_from_report_text,
    resolve_episode_dir,
)
from emet.memory.graph_eqa.question_bank import load_question_bank, score_eqa_results


def test_resolve_episode_dir_nested():
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    assert resolve_episode_dir(fixtures).is_dir()


def test_graph_stats_from_report_text():
    text = "Scene graph\nNodes (3)\nEdges (2)"
    s = graph_stats_from_report_text(text)
    assert s["node_count"] == 3.0


def test_question_bank_load():
    bank_path = Path(__file__).resolve().parents[2] / "emet/config/benchmarks/dynagraph_questions.yaml"
    bank = load_question_bank(bank_path, env_filter="default_table")
    assert len(bank) >= 2
    assert bank[0]["expected_tokens"]


def test_score_eqa_results_tokens():
    rows = [
        {
            "question": "What colors?",
            "answer": "I see a red cylinder and a blue cube.",
            "expected_tokens": ["red", "blue"],
        }
    ]
    scored = score_eqa_results(rows)
    assert scored["accuracy"] == 1.0
    assert scored["questions"][0]["pass"] is True


def test_dynagraph_eval_on_fixture_gt_snippet():

    # Minimal: only tests import + empty dir handling would fail; use fusion fixture eval path
    gt = json.loads((Path(__file__).resolve().parents[1] / "fixtures/gt_robocasa_seed0_snippet.json").read_text())
    from emet.memory.graph_eqa.graph_object_fusion.calibrate import load_calibration_frames_jsonl
    from emet.memory.graph_eqa.graph_object_fusion.evaluate import score_detections_vs_gt

    frames = load_calibration_frames_jsonl(
        Path(__file__).resolve().parents[1] / "fixtures/calibration_frames_snippet.jsonl"
    )
    m = score_detections_vs_gt(gt, frames)
    assert m["spatial_recall"] >= 0.5


_run_bench = os.environ.get("RUN_DYNAGRAPH_BENCHMARK_SMOKE", "0").strip().lower()
RUN_BENCH_SIM = _run_bench not in ("0", "false", "no", "off")


@pytest.mark.skipif(not RUN_BENCH_SIM, reason="RUN_DYNAGRAPH_BENCHMARK_SMOKE=0")
@pytest.mark.timeout(1800)
def test_benchmark_smoke_default_only():
    """Full sim smoke (default table tiers only). Set RUN_DYNAGRAPH_BENCHMARK_SMOKE=1."""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[3]
    r = subprocess.run(
        [sys.executable, str(repo / "src/test/app/run_dynagraph_benchmark_smoke.py"), "--default"],
        cwd=repo,
        timeout=1200,
    )
    assert r.returncode == 0
