"""Tests for scripts/build_eval_figure_pack.py OVMM aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from build_eval_figure_pack import _ovmm_success, _summarize_ovmm  # noqa: E402


def test_ovmm_success_handles_bool_and_float():
    assert _ovmm_success(True) is True
    assert _ovmm_success(False) is False
    assert _ovmm_success(1.0) is True
    assert _ovmm_success(0.5) is True
    assert _ovmm_success(0.0) is False


def test_summarize_ovmm_splits_object_and_recep(tmp_path: Path):
    run_id = "eval_smoke_test"
    root = tmp_path / f"{run_id}_dynamem"
    root.mkdir()
    (root / "ep_a_dynamem.json").write_text(
        json.dumps(
            {
                "episode_id": "ep_a",
                "backend": "dynamem",
                "object_query": "lamp",
                "start_recep": "bed",
                "goal_recep": "table",
                "find_object_success": True,
                "find_recep_success": False,
                "find_partial_success": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (root / "ep_b_dynamem.json").write_text(
        json.dumps(
            {
                "episode_id": "ep_b",
                "backend": "dynamem",
                "object_query": "lamp",
                "start_recep": "bed",
                "goal_recep": "table",
                "find_object_success": True,
                "find_recep_success": True,
                "find_partial_success": 1.0,
            }
        ),
        encoding="utf-8",
    )

    summary = _summarize_ovmm(run_id, tmp_path)
    stats = summary["backends"]["dynamem"]
    assert summary["episodes"][0]["task"] == "move lamp from bed to table"
    assert stats["n"] == 2
    assert stats["find_object_success"] == 2
    assert stats["find_recep_success"] == 1
    assert stats["partial_success"] == 2
    assert stats["find_object_success_rate"] == 1.0
    assert stats["find_recep_success_rate"] == 0.5
    assert stats["find_partial_success_rate"] == 1.0
    assert stats["find_both_success"] == 1
    assert stats["find_object_only"] == 1
    assert stats["find_both_success_rate"] == 0.5
    assert stats["find_object_only_rate"] == 0.5
    assert summary["episodes"][0]["outcome"] == "object_only"
    assert summary["episodes"][1]["outcome"] == "both"
