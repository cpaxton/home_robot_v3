# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for scripts/hmeqa_significance.py (no GPU)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "hmeqa_significance.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("hmeqa_significance", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sig():
    return _load_mod()


def test_wilson_ci_bounds(sig):
    lo, hi = sig._wilson_ci(5, 8)
    assert lo is not None and hi is not None
    assert 0.0 <= lo < 5 / 8 < hi <= 1.0
    assert sig._wilson_ci(0, 0) == (None, None)


def test_mcnemar_identical_arms_p_one(sig):
    pairs = [
        {"classic_correct": True, "agentic_correct": True},
        {"classic_correct": False, "agentic_correct": False},
        {"classic_correct": True, "agentic_correct": True},
    ]
    m = sig.mcnemar_exact(pairs)
    assert m["discordant"] == 0
    assert m["both_correct"] == 2
    assert m["both_wrong"] == 1
    assert m["p_value"] == 1.0


def test_mcnemar_discordant_tallies(sig):
    # 1 classic-only, 3 agentic-only → discordant=4, agentic favored
    pairs = [
        {"classic_correct": True, "agentic_correct": False},  # classic_only
        {"classic_correct": False, "agentic_correct": True},  # agentic_only
        {"classic_correct": False, "agentic_correct": True},
        {"classic_correct": False, "agentic_correct": True},
        {"classic_correct": True, "agentic_correct": True},  # both
        {"classic_correct": False, "agentic_correct": False},  # neither
    ]
    m = sig.mcnemar_exact(pairs)
    assert m["classic_only"] == 1
    assert m["agentic_only"] == 3
    assert m["both_correct"] == 1
    assert m["both_wrong"] == 1
    assert m["discordant"] == 4
    assert m["p_value"] is not None
    assert 0.0 < m["p_value"] <= 1.0


def test_paired_rows_join_on_shared_ids(sig):
    classic = {
        1: {"correct": True, "planning_steps": 10, "predicted_answer": "A"},
        2: {"correct": False, "planning_steps": 20, "predicted_answer": "B"},
        99: {"correct": True, "planning_steps": 5, "predicted_answer": "C"},
    }
    agentic = {
        1: {"correct": True, "planning_steps": 4, "predicted_answer": "A"},
        2: {"correct": True, "planning_steps": 8, "predicted_answer": "D"},
        3: {"correct": False, "planning_steps": 1, "predicted_answer": ""},
    }
    pairs = sig.paired_rows(classic, agentic)
    assert [p["question_id"] for p in pairs] == [1, 2]
    assert pairs[0]["classic_correct"] is True and pairs[0]["agentic_correct"] is True
    assert pairs[1]["classic_correct"] is False and pairs[1]["agentic_correct"] is True


def test_analyze_pairs_end_to_end(sig, tmp_path: Path):
    pairs = [
        {
            "question_id": 1,
            "classic_correct": True,
            "agentic_correct": True,
            "classic_steps": 40,
            "agentic_steps": 15,
        },
        {
            "question_id": 2,
            "classic_correct": False,
            "agentic_correct": True,
            "classic_steps": 50,
            "agentic_steps": 18,
        },
        {
            "question_id": 3,
            "classic_correct": True,
            "agentic_correct": False,
            "classic_steps": 30,
            "agentic_steps": 20,
        },
        {
            "question_id": 4,
            "classic_correct": False,
            "agentic_correct": False,
            "classic_steps": 60,
            "agentic_steps": 22,
        },
    ]
    result = sig.analyze_pairs(pairs, n_boot=200, seed=1)
    assert result["n_paired"] == 4
    assert result["classic"]["correct"] == 2
    assert result["agentic"]["correct"] == 2
    assert result["mcnemar"]["classic_only"] == 1
    assert result["mcnemar"]["agentic_only"] == 1
    assert result["wilcoxon_steps"]["n"] == 4
    assert result["wilcoxon_steps"]["mean_diff"] == pytest.approx((25 + 32 + 10 + 38) / 4)
    # agentic fewer steps on every pair → Wilcoxon should favor classic>agentic
    assert result["wilcoxon_steps"]["p_value"] is not None
    assert result["wilcoxon_steps"]["p_value"] < 0.1

    # Round-trip via summary JSON shape
    summary = {
        "classic": {
            "per": [
                {"q": p["question_id"], "correct": p["classic_correct"], "planning_steps": p["classic_steps"]}
                for p in pairs
            ]
        },
        "agentic": {
            "per": [
                {"q": p["question_id"], "correct": p["agentic_correct"], "planning_steps": p["agentic_steps"]}
                for p in pairs
            ]
        },
    }
    path = tmp_path / "h2h_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    again = sig.analyze_summary(loaded, n_boot=50, seed=0)
    assert again["n_paired"] == 4
    assert again["mcnemar"]["discordant"] == 2


def test_analyze_run_dir_jsonl(sig, tmp_path: Path):
    for arm, rows in (
        (
            "classic",
            [
                {"question_id": 10, "correct": True, "planning_steps": 40},
                {"question_id": 11, "correct": False, "planning_steps": 50},
            ],
        ),
        (
            "agentic",
            [
                {"question_id": 10, "correct": True, "planning_steps": 12},
                {"question_id": 11, "correct": True, "planning_steps": 14},
            ],
        ),
    ):
        (tmp_path / f"{arm}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )
    result = sig.analyze_run_dir(tmp_path, n_boot=100, seed=0)
    assert result["n_paired"] == 2
    assert result["mcnemar"]["agentic_only"] == 1
    assert result["mcnemar"]["classic_only"] == 0
