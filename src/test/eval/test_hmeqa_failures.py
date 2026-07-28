# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for emet.eval.hmeqa_failures (no GPU)."""

from __future__ import annotations

from pathlib import Path

from emet.eval.hmeqa_failures import _trace_path_for_row, classify_pair


def test_classify_scored_vs_submit_mismatch():
    row = classify_pair(
        qid=28,
        classic={
            "correct": True,
            "predicted_answer": "D",
            "gold_answer_letter": "D",
            "planning_steps": 19,
            "graph_health": {"prompt_obs_count": 4, "prompt_node_count": 14, "n_object": 8},
            "question": "How many red pillows? A)1 B)3 C)4 D)2",
        },
        agentic={
            "correct": False,
            "predicted_answer": "A",
            "gold_answer_letter": "D",
            "planning_steps": 14,
            "raw_eqa_output": "Caption:\n...\n[salvage]\nanswer:\nA\n",
            "graph_health": {"prompt_obs_count": 4, "prompt_node_count": 16, "n_object": 9},
            "debug_bundle_dir": "/tmp/fake",
            "question": "How many red pillows? A)1 B)3 C)4 D)2",
        },
        trace={
            "n_rows": 5,
            "n_verify": 1,
            "n_assess": 1,
            "n_explore": 0,
            "tools": {},
            "last_assess": {"suggested_answer": "D", "target": "red pillows"},
            "last_verify": {
                "fused_verified": True,
                "decision": "ABSENT",
                "phrase": "red pillows",
            },
            "last_submit": {
                "final_answer": "D",
                "answer_source": "vlm_suggested",
                "vlm_suggested": "D",
            },
            "sync_scored": None,
            "trace_path": None,
        },
    )
    assert row["pair_kind"] == "classic_only"
    assert row["bucket"] == "scored_vs_submit_mismatch"


def test_classify_empty_abstain():
    row = classify_pair(
        qid=11,
        classic={"correct": True, "predicted_answer": "D", "gold_answer_letter": "D"},
        agentic={
            "correct": False,
            "predicted_answer": "",
            "gold_answer_letter": "D",
            "debug_bundle_dir": "/tmp/x",
            "question": "Where is the silver trash can? A)x B)y C)z D)w",
        },
        trace={
            "n_rows": 10,
            "n_verify": 6,
            "n_assess": 6,
            "n_explore": 5,
            "tools": {},
            "last_assess": {"suggested_answer": None, "target": "silver trash can"},
            "last_verify": {"fused_verified": False, "decision": "ABSENT", "phrase": "silver trash can"},
            "last_submit": {"final_answer": "unknown", "answer_source": "query"},
            "sync_scored": None,
        },
    )
    assert row["bucket"] == "empty_or_abstain"


def test_trace_path_prefers_newest_mtime(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    root = home / ".cache" / "habitat_eqa" / "episodes" / "h2h_agentic_q0042"
    older = root / "run_a" / "agentic_trace.jsonl"
    newer = root / "run_b" / "agentic_trace.jsonl"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text('{"tool":"explore_frontier"}\n', encoding="utf-8")
    newer.write_text('{"tool":"submit_answer"}\n', encoding="utf-8")
    # Ensure newer mtime even on coarse filesystems.
    import os
    import time

    os.utime(older, (time.time() - 100, time.time() - 100))
    os.utime(newer, (time.time(), time.time()))
    got = _trace_path_for_row({}, 42)
    assert got == newer.resolve() or got == newer
    assert got is not None
    assert got.name == "agentic_trace.jsonl"
    assert got.parent.name == "run_b"