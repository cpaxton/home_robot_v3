# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for emet.eval.hmeqa_inspect (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from emet.eval.hmeqa_inspect import (
    format_inspect_text,
    inspect_episode,
    list_scored_episodes,
    media_paths,
    summarize_trace,
)


def test_summarize_trace_assess_and_explore():
    rows = [
        {
            "tool": "vlm_assess",
            "obs_id": 0,
            "present": False,
            "answerable": False,
            "suggested_answer": None,
            "reason": "no bowl",
        },
        {"tool": "explore_frontier", "source": "vlm_frontier", "toward": [1.0, 2.0]},
        {
            "tool": "submit_answer",
            "final_answer": "B",
            "answer_source": "vlm_suggested",
            "vlm_suggested": "B",
            "verified": False,
        },
        {"tool": "summary", "budget_hit": True, "n_nav": 8, "n_explore": 3, "n_rounds": 4},
    ]
    s = summarize_trace(rows)
    assert s["n_assess"] == 1
    assert s["n_explore"] == 1
    assert s["present_any"] is False
    assert s["explore_sources"] == {"vlm_frontier": 1}
    assert s["last_submit"]["final_answer"] == "B"
    assert s["budget_hit"] is True


def test_inspect_episode_bundle(tmp_path: Path):
    out = tmp_path / "run"
    bundle = out / "bundles" / "agentic_q105"
    frames = bundle / "frames"
    frames.mkdir(parents=True)
    (frames / "rgb_0000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (bundle / "agentic_trace.jsonl").write_text(
        json.dumps(
            {
                "tool": "vlm_assess",
                "obs_id": 0,
                "present": False,
                "answerable": False,
                "suggested_answer": None,
                "reason": "outdoor",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "agentic_q105.jsonl").write_text(
        json.dumps(
            {
                "question_id": 105,
                "question": "Where is the fruit bowl? A) kitchen B) patio",
                "predicted_answer": "B",
                "gold_answer_letter": "A",
                "correct": False,
                "planning_steps": 12,
                "observations": 40,
                "graph_health": {"top_labels": [{"label": "chair", "count": 3}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = inspect_episode(out, 105)
    assert payload["episode"]["gold"] == "A"
    assert payload["trace"]["n_assess"] == 1
    assert payload["media"]["frames_n"] == 1
    assert payload["media"]["primary_rgb_n"] == 1
    text = format_inspect_text(payload)
    assert "MISS" in text
    assert "feh" in text
    misses = [r for r in list_scored_episodes(out) if not r["correct"]]
    assert len(misses) == 1
    assert misses[0]["qid"] == 105


def test_media_paths_prefers_dense_frames_all(tmp_path: Path):
    bundle = tmp_path / "b"
    sparse = bundle / "frames"
    dense = bundle / "frames_all"
    sparse.mkdir(parents=True)
    dense.mkdir()
    (sparse / "rgb_0000.png").write_bytes(b"x")
    for i in range(5):
        (dense / f"frame_{i:04d}.png").write_bytes(b"x")
    media = media_paths(bundle)
    assert media["primary_rgb_kind"] == "frames_all"
    assert media["primary_rgb_n"] == 5
