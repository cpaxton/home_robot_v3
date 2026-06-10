# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from pathlib import Path

from emet.benchmarks.sqa3d.aggregate import aggregate_sqa3d_jsonl, write_aggregate_csv


def test_aggregate_sqa3d_jsonl(tmp_path: Path):
    path = tmp_path / "batch.jsonl"
    path.write_text(
        '{"question_id": 1, "method": "dynagraph", "replay_backend": "sens", '
        '"sens_match_xy_m": 0.2, "gold_answers": ["white"], "predicted_answer": "white", '
        '"em": true, "em_refined": true, "planning_steps": 10, "question": "What color?"}\n'
        '{"question_id": 2, "method": "dynagraph", "replay_backend": "mesh", '
        '"gold_answers": ["yes"], "predicted_answer": "no", "em": false, "em_refined": false, '
        '"planning_steps": 8, "question": "Is it on?"}\n',
        encoding="utf-8",
    )
    row = aggregate_sqa3d_jsonl(path)
    assert row["method"] == "dynagraph"
    assert row["n_episodes"] == 2
    assert row["em@1"] == 0.5
    assert row["tp"] == 1
    assert row["fp"] == 1
    assert row["mean_sens_match_xy_m"] == 0.2

    csv_path = tmp_path / "aggregate.csv"
    write_aggregate_csv([row], csv_path)
    text = csv_path.read_text(encoding="utf-8")
    assert "em@1" in text
    assert "dynagraph" in text
