# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

from emet.benchmarks.sqa3d.metrics import (
    is_episode_metrics_jsonl,
    score_sqa3d_episode_jsonl,
)


def test_score_episode_jsonl(tmp_path: Path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        '{"question_id": 1, "scene_id": "s", "question": "q", '
        '"gold_answers": ["brown"], "predicted_answer": "brown", "em": true, "em_refined": true}\n'
        '{"question_id": 2, "scene_id": "s", "question": "q2", '
        '"gold_answers": ["yes"], "predicted_answer": "no", "em": false, "em_refined": false}\n',
        encoding="utf-8",
    )
    assert is_episode_metrics_jsonl(path)
    out = score_sqa3d_episode_jsonl(path)
    assert out["em@1"] == 0.5
    assert out["n_questions"] == 2.0
