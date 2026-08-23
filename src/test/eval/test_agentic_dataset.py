# Copyright (c) Chris Paxton 2026

import json

import numpy as np

from emet.eval.agentic_dataset import mine_episode_records, mine_evidence_dataset, scene_split
from emet.habitat.hm3d_semantics import Hm3dSemanticLabeler


def test_scene_split_is_stable_and_scene_level():
    assert scene_split("scene-a") == scene_split("scene-a")
    assert scene_split("scene-a", salt="other") in {"train", "val", "test"}


def test_semantic_visibility_is_view_level():
    labeler = Hm3dSemanticLabeler({2: "woven basket", 3: "chair"})
    semantic = np.array([[0, 2, 2], [0, 3, 2]], dtype=np.uint32)
    depth = np.ones_like(semantic, dtype=np.float32)
    visible = labeler.visibility_for_phrase(semantic, "basket", depth)
    assert visible["gt_in_view"] is True
    assert visible["gt_visible_pixels"] == 3
    assert visible["gt_bbox_xyxy"] == [1, 0, 3, 2]
    absent = labeler.visibility_for_phrase(semantic, "towel", depth)
    assert absent["gt_in_view"] is False
    assert absent["gt_visible_pixels"] == 0


def test_mine_records_uses_prefix_only_and_view_labels(tmp_path):
    episode = tmp_path / "run" / "q0017_dynagraph"
    (episode / "images").mkdir(parents=True)
    (episode / "images" / "rgb_0007.png").write_bytes(b"png")
    (episode / "metrics.json").write_text(
        json.dumps(
            {
                "scene": "scene-a",
                "question_id": 17,
                "question": "Where is the basket?",
                "gold_answer_letter": "D",
                "correct": True,
            }
        )
    )
    rows = [
        {"tool": "inspect_graph", "round": 0, "hypotheses": [{"obs_id": 7, "score": 1.0}]},
        {
            "tool": "verify_siglip",
            "round": 1,
            "obs_id": 7,
            "phrase": "basket",
            "decision": "PRESENT",
            "sim": 0.13,
            "dense_sim": 0.13,
            "gt_view_label_available": True,
            "gt_in_view": True,
            "gt_visible_pixels": 120,
        },
        {"tool": "submit_answer", "round": 2, "verified": True},
    ]
    (episode / "agentic_trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    mined = mine_episode_records(episode)
    verify = next(record for record in mined if record.action_taken == "verify_siglip")
    assert verify.prior_verifies == []
    assert verify.gt["gt_in_view"] is True
    assert verify.label_source == "hm3d_semantic_sensor"
    assert verify.rgb_path and verify.rgb_path.endswith("rgb_0007.png")
    submit = next(record for record in mined if record.action_taken == "submit_answer")
    assert len(submit.prior_verifies) == 1

    output = tmp_path / "dataset.jsonl"
    manifest = mine_evidence_dataset(tmp_path, output)
    assert manifest["n_view_labeled"] == 1
    assert manifest["scene_leakage"] == []
    assert output.is_file()
