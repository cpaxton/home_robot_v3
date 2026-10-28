# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
import hashlib
from dataclasses import replace

import pytest
from PIL import Image

from emet.eval.embodied_qa_datasets import abstain_questions, express_questions
from emet.eval.embodied_qa_evidence import bind_recorded_evidence


def fixture(tmp_path):
    frames = []
    for index, color in ((3, "red"), (1, "blue")):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8), color).save(path)
        frames.append({"index": index, "rgb": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    row = {
        "question_id": "id",
        "question": "What is on the wall?",
        "answer": "private reference",
        "category": "private category",
        "episode_history": "opaque/video",
        "frame_list": [1, 3],
    }
    questions = abstain_questions([row], [{**row, "question": "Which object?", "answer": "unknown"}])
    manifest = {
        "schema_version": 1,
        "benchmark": "abstaineqa",
        "episodes": {"id": {"evidence_id": "opaque/video", "scene": None, "frames": frames}},
    }
    return questions, manifest


def test_paired_evidence_preserves_official_order_and_hides_annotations(tmp_path):
    questions, manifest = fixture(tmp_path)
    a, b = [bind_recorded_evidence(q, manifest, root=tmp_path) for q in questions]
    assert [f.index for f in a.frames] == [1, 3]
    assert a.frames == b.frames
    payload = a.model_input(max_images=2)
    assert payload[0] == "What is on the wall?" and len(payload) == 3
    assert payload[1].getpixel((0, 0)) == (0, 0, 255)
    assert b.model_input(max_images=2)[0] == "Which object?"
    with pytest.raises(ValueError, match="budget"):
        a.model_input(max_images=1)  # No silent selection/truncation.


@pytest.mark.parametrize(
    "failure",
    ["scene", "evidence_id", "missing_index", "duplicate_index", "missing_file", "hash", "escape", "benchmark"],
)
def test_incorrect_or_unavailable_evidence_fails_before_inference(tmp_path, failure):
    questions, manifest = fixture(tmp_path)
    entry = manifest["episodes"]["id"]
    if failure in ("scene", "evidence_id"):
        entry[failure] = "wrong"
    elif failure == "missing_index":
        entry["frames"] = entry["frames"][:1]
    elif failure == "duplicate_index":
        entry["frames"].append(entry["frames"][0])
    elif failure == "missing_file":
        entry["frames"][0]["rgb"] = "missing.png"
    elif failure == "hash":
        entry["frames"][0]["sha256"] = "0" * 64
    elif failure == "escape":
        entry["frames"][0]["rgb"] = "../outside.png"
    else:
        manifest["benchmark"] = "express"
    with pytest.raises(ValueError):
        bind_recorded_evidence(questions[0], manifest, root=tmp_path)


def test_changed_image_after_binding_is_not_fed_to_model(tmp_path):
    questions, manifest = fixture(tmp_path)
    packet = bind_recorded_evidence(questions[0], manifest, root=tmp_path)
    Image.new("RGB", (8, 8), "green").save(tmp_path / "1.png")
    with pytest.raises(ValueError, match="changed"):
        packet.model_input(max_images=2)


def test_express_requires_explicit_identity_and_uses_manifest_sequence(tmp_path):
    _, manifest = fixture(tmp_path)
    question = express_questions(
        [
            {
                "episode_id": 7,
                "question": "What color?",
                "type": "attribute",
                "scene_id": "scene",
                "answer": "private reference",
                "actions": {"oracle": True},
            }
        ]
    )[0]
    with pytest.raises(ValueError):
        bind_recorded_evidence(question, manifest, root=tmp_path)
    entry = manifest["episodes"]["id"]
    manifest.update(benchmark="express", episodes={"7": {**entry, "scene": "scene", "evidence_id": None}})
    packet = bind_recorded_evidence(question, manifest, root=tmp_path)
    assert [f.index for f in packet.frames] == [3, 1]  # Explicit sequence, no filename sorting.
    assert packet.model_input(max_images=2)[0] == "What color?"
    with pytest.raises(ValueError):
        bind_recorded_evidence(replace(question, scene="other"), manifest, root=tmp_path)
