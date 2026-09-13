"""QA evidence contracts, scoring and real shared runtime with controlled responses."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from emet.eval.agent_tasks.qa import load_public, readable, run_question, score_answers, validate_qa_scene
from emet.eval.agent_tasks.recording import write_json
from emet.eval.agent_tasks.spec import fingerprint, load_suite


@pytest.mark.parametrize(
    "pixels,box,clipped,passed",
    [
        (128, [1, 1, 16, 16], False, True),
        (127, [1, 1, 16, 16], False, False),
        (200, [1, 1, 6, 40], False, False),
        (500, [0, 1, 20, 25], True, False),
    ],
)
def test_readability_requires_size_and_complete_target(pixels, box, clipped, passed):
    assert readable({"pixels": pixels, "bbox_xyxy": box, "clipped": clipped}) is passed


@pytest.fixture
def pack(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    views = {}
    for i in range(2):
        path = public / f"view{i}.png"
        Image.fromarray(np.full((12, 12, 3), i * 100, dtype=np.uint8)).save(path)
        views[f"view:{i}"] = {"image": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    question = {
        "id": "changed",
        "question": "Which way did the object move?",
        "answer_format": "left or right",
        "view_ids": list(views),
        "metadata": {},
    }
    data = {"views": views, "questions": [question]}
    write_json(public / "questions.json", data)
    write_json(
        tmp_path / "private_answers.json",
        {
            "public_fingerprint": fingerprint(data),
            "keys": {"changed": {"answer": "left", "required_evidence": list(views)}},
        },
    )
    return tmp_path, public, data, question


@pytest.mark.parametrize(
    "response,passed",
    [
        ({"answer": "LEFT.", "evidence": ["view:0", "view:1"]}, True),
        ({"answer": "left", "evidence": ["view:0"]}, False),
        ({"answer": "right", "evidence": ["view:0", "view:1"]}, False),
        ({"answer": "left", "evidence": ["view:0", "view:1", "hidden"]}, False),
        ({"answer": "left", "evidence": [{}]}, False),
    ],
)
def test_answer_requires_correct_value_and_complete_valid_evidence(pack, response, passed):
    root, *_ = pack
    assert score_answers(root, {"changed": response})["success"] is passed


def test_public_question_tampering_rejected(pack):
    root, public, data, _ = pack
    data["questions"][0]["question"] = "Different question"
    write_json(public / "questions.json", data)
    with pytest.raises(ValueError, match="do not match"):
        score_answers(root, {})


def test_image_tampering_rejected(pack):
    _, public, *_ = pack
    (public / "view0.png").write_bytes(b"altered")
    with pytest.raises(ValueError, match="image evidence"):
        load_public(public)


def test_changed_colors_cannot_keep_old_answer_key():
    suite = load_suite()
    validate_qa_scene(suite)
    broken = copy.deepcopy(suite)
    broken["scene"]["objects"]["red_cylinder"]["color"] = [0, 0, 1, 1]
    with pytest.raises(ValueError, match="color contract"):
        validate_qa_scene(broken)


def test_qa_runtime_attaches_each_rgb_and_rejects_unseen_citations(pack, monkeypatch):
    from emet.agent import loop

    _, public, data, q = pack
    inputs = []
    responses = iter(
        [
            {"name": "submit_answer", "arguments": {"answer": "left", "evidence": ["view:0", "view:1"]}},
            {"name": "next_view", "arguments": {}},
            {"name": "submit_answer", "arguments": {"answer": "left", "evidence": ["view:0", "view:1"]}},
        ]
    )

    def call(client, text, *args, **kw):
        assert kw["image"] is not None
        inputs.append((text, kw["image"].copy()))
        assert "positions" not in text and "required_evidence" not in text and "private_answers" not in text
        return json.dumps({"tool_calls": [next(responses)]}), 0.01

    monkeypatch.setattr(loop, "_call_llm", call)
    result = run_question(q, data, public, SimpleNamespace(), max_rounds=3)
    assert result["response"]["answer"] == "left"
    assert result["outcome"]["status"] == "tool_completed"
    assert len(inputs) == 3
    assert inputs[0][1].max() == 0 and inputs[-1][1].min() == 100
    assert "already supplied" in inputs[1][0]
    assert result["seen_views"] == ["view:0", "view:1"]
    assert all("hash" in e for e in result["events"])
