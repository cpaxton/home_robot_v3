# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from emet.eval.eqa_decision_trace import finalize_eqa_decision_trace, record_eqa_decision_iteration


def test_audit_collision_preserves_both_requests_and_reports_relocation(tmp_path):
    from emet.eval.eqa_decision_trace import record_vlm_request

    kwargs = {"system_prompt": "system", "user_text": "one", "images": [], "metadata": {}}
    first = record_vlm_request(tmp_path / "call_1", **kwargs)
    with pytest.warns(RuntimeWarning, match="audit path collision"):
        second = record_vlm_request(tmp_path / "call_1", **{**kwargs, "user_text": "two"})
    assert first != second
    assert json.loads(first.read_text())["user_text"] == "one"
    assert json.loads(second.read_text())["user_text"] == "two"
    assert json.loads(second.read_text())["metadata"]["audit_path_collision"] == str(first.parent)


def test_audit_rejects_unsupported_raw_history_image(tmp_path):
    from emet.eval.eqa_decision_trace import record_vlm_request

    with pytest.raises(TypeError, match="Unsupported raw conversation image"):
        record_vlm_request(
            tmp_path / "call",
            system_prompt="",
            user_text="",
            images=[],
            metadata={},
            history=[{"role": "user", "content": [{"type": "image_url"}]}],
        )


def test_router_request_preserves_system_user_image_order_and_pixels(tmp_path):
    import hashlib

    from emet.eval.eqa_decision_trace import record_vlm_request

    images = [Image.fromarray(np.full((4, 5, 3), value, dtype=np.uint8)) for value in (20, 40)]
    request = record_vlm_request(
        tmp_path / "call_1",
        system_prompt="system",
        user_text="exact\nstate",
        images=images,
        metadata={"reset_context": True},
        history=[
            {"role": "user", "content": ["old view", images[0]]},
            {"role": "assistant", "content": "inspect_graph"},
            {"role": "user", "content": "TOOL_RESULT no new detections"},
        ],
    )
    saved = json.loads(request.read_text())
    assert saved["system_prompt"] == "system"
    assert saved["user_text"] == "exact\nstate"
    assert saved["conversation_history"][1]["content"] == "inspect_graph"
    assert (request.parent / saved["conversation_history"][0]["content"][1]["path"]).is_file()
    for entry, original in zip(saved["images"], images, strict=True):
        assert entry["pixel_sha256"] == hashlib.sha256(original.tobytes()).hexdigest()
        np.testing.assert_array_equal(np.asarray(Image.open(request.parent / entry["path"])), np.asarray(original))


def test_explicit_camera_roles_are_not_mislabeled_as_navigation_fallback(tmp_path):
    images = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    kwargs = {
        "question": "Where?",
        "text_blocks": ["Current and earlier context"],
        "obs_ids": [],
        "crop_obs_id": None,
        "nav_fallback_count": 0,
        "relevant_images": images,
    }
    path = record_eqa_decision_iteration(tmp_path, 1, **kwargs, image_roles=["assessed_view", "pre_approach_context"])
    meta = json.loads((path / "meta.json").read_text())
    assert [im["kind"] for im in meta["images"]] == ["assessed_view", "pre_approach_context"]
    assert meta["nav_fallback_count"] == 0
    with pytest.raises(ValueError, match="every attached image"):
        record_eqa_decision_iteration(tmp_path, 2, **kwargs, image_roles=["assessed_view"])


def test_record_eqa_decision_iteration_writes_prompt_and_images(tmp_path: Path) -> None:
    trace_root = tmp_path / "eqa_decisions"
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[1:3, 1:3] = (255, 0, 0)
    im = Image.fromarray(rgb, mode="RGB")
    record_eqa_decision_iteration(
        trace_root,
        1,
        question="What time is it?",
        text_blocks=["Question: What time is it?", "SCENE_GRAPH\nnode"],
        obs_ids=[68],
        crop_obs_id=None,
        nav_fallback_count=0,
        relevant_images=[im],
        view_status="VIEW_STATUS: attached=68",
        close_look_status="CLOSE_LOOK: resolved=yes",
        vlm_raw='{"answer":"9-11pm"}',
        parsed={"answer": "9-11pm", "confidence": True},
    )
    iter_dir = trace_root / "iter_001"
    assert (iter_dir / "prompt.txt").is_file()
    assert (iter_dir / "image_1_obs68.png").is_file()
    meta = json.loads((iter_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["obs_ids"] == [68]
    assert meta["close_look_status"].startswith("CLOSE_LOOK")
    finalize_eqa_decision_trace(trace_root, n_iterations=1)
    assert (trace_root / "README.md").is_file()
    index_lines = (trace_root / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(index_lines) == 1
