# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from emet.eval.eqa_decision_trace import finalize_eqa_decision_trace, record_eqa_decision_iteration


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
