# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import hashlib
import json
import runpy
from pathlib import Path

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/replay_final_context.py"))


def test_context_arms_preserve_question_and_do_not_choose_answers():
    original = (
        "Question: where? A) kitchen B) room\nQUESTION_ROOM (scope): kitchen, room\n"
        "MEMORY (hints; for location MCQs, prefer option landmarks visible in Image 1 over nearest-furniture guesses)\n"
        "Capture sequence is chronological\nImage 1: current\nImage 2: earlier"
    )
    transform = MODULE["user_text"]
    assert transform(original, "original") == original
    assert transform(original, "neutral_system") == original
    neutral = transform(original, "neutral_user")
    assert "QUESTION_ROOM" not in neutral and "prefer option landmarks" not in neutral
    assert "MEMORY" in neutral
    minimal = transform(original, "image_only")
    assert "MEMORY" not in minimal
    assert original.splitlines()[0] in minimal
    assert minimal.endswith("Image 2: earlier")


def test_prepare_copies_exact_images_and_keeps_gold_out_of_prompts(tmp_path):
    iteration = tmp_path / "episode/eqa_decisions/iter_001"
    iteration.mkdir(parents=True)
    data = b"unchanged fixture PNG bytes"
    (iteration / "image.png").write_bytes(data)
    (iteration / "meta.json").write_text(json.dumps({"question": "Where?", "images": [{"path": "image.png"}]}))
    (iteration / "prompt.txt").write_text("Question: Where?\nCapture sequence is chronological\nImage 1: current")
    source = tmp_path / "source"
    source.mkdir()
    (source / "q14.jsonl").write_text(
        json.dumps(
            {
                "debug_bundle_dir": str(tmp_path / "episode"),
                "gold_answer_letter": "SECRET_GOLD",
                "choices": ["A", "B"],
                "predicted_answer": "A",
                "correct": False,
            }
        )
    )
    output = tmp_path / "audit"
    cases = MODULE["prepare"](source, output, [14])
    assert cases[0]["images"][0]["sha256"] == hashlib.sha256(data).hexdigest()
    assert (output / "q14/image_1.png").read_bytes() == data
    for arm in MODULE["ARMS"]:
        assert "SECRET_GOLD" not in (output / f"q14/{arm}.txt").read_text()
