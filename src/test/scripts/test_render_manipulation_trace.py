# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/render_manipulation_trace.py"
spec = importlib.util.spec_from_file_location("render_manipulation_trace", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_replay_frames_include_failure_final_state_and_do_not_invent_pickup():
    rows = [{"sim_time": i, "gripper_contact": i < 4} for i in range(6)]
    assert module.select_frames(rows, {"physical_pick_success": False}) == [("initial", 2), ("final", 5)]
    assert module.select_frames(rows, {"physical_pick_success": True, "pick_time": 1}) == [
        ("initial", 2),
        ("picked", 1),
        ("last_gripper_contact", 3),
        ("final", 5),
    ]


def test_replay_rejects_empty_trace():
    with pytest.raises(ValueError, match="Empty"):
        module.select_frames([], {})
