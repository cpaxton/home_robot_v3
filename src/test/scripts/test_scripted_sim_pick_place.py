# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit coverage for the live-CHAT branch of scripted_sim_pick_place."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from emet.agent import tools as agent_tools

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "scripted_sim_pick_place.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("scripted_sim_pick_place_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_explicit_kinematic_calls_use_live_chat_tools(monkeypatch):
    script = _load_script_module()
    robot = object()
    context_seen = {}
    args_seen = {}

    def plan_pick_place(*, task_ref: str) -> str:
        args_seen["task_ref"] = task_ref
        return "TAMP plan plan:1: mode=kinematic."

    def get_tools(context):
        context_seen.update(context)
        return [SimpleNamespace(name="plan_pick_place", func=plan_pick_place)]

    monkeypatch.setattr(agent_tools, "get_tools", get_tools)

    assert script.run_scripted_tool_calls(
        robot,
        [{"name": "plan_pick_place", "arguments": {"task_ref": "task:1"}}],
        manip_mode="kinematic",
    )
    assert context_seen == {"robot": robot, "manip_mode": "kinematic"}
    assert args_seen == {"task_ref": "task:1"}
