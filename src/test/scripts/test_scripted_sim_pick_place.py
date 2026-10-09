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
        return '{"schema_version": 1, "status": "ok", "data": {"plan_ref": "plan:abc:7"}}'

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


def test_runner_resolves_returned_handles_and_scoring(monkeypatch):
    import json

    script = _load_script_module()
    seen = []

    def get_tools(context):
        context["_tamp_task_refs"] = {"task:random:43": SimpleNamespace(receptacle_body="selected_private")}

        def discover():
            return json.dumps(
                {"schema_version": 1, "status": "ok", "data": {"tasks": [{"task_ref": "task:random:43"}]}}
            )

        def plan(**args):
            seen.append(args)
            return json.dumps({"schema_version": 1, "status": "ok", "data": {"plan_ref": "plan:random:99"}})

        def execute(**args):
            seen.append(args)
            return json.dumps({"schema_version": 1, "status": "ok", "data": {}})

        return [
            SimpleNamespace(name=name, func=fn)
            for name, fn in [("scene_tasks", discover), ("plan_pick_place", plan), ("execute_pick_place_plan", execute)]
        ]

    monkeypatch.setattr(agent_tools, "get_tools", get_tools)
    calls = [
        {"name": "scene_tasks"},
        {"name": "plan_pick_place", "arguments": {"task_ref": "$task_ref"}},
        {"name": "execute_pick_place_plan", "arguments": {"plan_ref": "$plan_ref"}},
    ]
    context = {}
    assert script.run_scripted_tool_calls(object(), calls, manip_mode="kinematic", context_out=context)
    assert seen == [{"task_ref": "task:random:43"}, {"plan_ref": "plan:random:99"}]
    assert script._planned_receptacle_body(calls, context) == "selected_private"


def test_failed_planning_cannot_reuse_previous_successful_handle(monkeypatch):
    from unittest.mock import Mock

    script = _load_script_module()
    plan = Mock(
        side_effect=[
            '{"schema_version":1,"status":"ok","data":{"plan_ref":"old"}}',
            '{"schema_version":1,"status":"error","data":{}}',
        ]
    )
    execute = Mock()
    monkeypatch.setattr(
        agent_tools,
        "get_tools",
        lambda ctx: [
            SimpleNamespace(name="plan_pick_place", func=plan),
            SimpleNamespace(name="execute_pick_place_plan", func=execute),
        ],
    )
    calls = [
        {"name": "plan_pick_place"},
        {"name": "plan_pick_place"},
        {"name": "execute_pick_place_plan", "arguments": {"plan_ref": "$plan_ref"}},
    ]
    assert not script.run_scripted_tool_calls(object(), calls, manip_mode="kinematic")
    assert not execute.called
