# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Pre-command CLI exits remain failed, inspectable agent trials."""

import importlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


def test_cli_startup_failure_without_audit_writes_failed_summary(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    runner = importlib.import_module("eval_native_chat_session")
    config = SimpleNamespace(no_cameras=True, port_offset=0)
    monkeypatch.setattr(runner, "load_sim_launch_config_from_path", lambda _: config)
    monkeypatch.setattr(runner, "load_placement_regions", lambda _: [[], []])
    monkeypatch.setattr(runner, "mujoco", SimpleNamespace(
        MjModel=SimpleNamespace(from_xml_path=lambda _: object())))
    robot = SimpleNamespace(step=lambda _: None)
    monkeypatch.setattr(runner, "live_table_robot", lambda *a, **kw: nullcontext(robot))
    monkeypatch.setattr(runner, "goal_predicates", lambda *a: {
        "object1": {"satisfied": False}, "object2": {"satisfied": False}})
    terminated = []
    process = SimpleNamespace(wait=lambda **kw: 1)
    monkeypatch.setattr(runner, "popen_session", lambda *a, **kw: process)
    monkeypatch.setattr(runner, "terminate_process_tree", terminated.append)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "test-source\n")
    config_path = tmp_path / "sim.json"
    config_path.write_text("{}")
    out = tmp_path / "episode"
    monkeypatch.setattr("sys.argv", ["eval_native_chat_session", "--output", str(out),
        "--live-sim", str(config_path), "--placement-regions", "unused.json",
        "--base-url", "http://unused.invalid", "--remote-model", "unused"])

    assert runner.main() == 1
    result = json.loads((out / "summary.json").read_text())
    assert result["cli_exit_code"] == 1
    assert result["passed"] is False and result["task_passed"] is False
    assert result["audit_present"] is False and result["audit_commands"] == 0
    assert result["execution_completion_verified"] is False
    assert terminated == [process]
