# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Exercise expanded-evaluation orchestration without starting native workers."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "feedback_ablation_runner", ROOT / "scripts/run_mars_feedback_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_dependencies(monkeypatch):
    mastery = types.ModuleType("emet.rl.manipulation.mastery")
    mastery.load_suite = lambda path: {"config": {}, "cases": []}
    gate = types.ModuleType("emet.rl.regression_gate")
    gate.verify_regressions = lambda path: None
    policy = types.ModuleType("emet.rl.manipulation.policy")
    policy.ManipulationPolicy = lambda path: types.SimpleNamespace(
        model=types.SimpleNamespace(policy=types.SimpleNamespace(executor_mask=lambda obs: None))
    )
    for module in (mastery, gate, policy):
        monkeypatch.setitem(sys.modules, module.__name__, module)


def prepare(monkeypatch, tmp_path, *, source=False):
    fake_dependencies(monkeypatch)
    runner = load_runner()
    output = tmp_path / "experiment"
    argv = [
        "runner",
        "--output",
        str(output),
        "--regression-receipt",
        str(tmp_path / "receipt.json"),
        "--variants",
        "baseline",
        "velocity",
        "--seeds",
        "90",
        "91",
        "--steps",
        "256",
        "--evaluation-stages",
        "base",
        "arm",
        "--evaluation-population",
        "admission",
    ]
    source_path = tmp_path / "source"
    if source:
        source_path.mkdir()
        (source_path / "manifest.json").write_text('{"weights_sha256":"original"}')
        argv += ["--source", str(source_path)]
    monkeypatch.setattr(sys, "argv", argv)
    return runner, output, source_path


def fake_child(command, *, cwd, env, stdout, stderr, check):
    assert check and env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    action = command[3]
    destination = Path(command[command.index("--output") + 1])
    summary = {"complete": True, "groups": {"arm/validation": {"cases": 4, "mastered": 4}}}
    if action == "train":
        for label in ("step-0", "last"):
            path = destination / label
            path.mkdir(parents=True)
            (path / "summary.json").write_text(json.dumps(summary))
    else:
        destination.mkdir(parents=True)
        (destination / "summary.json").write_text(json.dumps(summary))


def test_expanded_evaluation_preserves_masks_and_checks_both_checkpoints(monkeypatch, tmp_path):
    runner, output, _ = prepare(monkeypatch, tmp_path, source=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return fake_child(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    runner.main()
    result = json.loads((output / "results.json").read_text())
    assert result["complete"] and result["executor_action_mask"]
    assert result["evaluation_stages"] == ["base", "arm"]
    assert result["evaluation_population"] == "admission"
    assert len(result["runs"]) == 4
    assert all("expanded_initial" in run and "expanded_final" in run for run in result["runs"])
    assert len([c for c in calls if c[3] == "preflight"]) == 2
    assert len([c for c in calls if c[3] == "evaluate"]) == 8
    for command in calls:
        stages = command[
            command.index("--stages") + 1 : command.index("--policy")
            if "--policy" in command
            else command.index("--output")
            if command[3] == "preflight"
            else command.index("--preflight")
        ]
        assert stages == (
            ["arm"] if command[3] == "train" else ["arm", "base"] if command[3] == "preflight" else ["base", "arm"]
        )
        if command[3] == "train":
            assert "--freeze-normalization" in command
        if command[3] == "evaluate":
            assert command[command.index("--population") + 1] == "admission"
    assert json.loads((output / "velocity-suite.json").read_text())["config"]["feedback_reference_velocity"]


def test_source_change_stops_the_next_seed(monkeypatch, tmp_path):
    runner, _, source_path = prepare(monkeypatch, tmp_path, source=True)
    trained = []

    def execute(command, **kwargs):
        fake_child(command, **kwargs)
        if command[3] == "train":
            trained.append(command)
            (source_path / "manifest.json").write_text('{"weights_sha256":"changed"}')

    monkeypatch.setattr(runner.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="source checkpoint changed"):
        runner.main()
    assert len(trained) == 1
