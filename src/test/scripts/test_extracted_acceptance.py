# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Offline acceptance-driver guards; no simulator or model startup."""

import hashlib
import json
import os
import runpy
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "scripts/run_extracted_acceptance.sh"


@pytest.mark.parametrize("ok", [True, False])
def test_full_runner_probe_exit_status(monkeypatch, capsys, ok):
    @dataclass
    class Result:
        ok: bool

    package = ModuleType("emet_habitat")
    package.__path__ = []
    runner = ModuleType("emet_habitat.runner")
    probe = ModuleType("emet_habitat.egl_probe")

    def run_probe(*, question_id):
        assert question_id == 15
        return Result(ok)

    probe.run_egl_probe = run_probe
    for module in (package, runner, probe):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(ROOT / "scripts/check_habitat_runtime.py"), run_name="__main__")
    assert exc.value.code == (0 if ok else 1)
    assert json.loads(capsys.readouterr().out) == {"ok": ok}


def test_acceptance_preflight_precedes_episode_loop():
    driver = DRIVER.read_text()
    assert driver.index("scripts/check_habitat_runtime.py") < driver.index(
        'bash "$DRIVER_ROOT/scripts/run_dev_loop.sh"'
    )
    assert "set -euo pipefail" in driver


def test_missing_metrics_has_infrastructure_diagnostic(tmp_path):
    validate = runpy.run_path(str(ROOT / "scripts/check_acceptance_metrics.py"))["validate_metrics"]
    with pytest.raises(ValueError, match="Infrastructure error: cannot read episode metrics"):
        validate("eqa", tmp_path / "q12.jsonl")


@pytest.mark.parametrize("pretty", [False, True])
def test_room_metric_object_accepts_pretty_and_jsonl_encoding(tmp_path, pretty):
    validate = runpy.run_path(str(ROOT / "scripts/check_acceptance_metrics.py"))["validate_metrics"]
    path = tmp_path / "room.jsonl"
    path.write_text(
        json.dumps({"find_object_success": True, "find_recep_success": False}, indent=2 if pretty else None) + "\n"
    )
    validate("room", path)


@pytest.mark.parametrize("key", ["agentic_find_error", "agentic_find_error_recep"])
def test_phase_exception_is_not_a_completed_task_miss(tmp_path, key):
    validate = runpy.run_path(str(ROOT / "scripts/check_acceptance_metrics.py"))["validate_metrics"]
    path = tmp_path / "room.json"
    path.write_text(json.dumps({"find_object_success": False, "find_recep_success": False, key: "missing weights"}))
    with pytest.raises(ValueError, match="Infrastructure error"):
        validate("room", path)


@pytest.mark.parametrize("phase", ["eqa", "room"])
@pytest.mark.parametrize("kind", ["pass", "miss", "error", "empty", "malformed", "missing", "duplicate"])
def test_metrics_require_a_scored_episode(tmp_path, phase, kind):
    validate = runpy.run_path(str(ROOT / "scripts/check_acceptance_metrics.py"))["validate_metrics"]
    outcome = kind == "pass"
    row = {"correct": outcome} if phase == "eqa" else {"find_object_success": outcome, "find_recep_success": outcome}
    if kind == "error":
        row["error"] = "simulator startup failed"
    if kind == "missing":
        row = {}
    text = json.dumps(row)
    if kind == "empty":
        text = ""
    elif kind == "malformed":
        text = "{bad"
    elif kind == "duplicate":
        text += "\n" + text
    path = tmp_path / "metrics.json"
    path.write_text(text)
    if kind in ("pass", "miss"):
        validate(phase, path)
    else:
        with pytest.raises(ValueError):
            validate(phase, path)


def test_driver_syntax_and_invalid_phase(tmp_path):
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)
    result = subprocess.run(["bash", str(DRIVER), "invalid"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_relative_driver_invocation_preserves_driver_after_source_chdir(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    for name in ("uv.lock", "configs/emet/default.yaml", "configs/ovmm/find_phase_episodes.yaml"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    out = tmp_path / "results"
    # Stop at package inventory: exercise real provenance setup without a model or simulator.
    env = dict(os.environ, SOURCE_DIR=str(source), SOURCE_SHA=sha, OUT_DIR=str(out), EMET_PY="/bin/false")
    result = subprocess.run(
        ["bash", "scripts/run_extracted_acceptance.sh", "eqa"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (out / "driver.sh").read_bytes() == DRIVER.read_bytes()
    hashes = (out / "input_hashes.txt").read_text()
    assert len(hashes.splitlines()) == 7
    assert f"{hashlib.sha256(DRIVER.read_bytes()).hexdigest()}  {DRIVER}" in hashes
    assert str(ROOT / "scripts/check_habitat_runtime.py") in hashes


@pytest.mark.parametrize("failure", ["wrong_sha", "dirty", "staged", "untracked", "existing_output"])
def test_driver_rejects_unfrozen_inputs_before_creating_output(tmp_path, failure):
    repo = tmp_path / "source"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    git("init", "-q")
    tracked = repo / "tracked.txt"
    tracked.write_text("original\n")
    git("add", "tracked.txt")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "fixture",
    )
    sha = git("rev-parse", "HEAD")
    out = tmp_path / "results"
    if failure == "wrong_sha":
        sha = "0" * 40
    elif failure in ("dirty", "staged"):
        tracked.write_text("modified\n")
        if failure == "staged":
            git("add", "tracked.txt")
    elif failure == "untracked":
        (repo / "untracked.txt").touch()
    else:
        out.mkdir()
        (out / "preserve.txt").write_text("evidence")
    env = dict(os.environ, SOURCE_DIR=str(repo), SOURCE_SHA=sha, OUT_DIR=str(out))
    result = subprocess.run(["bash", str(DRIVER), "eqa"], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    if failure == "existing_output":
        assert list(out.iterdir()) == [out / "preserve.txt"]
        assert (out / "preserve.txt").read_text() == "evidence"
    else:
        assert not out.exists()


@pytest.mark.parametrize("robot", ["stretch", "rby1"])
def test_acceptance_config_survives_runtime_profile(monkeypatch, robot):
    from emet.core.parameters import get_parameters
    from emet.eval.benchmark_dynagraph import enable_query_driven_memory
    from emet.eval.ovmm_find_phase import apply_backend_parameters

    monkeypatch.setenv("EMET_CONFIG", str(ROOT / "configs/emet/shared_find_acceptance.yaml"))
    params = apply_backend_parameters(get_parameters("dynav_config.yaml", robot=robot), "lazy_graph")
    enable_query_driven_memory(params, "lazy_graph")
    assert params.get("query_driven_memory") is True
    assert params.get("query_memory")["grounding_backend"] == "vlm"
    assert params.get("query_memory")["mask_backend"] == "yoloe_sam2"
    assert params.get("eqa")["collect_agentic_trace"] is True
    assert params.get("eqa")["vl_hf_model_id"] == "Qwen/Qwen3-VL-8B-Instruct"


def test_every_inventory_file_has_a_disposition():
    inventory = json.loads((ROOT / "docs/plans/shared_stack_inventory.json").read_text())
    allowed = {
        "already_main",
        "superseded_by_main",
        "represented_in_stack",
        "deferred",
        "partial_extraction_residual_deferred",
    }
    for branch in inventory["branches"].values():
        assert len(branch["source_sha"]) == 40
        paths = [row["path"] for row in branch["files"]]
        assert len(paths) == len(set(paths))
        for row in branch["files"]:
            assert row["status"] in allowed
            if "deferred" in row["status"] or row["status"] == "superseded_by_main":
                assert row["residual"]
            if row["status"] == "represented_in_stack":
                assert row["source_blob"] == row["stack_blob"]
