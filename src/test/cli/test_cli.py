# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Tests for the emet CLI."""

import json
import subprocess
import sys
import time


def test_cli_help():
    """CLI --help runs without error."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "emet" in result.stdout.lower()
    assert "serve" in result.stdout
    assert "run" in result.stdout
    assert "sync" in result.stdout
    assert "install" in result.stdout
    assert "show" in result.stdout
    assert "test" in result.stdout
    assert "eval" in result.stdout
    assert "jobs" in result.stdout
    assert "hmeqa" in result.stdout
    assert "debug-da3-depth" in result.stdout


def test_eval_group_help():
    """emet eval --help lists GPU preflight subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "eval", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "status" in result.stdout
    assert "diagnose" in result.stdout
    assert "check" in result.stdout
    assert "wait" in result.stdout
    assert "kill-stale" in result.stdout
    assert "affinity" in result.stdout
    assert "recover" in result.stdout


def test_habitat_group_help_lists_safe_start():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "habitat", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "safe-start" in result.stdout
    assert "egl-probe" in result.stdout
    assert "info" in result.stdout


def test_habitat_safe_start_help():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "habitat", "safe-start", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "need-mib" in result.stdout
    assert "force-inline" in result.stdout
    assert "smoke-episode" in result.stdout
    assert "queued" in result.stdout.lower() or "detach" in result.stdout.lower()


def test_jobs_run_id_from_output():
    from emet.cli import _jobs_run_id_from_output

    assert _jobs_run_id_from_output(None) is None
    assert _jobs_run_id_from_output("") is None
    assert _jobs_run_id_from_output("registered  abc\njob_20260101_abc\n") == "job_20260101_abc"
    assert _jobs_run_id_from_output("only-id\n") == "only-id"


def test_hmeqa_group_help():
    """emet hmeqa --help lists H2H helpers."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "hmeqa", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "h2h" in result.stdout
    assert "resume" in result.stdout
    assert "status" in result.stdout
    assert "summarize" in result.stdout
    assert "overnight" in result.stdout
    assert "significance" in result.stdout
    assert "ladder" in result.stdout
    assert "failures" in result.stdout


def test_hmeqa_h2h_help_lists_evidence_policy_flags():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "hmeqa", "h2h", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "agentic-verifier" in result.stdout
    assert "require-verified" in result.stdout
    assert "agentic-router" in result.stdout
    assert "--use-hm3d-semantics" in result.stdout
    assert "--enrich-labels" in result.stdout
    assert "paper-router" in result.stdout
    assert "eqa-hf-model-id" in result.stdout
    assert "eqa-vl-family" in result.stdout
    assert "--description" in result.stdout
    assert "--host" in result.stdout
    assert "--vl-endpoint" in result.stdout
    assert "--vl-port" in result.stdout
    assert "--decision-policy" in result.stdout
    assert "grounded_v2" in result.stdout
    assert "--graph-evidence-mode" in result.stdout
    assert "--room-history-mode" in result.stdout
    assert "--room-policy" in result.stdout
    assert "--room-target-hints" in result.stdout
    assert "--investigate-stamp" in result.stdout
    assert "--attempt-ledger-mode" in result.stdout
    assert "--variant-id" in result.stdout
    assert "--episode-timeout" in result.stdout
    assert "--max-planning-steps" in result.stdout
    assert "--max-movement-step" in result.stdout
    # Paper-router / Click default is Qwen-first (none), OWL opt-in only.
    assert "none" in result.stdout
    assert "owlv2" in result.stdout


def test_hmeqa_resume_help_lists_frozen_variant_flags():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "hmeqa", "resume", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for option in (
        "--decision-policy",
        "--use-hm3d-semantics",
        "--enrich-labels",
        "--graph-evidence-mode",
        "--room-history-mode",
        "--room-policy",
        "--room-target-hints",
        "--investigate-stamp",
        "--attempt-ledger-mode",
        "--variant-id",
        "--eqa-hf-model-id",
        "--eqa-vl-family",
        "--eqa-vl-quantization",
        "--eqa-answer-max-new-tokens",
        "--episode-timeout",
        "--max-planning-steps",
        "--max-movement-step",
    ):
        assert option in result.stdout


def test_hmeqa_paper_router_does_not_enable_variant_axes(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from emet.cli import main

    captured = {}
    monkeypatch.setattr("emet.cli._hmeqa_launch", lambda **kwargs: captured.update(kwargs))
    result = CliRunner().invoke(
        main,
        [
            "hmeqa",
            "h2h",
            str(tmp_path),
            "--ids",
            "2,104",
            "--preset",
            "paper-router",
        ],
    )
    assert result.exit_code == 0, result.output
    frozen = captured["frozen_values"]
    assert frozen["agentic_router"] is True
    assert frozen["decision_policy"] == "legacy"
    assert frozen["graph_evidence_mode"] == "off"
    assert frozen["room_history_mode"] == "off"
    assert frozen["investigate_stamp"] is False
    assert frozen["attempt_ledger_mode"] == "off"
    assert frozen["variant_id"] == "legacy"
    assert frozen["use_hm3d_semantics"] is False
    assert frozen["use_enrich_labels"] is False


def test_hmeqa_resume_reuses_frozen_variant_and_allows_operational_override(
    monkeypatch,
    tmp_path,
):
    from click.testing import CliRunner

    from emet.cli import main
    from emet.eval.hmeqa_launch import build_hmeqa_run_config, prepare_hmeqa_run_manifest

    config = build_hmeqa_run_config(
        arms="agentic",
        ids="2,104",
        agentic_verifier="none",
        require_verified=False,
        agentic_router=True,
        use_hm3d_semantics=True,
        use_enrich_labels=True,
        decision_policy="grounded_v2",
        graph_evidence_mode="shadow",
        room_history_mode="agent",
        room_policy="llm",
        room_target_hints=False,
        investigate_stamp=True,
        attempt_ledger_mode="shadow",
        variant_id="grounded-shadow-r1",
        eqa_answer_max_new_tokens=512,
        episode_timeout_seconds=3600,
        max_planning_steps=12,
        max_movement_step=6,
    )
    prepare_hmeqa_run_manifest(
        tmp_path,
        project_root=tmp_path,
        config=config,
        sources={"variant.id": "command_line"},
        resume=False,
        git_state={
            "commit": "a" * 40,
            "dirty": False,
            "dirty_digest": None,
            "status": [],
        },
    )

    captured = {}
    monkeypatch.setattr("emet.cli._hmeqa_launch", lambda **kwargs: captured.update(kwargs))
    result = CliRunner().invoke(
        main,
        ["hmeqa", "resume", str(tmp_path), "--cooldown", "7"],
    )
    assert result.exit_code == 0, result.output
    frozen = captured["frozen_values"]
    assert frozen["arms"] == "agentic"
    assert frozen["holdout_ids"] == "2,104"
    assert frozen["decision_policy"] == "grounded_v2"
    assert frozen["graph_evidence_mode"] == "shadow"
    assert frozen["room_history_mode"] == "agent"
    assert frozen["room_policy"] == "llm"
    assert frozen["use_hm3d_semantics"] is True
    assert frozen["use_enrich_labels"] is True
    assert frozen["room_target_hints"] is False
    assert frozen["investigate_stamp"] is True
    assert frozen["attempt_ledger_mode"] == "shadow"
    assert frozen["variant_id"] == "grounded-shadow-r1"
    assert frozen["eqa_answer_max_new_tokens"] == 512
    assert frozen["episode_timeout"] == 3600
    assert frozen["max_planning_steps"] == 12
    assert frozen["max_movement_step"] == 6
    assert captured["cooldown"] == 7


def test_hmeqa_overnight_help():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "hmeqa", "overnight", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "skip-bal32" in result.stdout
    assert "gate-min-acc" in result.stdout
    assert "agentic-router" in result.stdout
    assert "none" in result.stdout
    assert "owlv2" in result.stdout
    assert "RESUME" in result.stdout or "resume" in result.stdout.lower()
    assert "--base" in result.stdout


def test_jobs_cancel_help():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "jobs", "cancel", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "JOB_ID" in result.stdout or "job" in result.stdout.lower()


def test_status_group_help():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "status", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "tail" in result.stdout
    assert "path" in result.stdout
    assert "latest" in result.stdout


def test_test_help_mentions_agent_regression():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "test", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # Click may soft-wrap "agent-regression" across lines
    compact = result.stdout.replace("\n", "").replace(" ", "")
    assert "agent-regression" in compact or ("agent-" in result.stdout and "regression" in result.stdout)


def test_jobs_group_help():
    """emet jobs --help lists management subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "jobs", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "list" in result.stdout
    assert "cancel" in result.stdout
    assert "register" in result.stdout
    assert "run" in result.stdout
    assert "report" in result.stdout


def test_jobs_report_help():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "jobs", "report", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "JOB_ID" in result.stdout or "job" in result.stdout.lower()
    assert "--rooms" in result.stdout
    assert "--out-dir" in result.stdout
    assert "--fail-only" in result.stdout
    assert "--question" in result.stdout


def test_jobs_run_help_lists_safety_flags():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "jobs", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "cpu-safe" in result.stdout
    assert "gpu-exclusive" in result.stdout
    assert "--description" in result.stdout


def test_jobs_run_detached_supervisor_registers_itself(tmp_path):
    import os

    jobs_dir = tmp_path / "jobs"
    out_dir = tmp_path / "out"
    marker = out_dir / "child-ran"
    env = os.environ.copy()
    env["EMET_JOBS_DIR"] = str(jobs_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "emet.cli",
            "jobs",
            "run",
            "--name",
            "self-register-test",
            "--out-dir",
            str(out_dir),
            "--",
            sys.executable,
            "-c",
            f"import time; from pathlib import Path; time.sleep(0.5); Path({str(marker)!r}).write_text('ok')",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    job_id = result.stdout.strip().splitlines()[-1]
    record_path = jobs_dir / f"{job_id}.json"

    deadline = time.monotonic() + 10.0
    record = {}
    while time.monotonic() < deadline:
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") in {"done", "failed"}:
                break
        time.sleep(0.05)

    assert record.get("status") == "done", record
    assert isinstance(record.get("pid"), int)
    assert marker.read_text(encoding="utf-8") == "ok"
    wrapper = (out_dir / "job_wrapper.sh").read_text(encoding="utf-8")
    assert "jobs register --job-id" in wrapper
    assert wrapper.index("jobs register --job-id") < wrapper.index('jobs update "$JOB_ID" --status running')


def test_jobs_run_spawn_failure_leaves_no_phantom_record(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from emet.cli import main

    jobs_dir = tmp_path / "jobs"
    monkeypatch.setenv("EMET_JOBS_DIR", str(jobs_dir))

    def fail_spawn(*args, **kwargs):
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(subprocess, "Popen", fail_spawn)
    result = CliRunner().invoke(
        main,
        [
            "jobs",
            "run",
            "--name",
            "must-not-register",
            "--out-dir",
            str(tmp_path / "out"),
            "--",
            sys.executable,
            "-c",
            "print('never runs')",
        ],
    )
    assert result.exit_code != 0
    assert not jobs_dir.exists() or not list(jobs_dir.glob("*.json"))


def test_jobs_update_help_lists_description():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "jobs", "update", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--description" in result.stdout


def test_jobs_list_runs(tmp_path):
    import os

    env = os.environ.copy()
    env["EMET_JOBS_DIR"] = str(tmp_path / "jobs")
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "jobs", "list", "--no-scan"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0


def test_eval_status_runs():
    """emet eval status is read-only and exits 0 even without a GPU."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "eval", "status"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "GPU:" in result.stdout


def test_cli_version():
    """CLI --version prints version."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "0.3" in result.stdout


def test_serve_help():
    """emet serve --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "serve", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "mujoco" in result.stdout
    assert "headless" in result.stdout


def test_run_help():
    """emet run --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "dynamem" in result.stdout
    assert "dynagraph" in result.stdout
    assert "graph" in result.stdout and "eqa" in result.stdout
    assert "mapping" in result.stdout
    assert "debug-da3-depth" in result.stdout


def test_run_graph_eqa_help():
    """emet run graph-eqa --help works (GraphEQA app is wired)."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "run", "graph-eqa", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "robot" in result.stdout.lower() or "robot_ip" in result.stdout


def test_run_dynagraph_help():
    """emet run --help lists dynagraph; app module --help lists merge/staleness."""
    r1 = subprocess.run(
        [sys.executable, "-m", "emet.cli", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert r1.returncode == 0
    assert "dynagraph" in r1.stdout.lower()
    r2 = subprocess.run(
        [sys.executable, "-m", "emet.app.run_dynagraph", "--help"],
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 0
    out2 = (r2.stdout + r2.stderr).lower()
    assert "staleness" in out2 or "merge" in out2


def test_sync_help():
    """emet sync --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "sync", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--all" in result.stdout
    assert "--sim" in result.stdout
    assert "--dynamem" in result.stdout


def test_show_help():
    """emet show --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "show", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "rrd" in result.stdout.lower()


def test_test_help():
    """emet test --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "test", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "pytest" in result.stdout.lower()


def test_install_submodules_help():
    """emet install submodules --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "submodules", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "submodule" in result.stdout.lower()


def test_install_sim_help():
    """emet install sim --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "sim", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "robocasa" in result.stdout.lower() or "sim" in result.stdout.lower()


def test_install_robocasa_help():
    """emet install robocasa --help works (same as install sim)."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "robocasa", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "robocasa" in result.stdout.lower()


def test_install_full_help():
    """emet install full --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "full", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "install" in result.stdout.lower()
    assert "--profile" in result.stdout


def test_install_menu_help():
    """emet install menu --help lists --text-only."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "menu", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--text-only" in result.stdout


def test_install_pre_commit_help():
    """emet install pre-commit --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install", "pre-commit", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "pre-commit" in result.stdout.lower()


def test_install_completion_help():
    """emet install-completion --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install-completion", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "bash" in result.stdout or "zsh" in result.stdout


def test_ovmm_help():
    """emet ovmm --help lists find/full/prepare/sweep/rates/status."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "ovmm", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "find" in out
    assert "full" in out
    assert "prepare" in out
    assert "sweep" in out
    assert "rates" in out
    assert "status" in out


def test_sqa3d_help():
    """emet sqa3d --help lists embodied subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "sqa3d", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "run-episode" in out
    assert "run-batch" in out
    assert "run-real-sweep" in out
    assert "plot-results" in out
    assert "info" in out


def test_robovista_help():
    """emet robovista --help lists info and run-batch."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "robovista", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "info" in out
    assert "run-batch" in out


def test_robovista_run_batch_help():
    """run-batch documents domain, mock-llm, and VLM flags."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "robovista", "run-batch", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "--domain" in out
    assert "--mock-llm" in out
    assert "--eqa-vl-family" in out
    assert "--max-questions" in out
    assert "--resume" in out


def test_sqa3d_run_real_sweep_help():
    """run-real-sweep documents replay-mode and isolate-episodes."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "sqa3d", "run-real-sweep", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "--replay-mode" in out
    assert "--isolate-episodes" in out
    assert "--with-sens" in out
    assert "--all" in out
    assert "--resume" in out
    assert "--export-root" in out


def test_eval_sqa3d_help():
    """emet eval-sqa3d --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "eval-sqa3d", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--predictions" in result.stdout or "-p" in result.stdout
    assert "split" in result.stdout.lower()


def test_debug_da3_depth_help():
    """emet debug-da3-depth --help lists DA3 options (top-level subcommand)."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "debug-da3-depth", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--model-id" in result.stdout
    assert "--depth-source" in result.stdout
    assert "--meshes" in result.stdout or "--no-meshes" in result.stdout


def test_install_completion_bash():
    """install-completion --shell bash produces valid script."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "install-completion", "--shell", "bash"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "_emet_completion" in result.stdout or "emet" in result.stdout


def test_mars_help():
    """emet mars --help and start --help work."""
    for args in (["mars", "--help"], ["mars", "start", "--help"]):
        result = subprocess.run(
            [sys.executable, "-m", "emet.cli", *args],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    assert "username" in result.stdout.lower() or "--user" in result.stdout
    assert "deploy" in result.stdout


def test_capture_help():
    """emet capture --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "capture", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "127.0.0.1" in result.stdout
    assert "--backend" in result.stdout
    assert "voxel_only" in result.stdout
    assert "docs/zmq_obs.md" in result.stdout or "zmq_obs" in result.stdout.lower()
    assert "--robot" in result.stdout


def test_stream_help():
    """emet stream --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "stream", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "127.0.0.1" in result.stdout
    assert "--robot" in result.stdout
    assert "--backend" in result.stdout
    assert "voxel_only" in result.stdout
    assert "--cameras-only" in result.stdout
    assert "docs/zmq_obs.md" in result.stdout or "zmq_obs" in result.stdout.lower()
    assert "Rerun" in result.stdout or "rerun" in result.stdout.lower()
