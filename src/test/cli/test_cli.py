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

import subprocess
import sys


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
    assert "paper-router" in result.stdout
    assert "eqa-hf-model-id" in result.stdout
    assert "eqa-vl-family" in result.stdout
    assert "--description" in result.stdout
    assert "--host" in result.stdout
    assert "--vl-endpoint" in result.stdout
    assert "--vl-port" in result.stdout
    # Paper-router / Click default is Qwen-first (none), OWL opt-in only.
    assert "none" in result.stdout
    assert "owlv2" in result.stdout


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
