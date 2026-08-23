# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""No-GPU tests for the managed OVMM VL worker."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from emet.eval.ovmm_vl_worker import (
    ManagedVLWorker,
    allocate_local_vl_port,
    local_vl_endpoint,
)


def test_allocate_local_vl_port_and_endpoint():
    port = allocate_local_vl_port()
    assert 0 < port < 65536
    assert local_vl_endpoint(port) == f"openai@http://127.0.0.1:{port}/v1"


@patch("emet.eval.ovmm_vl_worker.wait_for_vl_worker")
@patch("emet.eval.ovmm_vl_worker.popen_session")
@patch("emet.eval.ovmm_vl_worker.terminate_process_tree")
def test_worker_starts_with_serve_command(mock_terminate, mock_popen, mock_wait):
    proc = MagicMock()
    proc.poll.return_value = None
    mock_popen.return_value = proc
    mock_wait.return_value = {"ready": True, "multimodal": True}

    worker = ManagedVLWorker(port=8123)
    assert worker.start() == "openai@http://127.0.0.1:8123/v1"
    command = mock_popen.call_args.args[0]
    assert "python" in Path(command[0]).name
    assert command[-5:] == ["--vl", "--host", "127.0.0.1", "--port", "8123"]
    assert mock_popen.call_args.kwargs["env"]["EMET_ALLOW_SDPA_ATTN"] == "1"
    assert mock_popen.call_args.kwargs["env"]["OMP_NUM_THREADS"] == "4"
    assert mock_popen.call_args.kwargs["env"]["TOKENIZERS_PARALLELISM"] == "false"
    mock_wait.assert_called_once()

    worker.stop()
    mock_terminate.assert_called_once_with(proc)


@patch("emet.eval.ovmm_vl_worker.wait_for_vl_worker")
@patch("emet.eval.ovmm_vl_worker.popen_session")
@patch("emet.eval.ovmm_vl_worker.terminate_process_tree")
def test_worker_cleans_up_on_readiness_failure(mock_terminate, mock_popen, mock_wait):
    proc = SimpleNamespace(poll=lambda: None, returncode=None)
    mock_popen.return_value = proc
    mock_wait.side_effect = RuntimeError("not ready")

    worker = ManagedVLWorker(port=8124)
    try:
        worker.start()
    except RuntimeError as exc:
        assert str(exc) == "not ready"
    else:
        raise AssertionError("worker start unexpectedly succeeded")
    mock_terminate.assert_called_once_with(proc)
    assert worker.process is None


@patch("emet.eval.ovmm_find_phase.run_episode_find_phase")
@patch("emet.eval.ovmm_batch._configured_vl_endpoint", return_value=None)
@patch("emet.eval.ovmm_vl_worker.ManagedVLWorker")
def test_run_ovmm_batch_defers_managed_vl_to_episode(mock_worker_cls, _mock_ep, mock_run, tmp_path):
    """The episode starts VL after mapping, not during simulator/agent initialization."""
    from emet.eval.ovmm_batch import OvmmBatchOptions, run_ovmm_batch
    from emet.eval.ovmm_find_phase import FindPhaseEpisode

    worker = MagicMock()
    worker.start.return_value = "openai@http://127.0.0.1:8123/v1"
    mock_worker_cls.return_value = worker
    mock_run.return_value = {
        "episode_id": "ep0",
        "tier": "S0",
        "backend": "dynagraph",
        "find_object_success": True,
        "find_recep_success": True,
        "find_partial_success": 1.0,
    }

    episode = FindPhaseEpisode(
        id="ep0",
        tier="S0",
        sim="configs/sim/default_table_stretch.yaml",
        object="red cylinder",
        start_recep="table",
        goal_recep="table",
    )
    with patch(
        "emet.eval.ovmm_find_phase.load_find_phase_episodes",
        return_value=[episode],
    ):
        with patch("emet.eval.ovmm_benchmark_config.load_ovmm_benchmark_config") as mock_bench:
            mock_bench.return_value = SimpleNamespace(
                sim_episodes_yaml="configs/ovmm/find_phase_episodes.yaml",
                full_episodes_yaml="configs/ovmm/full_episodes.yaml",
                paths=SimpleNamespace(
                    output_dir_sim=tmp_path / "sim",
                    output_dir_full=tmp_path / "full",
                ),
            )
            rc = run_ovmm_batch(
                OvmmBatchOptions(
                    episodes="configs/ovmm/find_phase_episodes.yaml",
                    backends=["dynagraph"],
                    output_dir=tmp_path / "out",
                    agentic_find=True,
                )
            )

    assert rc == 0
    worker.start.assert_not_called()
    worker.stop.assert_not_called()
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["vl_worker"] is worker


@patch("emet.eval.ovmm_find_phase.run_episode_find_phase")
@patch("emet.eval.ovmm_batch._configured_vl_endpoint", return_value="openai@http://remote/v1")
@patch("emet.eval.ovmm_vl_worker.ManagedVLWorker")
def test_run_ovmm_batch_skips_worker_when_endpoint_set(mock_worker_cls, _mock_ep, mock_run, tmp_path):
    from emet.eval.ovmm_batch import OvmmBatchOptions, run_ovmm_batch
    from emet.eval.ovmm_find_phase import FindPhaseEpisode

    mock_run.return_value = {
        "episode_id": "ep0",
        "tier": "S0",
        "backend": "dynagraph",
        "find_object_success": True,
        "find_recep_success": True,
        "find_partial_success": 1.0,
    }
    episode = FindPhaseEpisode(
        id="ep0",
        tier="S0",
        sim="configs/sim/default_table_stretch.yaml",
        object="red cylinder",
        start_recep="table",
        goal_recep="table",
    )
    with patch(
        "emet.eval.ovmm_find_phase.load_find_phase_episodes",
        return_value=[episode],
    ):
        with patch("emet.eval.ovmm_benchmark_config.load_ovmm_benchmark_config") as mock_bench:
            mock_bench.return_value = SimpleNamespace(
                sim_episodes_yaml="configs/ovmm/find_phase_episodes.yaml",
                full_episodes_yaml="configs/ovmm/full_episodes.yaml",
                paths=SimpleNamespace(
                    output_dir_sim=tmp_path / "sim",
                    output_dir_full=tmp_path / "full",
                ),
            )
            rc = run_ovmm_batch(
                OvmmBatchOptions(
                    episodes="configs/ovmm/find_phase_episodes.yaml",
                    backends=["dynagraph"],
                    output_dir=tmp_path / "out",
                    agentic_find=True,
                )
            )

    assert rc == 0
    mock_worker_cls.assert_not_called()
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["vl_worker"] is None
