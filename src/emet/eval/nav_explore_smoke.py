# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""VLM-free spawn → rotate → explore smoke across Robocasa / OVMM kitchens / MolmoSpaces.

No Qwen / ManagedVLWorker. Dynamem still loads mapping models (SigLIP / YoloE) when
``cpu_only=True``. Used by ``src/test/simulation/test_multi_env_nav_explore_smoke.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from emet.config.sim_launch_config import (
    SimLaunchConfig,
    SimLaunchMolmospaces,
    SimLaunchRobocasa,
)

MIN_TRANSLATION_M = 0.08
MIN_EXPLORED_DELTA = 50
# Robocasa/Molmo merge + EGL bind can exceed 2 minutes on a busy workstation GPU.
DEFAULT_SIM_BIND_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class NavExploreSmokeCase:
    """One environment case for :func:`run_nav_explore_smoke`."""

    name: str
    kind: str
    launch: SimLaunchConfig
    robot: str = "stretch"
    parameters_name: str = "dynav_config.yaml"


def default_nav_explore_cases() -> list[NavExploreSmokeCase]:
    """Canonical three-case matrix from the multi-env nav/explore plan."""
    return [
        NavExploreSmokeCase(
            name="robocasa_l1",
            kind="robocasa",
            launch=SimLaunchRobocasa(
                robot="stretch",
                robocasa_task="PickPlaceCounterToCabinet",
                robocasa_style=1,
                robocasa_layout=1,
                seed=0,
                headless=True,
            ),
        ),
        NavExploreSmokeCase(
            name="ovmm_robocasa_l2",
            kind="ovmm",
            launch=SimLaunchRobocasa(
                robot="stretch",
                robocasa_task="PickPlaceCounterToCabinet",
                robocasa_style=2,
                robocasa_layout=2,
                seed=0,
                headless=True,
            ),
        ),
        NavExploreSmokeCase(
            name="molmo_ithor_0",
            kind="molmospaces",
            launch=SimLaunchMolmospaces(
                robot="stretch",
                scene="ithor",
                split="train",
                index=0,
                headless=True,
            ),
        ),
    ]


def molmospaces_wrapper_available() -> bool:
    """True when ``.venv-molmospaces`` / ``emet-molmospaces`` can merge scenes."""
    from emet.simulation.molmospaces_config import build_molmospaces_wrapper_command

    return build_molmospaces_wrapper_command(["--help"]) is not None


def _unique_port_offset(*, salt: int = 0) -> int:
    # Keep away from OVMM find defaults (220/280) and common interactive offsets.
    return int(400 + (os.getpid() % 80) * 2 + int(salt) % 20)


def _base_xyt_world(robot: Any) -> np.ndarray:
    getter = getattr(robot, "get_base_pose_world", None)
    if callable(getter):
        pose = getter(timeout=10.0)
        if pose is not None:
            return np.asarray(pose, dtype=np.float64).reshape(-1)[:3]
    pose = robot.get_base_pose(timeout=10.0)
    return np.asarray(pose, dtype=np.float64).reshape(-1)[:3]


def explored_cell_count(agent: Any) -> int:
    """Count explored cells on the agent's 2D occupancy map (0 if unavailable)."""
    vm = getattr(agent, "voxel_map", None)
    if vm is None or not hasattr(vm, "get_2d_map"):
        return 0
    try:
        _obstacles, explored = vm.get_2d_map()
    except Exception:
        return 0
    if explored is None:
        return 0
    if hasattr(explored, "cpu"):
        explored = explored.cpu().numpy()
    arr = np.asarray(explored)
    if arr.size == 0:
        return 0
    return int(arr.sum())


def _read_log_tail(path: Path | None, *, max_chars: int = 4000) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:] if len(text) > max_chars else text


def _spawn_sim_with_log(
    launch: SimLaunchConfig,
    *,
    bind_timeout_s: float,
) -> tuple[Any, Path, Any]:
    """Spawn ``mujoco_server`` with stderr captured; wait for the ZMQ send port.

    Returns ``(proc, log_path, log_file)``. Caller must close *log_file* after terminating
    *proc* so the child keeps a live FD while running.
    """
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv
    from emet.simulation.sim_subprocess import wait_for_sim_tcp_port
    from emet.utils.port_utils import get_ports
    from emet.utils.process_tree import popen_session, terminate_process_tree

    argv = prepare_mujoco_server_argv(launch)
    cmd = [sys.executable, "-m", "emet.simulation.mujoco_server", *argv]
    repo_root = Path(__file__).resolve().parents[3]
    log_file = tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"nav_explore_smoke_{launch.kind}_",
        suffix=".log",
        delete=False,
        encoding="utf-8",
    )
    log_path = Path(log_file.name)
    proc = popen_session(
        cmd,
        cwd=str(repo_root),
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=log_file,
    )
    ports = get_ports(int(launch.port_offset))
    try:
        wait_for_sim_tcp_port(
            "127.0.0.1",
            int(ports.send),
            proc=proc,
            timeout_sec=float(bind_timeout_s),
        )
    except BaseException:
        terminate_process_tree(proc, grace_s=8.0)
        try:
            log_file.close()
        except Exception:
            pass
        raise
    return proc, log_path, log_file


def run_nav_explore_smoke(
    case: NavExploreSmokeCase,
    *,
    port_offset: int | None = None,
    zmq_startup_timeout: float = 180.0,
    sim_bind_timeout_s: float = DEFAULT_SIM_BIND_TIMEOUT_S,
) -> dict[str, Any]:
    """Spawn sim, rotate_in_place, one ``run_exploration``, return measured summary.

    Success when either translation after the full sequence is ``>= MIN_TRANSLATION_M``
    or explored cells grow by ``>= MIN_EXPLORED_DELTA``. Always tears down the sim.
    """
    from emet.app.robot_cli import create_robot_client_from_cli
    from emet.controller.controller_dynamem import DynamemController
    from emet.core.parameters import get_parameters
    from emet.utils.process_tree import terminate_process_tree

    if case.kind == "molmospaces" and not molmospaces_wrapper_available():
        return {
            "name": case.name,
            "kind": case.kind,
            "ok": False,
            "skipped": True,
            "skip_reason": "MolmoSpaces wrapper not installed (.venv-molmospaces)",
        }

    offset = int(port_offset) if port_offset is not None else _unique_port_offset()
    launch = replace(case.launch, port_offset=offset, headless=True)

    prev_env = {
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "EMET_SIM_NAV_DEBUG": os.environ.get("EMET_SIM_NAV_DEBUG"),
        "EMET_DYNAMEM_PERFECT_DEPTH": os.environ.get("EMET_DYNAMEM_PERFECT_DEPTH"),
        "EMET_ZMQ_STARTUP_TIMEOUT": os.environ.get("EMET_ZMQ_STARTUP_TIMEOUT"),
    }
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["EMET_SIM_NAV_DEBUG"] = "1"
    os.environ["EMET_DYNAMEM_PERFECT_DEPTH"] = "1"
    os.environ["EMET_ZMQ_STARTUP_TIMEOUT"] = str(float(zmq_startup_timeout))

    robot = None
    proc = None
    log_file = None
    log_path: Path | None = None
    summary: dict[str, Any] = {
        "name": case.name,
        "kind": case.kind,
        "port_offset": offset,
        "ok": False,
        "skipped": False,
    }
    try:
        proc, log_path, log_file = _spawn_sim_with_log(launch, bind_timeout_s=sim_bind_timeout_s)
        summary["sim_log"] = str(log_path)
        params = get_parameters(case.parameters_name)
        robot = create_robot_client_from_cli(
            case.robot,
            "127.0.0.1",
            port_offset=offset,
            parameters=params,
            enable_rerun_server=False,
            start_immediately=True,
            allow_missing_depth=False,
            zmq_startup_timeout=zmq_startup_timeout,
        )
        agent = DynamemController(
            robot,
            params,
            cpu_only=True,
            eqa=False,
            mllm=False,
            realtime_updates=False,
            save_rerun=False,
        )

        xyt0 = _base_xyt_world(robot)
        explored0 = explored_cell_count(agent)
        agent.rotate_in_place()
        xyt_after_rotate = _base_xyt_world(robot)
        explored_after_rotate = explored_cell_count(agent)

        explore_ok = bool(agent.run_exploration())
        xyt1 = _base_xyt_world(robot)
        explored1 = explored_cell_count(agent)

        delta_xy = float(np.hypot(float(xyt1[0] - xyt0[0]), float(xyt1[1] - xyt0[1])))
        explored_delta = int(explored1 - explored0)
        ok = bool(delta_xy >= MIN_TRANSLATION_M or explored_delta >= MIN_EXPLORED_DELTA)

        summary.update(
            {
                "ok": ok,
                "explore_ok": explore_ok,
                "xyt0": xyt0.tolist(),
                "xyt_after_rotate": xyt_after_rotate.tolist(),
                "xyt1": xyt1.tolist(),
                "delta_xy_m": delta_xy,
                "explored0": explored0,
                "explored_after_rotate": explored_after_rotate,
                "explored1": explored1,
                "explored_delta": explored_delta,
                "min_translation_m": MIN_TRANSLATION_M,
                "min_explored_delta": MIN_EXPLORED_DELTA,
            }
        )
        return summary
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
        tail = _read_log_tail(log_path)
        if tail:
            summary["sim_log_tail"] = tail
        return summary
    finally:
        if robot is not None:
            stop = getattr(robot, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
        if proc is not None:
            terminate_process_tree(proc, grace_s=8.0)
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass
        for key, val in prev_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


__all__ = [
    "DEFAULT_SIM_BIND_TIMEOUT_S",
    "MIN_EXPLORED_DELTA",
    "MIN_TRANSLATION_M",
    "NavExploreSmokeCase",
    "default_nav_explore_cases",
    "explored_cell_count",
    "molmospaces_wrapper_available",
    "run_nav_explore_smoke",
]
