# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Dynamic exploration benchmark runners.

Phase 1: explore-loop batch (``run_explore_episode_subprocess``).
Phase 2: single world-change episode (``run_world_change_episode``).
Lifelong: K-cycle checkpoint/fuzz/reload loop (``run_lifelong_episode``).
"""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from emet.eval.benchmark_dynagraph import apply_dynamic_explore_backend, profile_settings
from emet.eval.dynamic_exploration_config import (
    DynamicExploreConfig,
    DynamicExploreEpisode,
    ExploreRunSpec,
    LifelongEpisode,
    WorldChangeEpisode,
    flatten_eval_metrics,
)
from emet.eval.ovmm_find_phase import resolve_find_phase_nav_step_timeout
from emet.eval.sim_eval_session import benchmark_sim_server, connect_benchmark_robot
from emet.eval.world_fuzz import FuzzAction, apply_fuzz_actions, fuzz_actions_for_cycle

LIFELONG_QUESTION_ENV = "lifelong"
_GT_NODE_DESC_PREFIX = "ground_truth:"
_NODE_MATCH_RADIUS_M = 0.75


@dataclass
class DynamicExploreRunConfig:
    backend: str = "dynagraph"
    cpu_only: bool = False
    port_offset: int = 0
    no_sensor_perception: bool = True
    resume: bool = False
    skip_eqa: bool = False
    use_scene_cache: bool = True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def count_object_nodes(memory: Any, *, label_hint: str | None = None) -> int:
    """Count non-viewpoint graph nodes; optional substring match on any label."""
    if memory is None:
        return 0
    nodes = [n for n in memory.get_nodes() if not getattr(n, "is_viewpoint", False)]
    if label_hint:
        hint = label_hint.lower()
        nodes = [
            n
            for n in nodes
            if any(hint in str(lab).lower() for lab in (getattr(n, "labels", None) or []))
        ]
    return len(nodes)


def _dynagraph_subprocess_timeout_s(
    *,
    explore_max_iters: int = 0,
    sim_kind: str = "",
    cpu_only: bool = False,
    skip_eqa: bool = False,
) -> float:
    """Wall-clock budget for one ``emet run dynagraph`` subprocess.

    Override with ``EMET_DYNAMIC_EXPLORE_DYNAGRAPH_TIMEOUT_S`` (seconds).
    Robocasa explore K=3 on GPU is ~60--75 min; K=30 needs several hours.
    """
    override = os.environ.get("EMET_DYNAMIC_EXPLORE_DYNAGRAPH_TIMEOUT_S", "").strip()
    if override:
        return float(override)
    iters = max(0, int(explore_max_iters))
    timeout_s = 1800.0 + 1200.0 * iters
    if sim_kind in ("molmospaces", "robocasa"):
        timeout_s += 900.0
    if iters == 0:
        timeout_s = max(timeout_s, 3600.0)
    if not skip_eqa:
        # Post-explore question bank runs real VLM EQA (Qwen3-VL load + 2+ questions).
        timeout_s += 5400.0 if cpu_only else 3600.0
    if cpu_only:
        timeout_s *= 2.0
    return min(timeout_s, 43200.0)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _tail_text(path: Path, *, max_chars: int = 1200) -> str:
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _append_progress_event(progress_path: Path | None, event: dict[str, Any]) -> None:
    if progress_path is None:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.time(), **event}
    with progress_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()


def _kill_process_tree(proc: subprocess.Popen[str], *, label: str) -> None:
    """Kill ``proc`` and its process group (uv wrapper + run_dynagraph + mujoco_server)."""
    pid = proc.pid
    if pid is None:
        return
    print(f"[dynamic-explore] KILL_TREE {label} pid={pid}", flush=True)
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def run_logged_subprocess(
    cmd: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str] | None,
    log_path: Path,
    timeout_s: float,
    label: str,
    progress_path: Path | None = None,
    heartbeat_s: float | None = None,
    stale_log_s: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with stdout/stderr teed to ``log_path`` and heartbeats.

    Heartbeat / stale-log intervals (seconds):
    - ``EMET_DYNAMIC_EXPLORE_HEARTBEAT_S`` (default 120)
    - ``EMET_DYNAMIC_EXPLORE_STALE_LOG_S`` (default 900) — warn when log mtime is stale
    - ``EMET_DYNAMIC_EXPLORE_STALE_KILL_S`` (default 2× stale) — kill process group when log is stale
    """
    hb = float(heartbeat_s) if heartbeat_s is not None else _env_float("EMET_DYNAMIC_EXPLORE_HEARTBEAT_S", 120.0)
    stale = float(stale_log_s) if stale_log_s is not None else _env_float("EMET_DYNAMIC_EXPLORE_STALE_LOG_S", 900.0)
    stale_kill = _env_float("EMET_DYNAMIC_EXPLORE_STALE_KILL_S", stale * 2.0 if stale > 0 else 0.0)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    last_hb = t0
    last_mtime = 0.0
    stale_warned = False

    print(
        f"[dynamic-explore] START {label} timeout_s={timeout_s:.0f} log={log_path}",
        flush=True,
    )
    _append_progress_event(
        progress_path,
        {
            "event": "subprocess_start",
            "label": label,
            "timeout_s": timeout_s,
            "log": str(log_path),
            "cmd": cmd,
        },
    )

    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            while True:
                try:
                    returncode = proc.wait(timeout=min(5.0, max(1.0, hb)))
                    break
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    elapsed = now - t0
                    if elapsed >= timeout_s:
                        _kill_process_tree(proc, label=label)
                        tail = _tail_text(log_path)
                        raise subprocess.TimeoutExpired(cmd, timeout_s, output=tail) from None

                    mtime = log_path.stat().st_mtime if log_path.is_file() else 0.0
                    log_age = time.time() - mtime if mtime else elapsed
                    if now - last_hb >= hb:
                        last_hb = now
                        msg = (
                            f"[dynamic-explore] HEARTBEAT {label} elapsed_s={elapsed:.0f}/"
                            f"{timeout_s:.0f} log_age_s={log_age:.0f} pid={proc.pid}"
                        )
                        print(msg, flush=True)
                        _append_progress_event(
                            progress_path,
                            {
                                "event": "heartbeat",
                                "label": label,
                                "elapsed_s": elapsed,
                                "timeout_s": timeout_s,
                                "log_age_s": log_age,
                                "pid": proc.pid,
                            },
                        )
                    if stale_kill > 0 and log_age >= stale_kill:
                        print(
                            f"[dynamic-explore] STALE_KILL {label} log_age_s={log_age:.0f} "
                            f"(threshold={stale_kill:.0f})",
                            flush=True,
                        )
                        _kill_process_tree(proc, label=label)
                        tail = _tail_text(log_path)
                        raise subprocess.TimeoutExpired(cmd, timeout_s, output=tail) from None
                    if stale > 0 and log_age >= stale and (mtime != last_mtime or not stale_warned):
                        last_mtime = mtime
                        stale_warned = True
                        tail = _tail_text(log_path, max_chars=800)
                        warn = (
                            f"[dynamic-explore] STALE_LOG {label} log_age_s={log_age:.0f} "
                            f"(threshold={stale:.0f}). Tail:\n{tail}"
                        )
                        print(warn, flush=True)
                        _append_progress_event(
                            progress_path,
                            {
                                "event": "stale_log",
                                "label": label,
                                "log_age_s": log_age,
                                "stale_threshold_s": stale,
                                "log_tail": tail,
                            },
                        )
            wall = time.monotonic() - t0
            print(
                f"[dynamic-explore] END {label} returncode={returncode} wall_s={wall:.0f}",
                flush=True,
            )
            _append_progress_event(
                progress_path,
                {
                    "event": "subprocess_end",
                    "label": label,
                    "returncode": returncode,
                    "wall_s": wall,
                },
            )
            return subprocess.CompletedProcess(cmd, returncode)
        except Exception:
            if proc.poll() is None:
                _kill_process_tree(proc, label=label)
            raise


def _resolve_sim_cfg(episode: DynamicExploreEpisode):
    from emet.config.sim_launch_config import (
        SimLaunchMolmospaces,
        SimLaunchRobocasa,
        load_sim_launch_config_from_path,
    )

    sim_cfg = load_sim_launch_config_from_path(episode.sim)
    if episode.seed is not None and isinstance(sim_cfg, SimLaunchRobocasa):
        sim_cfg = replace(sim_cfg, seed=int(episode.seed), robot="stretch")
    if episode.molmo_index is not None and isinstance(sim_cfg, SimLaunchMolmospaces):
        sim_cfg = replace(sim_cfg, index=int(episode.molmo_index), robot="stretch")
    return sim_cfg


def _profile_cli_flags(backend: str, cfg: DynamicExploreConfig) -> list[str]:
    profile_name = cfg.profiles.get(backend, "interactive")
    settings = profile_settings(profile_name)
    flags: list[str] = []
    if settings.get("dynagraph_merge_xy_m") is not None:
        flags.extend(["--merge-xy-m", str(settings["dynagraph_merge_xy_m"])])
    if settings.get("dynagraph_staleness_horizon") is not None:
        flags.extend(["--staleness-horizon", str(settings["dynagraph_staleness_horizon"])])
    return flags


def build_dynagraph_subprocess_cmd(
    *,
    export_dir: Path,
    port_offset: int,
    backend: str,
    cfg: DynamicExploreConfig,
    cpu_only: bool,
    no_sensor_perception: bool,
    questions_yaml: Path | None = None,
    question_env: str | None = None,
    input_dir: Path | None = None,
    explore_iters: int = 0,
    export_voxel_pickle: bool = False,
    include_explore_loop: bool = True,
    skip_eqa: bool = False,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "emet",
        "run",
        "dynagraph",
        "--robot",
        "stretch",
        "--robot-ip",
        "127.0.0.1",
        "--port-offset",
        str(port_offset),
        "--no-rerun",
        "--export",
        str(export_dir),
    ]
    if export_voxel_pickle:
        cmd.append("--export-voxel-pickle")
    if not skip_eqa and questions_yaml is not None and question_env is not None:
        cmd.extend(
            [
                "--question-file",
                str(questions_yaml),
                "--question-env",
                question_env,
            ]
        )
    if input_dir is not None:
        cmd.extend(["--input-path", str(input_dir)])
    if cpu_only:
        cmd.append("--cpu-only")
    if no_sensor_perception:
        cmd.append("--no-sensor-perception")
    cmd.extend(_profile_cli_flags(backend, cfg))
    if include_explore_loop and explore_iters > 0:
        cmd.extend(["--explore-loop", "--explore-max-iters", str(explore_iters)])
    return cmd


def run_explore_episode_subprocess(
    run: ExploreRunSpec,
    run_cfg: DynamicExploreRunConfig,
    cfg: DynamicExploreConfig,
    *,
    output_dir: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Phase 1: sim subprocess + ``emet run dynagraph`` + ``eval-dynagraph``."""
    from emet.memory.graph_eqa.dynagraph_eval import compute_dynagraph_eval

    repo = repo_root or _repo_root()
    out_json = output_dir / f"{run.run_id}.json"
    export_dir = output_dir / "exports" / run.run_id

    if run_cfg.resume and out_json.is_file():
        return json.loads(out_json.read_text(encoding="utf-8"))

    sim_cfg = _resolve_sim_cfg(run.episode)
    port_offset = int(run_cfg.port_offset)
    sim_cfg = replace(sim_cfg, port_offset=port_offset, headless=True)

    cache_dir = None
    explore_iters = int(run.explore_max_iters)
    include_explore = run.mapping_mode == "explore"
    if run_cfg.use_scene_cache:
        from emet.eval.scene_map_cache import resolve_scene_cache_for_sim

        cache_dir = resolve_scene_cache_for_sim(sim_cfg, enabled=True)
        if cache_dir is not None:
            # Baseline already mapped — skip rotate/explore budget.
            explore_iters = 0
            include_explore = False

    dyn_cmd = build_dynagraph_subprocess_cmd(
        export_dir=export_dir,
        port_offset=port_offset,
        backend=run.backend,
        cfg=cfg,
        cpu_only=run_cfg.cpu_only,
        no_sensor_perception=run_cfg.no_sensor_perception,
        questions_yaml=cfg.paths.questions_yaml,
        question_env=run.episode.question_env,
        input_dir=cache_dir,
        explore_iters=explore_iters,
        include_explore_loop=include_explore,
        skip_eqa=run_cfg.skip_eqa,
    )

    t0 = time.monotonic()
    try:
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.parent.mkdir(parents=True, exist_ok=True)

        with benchmark_sim_server(sim_cfg, repo=repo, cpu_only=run_cfg.cpu_only, cwd=repo) as sim:
            env = sim.env
            sim_kind = sim.sim_kind

            dyn_log = export_dir / "dynagraph.log"
            export_dir.mkdir(parents=True, exist_ok=True)
            dyn_timeout = _dynagraph_subprocess_timeout_s(
                explore_max_iters=run.explore_max_iters,
                sim_kind=sim_kind,
                cpu_only=run_cfg.cpu_only,
                skip_eqa=run_cfg.skip_eqa,
            )
            progress_path = output_dir / "progress.jsonl"
            try:
                proc = run_logged_subprocess(
                    dyn_cmd,
                    cwd=repo,
                    env=env,
                    log_path=dyn_log,
                    timeout_s=dyn_timeout,
                    label=run.run_id,
                    progress_path=progress_path,
                )
            except subprocess.TimeoutExpired as exc:
                tail = str(exc.output or _tail_text(dyn_log))
                raise RuntimeError(
                    f"dynagraph timed out after {dyn_timeout:.0f}s\n{tail[-4000:]}"
                ) from exc
            combined = dyn_log.read_text(encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                raise RuntimeError(f"dynagraph exited {proc.returncode}\n{combined[-4000:]}")
            if not (export_dir / "memory").is_dir() and "Exported graph memory to" not in combined:
                raise RuntimeError(f"export missing under {export_dir}\n{combined[-2000:]}")

            metrics = compute_dynagraph_eval(
                export_dir,
                questions_path=cfg.paths.questions_yaml,
                question_env=run.episode.question_env,
            )
        wall_s = time.monotonic() - t0
        row = flatten_eval_metrics(metrics, run_spec=run, episode_wall_s=wall_s)
        row["export_dir"] = str(export_dir)
        row["map_source"] = "cache" if cache_dir is not None else "live"
        if cache_dir is not None:
            row["scene_cache_dir"] = str(cache_dir)
        health = metrics.get("graph_health") or {}
        failure = str(health.get("failure_class") or "")
        row["graph_health_ok"] = failure not in ("empty_graph", "thin_graph")
        if failure in ("empty_graph", "thin_graph"):
            print(
                f"[dynamic-explore] WARN {run.run_id}: graph_health={failure} "
                f"n_object={health.get('n_object')} — instance→graph may not have attached",
                flush=True,
            )
        payload = {"metrics": metrics, "summary": row}
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
    except Exception as exc:
        wall_s = time.monotonic() - t0
        err = str(exc)
        dyn_log = export_dir / "dynagraph.log"
        if dyn_log.is_file():
            tail = _tail_text(dyn_log, max_chars=2000)
            if tail and tail not in err:
                err = f"{err}\n--- dynagraph.log tail ---\n{tail}"
        print(f"[dynamic-explore] FAIL {run.run_id} wall_s={wall_s:.0f}: {err[:400]}", flush=True)
        _append_progress_event(
            output_dir / "progress.jsonl",
            {
                "event": "run_fail",
                "label": run.run_id,
                "wall_s": wall_s,
                "error": err[:2000],
            },
        )
        payload = {
            "metrics": {"error": err},
            "summary": flatten_eval_metrics(
                {"error": err},
                run_spec=run,
                episode_wall_s=wall_s,
            ),
        }
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload


def _run_eqa_single(agent: Any, robot: Any, qspec: dict[str, Any]) -> dict[str, Any]:
    import re

    from emet.controller.task.dynamem import EQAExecuter

    qtext = str(qspec.get("question", "")).strip()
    robot.move_to_nav_posture()
    robot.switch_to_navigation_mode()
    eq_executor = EQAExecuter(agent)
    try:
        discord_text, _imgs = eq_executor(qtext)
    except Exception as e:
        discord_text = f"EQA question failed: {e}"
    answer = ""
    m = re.search(r"(?i)answer:\s*(.+?)(?:\n|$)", discord_text or "")
    if m:
        answer = m.group(1).strip()
    elif discord_text:
        answer = discord_text.strip()
    return {**qspec, "question": qtext, "discord_text": discord_text, "answer": answer}


def _default_relocate_xy(session: dict[str, Any] | None, body: str) -> tuple[float, float, float]:
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    placements = read_sim_object_placements(session)
    if body not in placements:
        raise RuntimeError(f"body {body!r} not in sim_object_placements")
    pos = placements[body]["pos"]
    return float(pos[0]) + 1.5, float(pos[1]) + 0.5, float(pos[2])


def run_world_change_episode(
    wc: WorldChangeEpisode,
    base_episode: DynamicExploreEpisode,
    run_cfg: DynamicExploreRunConfig,
    cfg: DynamicExploreConfig,
    *,
    explore_max_iters: int = 15,
    output_dir: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Phase 2: explore → EQA pre → relocate body → recovery → EQA post → export."""
    from emet.app.dynagraph_explore import dynagraph_explore_until_terminated
    from emet.controller.controller_dynagraph import DynagraphController
    from emet.controller.task.dynamem import EQAExecuter
    from emet.core.parameters import get_parameters
    from emet.memory.graph_eqa.dynagraph_eval import compute_dynagraph_eval
    from emet.memory.graph_eqa.question_bank import load_question_bank, score_eqa_results
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    repo = repo_root or _repo_root()
    run_id = f"{wc.id}_{run_cfg.backend}"
    out_json = output_dir / f"{run_id}.json"
    export_dir = output_dir / "exports" / run_id

    if run_cfg.resume and out_json.is_file():
        return json.loads(out_json.read_text(encoding="utf-8"))

    sim_cfg = _resolve_sim_cfg(base_episode)
    port_offset = int(run_cfg.port_offset)
    sim_cfg = replace(sim_cfg, port_offset=port_offset, headless=True)

    questions = load_question_bank(cfg.paths.questions_yaml, env_filter=wc.question_env)
    pre_q = next((q for q in questions if q.get("phase") == "pre"), questions[0] if questions else None)
    post_q = next((q for q in questions if q.get("phase") == "post"), questions[-1] if len(questions) > 1 else None)
    if pre_q is None or post_q is None:
        raise RuntimeError(f"world-change question bank {wc.question_env!r} needs pre/post questions")

    t0 = time.monotonic()
    recovery_steps = 0
    n_stale_after_move = 0
    n_pruned_total = 0

    try:
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        with benchmark_sim_server(sim_cfg, repo=repo, cpu_only=run_cfg.cpu_only, cwd=repo) as sim:
            sim_kind = sim.sim_kind
            robot = connect_benchmark_robot(sim_cfg, port_offset)
            agent = None
            try:
                parameters = apply_dynamic_explore_backend(get_parameters("dynav_config.yaml"), run_cfg.backend)
                parameters["encoder"] = None
                parameters["debug_perfect_sensor_depth"] = True
                nav_timeout = resolve_find_phase_nav_step_timeout(
                    cpu_only=run_cfg.cpu_only,
                    sim_kind=sim_kind,
                )
                parameters["find_phase_nav_step_timeout_s"] = nav_timeout

                cache_dir = None
                if run_cfg.use_scene_cache:
                    from emet.eval.scene_map_cache import resolve_scene_cache_for_sim

                    cache_dir = resolve_scene_cache_for_sim(sim_cfg, enabled=True)

                agent = DynagraphController(
                    robot,
                    parameters,
                    save_rerun=False,
                    cpu_only=run_cfg.cpu_only,
                    use_instance_graph=True,
                    use_sensor_perception=not run_cfg.no_sensor_perception,
                    graph_memory_input_path=str(cache_dir) if cache_dir is not None else None,
                )
                agent.start()
                agent._fast_explore_lookaround = True

                executor = EQAExecuter(agent)
                if cache_dir is None:
                    executor.rotate_in_place()
                    dynagraph_explore_until_terminated(agent, max_iterations=int(explore_max_iters))
                else:
                    # Cached baseline already covers the static scene; world-change
                    # still invalidates nodes near the relocated body below.
                    agent.update()

                pre_row = _run_eqa_single(agent, robot, pre_q)
                pre_score = score_eqa_results([pre_row], episode_dir=None)

                session = robot.get_emet_session()
                placements_pre = read_sim_object_placements(session)
                old_pos = None
                if wc.relocate_body in placements_pre:
                    old_pos = list(placements_pre[wc.relocate_body]["pos"])
                rx, ry, rz = _default_relocate_xy(session, wc.relocate_body)
                from emet.simulation.sim_manipulation import robot_zmq_set_body_pose

                robot_zmq_set_body_pose(robot, wc.relocate_body, [rx, ry, rz])
                time.sleep(0.5)
                agent.update()

                mem = agent.graph_memory
                cur_step = int(getattr(agent, "obs_count", 0))
                n_stale_after_move = count_object_nodes(
                    mem,
                    label_hint=str(pre_q.get("gt_body_key") or "obj"),
                )
                if mem is not None:
                    # Known move: age nodes at the old pose so maintain can prune without
                    # waiting a full staleness_horizon of unobserved steps.
                    if old_pos is not None and hasattr(mem, "invalidate_nodes_near"):
                        _aged, n_inv = mem.invalidate_nodes_near(
                            old_pos,
                            radius_m=_NODE_MATCH_RADIUS_M,
                            current_step=cur_step,
                            prune=True,
                        )
                        n_pruned_total += int(n_inv)
                    elif mem.staleness_horizon > 0:
                        n_pruned_total += int(mem.maintain(cur_step))
                    if hasattr(mem, "clear_eqa_working_memory"):
                        mem.clear_eqa_working_memory()

                recovery_iters = int(cfg.recovery_explore_iters)
                _, _n_ok, nit = dynagraph_explore_until_terminated(agent, max_iterations=recovery_iters)
                recovery_steps = int(nit)
                if _n_ok == 0:
                    executor.rotate_in_place()
                    recovery_steps += 1

                post_row = _run_eqa_single(agent, robot, post_q)
                post_score = score_eqa_results([post_row], episode_dir=None)

                from emet.memory.headless_export import export_dynagraph_episode

                session = robot.get_emet_session()
                placements = read_sim_object_placements(session)
                export_dynagraph_episode(
                    agent.graph_memory,
                    getattr(agent, "voxel_map", None),
                    export_dir,
                    title="Scene graph (dynamic world-change export)",
                    robot="stretch",
                    environment=session.get("environment") if isinstance(session, dict) else None,
                    spawn_floor_map=session.get("spawn_floor_map") if isinstance(session, dict) else None,
                    sim_object_placements=placements,
                )

                eval_metrics = compute_dynagraph_eval(
                    export_dir,
                    questions_path=cfg.paths.questions_yaml,
                    question_env=wc.question_env,
                )
                wall_s = time.monotonic() - t0

                loc_err = None
                gt_key = str(post_q.get("gt_body_key") or "")
                if gt_key and gt_key in placements:
                    gt_xy = np.asarray(placements[gt_key]["pos"][:2], dtype=np.float64)
                    cited = post_row.get("cited_xyz")
                    if isinstance(cited, (list, tuple)) and len(cited) >= 2:
                        pred_xy = np.asarray(cited[:2], dtype=np.float64)
                        loc_err = float(np.linalg.norm(pred_xy - gt_xy))

                payload = {
                    "run_id": run_id,
                    "phase": "world-change",
                    "episode_id": base_episode.id,
                    "backend": run_cfg.backend,
                    "answer_correct_pre": bool(pre_score.get("accuracy", 0) >= 1.0),
                    "answer_correct_post": bool(post_score.get("accuracy", 0) >= 1.0),
                    "n_stale_nodes_after_move": n_stale_after_move,
                    "n_pruned_by_maintain": n_pruned_total,
                    "recovery_steps": recovery_steps,
                    "relocate_body": wc.relocate_body,
                    "relocate_xyz": [rx, ry, rz],
                    "localization_err_m": loc_err,
                    "episode_wall_s": wall_s,
                    "map_source": "cache" if cache_dir is not None else "live",
                    "scene_cache_dir": str(cache_dir) if cache_dir is not None else None,
                    "pre_eqa": pre_score,
                    "post_eqa": post_score,
                    "eval": eval_metrics,
                    "export_dir": str(export_dir),
                }
                out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                return payload
            finally:
                if agent is not None:
                    stop = getattr(agent, "stop", None)
                    if callable(stop):
                        stop()
                stop = getattr(robot, "stop", None)
                if callable(stop):
                    stop()
    except Exception as exc:
        payload = {
            "run_id": run_id,
            "phase": "world-change",
            "episode_id": base_episode.id,
            "backend": run_cfg.backend,
            "error": str(exc),
            "episode_wall_s": time.monotonic() - t0,
        }
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload


# ---------------------------------------------------------------------------
# Lifelong (K-cycle) episodes: explore/answer -> checkpoint -> fuzz -> reload.
# ---------------------------------------------------------------------------


def _write_cycle_questions_yaml(path: Path, questions: list[dict[str, Any]]) -> None:
    import yaml

    payload = {"environments": [{"env": LIFELONG_QUESTION_ENV, "questions": questions}]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_checkpoint_object_nodes(ckpt_dir: Path) -> list[dict[str, Any]]:
    """Object nodes (non-viewpoint, non-GT) from a checkpoint's graph.json."""
    graph_path = ckpt_dir / "graph.json"
    if not graph_path.is_file():
        return []
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for n in data.get("nodes", []):
        if n.get("is_viewpoint"):
            continue
        desc = n.get("description") or ""
        if isinstance(desc, str) and desc.startswith(_GT_NODE_DESC_PREFIX):
            continue
        out.append(n)
    return out


def _count_nodes_near(nodes: list[dict[str, Any]], pos: list[float], radius: float) -> int:
    target = np.asarray(pos[:2], dtype=np.float64)
    count = 0
    for n in nodes:
        xyz = np.asarray(n.get("xyz", [0, 0, 0]), dtype=np.float64).reshape(-1)
        if xyz.size >= 2 and float(np.linalg.norm(xyz[:2] - target)) <= radius:
            count += 1
    return count


def _churn_metrics_for_moves(
    ckpt_dir: Path,
    move_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-moved-body memory adaptation: nodes near the old vs new GT position."""
    nodes = _load_checkpoint_object_nodes(ckpt_dir)
    out: list[dict[str, Any]] = []
    for rec in move_records:
        old_pos = rec.get("old_pos")
        new_pos = rec.get("pos")
        if old_pos is None or new_pos is None:
            continue
        n_old = _count_nodes_near(nodes, list(old_pos), _NODE_MATCH_RADIUS_M)
        n_new = _count_nodes_near(nodes, list(new_pos), _NODE_MATCH_RADIUS_M)
        out.append(
            {
                "body": rec.get("target"),
                "old_pos": [float(x) for x in old_pos],
                "new_pos": [float(x) for x in new_pos],
                "nodes_near_old_pos": int(n_old),
                "nodes_near_new_pos": int(n_new),
                "adapted": bool(n_new > 0),
                "stale": bool(n_old > 0),
            }
        )
    return out


def invalidate_checkpoint_nodes_near_moves(
    ckpt_dir: Path,
    move_records: list[dict[str, Any]],
    *,
    radius_m: float = _NODE_MATCH_RADIUS_M,
) -> int:
    """Age/remove object nodes at pre-move poses in a lifelong checkpoint ``graph.json``.

    The next cycle reloads this checkpoint; without invalidation, nodes at the old
    pose linger until ``staleness_horizon`` elapses (often longer than one explore
    budget). Returns the number of object nodes removed.
    """
    graph_path = ckpt_dir / "graph.json"
    if not graph_path.is_file() or not move_records:
        return 0
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = list(data.get("nodes") or [])
    if not nodes:
        return 0
    old_targets: list[np.ndarray] = []
    for rec in move_records:
        old_pos = rec.get("old_pos")
        if old_pos is None:
            continue
        old_targets.append(np.asarray(old_pos[:2], dtype=np.float64))
    if not old_targets:
        return 0

    def _near_old(n: dict[str, Any]) -> bool:
        if n.get("is_viewpoint") or n.get("is_frontier"):
            return False
        desc = n.get("description") or ""
        if isinstance(desc, str) and desc.startswith(_GT_NODE_DESC_PREFIX):
            return False
        xyz = np.asarray(n.get("xyz", [0, 0, 0]), dtype=np.float64).reshape(-1)
        if xyz.size < 2:
            return False
        return any(float(np.linalg.norm(xyz[:2] - t)) <= float(radius_m) for t in old_targets)

    kept = [n for n in nodes if not _near_old(n)]
    n_removed = len(nodes) - len(kept)
    if n_removed <= 0:
        return 0
    # Drop observations that only supported removed object nodes (best-effort).
    drop_obs = {
        int(n.get("obs_id"))
        for n in nodes
        if _near_old(n) and n.get("obs_id") is not None and not n.get("is_viewpoint")
    }
    if drop_obs and isinstance(data.get("observations"), list):
        data["observations"] = [
            o for o in data["observations"] if int(o.get("obs_id", -1)) not in drop_obs
        ]
        kept = [
            n
            for n in kept
            if not (n.get("is_viewpoint") and int(n.get("obs_id", -1)) in drop_obs)
        ]
    for i, n in enumerate(kept, start=1):
        n["node_id"] = i
    data["nodes"] = kept
    graph_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return int(n_removed)


def _apply_lifelong_changes(
    change_spec: dict[str, Any],
    *,
    port_offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Connect a short-lived ZMQ client, apply fuzz actions, and verify moved bodies.

    Returns (applied_records, move_records); move_records add ``old_pos`` and
    ``verified_pos`` from the live GT placements for churn metrics + verification.
    """
    from emet.app.robot_cli import create_robot_client_from_cli
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    robot = create_robot_client_from_cli(
        "stretch",
        "127.0.0.1",
        port_offset=port_offset,
        enable_rerun_server=False,
        start_immediately=True,
        allow_missing_depth=True,
    )
    try:
        if hasattr(robot, "wait_for_obs"):
            robot.wait_for_obs(timeout=30.0)
        placements = read_sim_object_placements(robot.get_emet_session())
        actions: list[FuzzAction] = fuzz_actions_for_cycle(change_spec, placements)
        if not actions:
            return [], []
        old_pos_by_body = {
            a.target: [float(x) for x in placements[a.target]["pos"][:3]]
            for a in actions
            if a.kind == "move" and a.target in placements
        }
        applied = apply_fuzz_actions(robot, actions)
        time.sleep(0.5)
        after = read_sim_object_placements(robot.get_emet_session())
        move_records: list[dict[str, Any]] = []
        for rec in applied:
            if rec.get("kind") != "move":
                continue
            body = str(rec.get("target"))
            rec["old_pos"] = old_pos_by_body.get(body)
            entry = after.get(body) if after else None
            if entry is not None:
                measured = [float(x) for x in entry["pos"][:3]]
                rec["verified_pos"] = measured
                target = np.asarray(rec.get("pos", measured), dtype=np.float64)
                rec["verified"] = bool(
                    float(np.linalg.norm(np.asarray(measured) - target)) <= 0.05
                )
            move_records.append(rec)
        return applied, move_records
    finally:
        stop = getattr(robot, "stop", None)
        if callable(stop):
            stop()


def run_lifelong_episode(
    le: LifelongEpisode,
    base_episode: DynamicExploreEpisode,
    run_cfg: DynamicExploreRunConfig,
    cfg: DynamicExploreConfig,
    *,
    output_dir: Path,
    repo_root: Path | None = None,
    cycle_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Lifelong K-cycle episode against one persistent sim server.

    Each cycle runs ``emet run dynagraph`` in a fresh subprocess that reloads the
    previous checkpoint (graph + voxel pickle + step counter), explores, answers the
    cycle's questions, and exports checkpoint ``cycle_t``. Between cycles the world
    is fuzzed over ZMQ (object teleports + door joints) and moves are verified
    against the live GT placements.
    """
    from emet.memory.graph_eqa.question_bank import score_eqa_results

    repo = repo_root or _repo_root()
    run_id = f"{le.id}_{run_cfg.backend}"
    out_json = output_dir / f"{run_id}.json"
    ckpt_root = output_dir / "exports" / run_id

    if run_cfg.resume and out_json.is_file():
        return json.loads(out_json.read_text(encoding="utf-8"))

    sim_cfg = _resolve_sim_cfg(base_episode)
    port_offset = int(run_cfg.port_offset)
    sim_cfg = replace(sim_cfg, port_offset=port_offset, headless=True)

    t0 = time.monotonic()
    cycle_results: list[dict[str, Any]] = []
    pending_moves: list[dict[str, Any]] = []

    try:
        if ckpt_root.exists():
            shutil.rmtree(ckpt_root)
        ckpt_root.mkdir(parents=True, exist_ok=True)

        with benchmark_sim_server(sim_cfg, repo=repo, cpu_only=run_cfg.cpu_only, cwd=repo) as sim:
            env = sim.env
            sim_kind = sim.sim_kind

            prev_ckpt: Path | None = None
            for t in range(int(le.cycles)):
                cycle_t0 = time.monotonic()
                ckpt = ckpt_root / f"cycle_{t}"
                questions = le.questions_for_cycle(t)
                qyaml: Path | None = None
                if questions:
                    qyaml = ckpt_root / f"questions_cycle{t}.yaml"
                    _write_cycle_questions_yaml(qyaml, questions)

                explore_iters = le.explore_iters_first if t == 0 else le.explore_iters_resume
                input_dir = prev_ckpt
                if t == 0 and input_dir is None and run_cfg.use_scene_cache:
                    from emet.eval.scene_map_cache import resolve_scene_cache_for_sim

                    cache_dir = resolve_scene_cache_for_sim(sim_cfg, enabled=True)
                    if cache_dir is not None:
                        input_dir = cache_dir
                        explore_iters = 0
                cmd = build_dynagraph_subprocess_cmd(
                    export_dir=ckpt,
                    port_offset=port_offset,
                    backend=run_cfg.backend,
                    cfg=cfg,
                    cpu_only=run_cfg.cpu_only,
                    no_sensor_perception=run_cfg.no_sensor_perception,
                    questions_yaml=qyaml,
                    question_env=LIFELONG_QUESTION_ENV if qyaml is not None else None,
                    input_dir=input_dir,
                    explore_iters=int(explore_iters),
                    export_voxel_pickle=True,
                    skip_eqa=run_cfg.skip_eqa or qyaml is None,
                    include_explore_loop=int(explore_iters) > 0,
                )
                cycle_timeout = cycle_timeout_s
                if cycle_timeout is None:
                    cycle_timeout = _dynagraph_subprocess_timeout_s(
                        explore_max_iters=int(explore_iters),
                        sim_kind=sim_kind,
                        cpu_only=run_cfg.cpu_only,
                        skip_eqa=run_cfg.skip_eqa or qyaml is None,
                    )
                cycle_log = ckpt / "dynagraph.log"
                ckpt.mkdir(parents=True, exist_ok=True)
                progress_path = output_dir / "progress.jsonl"
                try:
                    proc = run_logged_subprocess(
                        cmd,
                        cwd=repo,
                        env=env,
                        log_path=cycle_log,
                        timeout_s=cycle_timeout,
                        label=f"{run_id}_cycle{t}",
                        progress_path=progress_path,
                    )
                except subprocess.TimeoutExpired as exc:
                    tail = str(exc.output or _tail_text(cycle_log))
                    raise RuntimeError(
                        f"cycle {t}: dynagraph timed out after {cycle_timeout:.0f}s\n{tail[-4000:]}"
                    ) from exc
                combined = cycle_log.read_text(encoding="utf-8", errors="replace")
                if proc.returncode != 0:
                    raise RuntimeError(f"cycle {t}: dynagraph exited {proc.returncode}\n{combined[-4000:]}")
                if not (ckpt / "manifest.json").is_file():
                    raise RuntimeError(f"cycle {t}: checkpoint missing under {ckpt}\n{combined[-2000:]}")

                eqa_rows: list[dict[str, Any]] = []
                eqa_path = ckpt / "eqa_results.json"
                if eqa_path.is_file():
                    eqa_rows = json.loads(eqa_path.read_text(encoding="utf-8")).get("questions", [])
                eqa_score = score_eqa_results(eqa_rows, episode_dir=ckpt) if eqa_rows else None

                nodes = _load_checkpoint_object_nodes(ckpt)
                graph_raw = {}
                n_obs = None
                if (ckpt / "graph.json").is_file():
                    graph_raw = json.loads((ckpt / "graph.json").read_text(encoding="utf-8"))
                n_total_nodes = len(graph_raw.get("nodes", []))
                if graph_raw.get("observations") is not None:
                    n_obs = len(graph_raw.get("observations") or [])
                from emet.memory.graph_eqa.graph_stats import (
                    classify_graph_failure,
                    graph_health_from_checkpoint_nodes,
                )

                health = graph_health_from_checkpoint_nodes(
                    list(graph_raw.get("nodes") or []),
                    n_obs=n_obs,
                )
                health["failure_class"] = classify_graph_failure(health)

                cycle_row: dict[str, Any] = {
                    "cycle": t,
                    "export_dir": str(ckpt),
                    "explore_iters": int(explore_iters),
                    "resumed_from": str(prev_ckpt) if prev_ckpt else None,
                    "eqa": eqa_score,
                    "eqa_accuracy": (eqa_score or {}).get("accuracy"),
                    "object_node_count": len(nodes),
                    "total_node_count": n_total_nodes,
                    "graph_health": health,
                    "moved_body_churn": _churn_metrics_for_moves(ckpt, pending_moves),
                    "cycle_wall_s": time.monotonic() - cycle_t0,
                }

                pending_moves = []
                change_spec = le.changes_after_cycle(t)
                if t < int(le.cycles) - 1 and change_spec:
                    applied, move_records = _apply_lifelong_changes(
                        change_spec,
                        port_offset=port_offset,
                    )
                    cycle_row["fuzz_applied"] = applied
                    # Patch this cycle's checkpoint so the next reload does not keep
                    # confident nodes at pre-move poses (CONFIRMED_MEMORY / find).
                    n_inv = invalidate_checkpoint_nodes_near_moves(ckpt, move_records)
                    cycle_row["checkpoint_nodes_invalidated"] = int(n_inv)
                    pending_moves = move_records

                cycle_results.append(cycle_row)
                prev_ckpt = ckpt

        accs = [c.get("eqa_accuracy") for c in cycle_results]
        churn = [m for c in cycle_results for m in (c.get("moved_body_churn") or [])]
        payload = {
            "run_id": run_id,
            "phase": "lifelong",
            "episode_id": base_episode.id,
            "backend": run_cfg.backend,
            "cycles": int(le.cycles),
            "cycle_results": cycle_results,
            "summary": {
                "eqa_accuracy_by_cycle": accs,
                "object_node_count_by_cycle": [c.get("object_node_count") for c in cycle_results],
                "graph_health_by_cycle": [c.get("graph_health") for c in cycle_results],
                "n_moves_total": len(churn),
                "n_moves_adapted": sum(1 for m in churn if m.get("adapted")),
                "n_moves_stale": sum(1 for m in churn if m.get("stale")),
            },
            "episode_wall_s": time.monotonic() - t0,
        }
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
    except Exception as exc:
        payload = {
            "run_id": run_id,
            "phase": "lifelong",
            "episode_id": base_episode.id,
            "backend": run_cfg.backend,
            "cycles": int(le.cycles),
            "cycle_results": cycle_results,
            "error": str(exc),
            "episode_wall_s": time.monotonic() - t0,
        }
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
