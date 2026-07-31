#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Quick backend smoke: one Robocasa scene → top-down map + GT vs predicted object boxes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from emet.eval.sim_eval_session import (
    connect_benchmark_robot,
    launch_benchmark_sim_server,
    terminate_benchmark_sim_server,
)

REPO = Path(__file__).resolve().parents[1]

# Backend display colors (RGB 0–1 for matplotlib).
BACKEND_COLORS: dict[str, tuple[float, float, float]] = {
    "gt": (1.0, 1.0, 1.0),
    "dynamem": (0.35, 0.65, 1.0),
    "static_graph": (1.0, 0.55, 0.15),
    "graph_eqa": (1.0, 0.55, 0.15),  # legacy alias display
    "dynagraph": (0.25, 0.85, 0.45),
    "vlm_only": (0.95, 0.35, 0.85),
}


@dataclass
class SmokeResult:
    backend: str
    pred_xyz: list[float] | None
    err_xy_m: float | None
    query_used: str
    localize_source: str | None
    mapping_wall_s: float
    query_wall_s: float
    robot_xy: list[float] | None = None
    gt_xyz: list[float] | None = None
    gt_cat: str | None = None
    hit_0_5m: bool | None = None


@dataclass
class SimSession:
    robot: Any
    sim_kind: str
    port_offset: int
    benchmark_sim: Any
    server_log_fh: Any | None = None

    @property
    def server(self) -> subprocess.Popen[Any]:
        return self.benchmark_sim.server


def _free_mujoco_ports(port_offset: int) -> None:
    """Best-effort cleanup so sequential smoke backends can bind ZMQ ports."""
    for port in (4401 + port_offset, 4402 + port_offset, 4403 + port_offset, 4404 + port_offset):
        subprocess.run(
            ["uv", "run", "emet", "kill-mujoco-server", "--port", str(port)],
            cwd=str(REPO),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    time.sleep(1.0)


def _voxel_map_arrays(vm: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    obstacles, explored = vm.get_2d_map()
    if hasattr(obstacles, "detach"):
        obstacles = obstacles.detach().cpu().numpy()
    if hasattr(explored, "detach"):
        explored = explored.detach().cpu().numpy()
    go = vm.grid_origin
    if hasattr(go, "detach"):
        go = go.detach().cpu().numpy()
    go_xy = np.asarray(go, dtype=np.float64).reshape(-1)[:2]
    res = float(getattr(vm, "grid_resolution", 0.05))
    return np.asarray(obstacles), np.asarray(explored), go_xy, res


def _create_smoke_agent(
    backend: str,
    robot: Any,
    parameters: Any,
    *,
    cpu_only: bool,
):
    from emet.eval.ovmm_find_phase import create_find_phase_agent

    if backend == "vlm_only":
        from emet.controller.controller_graph_eqa import GraphEQAController

        return GraphEQAController(
            robot,
            parameters,
            save_rerun=False,
            use_instance_graph=False,
            cpu_only=cpu_only,
            use_sensor_perception=True,
        )
    return create_find_phase_agent(
        robot,
        parameters,
        backend,  # type: ignore[arg-type]
        cpu_only=cpu_only,
        use_sensor_perception=backend != "dynamem",
    )


def run_light_mapping_protocol(
    agent: Any,
    robot: Any,
    *,
    explore_steps: int,
    nav_timeout: float,
) -> int:
    """Snap-yaw mapping for smoke figures (avoids 8× blocking relative rotate timeouts)."""
    steps = 0
    robot.move_to_nav_posture()
    robot.look_front(blocking=True, timeout=nav_timeout)
    wait_obs = getattr(robot, "wait_for_obs", None)
    update = getattr(agent, "update", None)

    base = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)[:3]
    yaws = [float(base[2]) + k * (np.pi / 2.0) for k in range(4)]
    for yaw in yaws:
        robot.move_base_to(
            [float(base[0]), float(base[1]), yaw],
            relative=False,
            blocking=False,
            timeout=nav_timeout,
        )
        if callable(wait_obs):
            wait_obs(timeout=nav_timeout)
        else:
            time.sleep(1.5)
        if callable(update):
            update()
        steps += 1

    for _ in range(max(0, int(explore_steps))):
        agent.execute_action("")
        steps += 1

    refresh = getattr(agent, "refresh_ground_truth", None)
    if callable(refresh):
        refresh()
    return steps


def _robot_world_xy(robot: Any, session: dict[str, Any] | None) -> list[float] | None:
    from emet.utils.geometry import nav_xyt_to_world_xyt

    obs = getattr(robot, "last_observation", None) or getattr(robot, "_last_observation", None)
    gps = getattr(obs, "gps", None) if obs is not None else None
    compass = getattr(obs, "compass", None) if obs is not None else None
    if gps is not None and compass is not None:
        g = np.asarray(gps, dtype=np.float64).reshape(-1)
        c = np.asarray(compass, dtype=np.float64).ravel()
        if g.size >= 2 and c.size >= 1:
            local = np.array([float(g[0]), float(g[1]), float(c[0])], dtype=np.float64)
            wxyt = nav_xyt_to_world_xyt(local, session)
            return [float(wxyt[0]), float(wxyt[1])]
    try:
        bp = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
        if bp.size >= 2:
            return [float(bp[0]), float(bp[1])]
    except Exception:
        pass
    return None


def _resolve_object_query(
    object_query: str | None,
    *,
    object_fallback: str,
    gt_body: str,
    placements: dict[str, Any] | None,
) -> str:
    from emet.eval.ovmm_find_phase import FindPhaseEpisode, resolve_object_query

    if object_query and str(object_query).strip():
        return str(object_query).strip()
    episode = FindPhaseEpisode(
        id="smoke",
        tier="S1",
        sim="",
        object=object_fallback,
        object_gt_body=gt_body,
        start_recep="counter",
        goal_recep="cab",
        success_radius_m=0.50,
        explore_steps=0,
    )
    return resolve_object_query(episode, placements)


def start_sim_session(
    *,
    sim_yaml: str,
    port_offset: int,
    cpu_only: bool,
    repo: Path,
    seed: int | None = None,
    server_log_dir: Path | None = None,
) -> SimSession:
    """Boot one MuJoCo server + ZMQ client (shared across backend rows)."""
    from dataclasses import replace as dc_replace

    from emet.config.sim_launch_config import load_sim_launch_config_from_path

    sim_cfg = load_sim_launch_config_from_path(sim_yaml)
    if seed is not None:
        sim_cfg = dc_replace(sim_cfg, seed=int(seed))
    sim_cfg = dc_replace(sim_cfg, port_offset=port_offset, headless=True)

    _free_mujoco_ports(port_offset)
    server_log_fh = None
    server_stderr: Any = subprocess.DEVNULL
    if server_log_dir is not None:
        server_log_dir.mkdir(parents=True, exist_ok=True)
        server_log_fh = open(server_log_dir / f"sim_offset{port_offset}.log", "w", encoding="utf-8")
        server_stderr = server_log_fh

    benchmark_sim = launch_benchmark_sim_server(
        sim_cfg,
        repo=repo,
        cpu_only=cpu_only,
        cwd=repo,
        server_stderr=server_stderr,
    )
    robot = connect_benchmark_robot(sim_cfg, port_offset)
    return SimSession(
        robot=robot,
        sim_kind=benchmark_sim.sim_kind,
        port_offset=port_offset,
        benchmark_sim=benchmark_sim,
        server_log_fh=server_log_fh,
    )


def stop_sim_session(session: SimSession) -> None:
    if session.server_log_fh is not None:
        session.server_log_fh.close()
    stop = getattr(session.robot, "stop", None)
    if callable(stop):
        stop()
    terminate_benchmark_sim_server(session.benchmark_sim)


def run_backend_on_session(
    backend: str,
    session: SimSession,
    *,
    object_query: str,
    gt_body: str,
    explore_steps: int,
    cpu_only: bool,
    nav_timeout_s: float | None,
    light_map: bool,
    spawn_xyt: list[float] | None,
) -> tuple[SmokeResult, Any | None]:
    """Map + localize on an already-running sim (same ``obj_main`` GT for all backends)."""
    from emet.core.parameters import get_parameters
    from emet.eval.ovmm_find_phase import (
        apply_backend_parameters,
        get_memory_backend_for_agent,
        query_find_phase_localization,
        resolve_find_phase_nav_step_timeout,
        run_mapping_protocol,
    )
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    robot = session.robot
    sim_kind = session.sim_kind
    agent = None
    try:
        if spawn_xyt is not None and len(spawn_xyt) >= 3:
            robot.move_base_to(
                [float(spawn_xyt[0]), float(spawn_xyt[1]), float(spawn_xyt[2])],
                relative=False,
                blocking=False,
                timeout=30.0,
            )
            wait_obs = getattr(robot, "wait_for_obs", None)
            if callable(wait_obs):
                wait_obs(timeout=30.0)
        robot.move_to_nav_posture()

        profile_backend = "static_graph" if backend == "vlm_only" else backend
        if profile_backend == "graph_eqa":
            profile_backend = "static_graph"
        parameters = apply_backend_parameters(get_parameters("dynav_config.yaml"), profile_backend)
        parameters["encoder"] = None
        parameters["debug_perfect_sensor_depth"] = True
        parameters["find_phase_nav_step_timeout_s"] = resolve_find_phase_nav_step_timeout(
            cpu_only=cpu_only,
            sim_kind=sim_kind,
            override=nav_timeout_s,
        )

        agent = _create_smoke_agent(backend, robot, parameters, cpu_only=cpu_only)
        agent.start()

        nav_timeout = float(
            nav_timeout_s
            if nav_timeout_s is not None
            else resolve_find_phase_nav_step_timeout(cpu_only=cpu_only, sim_kind=sim_kind)
        )
        t_map0 = time.monotonic()
        if light_map:
            run_light_mapping_protocol(
                agent,
                robot,
                explore_steps=0 if backend == "vlm_only" else explore_steps,
                nav_timeout=nav_timeout,
            )
        else:
            run_mapping_protocol(
                agent,
                explore_steps=0 if backend == "vlm_only" else explore_steps,
                not_rotate=False,
            )
        mapping_wall_s = time.monotonic() - t_map0

        zmq_session = robot.get_emet_session()
        placements = read_sim_object_placements(zmq_session)
        memory = get_memory_backend_for_agent(
            agent, profile_backend if backend != "vlm_only" else "static_graph"
        )
        vm = getattr(agent, "voxel_map", None)

        prefer_voxel = backend == "dynamem"
        t_q0 = time.monotonic()
        try:
            pred_xyz, ok, q_used, source = query_find_phase_localization(
                memory,
                object_query,
                placements=placements,
                session=zmq_session,
                near_recep="counter",
                voxel_map=vm,
                convert_nav_to_world=False,
                prefer_voxel=prefer_voxel,
            )
        except Exception as exc:
            print(f"WARN: localization failed for {backend}: {exc}", file=sys.stderr, flush=True)
            pred_xyz, q_used, source = None, object_query, None
        query_wall_s = time.monotonic() - t_q0

        gt_xyz = None
        gt_cat = None
        err_xy = None
        if placements and gt_body in placements:
            gt_info = placements[gt_body]
            gt_xyz = [float(x) for x in gt_info["pos"][:3]]
            gt_cat = str(gt_info.get("cat") or gt_body)
            if pred_xyz is not None:
                p = np.asarray(pred_xyz, dtype=np.float64).reshape(3)
                g = np.asarray(gt_xyz, dtype=np.float64).reshape(3)
                err_xy = float(np.linalg.norm(p[:2] - g[:2]))

        pred_list = [float(x) for x in pred_xyz.reshape(3)] if pred_xyz is not None else None
        hit = bool(err_xy is not None and err_xy <= 0.50)
        result = SmokeResult(
            backend=backend,
            pred_xyz=pred_list,
            err_xy_m=err_xy,
            query_used=q_used,
            localize_source=source,
            mapping_wall_s=float(mapping_wall_s),
            query_wall_s=float(query_wall_s),
            robot_xy=_robot_world_xy(robot, zmq_session),
            gt_xyz=gt_xyz,
            gt_cat=gt_cat,
            hit_0_5m=hit if pred_list is not None else None,
        )
        return result, vm
    finally:
        if agent is not None:
            stop = getattr(agent, "stop", None)
            if callable(stop):
                stop()


def render_localization_figure(
    *,
    rgb: np.ndarray,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    crop_ij: tuple[int, int, int, int],
    map_shape_hw: tuple[int, int],
    gt_xyz: list[float],
    predictions: list[SmokeResult],
    out_path: Path,
    title: str,
    robot_xy: list[float] | None = None,
    box_half_m: float = 0.22,
) -> None:
    """Draw cropped top-down map with GT + per-backend prediction boxes."""
    from emet.visualization.map_snapshot import world_xy_to_grid_ij

    i0, i1, j0, j1 = crop_ij
    crop = np.ascontiguousarray(rgb[i0:i1, j0:j1])
    h, w = crop.shape[0], crop.shape[1]
    full_h, full_w = int(map_shape_hw[0]), int(map_shape_hw[1])

    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    ax.imshow(crop, origin="upper")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, color="white", pad=8)
    fig.patch.set_facecolor("#1a1a22")

    def _draw_box(xy: list[float], color: tuple[float, float, float], label: str, *, dashed: bool) -> None:
        ri, rj = world_xy_to_grid_ij(xy, grid_origin_xy, grid_resolution, (full_h, full_w))
        ci, cj = ri - i0, rj - j0
        if not (0 <= ci < h and 0 <= cj < w):
            ax.plot(
                np.clip(cj, 0, w - 1),
                np.clip(ci, 0, h - 1),
                marker="x",
                color=color,
                markersize=8,
                markeredgewidth=2,
            )
            return
        half_cells = max(1.0, box_half_m / grid_resolution)
        rect = mpatches.Rectangle(
            (cj - half_cells, ci - half_cells),
            2 * half_cells,
            2 * half_cells,
            linewidth=2.5 if not dashed else 2.0,
            edgecolor=color,
            facecolor=(*color, 0.15),
            linestyle="-" if not dashed else "--",
            label=label,
        )
        ax.add_patch(rect)
        ax.plot(cj, ci, marker="x", color=color, markersize=8, markeredgewidth=2)

    if robot_xy is not None:
        ri, rj = world_xy_to_grid_ij(robot_xy, grid_origin_xy, grid_resolution, (full_h, full_w))
        ci, cj = ri - i0, rj - j0
        if 0 <= ci < h and 0 <= cj < w:
            ax.plot(cj, ci, marker="*", color="#ffee55", markersize=14, markeredgewidth=1, label="robot")

    gt_label = f"GT ({gt_xyz[0]:.2f}, {gt_xyz[1]:.2f})"
    _draw_box(gt_xyz, BACKEND_COLORS["gt"], gt_label, dashed=False)

    legend_handles = [
        mpatches.Patch(edgecolor=BACKEND_COLORS["gt"], facecolor="none", label=gt_label),
    ]
    for row in predictions:
        if row.pred_xyz is None:
            legend_handles.append(mpatches.Patch(edgecolor="gray", facecolor="none", label=f"{row.backend}: miss"))
            continue
        err = f"{row.err_xy_m:.2f}m" if row.err_xy_m is not None else "?"
        color = BACKEND_COLORS.get(row.backend, (0.8, 0.8, 0.8))
        lbl = f"{row.backend} ({err})"
        _draw_box(row.pred_xyz, color, lbl, dashed=True)
        legend_handles.append(mpatches.Patch(edgecolor=color, facecolor="none", label=lbl))

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
        facecolor="#2a2a32",
        edgecolor="#555",
        labelcolor="white",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim",
        default="configs/sim/robocasa_pick_place_stretch.yaml",
        help="Sim launch YAML",
    )
    parser.add_argument(
        "--object-query",
        default="",
        help="Memory query (default: resolve GT category from obj_main, OVMM-style)",
    )
    parser.add_argument(
        "--object",
        default="obj",
        help="Fallback object token when --object-query is empty (see find_phase_episodes.yaml)",
    )
    parser.add_argument("--gt-body", default="obj_main")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Robocasa placement seed (default: sim YAML seed, usually 0)",
    )
    parser.add_argument("--explore-steps", type=int, default=3)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["dynamem", "static_graph", "dynagraph", "vlm_only"],
        help="Backend rows (vlm_only = rotate-only voxel-VLM baseline; graph_eqa aliases static_graph)",
    )
    parser.add_argument("--port-offset-base", type=int, default=800)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--nav-timeout",
        type=float,
        default=None,
        help="Override find-phase nav step timeout (seconds)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke: static_graph + dynagraph, explore_steps=1",
    )
    parser.add_argument(
        "--full-protocol",
        action="store_true",
        help="Use OVMM rotate+explore mapping (slow; default is light snap-yaw mapping)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "runs" / "emet" / "backend_localization_smoke",
    )
    args = parser.parse_args()

    if args.quick:
        args.backends = ["static_graph", "dynagraph"]
        args.explore_steps = 1
    # Normalize legacy alias so create_find_phase_agent gets canonical ids.
    args.backends = ["static_graph" if b == "graph_eqa" else b for b in args.backends]

    out_dir = args.output_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

    sim_cfg_for_seed = load_sim_launch_config_from_path(args.sim)
    scene_seed = int(args.seed) if args.seed is not None else int(getattr(sim_cfg_for_seed, "seed", 0) or 0)

    port = int(args.port_offset_base)
    sim_session = start_sim_session(
        sim_yaml=args.sim,
        port_offset=port,
        cpu_only=bool(args.cpu_only),
        repo=REPO,
        seed=scene_seed,
        server_log_dir=out_dir / "sim_logs",
    )

    results: list[SmokeResult] = []
    map_vm = None
    gt_xyz: list[float] | None = None
    gt_cat: str | None = None
    robot_xy: list[float] | None = None
    spawn_xyt: list[float] | None = None
    object_query = args.object_query

    try:
        init_sess = sim_session.robot.get_emet_session()
        init_placements = read_sim_object_placements(init_sess)
        object_query = _resolve_object_query(
            args.object_query or None,
            object_fallback=str(args.object),
            gt_body=args.gt_body,
            placements=init_placements,
        )
        if init_placements and args.gt_body in init_placements:
            gt_info = init_placements[args.gt_body]
            gt_xyz = [float(x) for x in gt_info["pos"][:3]]
            gt_cat = str(gt_info.get("cat") or args.gt_body)
        try:
            spawn = np.asarray(sim_session.robot.get_base_pose(), dtype=np.float64).reshape(-1)[:3]
            spawn_xyt = [float(spawn[0]), float(spawn[1]), float(spawn[2])]
        except Exception:
            spawn_xyt = None

        scene_path = out_dir / "scene_gt.json"
        scene_path.write_text(
            json.dumps(
                {
                    "object_query": object_query,
                    "gt_body": args.gt_body,
                    "gt_xyz": gt_xyz,
                    "gt_cat": gt_cat,
                    "seed": scene_seed,
                    "shared_sim": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Scene GT: {args.gt_body} cat={gt_cat!r} xyz={gt_xyz} query={object_query!r}", flush=True)

        for backend in args.backends:
            print(f"=== backend={backend} (shared sim port_offset={port}) ===", flush=True)
            row, vm = run_backend_on_session(
                backend,
                sim_session,
                object_query=object_query,
                gt_body=args.gt_body,
                explore_steps=int(args.explore_steps),
                cpu_only=bool(args.cpu_only),
                nav_timeout_s=args.nav_timeout,
                light_map=not bool(args.full_protocol),
                spawn_xyt=spawn_xyt,
            )
            results.append(row)
            if vm is not None and map_vm is None:
                map_vm = vm
            elif vm is not None and backend == "dynagraph":
                map_vm = vm
            if robot_xy is None and row.robot_xy is not None:
                robot_xy = row.robot_xy
            print(json.dumps(asdict(row), indent=2), flush=True)
    finally:
        stop_sim_session(sim_session)

    summary_path = out_dir / "smoke_results.json"
    summary = {
        "scene": json.loads((out_dir / "scene_gt.json").read_text(encoding="utf-8"))
        if (out_dir / "scene_gt.json").is_file()
        else {},
        "success_radius_m": 0.50,
        "backends": [asdict(r) for r in results],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== Localization metrics (shared sim, OVMM query) ===", flush=True)
    for row in results:
        err = "miss" if row.err_xy_m is None else f"{row.err_xy_m:.3f} m"
        hit = "—" if row.hit_0_5m is None else ("HIT" if row.hit_0_5m else "miss")
        print(f"  {row.backend:12s}  err_xy={err:>10s}  @0.5m={hit:4s}  src={row.localize_source}", flush=True)

    if map_vm is None or gt_xyz is None:
        print("ERROR: no voxel map or GT position — cannot render figure", file=sys.stderr)
        return 1

    gt_label = f"{gt_cat or args.gt_body} ({gt_xyz[0]:.2f}, {gt_xyz[1]:.2f})"
    from emet.visualization.map_snapshot import (
        explored_crop_indices,
        render_topdown_map_rgb,
    )

    obs, exp, go_xy, res = _voxel_map_arrays(map_vm)
    map_shape = (int(obs.shape[0]), int(obs.shape[1]))
    robot_xy_t = tuple(robot_xy) if robot_xy is not None else None
    rgb_full = render_topdown_map_rgb(obs, exp, go_xy, res, robot_xy_t, max_side=None)
    bbox = explored_crop_indices(exp, robot_xy_t, go_xy, res, map_shape)
    if bbox is None:
        bbox = (0, map_shape[0], 0, map_shape[1])

    from emet.visualization.map_snapshot import world_xy_to_grid_ij

    i0, i1, j0, j1 = bbox
    for xy in [gt_xyz] + [r.pred_xyz for r in results if r.pred_xyz]:
        ri, rj = world_xy_to_grid_ij(xy, go_xy, res, map_shape)
        margin = 20
        i0 = max(0, min(i0, ri - margin))
        i1 = min(map_shape[0], max(i1, ri + margin + 1))
        j0 = max(0, min(j0, rj - margin))
        j1 = min(map_shape[1], max(j1, rj + margin + 1))
    bbox = (i0, i1, j0, j1)

    fig_path = out_dir / "backend_localization_topdown.png"
    render_localization_figure(
        rgb=rgb_full,
        grid_origin_xy=go_xy,
        grid_resolution=res,
        crop_ij=bbox,
        map_shape_hw=map_shape,
        gt_xyz=gt_xyz,
        predictions=results,
        out_path=fig_path,
        title=f"Object localization ({gt_label}) — explore={args.explore_steps}",
        robot_xy=robot_xy,
    )
    print(f"Wrote figure → {fig_path}", flush=True)
    print(f"Wrote metrics → {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
