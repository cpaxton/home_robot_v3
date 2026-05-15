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

"""Capture one **sim voxel + MuJoCo object GT** episode (`dataset_product`: ``sim_voxel_gt_episode``).

Produces ``dynamem/``, ``ground_truth.json``, and ``dataset_manifest.json`` under ``--output-dir``.
See ``docs/datasets/sim_voxel_gt.md``.

Run (from repo root, with sim extra / deps installed)::

    uv run python -m emet.app.capture_sim_dataset_episode --output-dir ./episode0

Visible sim (this process spawns the server; needs DISPLAY / GLFW for windowed GL)::

    uv run python -m emet.app.capture_sim_dataset_episode --output-dir ./episode0 --show-sim

With sim already running::

    uv run python -m emet.app.capture_sim_dataset_episode --no-server --output-dir ./episode0

Or via CLI::

    emet dataset capture-sim-episode --output-dir ./episode0
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import click
import numpy as np

from emet.config.sim_launch_config import SimLaunchConfig, load_sim_launch_config_from_path
from emet.dataset.graph_blob import gt_object_dicts_to_graph_blob
from emet.dataset.schema import GT_SCHEMA_VERSION
from emet.dataset.sim_health import RobotSimPhysicsExplodedError, check_robot_sim_stable
from emet.dataset.zmq_gt import read_gt_object_dicts_from_robot_client
from emet.utils.config import resolve_config_yaml_path

SimSource = Literal["default", "robosuite", "molmospaces"]

# Human-readable id for manifests and logs (what this tool produces).
DATASET_PRODUCT_ID = "sim_voxel_gt_episode"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _resolve_sim_yaml(
    *,
    source: SimSource,
    sim_config: str | None,
    robot: str | None,
) -> Path:
    root = _repo_root()
    if sim_config and str(sim_config).strip():
        return Path(resolve_config_yaml_path(str(sim_config).strip()))
    rk = (robot or "rby1").lower().replace("-", "_")
    if source == "default":
        rel = "configs/sim/default_table_stretch.yaml" if rk == "stretch" else "configs/sim/default_table_rby1.yaml"
    elif source == "robosuite":
        rel = "configs/sim/robocasa_pick_place.yaml"
    elif source == "molmospaces":
        rel = "configs/sim/molmospaces_ithor_train_0.yaml"
    else:
        raise click.UsageError(f"unknown source {source!r}")
    candidate = root / rel
    if not candidate.is_file():
        raise click.ClickException(f"Sim config not found: {candidate}")
    return candidate


def _merge_sim_cli(
    base: SimLaunchConfig,
    *,
    port_offset: int,
    headless: bool,
    seed: int,
    robot: str | None,
    show_viewer_ui: bool,
    kinematic_sim: bool = False,
) -> SimLaunchConfig:
    upd: dict[str, Any] = {
        "port_offset": int(port_offset),
        "headless": bool(headless),
        "seed": int(seed),
        "show_viewer_ui": bool(show_viewer_ui),
    }
    if kinematic_sim:
        upd["physics_mode"] = "kinematic"
    if robot is not None and str(robot).strip():
        upd["robot"] = str(robot).strip()
    return replace(base, **upd)


def _scene_key(cfg: SimLaunchConfig) -> str:
    return str(getattr(cfg, "kind", type(cfg).__name__))


@click.command("capture-sim-episode")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    required=True,
    help=(
        "Episode directory (created). Writes a **sim voxel + GT episode** dataset: "
        "`dynamem/` (DynaMem map), `ground_truth.json` (MuJoCo body poses), `dataset_manifest.json`, "
        "optional `gt_trajectory.jsonl`."
    ),
)
@click.option(
    "--source",
    type=click.Choice(["default", "robosuite", "molmospaces"]),
    default="default",
    show_default=True,
    help=(
        "Preset sim recipe: `default` = packaged table + robot YAML; `robosuite` = Robocasa kitchen YAML; "
        "`molmospaces` = packaged MolmoSpaces YAML. Use `--sim-config` to point at your own launch file instead."
    ),
)
@click.option(
    "--sim-config",
    type=str,
    default=None,
    help="Override sim launch YAML path (cwd-relative, absolute, or under emet/config).",
)
@click.option("--robot", type=str, default=None, help="Robot name; must match the sim server (default: YAML value).")
@click.option("--robot-ip", "--robot_ip", default="127.0.0.1", show_default=True)
@click.option("--port-offset", default=0, type=int, show_default=True)
@click.option(
    "--headless/--no-headless",
    default=True,
    show_default=True,
    help="When spawning sim: run MuJoCo server without a window (default). Use `--no-headless` or `--show-sim` for a visible sim.",
)
@click.option(
    "--show-sim",
    is_flag=True,
    help="When spawning sim: same as `--no-headless` (visible MuJoCo; needs DISPLAY / GLFW for windowed GL).",
)
@click.option(
    "--show-viewer-ui",
    is_flag=True,
    help="When spawning sim: pass `--show-viewer-ui` to the server (rby1 / MolmoSpaces path: side panels when not headless).",
)
@click.option("--seed", default=0, type=int, show_default=True)
@click.option(
    "--no-server",
    is_flag=True,
    help="Do not spawn sim; connect to an existing ZMQ server at --robot-ip with matching ports.",
)
@click.option(
    "--rotate-only",
    is_flag=True,
    help="Only run rotate_in_place (no extra idle --steps, no frontier step).",
)
@click.option(
    "--steps",
    default=0,
    type=int,
    show_default=True,
    help="After exploration, call agent.update() this many extra times (mapping refresh).",
)
@click.option(
    "--frontier-step",
    is_flag=True,
    help='After rotate, run one execute_action("") (may invoke planner / LLM depending on config).',
)
@click.option(
    "--gt-trajectory",
    is_flag=True,
    help="Append per-step rows to gt_trajectory.jsonl under --output-dir during exploration.",
)
@click.option(
    "--verbose-sim",
    is_flag=True,
    help="When spawning sim, inherit subprocess stdout/stderr (default: silenced).",
)
@click.option(
    "--kinematic-sim",
    "kinematic_sim",
    is_flag=True,
    help="Override sim YAML: spawn MuJoCo in kinematic mode (mj_forward only; snap poses).",
)
@click.option(
    "--agent-config",
    type=str,
    default="dynav_config.yaml",
    show_default=True,
    help="Planner / DynaMem parameters YAML (resolved like other emet configs).",
)
def capture_sim_episode(
    output_dir: str,
    source: SimSource,
    sim_config: str | None,
    robot: str | None,
    robot_ip: str,
    port_offset: int,
    headless: bool,
    show_sim: bool,
    show_viewer_ui: bool,
    seed: int,
    no_server: bool,
    rotate_only: bool,
    steps: int,
    frontier_step: bool,
    gt_trajectory: bool,
    verbose_sim: bool,
    kinematic_sim: bool,
    agent_config: str,
) -> None:
    """Build one **sim_voxel_gt_episode** dataset: DynaMem voxel map + MuJoCo body GT + manifest."""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = _repo_root()

    yaml_path = _resolve_sim_yaml(source=source, sim_config=sim_config, robot=robot)
    base_cfg = load_sim_launch_config_from_path(str(yaml_path))
    sim_headless = False if show_sim else bool(headless)
    cfg = _merge_sim_cli(
        base_cfg,
        port_offset=port_offset,
        headless=sim_headless,
        seed=seed,
        robot=robot,
        show_viewer_ui=show_viewer_ui,
        kinematic_sim=kinematic_sim,
    )
    eff_robot = str(cfg.robot)

    _print_run_banner(
        out=out,
        source=source,
        yaml_path=yaml_path,
        cfg=cfg,
        eff_robot=eff_robot,
        robot_ip=robot_ip,
        port_offset=port_offset,
        no_server=no_server,
        sim_headless=sim_headless,
        show_viewer_ui=show_viewer_ui,
        agent_config=agent_config,
        rotate_only=rotate_only,
        steps=steps,
        frontier_step=frontier_step,
        gt_trajectory=gt_trajectory,
    )

    proc = None
    if not no_server:
        from emet.simulation.sim_subprocess import shutdown_mujoco_server_subprocess, spawn_mujoco_server_subprocess

        click.echo("Spawning MuJoCo ZMQ sim subprocess…")
        try:
            proc = spawn_mujoco_server_subprocess(cfg, silence_sim_output=not verbose_sim)
        except RuntimeError as e:
            raise click.ClickException(str(e)) from e
        click.echo("Sim subprocess is accepting connections.")

    robot_client = None
    traj_fh = None
    try:
        from emet.app.robot_cli import create_robot_client_from_cli
        from emet.controller.task.dynamem import DynamemTaskExecutor
        from emet.core.parameters import get_parameters
        from emet.memory.backend import get_memory_backend

        workspace = out / "_agent_workspace"
        parameters = get_parameters(agent_config)
        click.echo(f"Connecting ZMQ robot client ({eff_robot!r} @ {robot_ip}, port_offset={port_offset})…")
        robot_client = create_robot_client_from_cli(
            eff_robot,
            robot_ip,
            port_offset=port_offset,
            enable_rerun_server=False,
            start_immediately=True,
            allow_missing_depth=True,
        )
        executor = DynamemTaskExecutor(
            robot_client,
            parameters,
            skip_confirmations=True,
            cpu_only=True,
            output_path=str(workspace),
            server_ip=robot_ip if robot_ip else "127.0.0.1",
        )

        def _require_stable(stage: str) -> None:
            try:
                check_robot_sim_stable(robot_client, stage=stage)
            except RobotSimPhysicsExplodedError as e:
                click.secho(str(e), fg="red", err=True)
                click.secho(
                    "Aborting capture: unstable MuJoCo sim (robot state diverged). "
                    "Check spawn / contacts, EGL vs GLFW, or asset versions; then retry.",
                    fg="red",
                    err=True,
                )
                raise click.ClickException(str(e)) from e

        robot_client.get_observation()
        _require_stable("after_agent_start")

        click.echo("Connected to sim. Running exploration (DynaMem voxel build)…")

        if gt_trajectory:
            traj_fh = (out / "gt_trajectory.jsonl").open("w", encoding="utf-8")

        def _log_traj(tag: str, step_hint: int) -> None:
            if traj_fh is None:
                return
            objs = read_gt_object_dicts_from_robot_client(robot_client)
            xyt = robot_client.get_base_pose()
            if xyt is None:
                xyv = [0.0, 0.0, 0.0]
            else:
                a = np.asarray(xyt, dtype=float).reshape(-1)
                xyv = [float(a[0]), float(a[1]), float(a[2])] if a.size >= 3 else [0.0, 0.0, 0.0]
            row = {
                "tag": tag,
                "step": int(step_hint),
                "wall_time": time.time(),
                "robot_xyt": xyv,
                "objects": objs,
            }
            traj_fh.write(json.dumps(row) + "\n")
            traj_fh.flush()

        executor.agent.rotate_in_place()
        robot_client.get_observation()
        _require_stable("after_rotate_in_place")
        _log_traj("after_rotate", int(getattr(robot_client, "_seq_id", 0)))

        if frontier_step and not rotate_only:
            executor.agent.execute_action("")
            robot_client.get_observation()
            _require_stable("after_frontier_step")
            _log_traj("after_frontier", int(getattr(robot_client, "_seq_id", 0)))

        if not rotate_only and steps > 0:
            for i in range(int(steps)):
                executor.agent.update()
                robot_client.get_observation()
                _require_stable(f"idle_update_{i}")
                _log_traj(f"idle_update_{i}", int(getattr(robot_client, "_seq_id", 0)))

        robot_client.get_observation()
        _require_stable("before_save")
        gt_objects = read_gt_object_dicts_from_robot_client(robot_client)

        click.echo(
            f"Saving episode dataset ({DATASET_PRODUCT_ID}): "
            f"{len(gt_objects)} GT object(s) from sim → dynamem/ + ground_truth.json + manifest…"
        )

        voxel_map = executor.agent.get_voxel_map()
        backend = get_memory_backend("dynamem", voxel_map=voxel_map)
        dynamem_dir = out / "dynamem"
        extra_graph = gt_object_dicts_to_graph_blob(gt_objects) if gt_objects else None
        backend.save(str(dynamem_dir), extra_graph=extra_graph)

        gt_payload = {
            "schema_version": GT_SCHEMA_VERSION,
            "dataset_product": DATASET_PRODUCT_ID,
            "robot": eff_robot,
            "scene_key": _scene_key(cfg),
            "sim_config": str(yaml_path),
            "objects": gt_objects,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        (out / "ground_truth.json").write_text(json.dumps(gt_payload, indent=2), encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "dataset_product": DATASET_PRODUCT_ID,
            "description": (
                "One sim episode: DynaMem voxel/sparse map under dynamem/, "
                "MuJoCo body GT under ground_truth.json (not vision detections)."
            ),
            "git_sha": _git_sha(root),
            "robot": eff_robot,
            "scene_key": _scene_key(cfg),
            "sim_yaml": str(yaml_path),
            "source": source,
            "port_offset": int(port_offset),
            "sim_spawn_headless": bool(cfg.headless) if not no_server else None,
            "sim_spawn_show_viewer_ui": bool(cfg.show_viewer_ui) if not no_server else None,
            "sim_physics_mode": str(getattr(cfg, "physics_mode", "dynamic")),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                "dynamem": "dynamem/",
                "ground_truth": "ground_truth.json",
                "gt_trajectory": "gt_trajectory.jsonl" if gt_trajectory else None,
            },
        }
        (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _print_done_summary(out=out, gt_count=len(gt_objects), gt_trajectory=gt_trajectory)
    finally:
        if traj_fh is not None:
            traj_fh.close()
        if robot_client is not None:
            try:
                robot_client.stop()
            except Exception:
                pass
        if proc is not None:
            from emet.simulation.sim_subprocess import shutdown_mujoco_server_subprocess

            shutdown_mujoco_server_subprocess()


def _print_run_banner(
    *,
    out: Path,
    source: SimSource,
    yaml_path: Path,
    cfg: SimLaunchConfig,
    eff_robot: str,
    robot_ip: str,
    port_offset: int,
    no_server: bool,
    sim_headless: bool,
    show_viewer_ui: bool,
    agent_config: str,
    rotate_only: bool,
    steps: int,
    frontier_step: bool,
    gt_trajectory: bool,
) -> None:
    recv_port = 4401 + int(port_offset)
    if no_server:
        sim_proc_desc = "connect only (--no-server; headless/visible controlled by how you started the server)"
    else:
        sim_proc_desc = (
            "headless (no MuJoCo window)" if sim_headless else "visible MuJoCo window (--show-sim or --no-headless)"
        )
        if show_viewer_ui:
            sim_proc_desc += " + server --show-viewer-ui (side panels when supported)"

    lines = [
        "",
        "=" * 72,
        f"  Dataset: {DATASET_PRODUCT_ID}  (one sim episode → voxel map + MuJoCo object GT)",
        "=" * 72,
        f"  Output directory:  {out}",
        f"  Sim preset (--source):  {source!r}",
        f"  Sim launch YAML:  {yaml_path}",
        f"  Resolved scene kind:  {_scene_key(cfg)!r}",
        f"  Sim physics_mode:  {str(getattr(cfg, 'physics_mode', 'dynamic'))!r}",
        f"  Robot (ZMQ):  {eff_robot!r}  @  {robot_ip}  (obs port ≈ {recv_port})",
        f"  Agent / DynaMem params:  {agent_config}",
        f"  Sim process:  {sim_proc_desc}",
        "",
        "  Files to be written in the output directory:",
        "    • dynamem/          — DynaMem memory directory (voxel map, frames, graph.json if GT objects exist)",
        "    • ground_truth.json — Simulator object poses (body names / pos / quat from MuJoCo, not YOLO/OWL)",
        "    • dataset_manifest.json — metadata + paths + dataset_product id",
    ]
    if gt_trajectory:
        lines.append("    • gt_trajectory.jsonl — per-step GT + robot_xyt (because --gt-trajectory)")
    lines.extend(
        [
            "",
            "  Exploration:",
            "    • rotate_in_place:  yes",
            f"    • extra agent.update steps:  {0 if rotate_only else steps}",
            f'    • execute_action("") frontier step:  {"yes" if frontier_step and not rotate_only else "no"}',
            "=" * 72,
            "",
        ]
    )
    click.echo("\n".join(lines))


def _print_done_summary(*, out: Path, gt_count: int, gt_trajectory: bool) -> None:
    lines = [
        "",
        "=" * 72,
        f"  Finished: {DATASET_PRODUCT_ID}",
        "=" * 72,
        f"  Episode root:  {out}",
        f"  GT objects written:  {gt_count}",
        "  Artifacts:",
        f"    • {out / 'dynamem'}",
        f"    • {out / 'ground_truth.json'}",
        f"    • {out / 'dataset_manifest.json'}",
    ]
    if gt_trajectory:
        lines.append(f"    • {out / 'gt_trajectory.jsonl'}")
    lines.extend(["=" * 72, ""])
    click.echo("\n".join(lines))


if __name__ == "__main__":
    capture_sim_episode()
