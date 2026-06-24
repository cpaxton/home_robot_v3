#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared MolmoSpaces navgrid session: rotate, explore, world raster (used by tier4 smoke scripts)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_MOLMO_ROBOTS = ("stretch", "rby1", "innate_mars")


@dataclass
class NavgridSessionResult:
    robot: str
    port_offset: int
    merged_xml: str
    explored_cells: int
    obstacle_cells: int
    explore_successes: int
    explore_iters: int
    world_raster: object  # WorldMapRaster


def wait_port(port: int, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"port {port} not open")


def wait_observation(port: int, timeout: float = 150.0) -> None:
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://127.0.0.1:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            sock.recv()
            return
        except zmq.Again:
            time.sleep(1)
    raise RuntimeError(f"no observation on port {port}")


def _resolve_base_body_id(model: mujoco.MjModel, base_body_name: str | None = None) -> int:
    if base_body_name:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        if bid >= 0:
            return int(bid)
    for name in ("base_link", "chassis", "base"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return int(bid)
    return -1


def collision_clip_from_merged_xml(
    merged_xml: str,
    *,
    inset: float = 0.2,
    base_body_name: str | None = None,
) -> tuple[float, float, float, float]:
    from emet.mapping.navgrid_compare import inset_clip_rect
    from emet.simulation import molmospaces_spawn as _molmo_spawn_mod

    model = mujoco.MjModel.from_xml_path(str(merged_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor_geom = _molmo_spawn_mod.resolve_floor_geom_name(model)
    bid = _resolve_base_body_id(model, base_body_name)
    if bid < 0:
        raise RuntimeError(f"no robot base body in merged MJCF {merged_xml!r}")
    robot_bodies = _molmo_spawn_mod._bodies_descending_from(model, bid)  # noqa: SLF001
    clip = _molmo_spawn_mod.collision_scene_xy_clip_rect(
        model,
        data,
        robot_bodies=robot_bodies,
        floor_geom_name=floor_geom,
        margin=0.42,
    )
    del model, data
    if clip is None:
        raise RuntimeError(f"collision_scene_xy_clip_rect returned None for {merged_xml}")
    return inset_clip_rect(clip, inset)


def run_navgrid_session(
    robot: str,
    *,
    port_offset: int,
    clip_rect: tuple[float, float, float, float],
    explore_iters: int = 3,
    scene: str = "ithor",
    split: str = "train",
    index: int = 0,
) -> NavgridSessionResult:
    """Start Molmo+robot server, scan + explore, return world-aligned occupancy raster."""
    from emet.app.robot_cli import create_robot_client_from_cli
    from emet.config.sim_launch_config import SimLaunchMolmospaces
    from emet.controller.controller_dynagraph import DynagraphController
    from emet.core.parameters import get_parameters
    from emet.mapping.navgrid_compare import world_raster_from_voxel_map
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv

    recv_port = 4401 + int(port_offset)
    argv = prepare_mujoco_server_argv(
        SimLaunchMolmospaces(
            robot=robot,
            scene=scene,
            split=split,
            index=index,
            headless=True,
            molmospaces_install=False,
            port_offset=port_offset,
        )
    )
    merged_xml: str | None = None
    for i, a in enumerate(argv):
        if a in ("--scene_path", "--scene-path") and i + 1 < len(argv):
            merged_xml = argv[i + 1]
            break
    if not merged_xml:
        raise RuntimeError(f"no --scene_path in argv for robot {robot!r}")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC)] + env.get("PYTHONPATH", "").split(os.pathsep))
    env.setdefault("MUJOCO_GL", "egl")
    proc = subprocess.Popen(
        [sys.executable, "-m", "emet.simulation.mujoco_server", *argv],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    client = None
    try:
        wait_port(recv_port)
        wait_observation(recv_port)

        params = get_parameters("dynav_config.yaml")
        params["use_instance_memory"] = False
        params["depth_source"] = "sensor"
        params.setdefault("graph_eqa_memory", {})["enabled"] = True

        client = create_robot_client_from_cli(
            robot,
            "127.0.0.1",
            port_offset=port_offset,
            enable_rerun_server=False,
            start_immediately=True,
            allow_missing_depth=False,
        )
        if hasattr(client, "set_velocity"):
            client.set_velocity(v=30.0, w=15.0)

        agent = DynagraphController(
            client,
            params,
            manipulation_only=True,
            cpu_only=True,
            use_instance_graph=False,
            use_sensor_perception=False,
        )
        agent.start()
        if hasattr(client, "move_to_nav_posture"):
            client.move_to_nav_posture()

        agent.rotate_in_place()
        explored_after_scan = int(agent.voxel_map.get_2d_map()[1].float().sum().item())
        explore_successes = 0
        explored_peak = explored_after_scan
        for _ in range(explore_iters):
            if agent.run_exploration():
                explore_successes += 1
            explored_now = int(agent.voxel_map.get_2d_map()[1].float().sum().item())
            explored_peak = max(explored_peak, explored_now)

        min_explored = int(os.environ.get("EMET_NAVGRID_MIN_EXPLORED_CELLS", "120"))
        if explored_peak < min_explored:
            raise RuntimeError(
                f"{robot}: too few explored cells after scan/explore (peak={explored_peak}, min={min_explored})"
            )
        if explored_peak <= explored_after_scan and explore_iters > 0 and explore_successes == 0:
            print(
                f"WARN: {robot}: explore steps did not grow map "
                f"(scan={explored_after_scan}); using scan map for compare",
                file=sys.stderr,
            )

        raster = world_raster_from_voxel_map(agent.voxel_map, clip_rect, resolution_m=0.1)
        obstacles, explored = agent.voxel_map.get_2d_map()
        return NavgridSessionResult(
            robot=robot,
            port_offset=port_offset,
            merged_xml=merged_xml,
            explored_cells=explored_peak,
            obstacle_cells=int(obstacles.float().sum().item()),
            explore_successes=explore_successes,
            explore_iters=explore_iters,
            world_raster=raster,
        )
    finally:
        if client is not None:
            try:
                client.stop()
            except Exception:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=4)
