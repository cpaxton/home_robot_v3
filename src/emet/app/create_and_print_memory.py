#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
#
# Run one rotate-in-place in the default MuJoCo scene (DynaMem), save memory to the
# common directory format, then load and print a summary.
#
# Usage:
#   # With sim auto-started (headless):
#   uv run python -m emet.app.create_and_print_memory
#
#   # With sim already running (e.g. emet serve mujoco in another terminal):
#   uv run python -m emet.app.create_and_print_memory --no-server
#
#   # Custom save path:
#   uv run python -m emet.app.create_and_print_memory --path ./my_memory

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import click

# Ensure subprocess can import emet
_SRC_ROOT = Path(__file__).resolve().parent.parent.parent


def _wait_for_port(host: str, port: int, timeout_sec: float = 30) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2) as _:
                return True
        except (TimeoutError, OSError):
            time.sleep(0.5)
    return False


def _scene_graph_to_graph_blob(scene_graph, semantic_sensor=None):
    """Convert mapping SceneGraph to memory format GraphBlob for save/print."""
    from emet.memory.format import GraphBlob, GraphEdgeView, GraphNodeView

    FLOOR_NODE_ID = 0
    nodes = [
        GraphNodeView(
            node_id=FLOOR_NODE_ID,
            labels=["floor"],
            xyz=[0.0, 0.0, 0.0],
            obs_id=0,
            description=None,
        )
    ]
    for idx, inst in enumerate(scene_graph.instances):
        gid = inst.global_id if inst.global_id is not None else idx
        label = f"id_{gid}"
        if semantic_sensor is not None and hasattr(semantic_sensor, "get_class_name_for_id"):
            cid = inst.get_category_id()
            if cid is not None:
                name = semantic_sensor.get_class_name_for_id(cid)
                if name:
                    label = name
        pos = scene_graph.get_ins_center_pos(idx)
        if hasattr(pos, "cpu"):
            pos = pos.detach().cpu().numpy()
        xyz = [float(pos.flat[0]), float(pos.flat[1]), float(pos.flat[2])]
        nodes.append(
            GraphNodeView(
                node_id=gid + 1,
                labels=[label],
                xyz=xyz,
                obs_id=gid,
                description=None,
            )
        )
    id_offset = 1
    edges = []
    for a, b, rel in scene_graph.relationships:
        id1 = a + id_offset if isinstance(a, int) else a
        id2 = FLOOR_NODE_ID if b == "floor" else (b + id_offset if isinstance(b, int) else b)
        edges.append(GraphEdgeView(id1=id1, id2=id2, relation=rel))
    return GraphBlob(nodes=nodes, edges=edges)


@click.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(),
    default="saved_memory",
    help="Directory path to save (and then load) memory.",
)
@click.option(
    "--no-server",
    is_flag=True,
    help="Do not start MuJoCo server; assume it is already running on 127.0.0.1.",
)
@click.option(
    "--robot-ip",
    "--robot_ip",
    "robot_ip",
    default="127.0.0.1",
    help="Robot/sim server IP (default 127.0.0.1).",
)
def main(path: str, no_server: bool, robot_ip: str):
    """Run rotate-in-place to build memory, save to directory, then load and print it."""
    proc = None
    robot = None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep))
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"

    if not no_server:
        print("Starting MuJoCo server (headless)...")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "emet.simulation.mujoco_server",
                "--headless",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _wait_for_port(robot_ip, 4401, timeout_sec=45):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
            raise SystemExit("MuJoCo server did not bind to 4401 within 45s.\n" + stderr)
        print("MuJoCo server ready.")

    try:
        from emet.controller.task.dynamem import DynamemTaskExecutor
        from emet.controller.zmq_client import StretchZmqClient
        from emet.core.parameters import get_parameters
        from emet.memory.backend import get_memory_backend
        from emet.memory.format import load_memory

        print("Connecting to robot/sim and running rotate_in_place...")
        robot = StretchZmqClient(
            robot_ip=robot_ip,
            enable_rerun_server=False,
            start_immediately=True,
        )
        parameters = get_parameters("dynav_config.yaml")
        executor = DynamemTaskExecutor(
            robot,
            parameters,
            skip_confirmations=True,
            cpu_only=True,
        )
        executor([("rotate_in_place", "")])

        agent = executor.agent
        voxel_map = agent.get_voxel_map()
        backend = get_memory_backend("dynamem", voxel_map=voxel_map)
        extra_graph = None
        if getattr(agent, "use_scene_graph", False) and getattr(agent, "scene_graph", None) is not None:
            extra_graph = _scene_graph_to_graph_blob(
                agent.scene_graph,
                getattr(agent, "semantic_sensor", None),
            )
        backend.save(path, extra_graph=extra_graph)
        from emet.memory.utils import print_memory_saved_help

        print_memory_saved_help(path)

        print("Loading memory and printing summary...")
        from emet.memory.utils import print_memory_state

        state = load_memory(path)
        print_memory_state(state)
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)


if __name__ == "__main__":
    main()
