#!/usr/bin/env python3
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
        except (OSError, socket.timeout):
            time.sleep(0.5)
    return False


def _print_memory_state(state):
    """Print a readable summary of a loaded MemoryState."""
    from emet.memory.format import MemoryState

    assert isinstance(state, MemoryState)
    print("\n" + "=" * 60)
    print("MEMORY STATE SUMMARY")
    print("=" * 60)
    if state.manifest:
        m = state.manifest
        print(f"  Backend:     {m.backend}")
        print(f"  Created:     {m.created_at or 'N/A'}")
        print(f"  Version:     {m.version}")
    print()
    if state.point_cloud is not None:
        pc = state.point_cloud
        n = pc.xyz.shape[0]
        print(f"  Point cloud: {n} points")
        if pc.xyz.size > 0:
            print(f"    XYZ range: x=[{pc.xyz[:, 0].min():.3f}, {pc.xyz[:, 0].max():.3f}] "
                  f"y=[{pc.xyz[:, 1].min():.3f}, {pc.xyz[:, 1].max():.3f}] "
                  f"z=[{pc.xyz[:, 2].min():.3f}, {pc.xyz[:, 2].max():.3f}]")
        if pc.rgb is not None:
            print(f"    RGB: present ({pc.rgb.shape})")
    else:
        print("  Point cloud: (none)")
    print(f"  Frames:      {len(state.frames)}")
    if state.graph is not None:
        g = state.graph
        print(f"  Graph:       {len(g.nodes)} nodes, {len(g.edges)} edges")
        for i, node in enumerate(g.nodes[:5]):
            labels = ", ".join(node.labels) if node.labels else "(no labels)"
            print(f"    Node {node.node_id}: xyz={node.xyz} labels=[{labels}]")
        if len(g.nodes) > 5:
            print(f"    ... and {len(g.nodes) - 5} more nodes")
    else:
        print("  Graph:       (none)")
    if state.obstacles_2d is not None:
        print(f"  2D obstacles: grid shape {state.obstacles_2d.shape}")
    if state.explored_2d is not None:
        print(f"  2D explored:  grid shape {state.explored_2d.shape}")
    if state.text_descriptions:
        print(f"  Text descriptions: {len(state.text_descriptions)} items")
    print("=" * 60 + "\n")


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
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep)
    )
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

        voxel_map = executor.agent.get_voxel_map()
        backend = get_memory_backend("dynamem", voxel_map=voxel_map)
        backend.save(path)
        from emet.memory.utils import print_memory_saved_help

        print_memory_saved_help(path)

        print("Loading memory and printing summary...")
        state = load_memory(path)
        _print_memory_state(state)
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
