# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""DynaMem floor-map coverage on MolmoSpaces iTHOR with XLeRobot (mirrors stretch test)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import pytest
import torch

from emet.config.sim_launch_config import SimLaunchMolmospaces
from emet.core.parameters import get_parameters
from emet.simulation import molmospaces_spawn as _molmo_spawn_mod
from emet.simulation.molmospaces_config import build_molmospaces_wrapper_command
from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv

_SRC_ROOT = Path(__file__).resolve().parents[2]


def _truthy(env: str) -> bool:
    return os.environ.get(env, "").strip().lower() in ("1", "true", "yes", "on")


_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _skip_reason() -> str | None:
    if not RUN_SIM_TESTS:
        return "RUN_SIM_TESTS=0 skips sim-heavy tests."
    if not _truthy("RUN_MOLMOSPACES_TESTS"):
        return "Set RUN_MOLMOSPACES_TESTS=1 when MolmoSpaces assets + wrapper are available."
    if not _truthy("RUN_XLEROBOT_MOLMO_DYNAMEM"):
        return "Heavy stack; set RUN_XLEROBOT_MOLMO_DYNAMEM=1 to run."
    return None


_SKIP = _skip_reason()


@pytest.mark.timeout(600)
@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "skipped")
def test_xlerobot_molmospaces_dynamem_floor_map_covers_collision_clip_and_populates_grid():
    if build_molmospaces_wrapper_command(["merge-scene", "--help"]) is None:
        pytest.skip("MolmoSpaces wrapper missing (emet-molmospaces / .venv-molmospaces).")

    recv_base = 4401
    port_extra = max(1, int(os.environ.get("EMET_MOLMO_DYNAMEM_PORT_OFFSET", "100")))
    recv_port = recv_base + port_extra

    sim = SimLaunchMolmospaces(
        robot="xlerobot",
        scene="ithor",
        split="train",
        index=0,
        headless=True,
        molmospaces_install=False,
        port_offset=port_extra,
    )
    argv = prepare_mujoco_server_argv(sim)
    merged_xml: str | None = None
    for i, a in enumerate(argv):
        if a in ("--scene_path", "--scene-path") and i + 1 < len(argv):
            merged_xml = argv[i + 1]
            break
    assert merged_xml and Path(merged_xml).is_file(), f"No merged MJCF in argv list: {argv!r}"

    m_ref = mujoco.MjModel.from_xml_path(str(merged_xml))
    d_ref = mujoco.MjData(m_ref)
    mujoco.mj_forward(m_ref, d_ref)
    floor_geom = _molmo_spawn_mod.resolve_floor_geom_name(m_ref)
    bid = mujoco.mj_name2id(m_ref, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    if bid < 0:
        bid = mujoco.mj_name2id(m_ref, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    robot_bodies = _molmo_spawn_mod._bodies_descending_from(m_ref, bid)  # noqa: SLF001
    clip = _molmo_spawn_mod.collision_scene_xy_clip_rect(
        m_ref,
        d_ref,
        robot_bodies=robot_bodies,
        floor_geom_name=floor_geom,
        margin=0.42,
    )
    del m_ref, d_ref
    if clip is None:
        pytest.skip("collision_scene_xy_clip_rect returned None for merged scene.")

    x0, x1, y0, y1 = _inset_clip_rect(clip, inset=0.2)
    corner_xy = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))

    proc = None
    robot_client = None
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT)] + env.get("PYTHONPATH", "").split(os.pathsep))
    if sys.platform == "linux":
        env["MUJOCO_GL"] = "egl"
    env.setdefault("EMET_ZMQ_STARTUP_TIMEOUT", "120")
    env["EMET_SIM_NAV_TELEPORT"] = "1"
    cmd = [sys.executable, "-m", "emet.simulation.mujoco_server", *argv]

    try:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if not _wait_for_port("127.0.0.1", recv_port, timeout_sec=120):
            err = proc.stderr.read().decode() if proc.stderr else ""
            pytest.fail(f"XLeRobot MuJoCo server did not open recv socket on {recv_port}. stderr:\n{err}")

        from emet.controller.generic_zmq_client import GenericZmqClient
        from emet.controller.task.dynamem import DynamemTaskExecutor
        from emet.robots.xlerobot import XLeRobotBackend

        spec = XLeRobotBackend().get_spec()
        robot_client = GenericZmqClient(
            robot_spec=spec,
            robot_ip="127.0.0.1",
            port_offset=int(port_extra),
            enable_rerun_server=False,
            start_immediately=True,
            allow_missing_depth=False,
        )
        params = get_parameters("dynav_config.yaml")
        params["use_instance_memory"] = False
        params["depth_source"] = "sensor"

        executor = DynamemTaskExecutor(
            robot_client,
            params,
            skip_confirmations=True,
            manipulation_only=True,
        )
        executor([("rotate_in_place", "")])

        voxel_map = executor.agent.get_voxel_map()
        grid = voxel_map.grid
        torch_device_str = voxel_map.map_2d_device or "cpu"
        td = torch.device(torch_device_str)
        obstacles, explored = voxel_map.get_2d_map()

        h_g, w_g = int(grid.grid_size[0]), int(grid.grid_size[1])
        assert obstacles.shape[0] == h_g and obstacles.shape[1] == w_g
        assert explored.shape[0] == h_g and explored.shape[1] == w_g

        for cx, cy in corner_xy:
            xy = torch.tensor([[float(cx), float(cy)]], dtype=torch.float32, device=td)
            mapped = grid.xy_to_grid_coords(xy)
            assert mapped is not None, (
                "MJCF collision-clip corners must map inside the DynaMem finite grid "
                f"({grid.grid_size}); corner=({cx},{cy}) clip={clip}"
            )

        explor_n = int(explored.float().sum().item())
        obs_n = int(obstacles.float().sum().item())
        assert explor_n >= 80, (
            "Expected substantive explored cells after rotate_in_place; "
            f"explored_sum={explor_n} (depth / manipulation_only voxel path)."
        )
        assert obs_n >= 8, f"Expected some obstacle occupancy; obstacle_sum={obs_n}"

        from emet.mapping.debug_navgrid_ascii import build_navgrid_from_voxel_map

        ascii_map = build_navgrid_from_voxel_map(voxel_map)
        assert "#" in ascii_map and "." in ascii_map, (
            "ASCII nav grid should show obstacles and explored cells after mapping"
        )
        assert ascii_map.count("#") >= 2 and ascii_map.count(".") >= 2
    finally:
        if robot_client is not None:
            try:
                robot_client.stop()
            except Exception:
                pass
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=4)
        if merged_xml is not None:
            try:
                Path(merged_xml).unlink(missing_ok=True)
            except OSError:
                pass


def _wait_for_port(host: str, port: int, timeout_sec: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (TimeoutError, OSError):
            time.sleep(0.5)
    return False


def _inset_clip_rect(rect: tuple[float, float, float, float], inset: float) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = rect
    if xmax - xmin < 2 * inset + 0.75 or ymax - ymin < 2 * inset + 0.75:
        return rect
    return (xmin + inset, xmax - inset, ymin + inset, ymax - inset)
