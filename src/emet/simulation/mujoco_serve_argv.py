# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Build ``emet.simulation.mujoco_server`` argv from :class:`SimLaunchConfig` (shared by CLI and ``--start-sim``)."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emet.config.sim_launch_config import SimLaunchConfig


def _merge_molmospaces_scene(
    *,
    scene: str,
    split: str,
    index: int,
    robot: str,
    install_if_missing: bool,
) -> str:
    """Run emet-molmospaces merge-scene; return path to merged MJCF."""
    from emet.simulation.molmospaces_config import (
        build_molmospaces_wrapper_command,
        ensure_molmospaces_assets_dir_env,
        galaxea_r1_assets_directory,
    )

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    ensure_molmospaces_assets_dir_env()
    fd, merged_path = tempfile.mkstemp(
        suffix=".xml", prefix="molmospaces_merged_", dir=str(galaxea_r1_assets_directory())
    )
    os.close(fd)
    merge_robot = robot.lower().replace("-", "_")
    if merge_robot in ("stretch", "hello_stretch", "hellostretch"):
        merge_robot = "rby1"
    merge_argv = [
        "merge-scene",
        "--scene",
        scene,
        "--split",
        split,
        "--index",
        str(index),
        "--robot",
        merge_robot,
        "--output",
        merged_path,
    ]
    if install_if_missing:
        merge_argv.append("--install-if-missing")
    cmd = build_molmospaces_wrapper_command(merge_argv)
    if cmd is None:
        try:
            os.unlink(merged_path)
        except OSError:
            pass
        raise RuntimeError(
            "MolmoSpaces wrapper not found (install: ./install.sh --molmospaces -y or full install)."
        )
    env = os.environ.copy()
    ensure_molmospaces_assets_dir_env(env)
    r = subprocess.run(cmd, cwd=str(repo_root), env=env)
    if r.returncode != 0:
        try:
            os.unlink(merged_path)
        except OSError:
            pass
        raise RuntimeError(f"molmospaces merge-scene failed with exit code {r.returncode}")
    return merged_path


def prepare_mujoco_server_argv(sim: SimLaunchConfig) -> list[str]:
    """Return CLI args for ``python -m emet.simulation.mujoco_server`` (no sys.argv[0])."""
    from emet.config.sim_launch_config import (
        SimLaunchDefaultMujoco,
        SimLaunchMolmospaces,
        SimLaunchRobocasa,
    )

    args: list[str] = []
    mol_session: tuple[str, str, int] | None = None
    robot_out = "stretch"

    if isinstance(sim, SimLaunchMolmospaces):
        merged = _merge_molmospaces_scene(
            scene=sim.scene,
            split=sim.split,
            index=int(sim.index),
            robot=sim.robot,
            install_if_missing=bool(sim.molmospaces_install),
        )
        args.extend(["--scene_path", merged])
        mol_session = (sim.scene, sim.split, int(sim.index))
        robot_out = sim.robot
        if robot_out.lower() in ("stretch", "hello_stretch", "hellostretch"):
            robot_out = "rby1"
    elif isinstance(sim, SimLaunchRobocasa):
        args.append("--use-robocasa")
        task = (sim.robocasa_task or "").strip() or "PickPlaceCounterToCabinet"
        args.extend(["--robocasa-task", task])
        args.extend(["--robocasa-style", str(int(sim.robocasa_style))])
        args.extend(["--robocasa-layout", str(int(sim.robocasa_layout))])
        if sim.robocasa_write_to_xml:
            args.append("--robocasa-write-to-xml")
        robot_out = sim.robot
    else:
        assert isinstance(sim, SimLaunchDefaultMujoco)
        robot_out = sim.robot
        if sim.scene_path and str(sim.scene_path).strip():
            args.extend(["--scene_path", str(sim.scene_path).strip()])

    if sim.headless:
        args.append("--headless")
    if sim.show_viewer_ui:
        args.append("--show-viewer-ui")
    if sim.no_cameras:
        args.append("--no-cameras")
    if sim.use_glx:
        args.append("--use-glx")
    if mol_session is not None:
        sc, sp, idx = mol_session
        args.extend(
            [
                "--molmospaces-session-scene",
                sc,
                "--molmospaces-session-split",
                sp,
                "--molmospaces-session-index",
                str(idx),
            ]
        )
    args.extend(["--seed", str(int(sim.seed))])
    if sim.steps is not None:
        args.extend(["--steps", str(int(sim.steps))])
    if sim.debug_molmospaces_spawn:
        args.append("--debug-molmospaces-spawn")
    if sim.port_offset:
        args.extend(["--port-offset", str(int(sim.port_offset))])
    if robot_out and robot_out.lower() != "stretch":
        args.extend(["--robot", robot_out])
    if sim.verbose:
        args.append("--verbose")
    return args
