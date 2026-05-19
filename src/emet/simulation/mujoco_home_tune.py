# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Interactive MuJoCo home-pose tuning: Simulate GUI, then emit ``<key ctrl=.../>`` text."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

import mujoco

from emet.simulation.mujoco_stationary_control import compute_stationary_ctrl_vector
from emet.simulation.robosuite_load_utils import apply_home_keyframe_preserving_base


def format_key_ctrl_attr(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    """Space-separated ``ctrl`` values in actuator order (matches MJCF ``<key ctrl=.../>``)."""
    u = compute_stationary_ctrl_vector(model, data)
    return " ".join(f"{float(x):.6g}" for x in u)


def print_home_keyframe_snippet(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    stream: TextIO = sys.stdout,
) -> None:
    """Print a copy-paste snippet for ``galaxea_r1.xml``-style ``<key name=\"home\" .../>``."""
    mujoco.mj_forward(model, data)
    ctrl = format_key_ctrl_attr(model, data)
    stream.write(
        "\n--- Paste into galaxea_r1.xml (or your robot MJCF) inside <keyframe> ---\n"
        f'    <key name="home" ctrl="{ctrl}"/>\n'
        "--- (torso + arms + grippers; swerve zeros preserved) ---\n\n"
    )


def run_tune_home_gui(
    mjcf_path: str | Path,
    *,
    apply_home_keyframe: bool,
    base_body_name: str,
    out: TextIO = sys.stdout,
) -> None:
    """Open MuJoCo **Simulate** (interactive); after the window closes, print ``ctrl=`` line.

    In Simulate you can drag joints, use controls, and let physics settle. Closing the window
    returns here; we then snapshot ``qpos`` → actuator ``ctrl`` string (same convention as
    stationary fill / MJCF home keyframe).
    """
    path = Path(mjcf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MJCF not found: {path}")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if apply_home_keyframe:
        if apply_home_keyframe_preserving_base(model, data, base_body_name=base_body_name):
            snap = format_key_ctrl_attr(model, data)
            out.write(
                "Started from MJCF keyframe 'home' (base pose preserved); "
                f"initial ctrl snapshot: {snap[:120]}{'…' if len(snap) > 120 else ''}\n"
            )
        else:
            out.write("No 'home' keyframe or no base free joint — starting from compiled defaults.\n")

    out.write(
        "Opening MuJoCo Simulate — pose the robot, then close the window to print the home ctrl line.\n"
    )
    out.flush()

    import mujoco.viewer as mjv

    mjv.launch(model, data)

    print_home_keyframe_snippet(model, data, stream=out)
