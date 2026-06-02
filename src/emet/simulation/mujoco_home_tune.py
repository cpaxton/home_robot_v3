# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Interactive MuJoCo home-pose tuning: Simulate GUI (physics) or kinematic viewer."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal, TextIO

import mujoco
import numpy as np

from emet.simulation.molmospaces_spawn import effective_floor_geom_name
from emet.simulation.mujoco_stationary_control import compute_stationary_ctrl_vector
from emet.simulation.robosuite_load_utils import (
    apply_home_keyframe_preserving_base,
    apply_zero_joint_pose_preserving_base,
    freejoint_qpos_qvel_addrs,
)

InitialPose = Literal["default", "zeros", "home"]


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


def configure_kinematic_tune(model: mujoco.MjModel) -> None:
    """Disable dynamics so accidental ``mj_step`` calls do not move the model."""
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SPRING)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)


def _spec_has_floor(spec: mujoco.MjSpec) -> bool:
    model = spec.compile()
    try:
        floor_nm = effective_floor_geom_name(model)
        return floor_nm is not None and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_nm) >= 0
    finally:
        del model


def _add_tune_floor_plane(spec: mujoco.MjSpec, *, kinematic: bool) -> None:
    """Ground plane for robot-only MJCF (contact on unless kinematic)."""
    g = spec.worldbody.add_geom()
    g.name = "emet_tune_floor"
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [12, 12, 0.01]
    g.pos = [0, 0, 0]
    g.rgba = [0.72, 0.72, 0.72, 1]
    if kinematic:
        g.contype = 0
        g.conaffinity = 0


def _apply_initial_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    initial_pose: InitialPose,
    base_body_name: str,
    logs: list[str],
) -> None:
    if initial_pose == "default":
        return
    if initial_pose == "zeros":
        apply_zero_joint_pose_preserving_base(model, data, base_body_name=base_body_name)
        logs.append("Set hinge/slide joints and ctrl to zero (base free joint pose preserved).")
        return
    if apply_home_keyframe_preserving_base(model, data, base_body_name=base_body_name):
        logs.append("Applied MJCF keyframe 'home' (base free joint pose preserved).")
    else:
        logs.append("No 'home' keyframe or no base free joint — using compiled defaults.")


def _hoist_free_base_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    z_world: float,
) -> None:
    """Raise the base free joint before freezing (reference height in the tune sandbox)."""
    addrs = freejoint_qpos_qvel_addrs(model, base_body_name)
    if addrs is None:
        return
    qadr, vadr = addrs
    data.qpos[qadr + 2] = float(z_world)
    if vadr >= 0:
        data.qvel[vadr : vadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def _freeze_base_at_current_world_pose(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Remove the base free joint and fix ``base_link`` at its current world pose."""
    addrs = freejoint_qpos_qvel_addrs(model, base_body_name)
    if addrs is None:
        return model, data
    qadr, _ = addrs
    pos = np.asarray(data.qpos[qadr : qadr + 3], dtype=np.float64)
    quat = np.asarray(data.qpos[qadr + 3 : qadr + 7], dtype=np.float64)
    base = spec.body(base_body_name)
    base.pos = [float(pos[0]), float(pos[1]), float(pos[2])]
    base.quat = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
    deleted = False
    for j in list(spec.joints):
        if j.type != mujoco.mjtJoint.mjJNT_FREE:
            continue
        parent_body = j.parent
        if parent_body is not None and getattr(parent_body, "name", None) == base_body_name:
            spec.delete(j)
            deleted = True
            break
    if not deleted:
        try:
            spec.delete(spec.joint("base_freejoint"))
        except Exception:
            pass
    model2 = spec.compile()
    data2 = mujoco.MjData(model2)
    mujoco.mj_forward(model2, data2)
    return model2, data2


def build_tune_model(
    mjcf_path: str | Path,
    *,
    initial_pose: InitialPose = "default",
    base_body_name: str,
    tune_base_z: float = 0.38,
    kinematic: bool = False,
) -> tuple[mujoco.MjModel, mujoco.MjData, list[str]]:
    """Load MJCF for home tuning (physics Simulate by default; optional kinematic sandbox).

    Returns:
        ``(model, data, log_lines)``
    """
    path = Path(mjcf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MJCF not found: {path}")
    logs: list[str] = []
    spec = mujoco.MjSpec.from_file(str(path))
    if not _spec_has_floor(spec):
        _add_tune_floor_plane(spec, kinematic=kinematic)
        if kinematic:
            logs.append("Added visual emet_tune_floor (reference only, no contact).")
        else:
            logs.append("Added emet_tune_floor plane (robot-only MJCF had no floor geom).")
    model = spec.compile()
    data = mujoco.MjData(model)
    if kinematic:
        configure_kinematic_tune(model)
        logs.append("Kinematic sandbox: gravity/contact/actuation disabled.")
    mujoco.mj_forward(model, data)
    _apply_initial_pose(
        model, data, initial_pose=initial_pose, base_body_name=base_body_name, logs=logs
    )
    if freejoint_qpos_qvel_addrs(model, base_body_name) is not None:
        _hoist_free_base_z(model, data, base_body_name=base_body_name, z_world=tune_base_z)
        logs.append(f"Hoisted base free joint to z={tune_base_z:g} m for tuning.")
        model, data = _freeze_base_at_current_world_pose(
            spec, model, data, base_body_name=base_body_name
        )
        if kinematic:
            configure_kinematic_tune(model)
        logs.append(
            f"Froze {base_body_name!r} at current pose (removed free joint)"
            + ("." if kinematic else " so Simulate does not drop the robot.")
        )
    elif not logs:
        logs.append("Loaded model (fixed base or no free joint).")
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return model, data, logs


def run_kinematic_tune_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    show_ui: bool = True,
) -> None:
    """Passive viewer loop: ``mj_forward`` only (no ``mj_step``)."""
    import mujoco.viewer as mjv

    with mjv.launch_passive(
        model,
        data,
        show_left_ui=show_ui,
        show_right_ui=show_ui,
    ) as viewer:
        while viewer.is_running():
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


def run_physics_tune_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """MuJoCo Simulate (full physics and control UI)."""
    import mujoco.viewer as mjv

    mjv.launch(model, data)


def run_tune_home_gui(
    mjcf_path: str | Path,
    *,
    initial_pose: InitialPose = "default",
    base_body_name: str,
    tune_base_z: float = 0.38,
    kinematic: bool = False,
    out: TextIO = sys.stdout,
) -> None:
    """Open MuJoCo viewer; after close, print ``<key name=\"home\" ctrl=.../>`` from ``qpos``."""
    model, data, logs = build_tune_model(
        mjcf_path,
        initial_pose=initial_pose,
        base_body_name=base_body_name,
        tune_base_z=tune_base_z,
        kinematic=kinematic,
    )
    for ln in logs:
        out.write(ln + "\n")
    if initial_pose != "default":
        snap = format_key_ctrl_attr(model, data)
        out.write(
            "Initial ctrl snapshot after setup: "
            f"{snap[:120]}{'…' if len(snap) > 120 else ''}\n"
        )

    if kinematic:
        out.write(
            "Opening kinematic MuJoCo viewer (no physics) — adjust joints or drag links, "
            "then close the window to print the home ctrl line.\n"
        )
    else:
        out.write(
            "Opening MuJoCo Simulate — pose the robot, tune physics/control, "
            "then close the window to print the home ctrl line.\n"
        )
    out.flush()

    if kinematic:
        run_kinematic_tune_viewer(model, data)
    else:
        run_physics_tune_viewer(model, data)

    print_home_keyframe_snippet(model, data, stream=out)
