# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from emet.robots.base import RobotSpec
from emet.simulation import molmospaces_spawn, scene_base_spawn
from emet.utils.geometry import angle_difference, xyt_base_to_global
from emet.utils.logger import Logger

logger = Logger(__name__)

DEFAULT_MOBILE_BASE_BODY = "base_link"


def base_body_free_joint_dofadr(model: mujoco.MjModel, base_body_name: str) -> tuple[int, int] | None:
    """Return ``(qposadr, dofadr)`` for the free joint on *base_body_name*, or None."""
    qadr = base_body_free_joint_qposadr(model, base_body_name)
    if qadr is None:
        return None
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) != bid:
            continue
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        return qadr, int(model.jnt_dofadr[j])
    return None


def write_base_freejoint_xyt(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    x: float,
    y: float,
    theta: float,
    z: float | None = None,
) -> bool:
    """Snap ``base_body_name`` free joint to world SE(2) ``(x, y, theta)``; preserve *z* if omitted."""
    addrs = base_body_free_joint_dofadr(model, base_body_name)
    if addrs is None:
        return False
    qadr, vadr = addrs
    z_use = float(data.qpos[qadr + 2]) if z is None else float(z)
    wt = float(theta)
    qw = float(np.cos(wt * 0.5))
    qz = float(np.sin(wt * 0.5))
    data.qpos[qadr] = float(x)
    data.qpos[qadr + 1] = float(y)
    data.qpos[qadr + 2] = z_use
    data.qpos[qadr + 3 : qadr + 7] = np.array([qw, 0.0, 0.0, qz], dtype=np.float64)
    data.qvel[vadr : vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return True


def read_body_se2_xyt(data: mujoco.MjData, base_body_name: str = DEFAULT_MOBILE_BASE_BODY) -> np.ndarray:
    """World SE(2) pose ``(x, y, theta)`` from ``body_xpos`` / ``body_xmat``."""
    xyz = data.body(base_body_name).xpos
    rotation = data.body(base_body_name).xmat.reshape(3, 3)
    theta = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return np.array([float(xyz[0]), float(xyz[1]), theta], dtype=np.float64)


def spawn_rel_xyt_to_world(goal_rel: np.ndarray, init_world_xyt: np.ndarray) -> np.ndarray:
    """Map spawn-frame ``(x, y, θ)`` to world using the spawn origin pose."""
    goal = np.asarray(goal_rel, dtype=np.float64).reshape(-1)[:3]
    origin = np.asarray(init_world_xyt, dtype=np.float64).reshape(-1)[:3]
    return xyt_base_to_global(goal, origin)


def se2_pose_at_goal(
    current_xyt: np.ndarray,
    goal_xyt: np.ndarray,
    *,
    xy_tol: float = 0.05,
    theta_tol: float = 0.15,
) -> bool:
    """True when planar position and yaw are within tolerances."""
    cur = np.asarray(current_xyt, dtype=np.float64).reshape(-1)[:3]
    goal = np.asarray(goal_xyt, dtype=np.float64).reshape(-1)[:3]
    xy_err = float(np.linalg.norm(cur[:2] - goal[:2]))
    th_err = abs(float(angle_difference(cur[2], goal[2])))
    return xy_err <= xy_tol and th_err <= theta_tol


def base_body_free_joint_qposadr(model: mujoco.MjModel, base_body_name: str) -> int | None:
    """Return the free-joint ``qpos`` slice start index for *base_body_name*, or None."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if bid < 0:
        return None
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) != bid:
            continue
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        return int(model.jnt_qposadr[j])
    return None


def maybe_prepare_molmospaces_meshes(scene_xml_path: str) -> None:
    """Ensure Molmo symlink layout when loading a CLI-merged MJCF named ``molmospaces_merged_*.xml``."""
    bn = scene_xml_path.rsplit("/", maxsplit=1)[-1]
    if not bn.startswith("molmospaces_merged"):
        return
    from emet.simulation.molmospaces_config import ensure_molmo_asset_layout_symlinks

    ensure_molmo_asset_layout_symlinks()


def apply_molmospaces_freejoint_base_autoplace(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    merged_mjcf_path: str | None,
    base_body_name: str,
    environment: dict[str, Any] | None,
    scene_source_basename: str | None,
    robot_key: str | None = None,
    debug: bool,
) -> bool:
    """If Molmo heuristic applies, reposition the base freejoint and snap ``model.qpos0``.

    Assumes *data* matches *model* and kinematics were initialized (``mj_forward``).

    Returns:
        True if placement ran and updated ``model.qpos0`` for the freejoint.
    """
    if not molmospaces_spawn.want_molmospaces_autoplace(
        environment=environment,
        scene_source_basename=scene_source_basename,
    ):
        return False
    qadr = base_body_free_joint_qposadr(model, base_body_name)
    if qadr is None:
        return False
    if debug:
        logger.info(
            "MolmoSpaces spawn debug: "
            f"scene_source_basename={scene_source_basename!r} "
            f"environment={environment!r} base_body_name={base_body_name!r}"
        )
    try:
        placed = molmospaces_spawn.find_molmospaces_freejoint_xyz(
            model,
            data,
            base_body_name=base_body_name,
            scene_label=scene_source_basename,
            merged_mjcf_path=merged_mjcf_path,
            environment=environment,
            robot_key=robot_key,
        )
    except Exception as e:
        logger.warning(f"MolmoSpaces base autoplace skipped ({e!r}).")
        return False
    if placed is None:
        if debug:
            logger.info(
                "MolmoSpaces base autoplace: find_molmospaces_freejoint_xyz returned None "
                "(see spawn debug lines above)."
            )
        return False
    x, y, z = placed
    logger.info(
        f"MolmoSpaces base autoplace: moved free joint on {base_body_name!r} to "
        f"({x:.3f}, {y:.3f}, {z:.3f}) to avoid origin clutter."
    )
    if debug:
        try:
            mujoco.mj_forward(model, data)
            for ln in molmospaces_spawn.format_spawn_contact_report(
                model,
                data,
                base_body_name=base_body_name,
                floor_geom_name="floor",
                max_lines=50,
                dist_report_threshold=0.15,
            ):
                logger.info(f"[molmospaces_spawn/post-place] {ln}")
            for ln in molmospaces_spawn.format_spawn_floor_alignment_report(
                model,
                data,
                base_body_name=base_body_name,
                floor_geom_name="floor",
                xy=(float(x), float(y)),
            ):
                logger.info(f"[molmospaces_spawn/post-place] {ln}")
        except Exception as e:
            logger.warning(f"MolmoSpaces spawn debug contact report failed: {e!r}")
    model.qpos0[qadr : qadr + 7] = data.qpos[qadr : qadr + 7]
    return True


def apply_robocasa_freejoint_base_autoplace(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_spec: RobotSpec,
    base_body_name: str,
    environment: dict[str, Any] | None,
    scene_source_basename: str | None = None,
    merged_mjcf_path: str | None = None,
    debug: bool = False,
) -> bool:
    """If Robocasa freejoint autoplace applies, reposition the base and snap ``model.qpos0``.

    Assumes *data* matches *model* and kinematics were initialized (``mj_forward``).

    Returns:
        True if placement ran and updated ``model.qpos0`` for the freejoint.
    """
    if not scene_base_spawn.want_robocasa_freejoint_autoplace(
        environment=environment,
        robot_spec=robot_spec,
    ):
        return False
    qadr = base_body_free_joint_qposadr(model, base_body_name)
    if qadr is None:
        return False
    spawn_hint: np.ndarray | None = None
    if isinstance(environment, dict):
        raw_hint = environment.get("spawn_hint_xyt")
        if raw_hint is not None:
            spawn_hint = np.asarray(raw_hint, dtype=np.float64).reshape(-1)[:3].copy()
    if debug:
        logger.info(
            "Robocasa freejoint spawn debug: "
            f"scene_source_basename={scene_source_basename!r} "
            f"environment={environment!r} base_body_name={base_body_name!r}"
        )
    try:
        placed = scene_base_spawn.find_robocasa_freejoint_xyz(
            model,
            data,
            base_body_name=base_body_name,
            robot_spec=robot_spec,
            scene_label=scene_source_basename,
            merged_mjcf_path=merged_mjcf_path,
            environment=environment,
            spawn_hint_xyt=spawn_hint,
        )
    except Exception as e:
        logger.warning(f"Robocasa freejoint autoplace skipped ({e!r}).")
        return False
    if placed is None:
        logger.info("Robocasa freejoint autoplace: no safer (x,y,z) found; keeping MJCF default base pose.")
        return False
    x, y, z = placed
    hint_dxy = ""
    if spawn_hint is not None and spawn_hint.size >= 2:
        hint_dxy = (
            f", Δxy from robosuite hint={float(np.hypot(x - float(spawn_hint[0]), y - float(spawn_hint[1]))):.3f}m"
        )
    logger.info(
        f"Robocasa freejoint autoplace: moved base on {base_body_name!r} to "
        f"({x:.3f}, {y:.3f}, {z:.3f}) for clearance from scene geometry{hint_dxy}."
    )
    if debug:
        try:
            mujoco.mj_forward(model, data)
            for ln in scene_base_spawn.format_spawn_contact_report(
                model,
                data,
                base_body_name=base_body_name,
                floor_geom_name="floor",
                max_lines=50,
                dist_report_threshold=0.12,
            ):
                logger.info(f"[scene_base_spawn/post-place] {ln}")
        except Exception as e:
            logger.warning(f"Robocasa freejoint spawn debug contact report failed: {e!r}")
    model.qpos0[qadr : qadr + 7] = data.qpos[qadr : qadr + 7]
    return True
