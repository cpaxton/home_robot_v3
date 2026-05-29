# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from emet.simulation import molmospaces_spawn
from emet.utils.logger import Logger

logger = Logger(__name__)


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
