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

"""Spawn placement for the packaged default table scene + merged mobile robot (``scene_environment.xml``).

The robot MJCF is merged at the world origin; without adjustment the base can sit inside the
floor plane or intersect the table. This module writes a better ``qpos0`` slice for the base
``freejoint`` before the ZMQ server creates ``MjData``.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from emet.robots import get_robot_spec
from emet.simulation.mujoco_ctrl_sync import stabilize_physics_inplace

logger = logging.getLogger(__name__)

_DEFAULT_BASE_BODY = "base_link"
# World XY in front / beside the packaged table (table is centered near y ≈ -1 m).
_SPAWN_XY_CANDIDATES: tuple[tuple[float, float], ...] = (
    (0.52, 0.58),
    (0.40, 0.72),
    (-0.48, 0.62),
    (0.0, 0.75),
    (-0.35, 0.50),
    (0.55, 0.40),
)


def snap_packaged_table_robot_to_scene_floor(
    model: mujoco.MjModel,
    *,
    base_body_name: str = _DEFAULT_BASE_BODY,
    robot_key: str | None = None,
) -> None:
    """Mutate ``model.qpos0`` so the merged robot stands on ``floor`` and clears the table box.

    When *robot_key* is set (same as ``emet serve mujoco --robot``), runs a short stepped settle with
    actuators re-synced to ``qpos`` each step so the stored default pose matches what the ZMQ server
    uses after load (avoids a large first-``mj_step`` PD kick / visible bounce).

    No-op if *base_body_name* is missing or has no free joint.
    """
    from emet.simulation import molmospaces_spawn as ms

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if bid < 0:
        return
    qadr = _base_freejoint_qposadr(model, base_body_name)
    if qadr is None:
        return

    data = mujoco.MjData(model)
    np.copyto(data.qpos, model.qpos0)

    chosen_xy: tuple[float, float] | None = None
    used_primary = False
    for x, y in _SPAWN_XY_CANDIDATES:
        z = ms._first_z_with_nonpenetrating_base(
            model,
            data,
            base_body_name=base_body_name,
            floor_geom_name="floor",
            x=float(x),
            y=float(y),
            z_floor=0.0,
            z_min_above_floor=0.08,
            z_max_above_floor=1.35,
            n_z=52,
            min_clearance=0.002,
        )
        if z is not None:
            ms.write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=float(x), y=float(y), z=float(z))
            mujoco.mj_forward(model, data)
            chosen_xy = (float(x), float(y))
            used_primary = True
            break

    if not used_primary:
        # Last resort: lift using collision hull vs floor at the first candidate XY.
        x0, y0 = _SPAWN_XY_CANDIDATES[0]
        if not ms.write_freejoint_base_xyzw(
            model, data, base_body_name=base_body_name, x=float(x0), y=float(y0), z=0.28
        ):
            return
        mujoco.mj_forward(model, data)
        robot_bodies = ms._bodies_descending_from(model, bid)
        zb = ms._min_robot_collision_geom_bottom_z(model, data, robot_bodies)
        if zb is None:
            return
        dz = float(0.04 - zb)
        data.qpos[qadr + 2] += dz
        mujoco.mj_forward(model, data)
        chosen_xy = (float(x0), float(y0))

    spec = get_robot_spec(robot_key) if robot_key else None
    # Stretch's registry spec is abstract (no MuJoCo actuator↔joint layout); skip PD settle here.
    if (
        spec is not None
        and robot_key is not None
        and robot_key.lower().replace("-", "_")
        in (
            "stretch",
            "hello_stretch",
            "hellostretch",
        )
    ):
        spec = None
    if spec is not None:
        np.copyto(model.qpos0, data.qpos)
        settle = mujoco.MjData(model)
        np.copyto(settle.qpos, model.qpos0)
        stabilize_physics_inplace(model, settle, spec, n_steps=48)
        np.copyto(model.qpos0, settle.qpos)
    else:
        model.qpos0[qadr : qadr + 7] = np.asarray(data.qpos[qadr : qadr + 7], dtype=np.float64)
    z_final = float(model.qpos0[qadr + 2])
    if used_primary and chosen_xy is not None:
        logger.info(
            "default_table_spawn: base freejoint qpos0 set to xy=(%.3f, %.3f) z=%.3f",
            chosen_xy[0],
            chosen_xy[1],
            z_final,
        )
    elif not used_primary:
        logger.warning(
            "default_table_spawn: primary XY/Z search failed; used fallback lift. Final z=%.3f xy=%s",
            z_final,
            chosen_xy,
        )


def _base_freejoint_id(model: mujoco.MjModel, base_body_name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if bid < 0:
        return -1
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) != bid:
            continue
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(j)
    return -1


def _base_freejoint_qposadr(model: mujoco.MjModel, base_body_name: str) -> int | None:
    j = _base_freejoint_id(model, base_body_name)
    if j < 0:
        return None
    return int(model.jnt_qposadr[j])
