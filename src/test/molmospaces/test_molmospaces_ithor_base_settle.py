# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Integration: iTHOR + rby1 merged MJCF — spawn, physics settle, base stays upright and orientation-stable.

Requires MolmoSpaces assets, ``emet_molmospaces`` (same as other RUN_MOLMOSPACES_TESTS checks), and
MuJoCo. Orthographic occupancy (iTHORMap) may require GL; if it fails, spawn falls back to heuristics.

Run (from repo root)::

    RUN_MOLMOSPACES_TESTS=1 uv run emet test src/test/molmospaces/test_molmospaces_ithor_base_settle.py -v

Optional: ``EMET_MOLMOSPACES_SETTLE_STEPS`` (default 900), ``EMET_MOLMOSPACES_ORIENTATION_MAX_DEG`` (default 8).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

import emet.simulation.molmospaces_spawn as molmospaces_spawn

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MOLMOSPACES_TESTS", "") != "1",
    reason="RUN_MOLMOSPACES_TESTS=1 (MolmoSpaces assets + emet_molmospaces in env)",
)


def _base_freejoint_quat_wxyz(model: mujoco.MjModel, data: mujoco.MjData, base_body_name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    assert bid >= 0, f"body {base_body_name!r} not found"
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) != bid:
            continue
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qadr = int(model.jnt_qposadr[j])
        return np.array(data.qpos[qadr + 3 : qadr + 7], dtype=np.float64)
    raise RuntimeError(f"no free joint on {base_body_name!r}")


def _body_z_axis_world(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    """World-frame direction of the body's +Z axis (third column of body-to-world rotation)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    assert bid >= 0
    R = data.xmat[bid].reshape(3, 3)
    return R[:, 2].astype(np.float64)


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    """Unsigned rotation angle in degrees between two unit quaternions (wxyz), double-cover safe."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    d = abs(float(np.dot(q0, q1)))
    d = min(1.0, max(0.0, d))
    return math.degrees(2.0 * math.acos(d))


def test_ithor_train_indices_base_flat_after_physics_settle():
    """Load ~10 iTHOR merges, autoplace base, step physics; base stays ~horizontal and ~same yaw/pitch/roll."""
    try:
        from emet_molmospaces.runner import _find_installed_scene_xml, _get_robot_mjcf_path, _merge_robot_into_scene
    except ImportError:
        pytest.skip("emet_molmospaces not installed in this environment")

    n_scenes = int(os.environ.get("EMET_MOLMOSPACES_ORIENTATION_N", "10"))
    settle_steps = int(os.environ.get("EMET_MOLMOSPACES_SETTLE_STEPS", "900"))
    max_deg = float(os.environ.get("EMET_MOLMOSPACES_ORIENTATION_MAX_DEG", "8"))
    min_up_dot = float(os.environ.get("EMET_MOLMOSPACES_MIN_UP_DOT", "0.92"))

    robot = _get_robot_mjcf_path("rby1")
    if robot is None or not robot.is_file():
        pytest.skip("rby1 MJCF not found")

    env_desc = {"kind": "molmospaces", "scene": "ithor", "split": "train"}

    for idx in range(n_scenes):
        scene = _find_installed_scene_xml("ithor", idx)
        if scene is None or not scene.is_file():
            pytest.skip(f"iTHOR scene index {idx} not installed (FloorPlan or assets missing)")
        merged = _merge_robot_into_scene(Path(scene), robot)
        try:
            m = mujoco.MjModel.from_xml_path(str(merged))
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            placed = molmospaces_spawn.find_molmospaces_freejoint_xyz(
                m,
                d,
                base_body_name="base_link",
                min_nonfloor_clearance=-5e-5,
                scene_label=merged.name,
                merged_mjcf_path=str(merged),
                environment={**env_desc, "index": idx},
            )
            assert placed is not None, f"index {idx}: find_molmospaces_freejoint_xyz returned None"
            x, y, z = placed
            assert molmospaces_spawn.write_freejoint_base_xyzw(m, d, base_body_name="base_link", x=x, y=y, z=float(z))
            mujoco.mj_forward(m, d)
            q0 = _base_freejoint_quat_wxyz(m, d, "base_link")
            z0 = _body_z_axis_world(m, d, "base_link")
            assert float(z0[2]) >= min_up_dot, f"index {idx}: base not upright before settle (z·ê_z={z0[2]:.3f})"

            d.ctrl[:] = 0.0
            d.qvel[:] = 0.0
            for _ in range(settle_steps):
                mujoco.mj_step(m, d)

            q1 = _base_freejoint_quat_wxyz(m, d, "base_link")
            z1 = _body_z_axis_world(m, d, "base_link")
            ang = _quat_angle_deg(q0, q1)
            assert ang <= max_deg, (
                f"index {idx}: orientation drift {ang:.2f}° > {max_deg}° (settle_steps={settle_steps})"
            )
            assert float(z1[2]) >= min_up_dot, f"index {idx}: base not upright after settle (z·ê_z={z1[2]:.3f})"
        finally:
            merged.unlink(missing_ok=True)
