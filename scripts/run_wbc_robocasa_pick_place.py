#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Multi-step PandaOmron WBC pick-place harness in a RoboCasa kitchen.

This is the task-target bridge above the reach adapter: a privileged teacher
(or a trained policy) tracks a sequence of world-frame EE and base targets for
approach, pregrasp, grasp, bilateral contact verification, lift, retract,
dynamic base transport, receptacle placement, release, and retreat.
Object/receptacle poses are task metadata for this teacher/ceiling run; the
actor still consumes the WBC contract observation. Do not present this script
alone as a no-GT benchmark result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_env import WbcGymEnv
from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend
from emet.rl.manipulation.wbc_training import WbcPPO

# Exact collision offsets of Robosuite's PandaGripper relative to gripper0_right_eef
# when pointing top-down (+Z_eef = -Z_world):
# - hand_collision palm extends to 31.0 mm above eef
# - finger*_collision tip extends to 9.2 mm below eef
# - finger*_pad_collision spans [eef_z - 4.4 mm, eef_z + 11.6 mm] (center +3.6 mm, half-height 8.0 mm)
# - open inner pad separation at q = [0.04, -0.04] is 79.0 mm
_PALM_CLEARANCE_ABOVE_EEF_M = 0.031
_FINGER_TIP_BELOW_EEF_M = 0.0092
_PAD_CENTER_ABOVE_EEF_M = 0.0036
_PAD_HALF_HEIGHT_M = 0.0080

_BOX_CORNERS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
_BOX_EDGE_INDICES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=int,
)


def _resolve_body(env: Any, requested: str) -> str:
    names = list(env.sim.model.body_names)
    if requested in names:
        return requested
    tokens = [token for token in requested.replace("_main", "").split("_") if token]
    candidates = [name for name in names if tokens and all(token in name for token in tokens)]
    if not candidates:
        raise ValueError(f"body {requested!r} is not in the RoboCasa scene; known={sorted(names)[:20]}...")
    return sorted(candidates, key=lambda name: (len(name), name))[0]


def _body_pose(env: Any, body: str) -> tuple[np.ndarray, np.ndarray]:
    body = _resolve_body(env, body)
    return (
        np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64).copy(),
        np.asarray(env.sim.data.get_body_xquat(body), dtype=np.float64).copy(),
    )


def _body_T(env: Any, body: str) -> np.ndarray:
    pos = np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64)
    quat = np.asarray(env.sim.data.get_body_xquat(body), dtype=np.float64)
    rotation = Rotation.from_quat(np.r_[quat[1:], quat[0]])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation.as_matrix()
    T[:3, 3] = pos
    return T


def _descendant_body_ids(model: Any, root_id: int) -> set[int]:
    ids = {int(root_id)}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            if body_id not in ids and int(model.body_parentid[body_id]) in ids:
                ids.add(body_id)
                changed = True
    return ids


def _body_geometry(env: Any, body: str) -> tuple[np.ndarray, np.ndarray]:
    """Return world-space vertices ``(N, 3)`` and segment edges ``(E, 2, 3)`` of collidable geoms."""
    model = env.sim.model
    data = env.sim.data
    root_id = int(model.body_name2id(_resolve_body(env, body)))
    body_ids = _descendant_body_ids(model, root_id)
    points: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    for geom in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom]) not in body_ids:
            continue
        if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
            continue
        geom_type = int(model.geom_type[geom])
        rotation = np.asarray(data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
        center = np.asarray(data.geom_xpos[geom], dtype=np.float64)
        if geom_type == 7:
            mesh = int(model.geom_dataid[geom])
            v_start = int(model.mesh_vertadr[mesh])
            v_count = int(model.mesh_vertnum[mesh])
            verts = np.asarray(model.mesh_vert[v_start : v_start + v_count], dtype=np.float64)
            world_verts = verts @ rotation.T + center
            points.append(world_verts)
            f_start = int(model.mesh_faceadr[mesh])
            f_count = int(model.mesh_facenum[mesh])
            if f_count > 0:
                faces = np.asarray(model.mesh_face[f_start : f_start + f_count], dtype=int)
                tri_edges = np.concatenate(
                    [
                        world_verts[faces[:, [0, 1]]],
                        world_verts[faces[:, [1, 2]]],
                        world_verts[faces[:, [2, 0]]],
                    ],
                    axis=0,
                )
                edges.append(tri_edges)
        else:
            size = np.asarray(model.geom_size[geom], dtype=np.float64)
            if geom_type == 2:  # sphere
                half = np.repeat(size[0], 3)
            elif geom_type in (3, 5):  # capsule, cylinder
                half = np.array([size[0], size[0], size[1]], dtype=np.float64)
            else:
                half = np.where(size > 0, size, 0.01)
            corners = (_BOX_CORNERS * half) @ rotation.T + center
            points.append(corners)
            edges.append(corners[_BOX_EDGE_INDICES])
    if not points:
        raise ValueError(f"body {body!r} has no collidable geometry")
    all_points = np.concatenate(points, axis=0)
    all_edges = np.concatenate(edges, axis=0) if edges else np.zeros((0, 2, 3), dtype=np.float64)
    return all_points, all_edges


def _slice_2d(points: np.ndarray, edges: np.ndarray, z_center: float, half_height: float) -> np.ndarray:
    """Exact 2D cross-section over ``[z_center - half_height, z_center + half_height]``."""
    chunks: list[np.ndarray] = []
    in_band = points[np.abs(points[:, 2] - z_center) <= half_height]
    if len(in_band) > 0:
        chunks.append(in_band[:, :2])
    if len(edges) > 0:
        p0 = edges[:, 0]
        p1 = edges[:, 1]
        dz = p1[:, 2] - p0[:, 2]
        non_flat = np.abs(dz) > 1e-9
        for z_plane in np.linspace(z_center - half_height, z_center + half_height, 5):
            crosses = non_flat & ((p0[:, 2] - z_plane) * (p1[:, 2] - z_plane) <= 0.0)
            if np.any(crosses):
                t = (z_plane - p0[crosses, 2]) / dz[crosses]
                pts = p0[crosses, :2] + t[:, None] * (p1[crosses, :2] - p0[crosses, :2])
                chunks.append(pts)
    if not chunks:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _plan_topdown_grasp(
    points: np.ndarray,
    edges: np.ndarray,
    *,
    base_yaw: float,
    opening_m: float,
    margin_m: float,
) -> tuple[float, np.ndarray, float, float, list[tuple[float, float]]]:
    """Select ``(eef_z, grasp_xy, grasp_yaw, width_along_fingers, table)``.

    Respects both counter clearance (finger tips 9.2 mm below EEF) and palm
    clearance (hand collision 31 mm above EEF), and sweeps grasp yaw across a
    180-degree wrist-comfortable span so elongated objects (soap bars, pears,
    cartons) are pinched across their narrow axis.
    """
    low_z = float(points[:, 2].min())
    high_z = float(points[:, 2].max())
    max_ok = opening_m - margin_m

    z_min_counter = low_z + _FINGER_TIP_BELOW_EEF_M + 0.005
    z_min_palm = high_z - (_PALM_CLEARANCE_ABOVE_EEF_M - 0.005)
    z_max = max(z_min_counter, high_z - 0.012)
    z_lo = min(z_max, max(z_min_counter, z_min_palm))
    z_hi = max(z_lo, z_max)

    # _READY_Q[6] = +0.785 rad corresponds to base_yaw + 90 deg; sweeping +-90 deg
    # around base_yaw + 90 deg covers every undirected 2D finger axis while keeping
    # wrist rotation within 90 deg of the ready posture.
    yaw_pref = base_yaw + 0.50 * np.pi
    candidate_yaws = yaw_pref + np.linspace(-0.5 * np.pi, 0.5 * np.pi, 18, endpoint=False)

    table: list[tuple[float, float]] = []
    best_valid: tuple[float, float, np.ndarray, float, float] | None = None
    best_fallback: tuple[float, float, np.ndarray, float, float] | None = None

    z_values = np.arange(z_lo, z_hi + 1e-9, 0.002)
    if len(z_values) == 0:
        z_values = np.array([z_lo], dtype=np.float64)

    for eef_z in z_values:
        pts_2d = _slice_2d(points, edges, float(eef_z) + _PAD_CENTER_ABOVE_EEF_M, _PAD_HALF_HEIGHT_M)
        if len(pts_2d) == 0:
            continue
        min_w_at_z = float("inf")
        for yaw in candidate_yaws:
            close_axis = np.array([np.sin(yaw), -np.cos(yaw)], dtype=np.float64)
            ortho_axis = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
            p_close = pts_2d @ close_axis
            p_ortho = pts_2d @ ortho_axis
            w_close = float(p_close.max() - p_close.min())
            min_w_at_z = min(min_w_at_z, w_close)
            center_xy = (
                0.5 * float(p_ortho.max() + p_ortho.min()) * ortho_axis
                + 0.5 * float(p_close.max() + p_close.min()) * close_axis
            )
            yaw_diff = abs(float(np.arctan2(np.sin(yaw - yaw_pref), np.cos(yaw - yaw_pref))))
            # Prefer deepest safe height (closest to z_lo, away from sloped top
            # caps/stems) first, then narrowest pinch width at that height.
            score = 2.0 * float(eef_z - z_lo) + 0.25 * w_close + 0.002 * yaw_diff
            if 0.012 <= w_close <= max_ok:
                if best_valid is None or score < best_valid[0]:
                    best_valid = (score, float(eef_z), center_xy, float(yaw), w_close)
            fallback_score = abs(w_close - 0.5 * max_ok) + 0.002 * yaw_diff
            if best_fallback is None or fallback_score < best_fallback[0]:
                best_fallback = (fallback_score, float(eef_z), center_xy, float(yaw), w_close)
        if np.isfinite(min_w_at_z):
            table.append((float(eef_z), min_w_at_z))

    chosen = best_valid if best_valid is not None else best_fallback
    if chosen is None:
        center_xy = 0.5 * (points[:, :2].min(axis=0) + points[:, :2].max(axis=0))
        return float(z_lo), center_xy, float(yaw_pref), 0.0, table
    _, eef_z, center_xy, yaw, width = chosen
    return eef_z, center_xy, yaw, width, table


def _receptacle_place_target(
    env: Any,
    receptacle_body: str,
    *,
    object_bottom_offset_m: float,
    place_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return ``(place_xy, preplace_eef_xyz, place_eef_xyz, front_out_xy)``.

    Top-open receptacles (sink) enter from above; front-open receptacles
    (cabinet) stage outside the open door along ``front_out_xy``, slide in
    horizontally at shelf height, and retreat straight back out the door.
    """
    fixture = None
    if "sink" in receptacle_body and hasattr(env, "sink"):
        fixture = env.sink
    elif "cab" in receptacle_body and hasattr(env, "cab"):
        fixture = env.cab
    if fixture is not None and hasattr(fixture, "get_int_sites"):
        int_sites = fixture.get_int_sites(relative=False)
        if int_sites:
            p0, px, py, pz = next(iter(int_sites.values()))
            p0 = np.asarray(p0, dtype=np.float64)
            u = np.asarray(px, dtype=np.float64) - p0
            v = np.asarray(py, dtype=np.float64) - p0
            w = np.asarray(pz, dtype=np.float64) - p0
            rim_z = float((p0 + w)[2])
            floor_z = float(p0[2])
            if "cab" in receptacle_body:
                v_xy = v[:2]
                front_out = -v_xy / max(float(np.linalg.norm(v_xy)), 1e-6)
                place_xy = (p0 + 0.50 * u + 0.28 * v)[:2]
                shelf_eef_z = floor_z + 0.035 + object_bottom_offset_m
                preplace_eef = np.array(
                    [place_xy[0] + 0.26 * front_out[0], place_xy[1] + 0.26 * front_out[1], shelf_eef_z + 0.02],
                    dtype=np.float64,
                )
                place_eef = np.array([place_xy[0], place_xy[1], shelf_eef_z], dtype=np.float64)
                print(
                    f"cab_interior floor_z={floor_z:.3f} rim_z={rim_z:.3f} "
                    f"place_xy={np.round(place_xy, 3).tolist()} front_out={np.round(front_out, 3).tolist()}",
                    flush=True,
                )
                return place_xy, preplace_eef, place_eef, front_out

            place_xy = (p0 + 0.50 * u + 0.35 * v)[:2]
            place_eef_z = max(floor_z + 0.05, rim_z - 0.01) + object_bottom_offset_m
            preplace_eef_z = rim_z + max(place_height_m, 0.14) + object_bottom_offset_m
            return (
                place_xy,
                np.array([place_xy[0], place_xy[1], preplace_eef_z], dtype=np.float64),
                np.array([place_xy[0], place_xy[1], place_eef_z], dtype=np.float64),
                None,
            )

    fallback_pos, _ = _body_pose(env, receptacle_body)
    if "sink" in receptacle_body and hasattr(env, "sink") and hasattr(env.sink, "pos"):
        fallback_pos = np.asarray(env.sink.pos, dtype=np.float64).reshape(3)
    place_xy = fallback_pos[:2].copy()
    place_eef_z = float(fallback_pos[2] + 0.06 + object_bottom_offset_m)
    preplace_eef_z = float(fallback_pos[2] + max(place_height_m, 0.16) + object_bottom_offset_m)
    return (
        place_xy,
        np.array([place_xy[0], place_xy[1], preplace_eef_z], dtype=np.float64),
        np.array([place_xy[0], place_xy[1], place_eef_z], dtype=np.float64),
        None,
    )


def _close_gate(
    wbc: WbcGymEnv,
    backend: RoboCasaPandaBackend,
    *,
    object_body: str,
    action_source,
    force_n: float,
    hold_s: float,
    timeout_s: float,
) -> tuple[bool, float, float, float]:
    stable = 0.0
    left = right = 0.0
    limit = int(round(timeout_s / wbc.control_dt))
    for _ in range(limit):
        _, _, terminated, truncated, _ = wbc.step(action_source())
        left, right = backend.finger_target_forces(object_body)
        if left > force_n and right > force_n:
            stable += wbc.control_dt
        else:
            stable = 0.0
        if stable >= hold_s or terminated or truncated:
            break
    return stable >= hold_s, left, right, stable


def _pose_from_T(T: np.ndarray) -> tuple[float, ...]:
    return _pose(T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat())


def _pose(position: np.ndarray, quat: np.ndarray) -> tuple[float, ...]:
    return tuple(float(v) for v in np.r_[position, quat])


def _ramp_ee(
    wbc: WbcGymEnv,
    backend: RoboCasaPandaBackend,
    *,
    action_source,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    quat: np.ndarray,
    start_quat: np.ndarray | None = None,
    speed_m_s: float,
    solve_ik: bool = False,
) -> int:
    distance = float(np.linalg.norm(end_pos - start_pos))
    ang_dist = 0.0
    slerp = None
    if start_quat is not None:
        r0 = Rotation.from_quat(start_quat)
        r1 = Rotation.from_quat(quat)
        ang_dist = float((r1 * r0.inv()).magnitude())
        if ang_dist > 1e-3:
            slerp = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
    duration_s = max(distance / max(speed_m_s, 1e-3), ang_dist / 0.8)
    steps = max(1, int(np.ceil(duration_s / wbc.control_dt)))
    for index in range(1, steps + 1):
        alpha = 0.5 - 0.5 * float(np.cos(np.pi * index / steps))
        position = start_pos + (end_pos - start_pos) * alpha
        q_step = slerp([alpha])[0].as_quat() if slerp is not None else quat
        backend.set_ee_target(_pose(position, q_step), solve_ik=solve_ik)
        _, _, terminated, truncated, _ = wbc.step(action_source())
        if terminated or truncated:
            return index
    return steps


def _run_base(
    wbc: WbcGymEnv,
    backend: RoboCasaPandaBackend,
    *,
    env: Any,
    action_source,
    timeout_s: float,
    label: str,
    tolerance_m: float = 0.035,
    settle_steps: int = 4,
    teleport: bool = False,
    carried_object: str | None = None,
) -> None:
    if teleport and carried_object is None and backend.teleport_base_target():
        print(
            f"{label}: teleported base={np.round(backend._base_pose(), 3).tolist()} "
            f"target={np.round(backend._target_base, 3).tolist()}",
            flush=True,
        )
        return
    steps = 0
    good = 0
    xy_error = float("inf")
    yaw_error = float("inf")
    limit = int(round(timeout_s / wbc.control_dt))
    target = np.asarray(backend._target_base, dtype=np.float64)
    resolved_carried = _resolve_body(env, carried_object) if carried_object is not None else None
    while steps < limit:
        _, _, terminated, truncated, _ = wbc.step(action_source())
        base = np.asarray(backend._base_pose(), dtype=np.float64)
        xy_error = float(np.linalg.norm(base[:2] - target[:2]))
        yaw_error = abs(float(np.arctan2(np.sin(base[2] - target[2]), np.cos(base[2] - target[2]))))
        error = max(xy_error, 0.25 * yaw_error)
        good = good + 1 if error <= tolerance_m else 0
        steps += 1
        if resolved_carried is not None and steps % 20 == 0:
            carried_pos, _ = _body_pose(env, resolved_carried)
            left_n, right_n = backend.finger_target_forces(resolved_carried)
            print(
                f"{label}: step={steps} xy_err={xy_error:.3f} yaw_err={yaw_error:.3f} "
                f"object={np.round(carried_pos, 3).tolist()} left_N={left_n:.2f} right_N={right_n:.2f}",
                flush=True,
            )
        if good >= settle_steps or terminated or truncated:
            break
    print(
        f"{label}: steps={steps} xy_error={xy_error:.4f} yaw_error={yaw_error:.4f} "
        f"base={np.round(backend._base_pose(), 3).tolist()} "
        f"target={np.round(backend._target_base, 3).tolist()}",
        flush=True,
    )


def _run_until(
    wbc: WbcGymEnv,
    backend: RoboCasaPandaBackend,
    *,
    action_source,
    target_tolerance_m: float,
    timeout_s: float,
    label: str,
    settle_steps: int = 3,
) -> dict[str, Any]:
    steps = 0
    good = 0
    last: dict[str, Any] = {}
    limit = int(round(timeout_s / wbc.control_dt))
    while steps < limit:
        _, _, terminated, truncated, info = wbc.step(action_source())
        last = info
        terms = info.get("reward_terms", {})
        error = float(terms.get("pos_err", float("inf")))
        good = good + 1 if error <= target_tolerance_m else 0
        steps += 1
        if good >= settle_steps or terminated or truncated:
            break
    print(
        f"{label}: steps={steps} pos_err={float(last.get('reward_terms', {}).get('pos_err', float('nan'))):.4f} "
        f"quat_err={float(last.get('reward_terms', {}).get('quat_err', float('nan'))):.4f} "
        f"base={np.round(backend._base_pose(), 3).tolist()}",
        flush=True,
    )
    return last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("logs/eval/wbc_multirobot_20261126/pandaomron_robust/best.pt"))
    parser.add_argument("--env-name", default="PickPlaceCounterToSink")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--object-body", default="obj_main")
    parser.add_argument("--receptacle-body", default="sink_main")
    parser.add_argument("--obj-groups", default="all", help="RoboCasa object group for deterministic object selection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-standoff", type=float, default=0.55)
    parser.add_argument("--place-standoff", type=float, default=0.58)
    parser.add_argument("--teacher", action="store_true", help="use the privileged teacher; no checkpoint")
    parser.add_argument("--teleport-base", action="store_true", help="teacher-only: snap unloaded base approach")
    parser.add_argument("--grasp-z-offset", type=float, default=0.02, help="EEF offset above object center in offset mode")
    parser.add_argument(
        "--grasp-mode",
        choices=("auto", "topdown", "side"),
        default="auto",
        help="geometry-aware top-down grasp or tilted grasp",
    )
    parser.add_argument(
        "--grasp-height-mode",
        choices=("geometry", "offset"),
        default="geometry",
        help="choose grasp height/yaw from collision geometry or fixed offset",
    )
    parser.add_argument("--gripper-opening", type=float, default=0.078, help="clear inner pad opening in meters")
    parser.add_argument("--grasp-margin", type=float, default=0.004, help="total finger-to-object clearance margin")
    parser.add_argument("--pregrasp-height", type=float, default=0.12, help="wrist standoff above the grasp")
    parser.add_argument("--grasp-tilt-deg", type=float, default=25.0, help="tilt from vertical in side mode")
    parser.add_argument("--descent-steps", type=int, default=60, help="steps for the final grasp descent")
    parser.add_argument("--grasp-force-n", type=float, default=1.0, help="per-finger force required to accept the grasp")
    parser.add_argument("--grip-hold-squeeze", type=float, default=0.0, help="optional measured-opening squeeze offset")
    parser.add_argument("--bilateral-hold-s", type=float, default=0.9, help="bilateral contact hold before lifting")
    parser.add_argument("--lift-height", type=float, default=0.03, help="verification lift height")
    parser.add_argument("--min-lift-m", type=float, default=0.01, help="object rise required to accept the grasp")
    parser.add_argument("--transport-height", type=float, default=0.14, help="carry height above the grasp")
    parser.add_argument("--carry-speed", type=float, default=0.05, help="bounded EE speed while carrying")
    parser.add_argument("--retract-m", type=float, default=0.22, help="pull the carried object back from the counter")
    parser.add_argument("--transport-base-scale", type=float, default=0.45, help="base speed scale while carrying")
    parser.add_argument("--place-height", type=float, default=0.14, help="clearance above receptacle rim before lowering")
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    import robocasa  # noqa: F401
    import robosuite
    from robosuite import load_part_controller_config

    env = robosuite.make(
        env_name=args.env_name,
        robots="PandaOmron",
        controller_configs=load_part_controller_config(default_controller="OSC_POSE"),
        translucent_robot=False,
        layout_and_style_ids=[[args.layout, args.style]],
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        obj_groups=args.obj_groups,
        seed=args.seed,
    )
    backend = RoboCasaPandaBackend(env, base_backoff=0.45, target_mode="offset")
    horizon_s = max(180.0, args.steps * 0.05)
    wbc = WbcGymEnv(
        backend,
        horizon_s=horizon_s,
        control_dt=0.05,
        success_hold_s=horizon_s,
        collision_force_n=1e9,
        hold_base=False,
    )
    policy = None
    device = args.device
    if not args.teacher:
        trainer, _ = WbcPPO.load(args.checkpoint, pandaomron_sim_profile(), device=args.device)
        policy = trainer.policy
        device = str(trainer.device)

    def action() -> np.ndarray:
        if args.teacher:
            return np.asarray(backend.teacher_action(), dtype=np.float32)
        assert policy is not None
        import torch

        with torch.no_grad():
            return (
                policy.deterministic_action(
                    torch.as_tensor(wbc.encode(backend._observation()), dtype=torch.float32, device=device).unsqueeze(0)
                )[0]
                .cpu()
                .numpy()
            )

    try:
        wbc.reset(seed=args.seed)
        lang = env.get_ep_meta().get("lang", "")
        object_pos, _ = _body_pose(env, args.object_body)
        base_world = backend._world_base_pose()

        # Estimate counter front normal from robot spawn -> counter object direction
        # (spawn faces the counter perpendicularly in RoboCasa).
        spawn_forward = np.array([np.cos(base_world[2]), np.sin(base_world[2])], dtype=np.float64)

        def base_target_for(target_xy: np.ndarray, *, standoff: float) -> np.ndarray:
            current_base = backend._world_base_pose()
            # Blend radial direction with the aisle normal (-spawn_forward) so a
            # recessed fixture like a sink does not pull the pedestal into the counter face.
            radial = np.asarray(current_base[:2], dtype=np.float64) - target_xy
            norm = float(np.linalg.norm(radial))
            radial_unit = radial / norm if norm > 1e-6 else -spawn_forward
            outward = radial_unit - 0.5 * spawn_forward
            outward /= max(float(np.linalg.norm(outward)), 1e-6)
            xy = target_xy + outward * standoff
            yaw = float(np.arctan2(target_xy[1] - xy[1], target_xy[0] - xy[0]))
            return np.array([xy[0], xy[1], yaw], dtype=np.float64)

        # Keep the collision-free ready posture (z_hand ~ 1.16 m) locked in the
        # base frame during unloaded base approach.
        ready_local = backend._observation().arms[0].ee_pose
        backend.set_local_ee_target(ready_local)
        if args.teacher:
            backend._target_q = np.asarray(backend.data.qpos[backend._arm_q], dtype=np.float64).copy()
        backend.hold_ee_in_base_frame(True)
        base_target = base_target_for(object_pos[:2], standoff=args.base_standoff)
        backend.set_world_base_target(tuple(float(v) for v in base_target))
        wbc.hold_base = False
        wbc._observation = backend._observation()
        print(
            f"lang={lang!r} object={args.object_body} pos={np.round(object_pos, 3).tolist()} "
            f"base_world={np.round(base_world, 3).tolist()} base_target={np.round(base_target, 3).tolist()}",
            flush=True,
        )
        _run_base(
            wbc,
            backend,
            env=env,
            action_source=action,
            timeout_s=15.0,
            label="base_approach",
            teleport=args.teleport_base,
        )
        backend.set_base_target(None)
        backend.hold_ee_in_base_frame(False)
        backend.reanchor_origin()
        wbc.hold_base = True
        wbc._observation = backend._observation()

        from emet.controller.task.tamp.grasp_frames import top_down_grasp_T

        grasp_object_pos, _ = _body_pose(env, args.object_body)
        resolved_object = _resolve_body(env, args.object_body)
        points, edges = _body_geometry(env, args.object_body)
        object_low = points.min(axis=0)
        object_high = points.max(axis=0)

        base_world = backend._world_base_pose()
        base_yaw = float(
            np.arctan2(grasp_object_pos[1] - base_world[1], grasp_object_pos[0] - base_world[0])
        )
        if args.grasp_height_mode == "geometry":
            grasp_z, grasp_xy, grasp_yaw, section_width, table = _plan_topdown_grasp(
                points,
                edges,
                base_yaw=base_yaw,
                opening_m=args.gripper_opening,
                margin_m=args.grasp_margin,
            )
        else:
            grasp_xy = grasp_object_pos[:2].copy()
            grasp_z = float(grasp_object_pos[2] + args.grasp_z_offset)
            grasp_yaw = base_yaw + 0.5 * np.pi
            section_width = 0.0
            table = []

        under_cabinet = "cab" in args.receptacle_body
        side = args.grasp_mode == "side"
        if side or under_cabinet:
            tilt = float(np.deg2rad(args.grasp_tilt_deg if side else 22.0))
            forward_vec = np.array([np.cos(base_yaw), np.sin(base_yaw), 0.0], dtype=np.float64)
            lateral_vec = np.array([-np.sin(base_yaw), np.cos(base_yaw), 0.0], dtype=np.float64)
            down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            approach = np.sin(tilt) * forward_vec + np.cos(tilt) * down
            approach /= max(float(np.linalg.norm(approach)), 1e-9)
            x_axis = np.cross(lateral_vec, approach)
            T_eef_grasp = np.eye(4, dtype=np.float64)
            T_eef_grasp[:3, :3] = np.column_stack([x_axis, lateral_vec, approach])
            T_eef_grasp[:3, 3] = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
        else:
            T_eef_grasp = top_down_grasp_T(
                np.array([grasp_xy[0], grasp_xy[1], grasp_z], dtype=np.float64),
                z_offset_m=0.0,
            )
            T_eef_grasp[:3, :3] = Rotation.from_euler("z", grasp_yaw).as_matrix() @ T_eef_grasp[:3, :3]

        T_hand_eef = np.linalg.inv(_body_T(env, "robot0_right_hand")) @ _body_T(env, "gripper0_right_eef")
        eef_in_hand = np.linalg.inv(T_hand_eef)
        T_hand_grasp = T_eef_grasp @ eef_in_hand
        T_hand_pregrasp = T_hand_grasp.copy()
        effective_pregrasp_h = min(args.pregrasp_height, 0.065) if under_cabinet else args.pregrasp_height
        if side:
            T_hand_pregrasp[:3, 3] -= T_eef_grasp[:3, 2] * effective_pregrasp_h
        else:
            # Always descend straight vertically so open pads straddle without lateral sweep.
            T_hand_pregrasp[2, 3] += effective_pregrasp_h
        grasp_pos = T_hand_grasp[:3, 3].copy()
        grasp_quat = Rotation.from_matrix(T_hand_grasp[:3, :3]).as_quat()
        pregrasp = T_hand_pregrasp[:3, 3].copy()
        print(
            f"grasp_plan object={np.round(grasp_object_pos, 3).tolist()} "
            f"aabb_z={[round(float(object_low[2]), 3), round(float(object_high[2]), 3)]} "
            f"grasp_xy={np.round(grasp_xy, 3).tolist()} z={grasp_z:.4f} "
            f"pinch_width={section_width:.4f} yaw={grasp_yaw:.3f} "
            f"hand_grasp={np.round(grasp_pos, 3).tolist()} "
            f"table={[(round(z, 3), round(w, 3)) for z, w in table[::4]]}",
            flush=True,
        )
        if not under_cabinet and float(object_high[2]) > 1.06:
            backend.set_torso_target(0.32)
            for _ in range(15):
                _, _, terminated, truncated, _ = wbc.step(action())
                if terminated or truncated:
                    break

        backend.set_gripper(True)
        T_hand_start = _body_T(env, "robot0_right_hand")
        _ramp_ee(
            wbc,
            backend,
            action_source=action,
            start_pos=T_hand_start[:3, 3].copy(),
            end_pos=pregrasp,
            quat=grasp_quat,
            start_quat=Rotation.from_matrix(T_hand_start[:3, :3]).as_quat(),
            speed_m_s=0.14,
            solve_ik=args.teacher,
        )
        _run_until(
            wbc,
            backend,
            action_source=action,
            target_tolerance_m=min(args.tolerance, 0.01),
            timeout_s=4.0,
            label="pregrasp",
        )

        pregrasp_object, _ = _body_pose(env, args.object_body)
        for index in range(1, args.descent_steps + 1):
            alpha = 0.5 - 0.5 * float(np.cos(np.pi * index / args.descent_steps))
            position = pregrasp + (grasp_pos - pregrasp) * alpha
            backend.set_ee_target(_pose(position, grasp_quat), solve_ik=args.teacher)
            _, _, terminated, truncated, _ = wbc.step(action())
            if terminated or truncated:
                break
        grasp_object_pos, _ = _body_pose(env, args.object_body)
        print(
            f"descent: object_moved={np.linalg.norm(grasp_object_pos - pregrasp_object):.4f} "
            f"object={np.round(grasp_object_pos, 3).tolist()}",
            flush=True,
        )

        backend.set_gripper(False)
        accepted, left_force, right_force, hold = _close_gate(
            wbc,
            backend,
            object_body=resolved_object,
            action_source=action,
            force_n=args.grasp_force_n,
            hold_s=args.bilateral_hold_s,
            timeout_s=2.5,
        )
        reading = backend._contact_sensor.read(backend.data)
        print(
            f"close state: accepted={accepted} fingers_q={np.round(backend.data.qpos[backend._grip_q], 4).tolist()} "
            f"left_N={left_force:.2f} right_N={right_force:.2f} hold_s={hold:.2f} "
            f"finger_force={np.round(reading.finger_force, 3).tolist()} "
            f"breakdown={ {k: round(v, 3) for k, v in backend.contact_breakdown(resolved_object).items()} }",
            flush=True,
        )
        if not accepted:
            print("GRASP FAILED: no bilateral finger contact on the target", flush=True)
            return 1
        if args.grip_hold_squeeze > 0:
            measured = np.asarray(backend.data.qpos[backend._grip_q], dtype=np.float64).copy()
            hold_q = measured + np.array([-args.grip_hold_squeeze, args.grip_hold_squeeze])
            backend.set_gripper_target((float(hold_q[0]), float(hold_q[1])))

        # Stage 1: vertical verification lift.
        lift_start, _ = _body_pose(env, args.object_body)
        verify_pos = grasp_pos + np.array([0.0, 0.0, args.lift_height])
        _ramp_ee(
            wbc,
            backend,
            action_source=action,
            start_pos=grasp_pos,
            end_pos=verify_pos,
            quat=grasp_quat,
            speed_m_s=0.03,
            solve_ik=args.teacher,
        )
        lift_end, _ = _body_pose(env, args.object_body)
        rise = float(lift_end[2] - lift_start[2])
        print(
            f"lift_check: object_rise={rise:.4f} object={np.round(lift_end, 3).tolist()}",
            flush=True,
        )
        if rise < args.min_lift_m:
            print("GRASP FAILED: object was not retained through the lift", flush=True)
            return 1

        # Stage 2: pull out from any overhead cabinet while rising to carry pose.
        forward_xy = np.array([np.cos(base_yaw), np.sin(base_yaw), 0.0], dtype=np.float64)
        if under_cabinet:
            out_pos = verify_pos - forward_xy * max(args.retract_m, 0.25)
            s1 = _ramp_ee(
                wbc,
                backend,
                action_source=action,
                start_pos=verify_pos,
                end_pos=out_pos,
                quat=grasp_quat,
                speed_m_s=args.carry_speed,
                solve_ik=args.teacher,
            )
            retract_pos = out_pos + np.array([0.0, 0.0, max(0.0, args.transport_height - args.lift_height)])
            s2 = _ramp_ee(
                wbc,
                backend,
                action_source=action,
                start_pos=out_pos,
                end_pos=retract_pos,
                quat=grasp_quat,
                speed_m_s=args.carry_speed,
                solve_ik=args.teacher,
            )
        else:
            high_pos = grasp_pos + np.array([0.0, 0.0, args.transport_height])
            s1 = _ramp_ee(
                wbc,
                backend,
                action_source=action,
                start_pos=verify_pos,
                end_pos=high_pos,
                quat=grasp_quat,
                speed_m_s=args.carry_speed,
                solve_ik=args.teacher,
            )
            retract_pos = high_pos - forward_xy * args.retract_m
            s2 = _ramp_ee(
                wbc,
                backend,
                action_source=action,
                start_pos=high_pos,
                end_pos=retract_pos,
                quat=grasp_quat,
                speed_m_s=args.carry_speed,
                solve_ik=args.teacher,
            )
        carried, _ = _body_pose(env, args.object_body)
        carry_left, carry_right = backend.finger_target_forces(resolved_object)
        print(
            f"lift_to_transport: steps={s1 + s2} object={np.round(carried, 3).tolist()} "
            f"left_N={carry_left:.2f} right_N={carry_right:.2f}",
            flush=True,
        )
        if carry_left < args.grasp_force_n or carry_right < args.grasp_force_n:
            print("GRASP FAILED: object was not retained while retracting to transport", flush=True)
            return 1

        # Stage 3: dynamic physics-integrated base transport (never teleport qpos while carrying!).
        object_bottom_offset = max(0.015, float(grasp_z - object_low[2]))
        place_xy, preplace_eef_xyz, place_eef_xyz, front_out = _receptacle_place_target(
            env,
            args.receptacle_body,
            object_bottom_offset_m=object_bottom_offset,
            place_height_m=args.place_height,
        )
        if front_out is not None:
            backend.set_torso_target(0.34)
            cab_standoff = max(args.place_standoff, 0.65)
            standoff_xy = place_xy + front_out * cab_standoff
            place_base_yaw = float(np.arctan2(place_xy[1] - standoff_xy[1], place_xy[0] - standoff_xy[0]))
            place_base_target = np.array([standoff_xy[0], standoff_xy[1], place_base_yaw], dtype=np.float64)
        else:
            place_base_target = base_target_for(place_xy, standoff=args.place_standoff)
        backend.set_ee_target(_pose(retract_pos, grasp_quat), solve_ik=args.teacher)
        backend.hold_ee_in_base_frame(True)
        backend.set_base_speed_scale(args.transport_base_scale)
        backend.set_world_base_target(tuple(float(v) for v in place_base_target))
        wbc.hold_base = False
        wbc._observation = backend._observation()
        _run_base(
            wbc,
            backend,
            env=env,
            action_source=action,
            timeout_s=25.0,
            label="transport",
            teleport=False,
            carried_object=args.object_body,
        )
        backend.set_base_target(None)
        backend.set_base_speed_scale(1.0)
        backend.hold_ee_in_base_frame(False)
        backend.reanchor_origin()
        wbc.hold_base = True
        wbc._observation = backend._observation()

        carried, _ = _body_pose(env, args.object_body)
        carried_left, carried_right = backend.finger_target_forces(resolved_object)
        print(
            f"transport_check: object={np.round(carried, 3).tolist()} "
            f"left_N={carried_left:.2f} right_N={carried_right:.2f}",
            flush=True,
        )
        if carried_left < 0.5 and carried_right < 0.5:
            print("TRANSPORT FAILED: object slipped out during base transport", flush=True)
            return 1

        # Stage 4: move to preplace, then into place & release.
        T_eef_now = _body_T(env, "gripper0_right_eef")
        T_hand_now = _body_T(env, "robot0_right_hand")
        if front_out is not None:
            # Front-open cabinet: pitch forearm forward 60 deg so hand/forearm enter
            # horizontally through the open cabinet door rather than through the roof.
            into_cab = np.array([-front_out[0], -front_out[1], 0.0], dtype=np.float64)
            lat_cab = np.array([-into_cab[1], into_cab[0], 0.0], dtype=np.float64)
            cab_tilt = float(np.deg2rad(60.0))
            cab_approach = np.sin(cab_tilt) * into_cab + np.cos(cab_tilt) * np.array([0.0, 0.0, -1.0])
            cab_approach /= max(float(np.linalg.norm(cab_approach)), 1e-9)
            cab_x = np.cross(lat_cab, cab_approach)
            T_eef_preplace = np.eye(4, dtype=np.float64)
            T_eef_preplace[:3, :3] = np.column_stack([cab_x, lat_cab, cab_approach])
            T_eef_preplace[:3, 3] = preplace_eef_xyz
        else:
            eef_x_horiz = np.array([T_eef_now[0, 0], T_eef_now[1, 0]], dtype=np.float64)
            place_yaw = (
                float(np.arctan2(eef_x_horiz[1], eef_x_horiz[0]))
                if float(np.linalg.norm(eef_x_horiz)) > 1e-4
                else float(backend._world_base_pose()[2] + 0.5 * np.pi)
            )
            T_eef_preplace = top_down_grasp_T(preplace_eef_xyz, z_offset_m=0.0)
            T_eef_preplace[:3, :3] = Rotation.from_euler("z", place_yaw).as_matrix() @ T_eef_preplace[:3, :3]
        T_eef_place = T_eef_preplace.copy()
        T_eef_place[:3, 3] = place_eef_xyz

        T_hand_preplace = T_eef_preplace @ eef_in_hand
        T_hand_place = T_eef_place @ eef_in_hand
        preplace_hand_pos = T_hand_preplace[:3, 3].copy()
        if front_out is None:
            preplace_hand_pos[2] = max(preplace_hand_pos[2], float(T_hand_now[2, 3]))
        place_hand_pos = T_hand_place[:3, 3].copy()
        place_quat = Rotation.from_matrix(T_hand_place[:3, :3]).as_quat()

        _ramp_ee(
            wbc,
            backend,
            action_source=action,
            start_pos=T_hand_now[:3, 3].copy(),
            end_pos=preplace_hand_pos,
            quat=place_quat,
            start_quat=Rotation.from_matrix(T_hand_now[:3, :3]).as_quat(),
            speed_m_s=args.carry_speed,
            solve_ik=args.teacher,
        )
        _run_until(
            wbc,
            backend,
            action_source=action,
            target_tolerance_m=0.03,
            timeout_s=3.0,
            label="preplace",
        )
        _ramp_ee(
            wbc,
            backend,
            action_source=action,
            start_pos=preplace_hand_pos,
            end_pos=place_hand_pos,
            quat=place_quat,
            speed_m_s=args.carry_speed,
            solve_ik=args.teacher,
        )
        _run_until(
            wbc,
            backend,
            action_source=action,
            target_tolerance_m=0.03,
            timeout_s=3.0,
            label="place",
        )

        # Stage 5: release and retreat >25 cm from the placed object for _check_success().
        backend.set_gripper(True)
        for _ in range(15):
            _, _, terminated, truncated, _ = wbc.step(action())
            if terminated or truncated:
                break
        print("release: opened gripper", flush=True)

        if front_out is not None:
            retreat_hand_pos = place_hand_pos + np.array(
                [0.30 * front_out[0], 0.30 * front_out[1], 0.02], dtype=np.float64
            )
        else:
            arrived_base = backend._world_base_pose()
            away_from_place = np.array(
                [np.cos(arrived_base[2]), np.sin(arrived_base[2]), 0.0], dtype=np.float64
            )
            retreat_hand_pos = preplace_hand_pos + np.array([0.0, 0.0, 0.14]) - away_from_place * 0.22
        _ramp_ee(
            wbc,
            backend,
            action_source=action,
            start_pos=place_hand_pos,
            end_pos=retreat_hand_pos,
            quat=place_quat,
            speed_m_s=0.08,
            solve_ik=args.teacher,
        )
        _run_until(
            wbc,
            backend,
            action_source=action,
            target_tolerance_m=0.05,
            timeout_s=3.0,
            label="retreat",
        )

        final_pos, _ = _body_pose(env, args.object_body)
        displacement = float(np.linalg.norm(final_pos - object_pos))
        placement_error = float(np.linalg.norm(final_pos[:2] - place_xy))
        task_success = bool(env._check_success()) if hasattr(env, "_check_success") else (
            displacement > 0.05 and placement_error < 0.25
        )
        print(
            f"object={args.object_body} start={np.round(object_pos, 3).tolist()} "
            f"final={np.round(final_pos, 3).tolist()} place_xy={np.round(place_xy, 3).tolist()} "
            f"displacement_m={displacement:.4f} receptacle_xy_error_m={placement_error:.4f} "
            f"task_success={task_success}",
            flush=True,
        )
        return 0 if task_success else 1
    finally:
        wbc.close()


if __name__ == "__main__":
    raise SystemExit(main())
