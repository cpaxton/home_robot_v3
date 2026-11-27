#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Multi-step PandaOmron WBC pick-place harness in a RoboCasa kitchen.

This is the task-target bridge above the reach adapter: a privileged teacher
(or a trained policy) tracks a sequence of world-frame EE targets for approach,
grasp, lift, transport, place, and retreat. Object/receptacle poses are task
metadata for this teacher/ceiling run; the actor still consumes the WBC contract
observation. Do not present this script alone as a no-GT benchmark result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_env import WbcGymEnv
from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend
from emet.rl.manipulation.wbc_training import WbcPPO


def _resolve_body(env: Any, requested: str) -> str:
    names = list(env.sim.model.body_names)
    if requested in names:
        return requested
    tokens = [token for token in requested.replace("_main", "").split("_") if token]
    candidates = [name for name in names if tokens and all(token in name for token in tokens)]
    if not candidates:
        raise ValueError(f"body {requested!r} is not in the RoboCasa scene; known={sorted(names)[:20]}...")
    # Prefer the shortest matching fixture body (and avoid mesh/door children).
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


def _finger_axis_center(env: Any) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(env.sim.data.get_body_xpos("gripper0_right_leftfinger"), dtype=np.float64)
    right = np.asarray(env.sim.data.get_body_xpos("gripper0_right_rightfinger"), dtype=np.float64)
    axis = right - left
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.array([0.0, 1.0, 0.0]), 0.5 * (left + right)
    return axis / norm, 0.5 * (left + right)


def _fixture_pos(env: Any, attr: str) -> np.ndarray | None:
    fixture = getattr(env, attr, None)
    if fixture is None or not hasattr(fixture, "pos"):
        return None
    return np.asarray(fixture.pos, dtype=np.float64).reshape(3)


def _body_points(env: Any, body: str) -> np.ndarray:
    """World-space vertices of the body's collidable geometry."""
    model = env.sim.model
    data = env.sim.data
    body_id = int(model.body_name2id(_resolve_body(env, body)))
    points: list[np.ndarray] = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) != body_id:
            continue
        if int(model.geom_contype[geom]) == 0 and int(model.geom_conaffinity[geom]) == 0:
            continue
        if int(model.geom_type[geom]) == 7:
            mesh = int(model.geom_dataid[geom])
            start = int(model.mesh_vertadr[mesh])
            count = int(model.mesh_vertnum[mesh])
            vertices = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
            rotation = np.asarray(data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
            points.append(vertices @ rotation.T + np.asarray(data.geom_xpos[geom], dtype=np.float64))
        else:
            center = np.asarray(data.geom_xpos[geom], dtype=np.float64)
            radius = float(np.linalg.norm(model.geom_size[geom]))
            points.append(np.array([center - radius, center + radius]))
    if not points:
        raise ValueError(f"body {body!r} has no collidable geometry")
    return np.concatenate(points, axis=0)


def _cross_section(points: np.ndarray, z: float, half_height: float) -> tuple[np.ndarray | None, float]:
    slab = points[np.abs(points[:, 2] - z) <= half_height]
    if len(slab) == 0:
        return None, 0.0
    low = slab[:, :2].min(axis=0)
    high = slab[:, :2].max(axis=0)
    return 0.5 * (low + high), float(np.max(high - low))


def _choose_grasp_z(
    points: np.ndarray,
    *,
    opening_m: float,
    margin_m: float,
    pad_half_height_m: float,
) -> tuple[float, np.ndarray, float, list[tuple[float, float]]]:
    high = float(points[:, 2].max())
    low = float(points[:, 2].min())
    max_width = 0.0
    probe = low
    while probe <= high:
        _, width = _cross_section(points, probe, 0.005)
        max_width = max(max_width, width)
        probe += 0.005
    min_width = max(0.015, 0.5 * max_width)
    # Tall, narrow objects: cap the grasp 5 cm below the top. This keeps the
    # gripper palm clear of the object while reducing the pendulum arm.
    if (high - low) > 1.2 * max_width and max_width <= opening_m - margin_m:
        capped = high - 0.05
        if capped >= low + 0.02:
            center_xy = 0.5 * (points[:, :2].min(axis=0) + points[:, :2].max(axis=0))
            return capped, center_xy, max_width, []
    candidate_top = high - 0.015
    candidate_bottom = max(low + 0.20 * (high - low), high - 0.08)
    center_z = 0.5 * (low + high)
    table: list[tuple[float, float]] = []
    valid: list[tuple[float, np.ndarray, float]] = []
    best: tuple[float, np.ndarray, float] | None = None
    z = candidate_top
    while z >= candidate_bottom - 1e-9:
        center, width = _cross_section(points, z, pad_half_height_m)
        table.append((z, width))
        if center is not None and min_width <= width <= opening_m - margin_m:
            valid.append((z, center, width))
        if center is not None and (
            best is None
            or abs(width - 0.5 * (min_width + opening_m)) < abs(best[2] - 0.5 * (min_width + opening_m))
        ):
            best = (z, center, width)
        z -= 0.002
    if valid:
        chosen = min(valid, key=lambda item: abs(item[0] - center_z))
        return chosen[0], chosen[1], chosen[2], table
    if best is None:
        raise ValueError("could not find a grasp cross-section")
    return best[0], best[1], best[2], table


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


def _run_base(
    wbc: WbcGymEnv,
    backend: RoboCasaPandaBackend,
    *,
    action_source,
    timeout_s: float,
    label: str,
    tolerance_m: float = 0.05,
    settle_steps: int = 3,
    teleport: bool = False,
) -> None:
    if teleport and backend.teleport_base_target():
        print(
            f"{label}: teleported base={np.round(backend._base_pose(), 3).tolist()} "
            f"target={np.round(backend._target_base, 3).tolist()}",
            flush=True,
        )
        return
    steps = 0
    good = 0
    limit = int(round(timeout_s / wbc.control_dt))
    target = np.asarray(backend._target_base, dtype=np.float64)
    while steps < limit:
        wbc.step(action_source())
        base = np.asarray(backend._base_pose(), dtype=np.float64)
        xy_error = float(np.linalg.norm(base[:2] - target[:2]))
        yaw_error = abs(float(np.arctan2(np.sin(base[2] - target[2]), np.cos(base[2] - target[2]))))
        error = max(xy_error, 0.25 * yaw_error)
        good = good + 1 if error <= tolerance_m else 0
        steps += 1
        if good >= settle_steps:
            break
    print(
        f"{label}: steps={steps} xy_error={xy_error:.4f} yaw_error={yaw_error:.4f} "
        f"base={np.round(backend._base_pose(), 3).tolist()} "
        f"target={np.round(backend._target_base, 3).tolist()}",
        flush=True,
    )


def _run_base_ramp(
    wbc: WbcGymEnv,
    backend: RoboCasaPandaBackend,
    *,
    env: Any,
    action_source,
    steps: int,
    label: str,
    hold_T: np.ndarray | None = None,
    object_body: str | None = None,
) -> bool:
    """Smooth kinematic base move while holding a pose in the base frame.

    Holding a world pose makes the arm counter-rotate as the base yaws, which
    swings a carried object. Holding the pose in the base frame keeps the arm
    configuration fixed and lets the object ride with the base.
    """
    target_q = backend.solve_base_target_qpos()
    if target_q is None:
        print(f"{label}: base target solve failed", flush=True)
        return False
    start_q = np.asarray(backend.data.qpos[backend._base_q], dtype=np.float64).copy()
    base_hold = None
    if hold_T is not None:
        base_hold = np.linalg.inv(_body_T(env, "mobilebase0_base")) @ hold_T
    # The kinematic ramp owns the base pose; suppress the teacher's base
    # velocity command so the base actuators do not fight it and shake the
    # carried object.
    saved_base_target = backend._target_base
    backend._target_base = None
    for index in range(1, steps + 1):
        # Smoothstep: zero velocity at both ends so the free object is not
        # stepped relative to the pads at the start of the base motion.
        alpha = 0.5 - 0.5 * np.cos(np.pi * index / steps)
        backend.apply_base_qpos(start_q + (target_q - start_q) * alpha)
        if base_hold is not None:
            backend.set_ee_target(_pose_from_T(_body_T(env, "mobilebase0_base") @ base_hold))
        _, _, terminated, truncated, _ = wbc.step(action_source())
        if object_body is not None and index % 10 == 0:
            carried = _body_pose(env, object_body)[0]
            left, right = backend.finger_target_forces(_resolve_body(env, object_body))
            print(
                f"{label}: step={index} object={np.round(carried, 3).tolist()} "
                f"left_N={left:.2f} right_N={right:.2f} "
                f"grip_q={np.round(backend.data.qpos[backend._grip_q], 4).tolist()}",
                flush=True,
            )
        if terminated or truncated:
            break
    backend.apply_base_qpos(target_q)
    backend._target_base = saved_base_target
    if base_hold is not None:
        backend.set_ee_target(_pose_from_T(_body_T(env, "mobilebase0_base") @ base_hold))
    print(
        f"{label}: ramped base={np.round(backend._base_pose(), 3).tolist()} "
        f"target={np.round(backend._target_base, 3).tolist()}",
        flush=True,
    )
    return True


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
        observation, reward, terminated, truncated, info = wbc.step(action_source())
        last = info
        terms = info.get("reward_terms", {})
        error = float(terms.get("pos_err", float("inf")))
        good = good + 1 if error <= target_tolerance_m else 0
        steps += 1
        if good >= settle_steps:
            break
        if terminated or truncated:
            break
    print(
        f"{label}: steps={steps} pos_err={float(last.get('reward_terms', {}).get('pos_err', float('nan'))):.4f} "
        f"quat_err={float(last.get('reward_terms', {}).get('quat_err', float('nan'))):.4f} "
        f"base={np.round(backend._base_pose(), 3).tolist()}",
        flush=True,
    )
    return last


def _run_lift(
    wbc: WbcGymEnv,
    backend: RoboCasaPandaBackend,
    *,
    env: Any,
    object_body: str,
    action_source,
    start: np.ndarray,
    end: np.ndarray,
    quat: np.ndarray,
    speed_m_s: float,
    settle_s: float,
    label: str = "lift",
) -> dict[str, Any]:
    """Ramp a Cartesian lift with a bounded EE speed, then settle."""
    if speed_m_s <= 0:
        raise ValueError("lift speed must be positive")
    distance = float(np.linalg.norm(end - start))
    steps = max(1, int(np.ceil(distance / speed_m_s / wbc.control_dt)))
    start_object, _ = _body_pose(env, object_body)
    last: dict[str, Any] = {}
    for index in range(1, steps + 1):
        target = start + (end - start) * (index / steps)
        backend.set_ee_target(_pose(target, quat))
        _, _, terminated, truncated, last = wbc.step(action_source())
        if terminated or truncated:
            break
    if settle_s > 0:
        last = _run_until(
            wbc,
            backend,
            action_source=action_source,
            target_tolerance_m=0.03,
            timeout_s=settle_s,
            label=f"{label}_settle",
        )
    final_object, _ = _body_pose(env, object_body)
    rise = float(final_object[2] - start_object[2])
    print(
        f"{label}: steps={steps} object_rise={rise:.4f} "
        f"object={np.round(final_object, 3).tolist()}",
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
    parser.add_argument("--obj-groups", default="all", help="RoboCasa object group for a deterministic object type")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-standoff", type=float, default=0.55)
    parser.add_argument("--teacher", action="store_true", help="use the privileged teacher; no checkpoint")
    parser.add_argument("--teleport-base", action="store_true", help="teacher-only: snap the base to its target")
    parser.add_argument("--grasp-z-offset", type=float, default=0.02, help="wrist target offset below object center")
    parser.add_argument(
        "--grasp-height-mode",
        choices=("geometry", "offset"),
        default="geometry",
        help="choose the grasp height from the object's collision geometry or a fixed offset",
    )
    parser.add_argument("--gripper-opening", type=float, default=0.063, help="usable finger opening in meters")
    parser.add_argument("--grasp-margin", type=float, default=0.0, help="clearance to reserve inside the opening")
    parser.add_argument("--pregrasp-height", type=float, default=0.12, help="wrist standoff above the grasp")
    parser.add_argument("--descent-steps", type=int, default=60, help="steps for the final grasp descent")
    parser.add_argument("--grasp-force-n", type=float, default=1.0, help="per-finger force required to accept the grasp")
    parser.add_argument("--bilateral-hold-s", type=float, default=0.9, help="bilateral contact hold before lifting")
    parser.add_argument("--lift-height", type=float, default=0.03, help="verification lift height")
    parser.add_argument("--min-lift-m", type=float, default=0.01, help="object rise required to accept the grasp")
    parser.add_argument("--transport-height", type=float, default=0.18, help="carry height above the grasp")
    parser.add_argument("--carry-speed", type=float, default=0.04, help="bounded lift speed for the carry raise")
    parser.add_argument("--transport-steps", type=int, default=60, help="kinematic base ramp steps for transport")
    parser.add_argument("--place-height", type=float, default=0.15, help="EEF height above the receptacle base")
    parser.add_argument("--lift-speed", type=float, default=0.02, help="Cartesian lift speed in m/s")
    parser.add_argument("--lift-settle", type=float, default=0.5, help="settle time after the lift ramp")
    parser.add_argument("--steps", type=int, default=1200)
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
    horizon_s = max(120.0, args.steps * 0.05)
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
    else:
        pass

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
        observation, _ = wbc.reset(seed=args.seed)
        object_pos, _ = _body_pose(env, args.object_body)
        receptacle_pos, _ = _body_pose(env, args.receptacle_body)
        fixture_pos = _fixture_pos(env, "sink")
        if fixture_pos is not None:
            receptacle_pos = fixture_pos
        current_pose = backend._observation().arms[0].ee_pose
        _, current_rot = backend._world_pose(current_pose)
        tool_quat = current_rot.as_quat()

        base_world = backend._world_base_pose()

        def base_target_for(target_xy: np.ndarray) -> np.ndarray:
            current_base = backend._world_base_pose()
            direction = np.asarray(current_base[:2], dtype=np.float64) - target_xy
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                direction = np.array([1.0, 0.0])
                norm = 1.0
            xy = target_xy + direction / norm * args.base_standoff
            yaw = float(np.arctan2(target_xy[1] - xy[1], target_xy[0] - xy[0]))
            return np.array([xy[0], xy[1], yaw], dtype=np.float64)

        base_target = base_target_for(object_pos[:2])
        backend.set_world_base_target(tuple(float(v) for v in base_target))
        print(
            f"object={args.object_body} pos={np.round(object_pos, 3).tolist()} "
            f"receptacle={args.receptacle_body} pos={np.round(receptacle_pos, 3).tolist()} "
            f"base_world={np.round(base_world, 3).tolist()} base_target={np.round(base_target, 3).tolist()}",
            flush=True,
        )
        backend.set_ee_target(_pose(np.asarray(current_pose[:3]) + np.array([0.0, 0.0, 0.0]), tool_quat))
        _run_base(
            wbc, backend, action_source=action, timeout_s=10.0, label="base_approach", teleport=args.teleport_base
        )

        from emet.controller.task.tamp.grasp_frames import top_down_grasp_T

        grasp_object_pos, _ = _body_pose(env, args.object_body)
        resolved_object = _resolve_body(env, args.object_body)
        if args.grasp_height_mode == "geometry":
            points = _body_points(env, args.object_body)
            grasp_z, grasp_xy, section_width, table = _choose_grasp_z(
                points,
                opening_m=args.gripper_opening,
                margin_m=args.grasp_margin,
                pad_half_height_m=0.012,
            )
        else:
            grasp_xy = grasp_object_pos[:2]
            grasp_z = float(grasp_object_pos[2] + args.grasp_z_offset)
            section_width = 0.0
            table = []
        base_world = backend._world_base_pose()
        base_yaw = float(
            np.arctan2(grasp_xy[1] - base_world[1], grasp_xy[0] - base_world[0])
        )
        points = _body_points(env, args.object_body)
        object_low = points.min(axis=0)
        object_high = points.max(axis=0)
        object_width = float(np.max(object_high[:2] - object_low[:2]))
        object_height = float(object_high[2] - object_low[2])
        side = args.grasp_mode == "side" or (
            args.grasp_mode == "auto"
            and object_height > 1.2 * object_width
            and object_width <= args.gripper_opening
        )
        if side:
            grasp_xy = 0.5 * (object_low[:2] + object_high[:2])
            grasp_z = float(0.5 * (object_low[2] + object_high[2]))
            approach = np.array([np.cos(base_yaw), np.sin(base_yaw), 0.0], dtype=np.float64)
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            lateral = np.cross(approach, up)
            T_eef_grasp = np.eye(4, dtype=np.float64)
            T_eef_grasp[:3, :3] = np.column_stack([up, lateral, approach])
            T_eef_grasp[:3, 3] = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
            grasp_yaw = base_yaw
        else:
            grasp_yaw = base_yaw + np.pi / 2.0
            T_eef_grasp = top_down_grasp_T(
                np.array([grasp_xy[0], grasp_xy[1], grasp_z], dtype=np.float64),
                z_offset_m=0.0,
            )
            T_eef_grasp[:3, :3] = Rotation.from_euler("z", grasp_yaw).as_matrix() @ T_eef_grasp[:3, :3]
        T_hand_eef = np.linalg.inv(_body_T(env, "robot0_right_hand")) @ _body_T(env, "gripper0_right_eef")
        eef_in_hand = np.linalg.inv(T_hand_eef)
        T_hand_grasp = T_eef_grasp @ eef_in_hand
        T_hand_pregrasp = T_hand_grasp.copy()
        if side:
            T_hand_pregrasp[:3, 3] -= T_eef_grasp[:3, 2] * args.pregrasp_height
        else:
            T_hand_pregrasp[2, 3] += args.pregrasp_height
        grasp_pos = T_hand_grasp[:3, 3].copy()
        grasp_quat = Rotation.from_matrix(T_hand_grasp[:3, :3]).as_quat()
        pregrasp = T_hand_pregrasp[:3, 3].copy()
        print(
            f"grasp_plan object={np.round(grasp_object_pos, 3).tolist()} "
            f"grasp_xy={np.round(grasp_xy, 3).tolist()} z={grasp_z:.4f} "
            f"section_width={section_width:.4f} yaw={grasp_yaw:.3f} side={side} "
            f"hand_grasp={np.round(grasp_pos, 3).tolist()} "
            f"table={[(round(z, 3), round(w, 3)) for z, w in table[::5]]}",
            flush=True,
        )
        backend.set_ee_target(_pose(pregrasp, grasp_quat))
        _run_until(
            wbc,
            backend,
            action_source=action,
            target_tolerance_m=args.tolerance,
            timeout_s=8.0,
            label="pregrasp",
        )

        pregrasp_object, _ = _body_pose(env, args.object_body)
        for index in range(1, args.descent_steps + 1):
            position = pregrasp + (grasp_pos - pregrasp) * (index / args.descent_steps)
            backend.set_ee_target(_pose(position, grasp_quat))
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

        lift_start, _ = _body_pose(env, args.object_body)
        lift_steps = 30
        for index in range(1, lift_steps + 1):
            position = grasp_pos + np.array([0.0, 0.0, args.lift_height * (index / lift_steps)])
            backend.set_ee_target(_pose(position, grasp_quat))
            _, _, terminated, truncated, _ = wbc.step(action())
            if terminated or truncated:
                break
        lift_end, _ = _body_pose(env, args.object_body)
        rise = float(lift_end[2] - lift_start[2])
        print(
            f"lift_check: object_rise={rise:.4f} object={np.round(lift_end, 3).tolist()}",
            flush=True,
        )
        if rise < args.min_lift_m:
            print("GRASP FAILED: object was not retained through the lift", flush=True)
            return 1

        # Raise the carried object to the transport height with a bounded speed.
        carry_steps = max(1, int(round(args.transport_height / args.carry_speed / wbc.control_dt)))
        for index in range(1, carry_steps + 1):
            position = grasp_pos + np.array(
                [0.0, 0.0, args.transport_height * (index / carry_steps)]
            )
            backend.set_ee_target(_pose(position, grasp_quat))
            _, _, terminated, truncated, _ = wbc.step(action())
            if terminated or truncated:
                break
        carried, _ = _body_pose(env, args.object_body)
        carry_left, carry_right = backend.finger_target_forces(resolved_object)
        print(
            f"lift_to_transport: steps={carry_steps} object={np.round(carried, 3).tolist()} "
            f"left_N={carry_left:.2f} right_N={carry_right:.2f}",
            flush=True,
        )
        if carry_left < args.grasp_force_n or carry_right < args.grasp_force_n:
            print("GRASP FAILED: object was not retained while raising to transport", flush=True)
            return 1
        lift = grasp_pos + np.array([0.0, 0.0, args.transport_height])
        T_hand_transport = T_hand_grasp.copy()
        T_hand_transport[2, 3] += args.transport_height

        # Move the base to the receptacle while holding the arm target.
        place_base = base_target_for(receptacle_pos[:2])
        backend.set_world_base_target(tuple(float(v) for v in place_base))
        backend.set_ee_target(_pose(lift, grasp_quat))
        if args.teleport_base:
            _run_base_ramp(
                wbc,
                backend,
                env=env,
                action_source=action,
                steps=args.transport_steps,
                label="transport",
                hold_T=T_hand_transport,
                object_body=args.object_body,
            )
        else:
            _run_base(
                wbc, backend, action_source=action, timeout_s=12.0, label="transport", teleport=False
            )
        carried, _ = _body_pose(env, args.object_body)
        carried_left, carried_right = backend.finger_target_forces(resolved_object)
        print(
            f"transport_check: object={np.round(carried, 3).tolist()} "
            f"left_N={carried_left:.2f} right_N={carried_right:.2f}",
            flush=True,
        )

        place_base = backend._world_base_pose()
        place_base_yaw = float(
            np.arctan2(receptacle_pos[1] - place_base[1], receptacle_pos[0] - place_base[0])
        )
        place_yaw = place_base_yaw + np.pi / 2.0
        T_eef_place = top_down_grasp_T(
            np.array([receptacle_pos[0], receptacle_pos[1], receptacle_pos[2] + args.place_height]),
            z_offset_m=0.0,
        )
        T_eef_place[:3, :3] = Rotation.from_euler("z", place_yaw).as_matrix() @ T_eef_place[:3, :3]
        T_hand_place = T_eef_place @ eef_in_hand
        place = T_hand_place[:3, 3].copy()
        place_quat = Rotation.from_matrix(T_hand_place[:3, :3]).as_quat()
        backend.set_ee_target(_pose(place, place_quat))
        _run_until(wbc, backend, action_source=action, target_tolerance_m=args.tolerance, timeout_s=8.0, label="place")
        backend.set_gripper(True)
        _run_until(wbc, backend, action_source=action, target_tolerance_m=0.05, timeout_s=2.0, label="release")
        backend.set_ee_target(_pose(place + np.array([0.0, 0.0, 0.10]), place_quat))
        _run_until(wbc, backend, action_source=action, target_tolerance_m=0.05, timeout_s=4.0, label="retreat")

        final_pos, _ = _body_pose(env, args.object_body)
        displacement = float(np.linalg.norm(final_pos - object_pos))
        placement_error = float(np.linalg.norm(final_pos - receptacle_pos))
        print(
            f"object={args.object_body} start={np.round(object_pos, 3).tolist()} "
            f"final={np.round(final_pos, 3).tolist()} displacement_m={displacement:.4f} "
            f"receptacle_error_m={placement_error:.4f}",
            flush=True,
        )
        return 0 if displacement > 0.05 and placement_error < 0.35 else 1
    finally:
        wbc.close()


if __name__ == "__main__":
    raise SystemExit(main())
