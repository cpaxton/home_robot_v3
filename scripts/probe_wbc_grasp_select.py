#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Select a top-down grasp height from object geometry and verify retention.

Privileged diagnostic. Reads the target object's mesh, picks a grasp height
where the horizontal cross-section fits the gripper opening, computes the hand
pose from the desired gripper-EEF pose, closes with a bilateral finger-force
gate, and measures object rise during a slow lift.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.torque_adapter import TorqueAdapter
from emet.controller.manipulation.wbc_contract import action_from_normalized_vector
from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.controller.task.tamp.grasp_frames import top_down_grasp_T
from emet.rl.contact import solved_contact_force
from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend

HAND_BODY = "robot0_right_hand"
FINGER_BODIES = ("gripper0_right_leftfinger", "gripper0_right_rightfinger")


def _pose(position: np.ndarray, quat: np.ndarray) -> tuple[float, ...]:
    return tuple(float(v) for v in np.r_[position, quat])


def _body_pos(env, body: str) -> np.ndarray:
    return np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64).copy()


def _body_T(env, body: str) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(env.sim.data.get_body_xmat(body), dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64)
    return T


def _pose_from_T(T: np.ndarray) -> tuple[float, ...]:
    return _pose(T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat())


def _step(backend: RoboCasaPandaBackend, adapter: TorqueAdapter, steps: int) -> None:
    dt = backend.control_dt
    for _ in range(steps):
        normalized = np.asarray(backend.teacher_action(), dtype=np.float32)
        command = adapter.command(action_from_normalized_vector(normalized, backend.profile), backend._observation())
        backend.step(command, dt)


def _object_points(env, body: str) -> np.ndarray:
    model = env.sim.model
    data = env.sim.data
    body_id = int(model.body_name2id(body))
    points = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) != body_id:
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
    return np.concatenate(points, axis=0)


def _cross_section(points: np.ndarray, z: float, half_height: float) -> tuple[float, float]:
    slab = points[np.abs(points[:, 2] - z) <= half_height]
    if len(slab) == 0:
        return 0.0, 0.0
    return float(slab[:, 0].max() - slab[:, 0].min()), float(slab[:, 1].max() - slab[:, 1].min())


def _choose_grasp_z(
    points: np.ndarray,
    *,
    opening_m: float,
    margin_m: float,
    pad_half_height_m: float,
) -> tuple[float, float, list[tuple[float, float]]]:
    high = float(points[:, 2].max())
    low = float(points[:, 2].min())
    candidate_top = high - 0.008
    candidate_bottom = max(low + 0.25 * (high - low), high - 0.06)
    table: list[tuple[float, float]] = []
    best: tuple[float, float] | None = None
    z = candidate_top
    while z >= candidate_bottom - 1e-9:
        width_x, width_y = _cross_section(points, z, pad_half_height_m)
        width = max(width_x, width_y)
        table.append((z, width))
        if width <= opening_m - margin_m:
            return z, width, table
        if best is None or width < best[1]:
            best = (z, width)
        z -= 0.002
    assert best is not None
    return best[0], best[1], table


def _finger_force(backend: RoboCasaPandaBackend, target_body: str) -> tuple[float, float]:
    model = backend.model
    data = backend.data
    target = int(model.body(target_body).id)
    fingers = [int(model.body(name).id) for name in FINGER_BODIES]
    totals = [0.0, 0.0]
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first = int(model.geom_bodyid[contact.geom1])
        second = int(model.geom_bodyid[contact.geom2])
        if target not in (first, second):
            continue
        other = first if second == target else second
        if other not in fingers:
            continue
        normal, _ = solved_contact_force(model, data, index)
        totals[fingers.index(other)] += normal
    return float(totals[0]), float(totals[1])


def _trial(
    env,
    backend: RoboCasaPandaBackend,
    adapter: TorqueAdapter,
    *,
    object_body: str,
    standoff: float,
    seed: int,
    opening_m: float,
    margin_m: float,
    pad_half_height_m: float,
    pregrasp_m: float,
    descent_steps: int,
    lift_m: float,
    settle_s: float,
) -> dict[str, object]:
    backend.reset(seed=seed)
    object_pos = _body_pos(env, object_body)
    if not np.array_equal(object_pos, object_pos):  # NaN guard
        return {"error": "object_pose_nan"}
    base_world = backend._world_base_pose()
    direction = np.asarray(base_world[:2]) - object_pos[:2]
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    target_xy = object_pos[:2] + direction * standoff
    base_yaw = float(np.arctan2(object_pos[1] - target_xy[1], object_pos[0] - target_xy[0]))
    backend.set_world_base_target((float(target_xy[0]), float(target_xy[1]), base_yaw))
    if not backend.teleport_base_target():
        return {"error": "teleport_failed"}

    points = _object_points(env, object_body)
    low = points.min(axis=0)
    high = points.max(axis=0)
    grasp_z, width, table = _choose_grasp_z(
        points,
        opening_m=opening_m,
        margin_m=margin_m,
        pad_half_height_m=pad_half_height_m,
    )
    # Fingers close along the base->object direction, so make the eef X axis lateral.
    yaw = base_yaw + np.pi / 2.0
    T_eef_des = top_down_grasp_T(np.array([object_pos[0], object_pos[1], grasp_z]), z_offset_m=0.0)
    T_eef_des[:3, :3] = Rotation.from_euler("z", yaw).as_matrix() @ T_eef_des[:3, :3]
    T_hand_eef = np.linalg.inv(_body_T(env, HAND_BODY)) @ _body_T(env, "gripper0_right_eef")
    eef_in_hand = np.linalg.inv(T_hand_eef)
    T_hand_des = T_eef_des @ eef_in_hand
    T_hand_pre = T_hand_des.copy()
    T_hand_pre[2, 3] += pregrasp_m

    backend.set_gripper(True)
    backend.set_ee_target(_pose_from_T(T_hand_pre))
    _step(backend, adapter, 50)
    at_pregrasp = _body_pos(env, object_body)
    start_hand = T_hand_pre[:3, 3].copy()
    end_hand = T_hand_des[:3, 3].copy()
    for index in range(descent_steps):
        alpha = (index + 1) / descent_steps
        T_step = T_hand_des.copy()
        T_step[:3, 3] = start_hand + (end_hand - start_hand) * alpha
        backend.set_ee_target(_pose_from_T(T_step))
        _step(backend, adapter, 1)
    at_grasp = _body_pos(env, object_body)

    backend.set_gripper(False)
    settled = 0.0
    left_force = right_force = 0.0
    steps = max(1, int(round(settle_s / backend.control_dt)))
    bilateral_s = 0.0
    for _ in range(steps):
        _step(backend, adapter, 1)
        left_force, right_force = _finger_force(backend, object_body)
        if left_force > 1.0 and right_force > 1.0:
            bilateral_s += backend.control_dt
        else:
            bilateral_s = 0.0
        settled = bilateral_s
    finger_q = np.asarray(backend.data.qpos[backend._grip_q], dtype=np.float64).copy()
    before_lift = _body_pos(env, object_body)
    T_hand_lift = T_hand_des.copy()
    T_hand_lift[2, 3] += lift_m
    for index in range(30):
        alpha = (index + 1) / 30
        T_step = T_hand_lift.copy()
        T_step[:3, 3] = T_hand_des[:3, 3] + np.array([0.0, 0.0, lift_m * alpha])
        backend.set_ee_target(_pose_from_T(T_step))
        _step(backend, adapter, 1)
    after_lift = _body_pos(env, object_body)
    rise = float(after_lift[2] - before_lift[2])
    retained = bool(left_force > 1.0 and right_force > 1.0 and rise > 0.01)
    return {
        "object": np.round(object_pos, 4).tolist(),
        "aabb_low": np.round(low, 4).tolist(),
        "aabb_high": np.round(high, 4).tolist(),
        "grasp_z": float(grasp_z),
        "section_width": float(width),
        "table": [(round(z, 3), round(w, 3)) for z, w in table[::5]],
        "pregrasp_moved": float(np.linalg.norm(at_pregrasp - object_pos)),
        "grasp_moved": float(np.linalg.norm(at_grasp - at_pregrasp)),
        "finger_q": np.round(finger_q, 4).tolist(),
        "left_N": float(left_force),
        "right_N": float(right_force),
        "bilateral_s": float(settled),
        "object_rise": rise,
        "retained": retained,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="PickPlaceCounterToSink")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--obj-groups", default="can")
    parser.add_argument("--object-body", default="obj_main")
    parser.add_argument("--standoffs", default="0.50,0.55")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opening-m", type=float, default=0.063)
    parser.add_argument("--margin-m", type=float, default=0.0)
    parser.add_argument("--pad-half-height-m", type=float, default=0.012)
    parser.add_argument("--pregrasp-m", type=float, default=0.12)
    parser.add_argument("--descent-steps", type=int, default=60)
    parser.add_argument("--lift-m", type=float, default=0.03)
    parser.add_argument("--settle-s", type=float, default=1.0)
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
    backend = RoboCasaPandaBackend(env, base_backoff=0.45)
    adapter = TorqueAdapter(pandaomron_sim_profile())
    backend.reset(seed=args.seed)
    print(f"lang={env.get_ep_meta().get('lang')!r}", flush=True)

    for standoff in (float(value) for value in args.standoffs.split(",") if value.strip()):
        result = _trial(
            env,
            backend,
            adapter,
            object_body=args.object_body,
            standoff=standoff,
            seed=args.seed,
            opening_m=args.opening_m,
            margin_m=args.margin_m,
            pad_half_height_m=args.pad_half_height_m,
            pregrasp_m=args.pregrasp_m,
            descent_steps=args.descent_steps,
            lift_m=args.lift_m,
            settle_s=args.settle_s,
        )
        print({"standoff": standoff, **result}, flush=True)
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
