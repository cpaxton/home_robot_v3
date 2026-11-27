#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Select a top-down grasp pose from collision geometry and verify retention.

Privileged diagnostic. Slices the target object's collidable mesh/primitive
edges across the finger-pad contact band, chooses the narrowest wrist-reachable
grasp yaw and counter/palm-safe height, closes with a bilateral finger-force
gate, and measures object rise during a slow lift.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_wbc_robocasa_pick_place import (  # noqa: E402
    _body_geometry,
    _body_T,
    _plan_topdown_grasp,
    _pose,
    _pose_from_T,
)

from emet.controller.manipulation.torque_adapter import TorqueAdapter  # noqa: E402
from emet.controller.manipulation.wbc_contract import action_from_normalized_vector  # noqa: E402
from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile  # noqa: E402
from emet.controller.task.tamp.grasp_frames import top_down_grasp_T  # noqa: E402
from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend  # noqa: E402

HAND_BODY = "robot0_right_hand"


def _body_pos(env, body: str) -> np.ndarray:
    return np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64).copy()


def _step(backend: RoboCasaPandaBackend, adapter: TorqueAdapter, steps: int) -> None:
    dt = backend.control_dt
    for _ in range(steps):
        normalized = np.asarray(backend.teacher_action(), dtype=np.float32)
        command = adapter.command(action_from_normalized_vector(normalized, backend.profile), backend._observation())
        backend.step(command, dt)


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
    pregrasp_m: float,
    descent_steps: int,
    lift_m: float,
    settle_s: float,
    reset_state: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None,
) -> dict[str, object]:
    if reset_state is None:
        backend.reset(seed=seed)
    else:
        qpos, qvel, origin_xy, origin_yaw = reset_state
        backend.data.qpos[:] = qpos
        backend.data.qvel[:] = qvel
        backend._origin_xy = np.asarray(origin_xy, dtype=np.float64).copy()
        backend._origin_yaw = float(origin_yaw)
        env.sim.forward()

    object_pos = _body_pos(env, object_body)
    base_world = backend._world_base_pose()
    direction = np.asarray(base_world[:2]) - object_pos[:2]
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    target_xy = object_pos[:2] + direction * standoff
    base_yaw = float(np.arctan2(object_pos[1] - target_xy[1], object_pos[0] - target_xy[0]))
    backend.set_world_base_target((float(target_xy[0]), float(target_xy[1]), base_yaw))
    if not backend.teleport_base_target():
        return {"error": "teleport_failed"}
    backend.set_base_target(None)

    points, edges = _body_geometry(env, object_body)
    low = points.min(axis=0)
    high = points.max(axis=0)
    grasp_z, grasp_center, yaw, width, table = _plan_topdown_grasp(
        points,
        edges,
        base_yaw=base_yaw,
        opening_m=opening_m,
        margin_m=margin_m,
    )
    T_eef_des = top_down_grasp_T(np.array([grasp_center[0], grasp_center[1], grasp_z]), z_offset_m=0.0)
    T_eef_des[:3, :3] = Rotation.from_euler("z", yaw).as_matrix() @ T_eef_des[:3, :3]
    T_hand_eef = np.linalg.inv(_body_T(env, HAND_BODY)) @ _body_T(env, "gripper0_right_eef")
    eef_in_hand = np.linalg.inv(T_hand_eef)
    T_hand_des = T_eef_des @ eef_in_hand
    T_hand_pre = T_hand_des.copy()
    T_hand_pre[2, 3] += pregrasp_m
    quat = Rotation.from_matrix(T_hand_des[:3, :3]).as_quat()

    backend.set_gripper(True)
    backend.set_ee_target(_pose_from_T(T_hand_pre), solve_ik=True)
    _step(backend, adapter, 50)
    start_hand = T_hand_pre[:3, 3].copy()
    end_hand = T_hand_des[:3, 3].copy()

    at_pregrasp = _body_pos(env, object_body)
    for index in range(descent_steps):
        alpha = 0.5 - 0.5 * float(np.cos(np.pi * (index + 1) / descent_steps))
        backend.set_ee_target(_pose(start_hand + (end_hand - start_hand) * alpha, quat), solve_ik=True)
        _step(backend, adapter, 1)
    at_grasp = _body_pos(env, object_body)

    backend.set_gripper(False)
    steps = max(1, int(round(settle_s / backend.control_dt)))
    bilateral_s = 0.0
    left_force = right_force = 0.0
    for _ in range(steps):
        _step(backend, adapter, 1)
        left_force, right_force = backend.finger_target_forces(object_body)
        bilateral_s = bilateral_s + backend.control_dt if (left_force > 1.0 and right_force > 1.0) else 0.0
    finger_q = np.asarray(backend.data.qpos[backend._grip_q], dtype=np.float64).copy()
    before_lift = _body_pos(env, object_body)
    for index in range(30):
        alpha = 0.5 - 0.5 * float(np.cos(np.pi * (index + 1) / 30))
        backend.set_ee_target(_pose(end_hand + np.array([0.0, 0.0, lift_m * alpha]), quat), solve_ik=True)
        _step(backend, adapter, 1)
    after_lift = _body_pos(env, object_body)
    rise = float(after_lift[2] - before_lift[2])
    return {
        "object": np.round(object_pos, 4).tolist(),
        "aabb_low": np.round(low, 4).tolist(),
        "aabb_high": np.round(high, 4).tolist(),
        "grasp_z": float(grasp_z),
        "grasp_center": np.round(grasp_center, 4).tolist(),
        "grasp_yaw": float(yaw),
        "pinch_width": float(width),
        "table": [(round(z, 3), round(w, 3)) for z, w in table[::4]],
        "pregrasp_moved": float(np.linalg.norm(at_pregrasp - object_pos)),
        "grasp_moved": float(np.linalg.norm(at_grasp - at_pregrasp)),
        "finger_q": np.round(finger_q, 4).tolist(),
        "left_N": float(left_force),
        "right_N": float(right_force),
        "bilateral_s": float(bilateral_s),
        "object_rise": rise,
        "retained": bool(left_force > 1.0 and right_force > 1.0 and rise > 0.01),
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
    parser.add_argument("--opening-m", type=float, default=0.078)
    parser.add_argument("--margin-m", type=float, default=0.004)
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
    reset_state = (
        backend.data.qpos.copy(),
        backend.data.qvel.copy(),
        np.asarray(backend._origin_xy, dtype=np.float64).copy(),
        float(backend._origin_yaw),
    )
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
            pregrasp_m=args.pregrasp_m,
            descent_steps=args.descent_steps,
            lift_m=args.lift_m,
            settle_s=args.settle_s,
            reset_state=reset_state,
        )
        print({"standoff": standoff, **result}, flush=True)
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
