#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Check whether the top-down grasp pose is kinematically reachable.

Privileged diagnostic. Solves damped-least-squares IK for the hand pose that
puts the gripper EEF at the object, then (optionally) drives the arm in joint
space and measures finger force plus object rise.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.torque_adapter import TorqueAdapter
from emet.controller.manipulation.wbc_contract import action_from_normalized_vector
from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.controller.task.tamp.grasp_frames import top_down_grasp_T
from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend

HAND_BODY = "robot0_right_hand"


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


def _top_down_T(object_pos: np.ndarray, z_offset: float, yaw: float) -> np.ndarray:
    T = top_down_grasp_T(object_pos + np.array([0.0, 0.0, z_offset]))
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix() @ T[:3, :3]
    return T


def _step(backend: RoboCasaPandaBackend, adapter: TorqueAdapter, steps: int) -> None:
    dt = backend.control_dt
    for _ in range(steps):
        normalized = np.asarray(backend.teacher_action(), dtype=np.float32)
        command = adapter.command(action_from_normalized_vector(normalized, backend.profile), backend._observation())
        backend.step(command, dt)


def _pose_error(env, body: str, target_T: np.ndarray) -> tuple[float, float]:
    current_T = _body_T(env, body)
    position = float(np.linalg.norm(target_T[:3, 3] - current_T[:3, 3]))
    rotation = Rotation.from_matrix(target_T[:3, :3]).inv() * Rotation.from_matrix(current_T[:3, :3])
    return position, float(np.linalg.norm(rotation.as_rotvec()))


def _solve_ik(
    backend: RoboCasaPandaBackend,
    env,
    target_T: np.ndarray,
    *,
    iterations: int = 300,
    damping: float = 1e-3,
    step_scale: float = 0.6,
) -> tuple[np.ndarray, float, float]:
    low = np.asarray(backend._arm_low, dtype=np.float64)
    high = np.asarray(backend._arm_high, dtype=np.float64)
    q = np.asarray(backend.data.qpos[backend._arm_q], dtype=np.float64).copy()
    best_q = q.copy()
    best_error = np.inf
    for _ in range(iterations):
        backend.data.qpos[backend._arm_q] = q
        env.sim.forward()
        current_T = _body_T(env, HAND_BODY)
        position_error = target_T[:3, 3] - current_T[:3, 3]
        rotation_error = (
            Rotation.from_matrix(target_T[:3, :3]) * Rotation.from_matrix(current_T[:3, :3]).inv()
        ).as_rotvec()
        error = np.r_[position_error, rotation_error]
        score = float(np.linalg.norm(position_error) + 0.1 * np.linalg.norm(rotation_error))
        if score < best_error:
            best_error = score
            best_q = q.copy()
        if np.linalg.norm(position_error) < 1e-3 and np.linalg.norm(rotation_error) < 0.02:
            break
        jacobian = backend._numerical_jacobian(q)
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * np.eye(6), error
        )
        q = np.clip(q + step_scale * delta, low, high)
    backend.data.qpos[backend._arm_q] = best_q
    env.sim.forward()
    position, angle = _pose_error(env, HAND_BODY, target_T)
    return best_q, position, angle


def _trial(
    env,
    backend: RoboCasaPandaBackend,
    adapter: TorqueAdapter,
    *,
    object_body: str,
    standoff: float,
    z_offset: float,
    yaw_offset_deg: float,
    pregrasp_m: float,
    lift_m: float,
    execute: bool,
    execute_mode: str,
    reset_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
) -> dict[str, object]:
    if reset_state is None:
        backend.reset(seed=0)
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
    backend.set_world_base_target(
        (
            float(target_xy[0]),
            float(target_xy[1]),
            float(np.arctan2(object_pos[1] - target_xy[1], object_pos[0] - target_xy[0])),
        )
    )
    if not backend.teleport_base_target():
        return {"standoff": standoff, "error": "teleport_failed"}
    base_world = backend._world_base_pose()
    q_initial = np.asarray(backend.data.qpos[backend._arm_q], dtype=np.float64).copy()
    T_hand_eef = np.linalg.inv(_body_T(env, "robot0_right_hand")) @ _body_T(env, "gripper0_right_eef")
    eef_in_hand = np.linalg.inv(T_hand_eef)
    base_yaw = float(np.arctan2(object_pos[1] - base_world[1], object_pos[0] - base_world[0]))
    yaw = base_yaw + float(np.deg2rad(yaw_offset_deg))
    T_eef_des = _top_down_T(object_pos, z_offset, yaw)
    T_hand_des = T_eef_des @ eef_in_hand
    T_eef_pre = T_eef_des.copy()
    T_eef_pre[2, 3] += pregrasp_m
    T_hand_pre = T_eef_pre @ eef_in_hand

    q_pre, pre_pos_err, pre_ang_err = _solve_ik(backend, env, T_hand_pre)
    q_grasp, grasp_pos_err, grasp_ang_err = _solve_ik(backend, env, T_hand_des)
    backend.data.qpos[backend._arm_q] = q_initial
    env.sim.forward()
    margins = np.minimum(q_grasp - backend._arm_low, backend._arm_high - q_grasp)
    result: dict[str, object] = {
        "standoff": standoff,
        "z_offset": z_offset,
        "yaw_offset_deg": yaw_offset_deg,
        "base_yaw": base_yaw,
        "base": np.round(base_world, 3).tolist(),
        "object": np.round(object_pos, 3).tolist(),
        "pregrasp_ik_err": [pre_pos_err, pre_ang_err],
        "grasp_ik_err": [grasp_pos_err, grasp_ang_err],
        "min_joint_margin": float(np.min(margins)),
        "q_grasp": np.round(q_grasp, 3).tolist(),
    }
    if not execute or grasp_pos_err > 0.01 or grasp_ang_err > 0.1:
        return result

    backend.set_gripper(True)
    if execute_mode == "joint":
        backend._target_q = q_pre.copy()
        backend._target_ee = None
        _step(backend, adapter, 60)
        start_object = _body_pos(env, object_body)
        at_pregrasp = _body_pos(env, object_body)
        backend._target_q = q_grasp.copy()
        _step(backend, adapter, 80)
    else:
        backend._target_q = None
        backend.set_ee_target(_pose_from_T(T_hand_pre))
        _step(backend, adapter, 60)
        start_object = _body_pos(env, object_body)
        at_pregrasp = _body_pos(env, object_body)
        for index in range(60):
            alpha = (index + 1) / 60
            T_step = T_hand_des.copy()
            T_step[:3, 3] = T_hand_pre[:3, 3] + (T_hand_des[:3, 3] - T_hand_pre[:3, 3]) * alpha
            backend.set_ee_target(_pose_from_T(T_step))
            _step(backend, adapter, 1)
    at_grasp = _body_pos(env, object_body)
    backend.set_gripper(False)
    _step(backend, adapter, 10)
    finger_q = np.asarray(backend.data.qpos[backend._grip_q], dtype=np.float64).copy()
    breakdown = backend.contact_breakdown(object_body)
    before_lift = _body_pos(env, object_body)
    q_before_lift = np.asarray(backend.data.qpos[backend._arm_q], dtype=np.float64).copy()
    T_hand_lift = T_hand_des.copy()
    T_hand_lift[2, 3] += lift_m
    T_eef_lift = T_eef_des.copy()
    T_eef_lift[2, 3] += lift_m
    q_lift, _, _ = _solve_ik(backend, env, T_hand_lift)
    backend.data.qpos[backend._arm_q] = q_before_lift
    env.sim.forward()
    if execute_mode == "joint":
        backend._target_q = q_lift.copy()
        _step(backend, adapter, 30)
    else:
        backend.set_ee_target(_pose_from_T(T_eef_lift))
        _step(backend, adapter, 30)
    after_lift = _body_pos(env, object_body)
    result.update(
        {
            "pregrasp_moved": float(np.linalg.norm(at_pregrasp - start_object)),
            "grasp_moved": float(np.linalg.norm(at_grasp - at_pregrasp)),
            "finger_q": np.round(finger_q, 3).tolist(),
            "finger_target_N": float(breakdown["finger_target_N"]),
            "finger_environment_N": float(breakdown["finger_environment_N"]),
            "object_rise": float(after_lift[2] - before_lift[2]),
            "retained": bool(breakdown["finger_target_N"] > 1.0 and (after_lift[2] - before_lift[2]) > 0.01),
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="PickPlaceCounterToSink")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--object-body", default="obj_main")
    parser.add_argument("--standoffs", default="0.40,0.50,0.55,0.60")
    parser.add_argument("--z-offsets", default="0.03,0.05")
    parser.add_argument("--yaw-offsets-deg", default="0,90,180,270")
    parser.add_argument("--pregrasp-m", type=float, default=0.12)
    parser.add_argument("--lift-m", type=float, default=0.03)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-mode", choices=("cartesian", "joint"), default="cartesian")
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
    )
    backend = RoboCasaPandaBackend(env, base_backoff=0.45)
    adapter = TorqueAdapter(pandaomron_sim_profile())
    backend.reset(seed=0)
    reset_state = (
        backend.data.qpos.copy(),
        backend.data.qvel.copy(),
        np.asarray(backend._origin_xy, dtype=np.float64).copy(),
        float(backend._origin_yaw),
    )
    print(f"lang={env.get_ep_meta().get('lang')!r}", flush=True)

    results = []
    for standoff in (float(value) for value in args.standoffs.split(",") if value.strip()):
        for z_offset in (float(value) for value in args.z_offsets.split(",") if value.strip()):
            for yaw_offset in (float(value) for value in args.yaw_offsets_deg.split(",") if value.strip()):
                result = _trial(
                    env,
                    backend,
                    adapter,
                    object_body=args.object_body,
                    standoff=standoff,
                    z_offset=z_offset,
                    yaw_offset_deg=yaw_offset,
                    pregrasp_m=args.pregrasp_m,
                    lift_m=args.lift_m,
                    execute=args.execute,
                    execute_mode=args.execute_mode,
                    reset_state=reset_state,
                )
                results.append(result)
                print(result, flush=True)
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
