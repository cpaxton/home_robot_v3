#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Compare a naive top-down wrist target against the model-derived hand frame.

Privileged diagnostic. `top_down_grasp_T` describes the gripper EEF frame, but
the WBC backend commands `robot0_right_hand`. The two frames differ by a fixed
rotation and translation. This probe prints that transform and measures
pregrasp/descend/lift tracking for each convention.
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


def _ee_world(backend: RoboCasaPandaBackend) -> np.ndarray:
    ee = np.asarray(backend._observation().arms[0].ee_pose, dtype=np.float64)
    position, _ = backend._world_pose(tuple(ee))
    return position


def _eef_world(env) -> tuple[np.ndarray, np.ndarray]:
    T = _body_T(env, "gripper0_right_eef")
    return T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()


def _pad_positions(env) -> tuple[np.ndarray, np.ndarray]:
    pads = [g for g in range(env.sim.model.ngeom) if "pad_collision" in (env.sim.model.geom_id2name(g) or "")]
    centers = [np.asarray(env.sim.data.geom_xpos[g], dtype=np.float64) for g in pads]
    if len(centers) != 2:
        return np.zeros(3), np.zeros(3)
    return centers[0], centers[1]


def _trial(
    env,
    backend: RoboCasaPandaBackend,
    adapter: TorqueAdapter,
    *,
    object_body: str,
    standoff: float,
    convention: str,
    descent_steps: int,
    pregrasp_m: float,
    lift_m: float,
) -> dict[str, float | bool | str]:
    backend.reset(seed=0)
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
        return {"convention": convention, "standoff": standoff, "error": "teleport_failed"}

    T_hand_eef = np.linalg.inv(_body_T(env, "robot0_right_hand")) @ _body_T(env, "gripper0_right_eef")
    eef_in_hand = np.linalg.inv(T_hand_eef)
    if convention == "naive":
        T_eef_des = top_down_grasp_T(object_pos)
        T_hand_des = T_eef_des
    else:
        T_eef_des = top_down_grasp_T(object_pos)
        T_hand_des = T_eef_des @ eef_in_hand
    T_eef_pre = T_eef_des.copy()
    T_eef_pre[2, 3] += pregrasp_m
    T_hand_pre = T_eef_pre @ eef_in_hand if convention == "hand" else T_eef_pre

    backend.set_gripper(True)
    backend.set_ee_target(_pose_from_T(T_hand_pre))
    _step(backend, adapter, 40)
    pregrasp_object = _body_pos(env, object_body)
    eef_pos, eef_quat = _eef_world(env)
    eef_error = Rotation.from_matrix(T_eef_des[:3, :3]).inv() * Rotation.from_quat(eef_quat)
    left_pad, right_pad = _pad_positions(env)
    pregrasp = {
        "object_moved": float(np.linalg.norm(pregrasp_object - object_pos)),
        "eef_error_m": float(np.linalg.norm(eef_pos - T_eef_des[:3, 3])),
        "eef_angle_deg": float(np.linalg.norm(eef_error.as_rotvec()) * 180.0 / np.pi),
        "pad_center": 0.5 * (left_pad + right_pad),
        "pad_gap": float(np.linalg.norm(right_pad - left_pad)),
        "object_z": float(pregrasp_object[2]),
    }

    # Descend linearly to the grasp pose, then close and lift.
    start_hand = T_hand_pre[:3, 3].copy()
    end_hand = T_hand_des[:3, 3].copy()
    start_object = _body_pos(env, object_body)
    for index in range(descent_steps):
        alpha = (index + 1) / descent_steps
        position = start_hand + (end_hand - start_hand) * alpha
        base_T = T_hand_des.copy()
        base_T[:3, 3] = position
        backend.set_ee_target(_pose_from_T(base_T))
        _step(backend, adapter, 1)
    descent_object = _body_pos(env, object_body)
    eef_pos, eef_quat = _eef_world(env)
    left_pad, right_pad = _pad_positions(env)
    descent = {
        "object_moved": float(np.linalg.norm(descent_object - start_object)),
        "eef_error_m": float(np.linalg.norm(eef_pos - T_eef_des[:3, 3])),
        "eef_angle_deg": float(
            np.linalg.norm((Rotation.from_matrix(T_eef_des[:3, :3]).inv() * Rotation.from_quat(eef_quat)).as_rotvec())
            * 180.0
            / np.pi
        ),
        "pad_center": 0.5 * (left_pad + right_pad),
        "pad_gap": float(np.linalg.norm(right_pad - left_pad)),
    }

    backend.set_gripper(False)
    _step(backend, adapter, 10)
    finger_q = np.asarray(backend.data.qpos[backend._grip_q], dtype=np.float64).copy()
    breakdown = backend.contact_breakdown(object_body)
    before_lift = _body_pos(env, object_body)
    T_hand_lift = T_hand_des.copy()
    T_hand_lift[2, 3] += lift_m
    backend.set_ee_target(_pose_from_T(T_hand_lift))
    _step(backend, adapter, 20)
    after_lift = _body_pos(env, object_body)
    return {
        "convention": convention,
        "standoff": standoff,
        "object_start": object_pos,
        "pregrasp": pregrasp,
        "descent": descent,
        "finger_q": finger_q,
        "finger_target_N": float(breakdown["finger_target_N"]),
        "finger_environment_N": float(breakdown["finger_environment_N"]),
        "object_rise": float(after_lift[2] - before_lift[2]),
        "retained": bool(breakdown["finger_target_N"] > 1.0 and (after_lift[2] - before_lift[2]) > 0.01),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="PickPlaceCounterToSink")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--object-body", default="obj_main")
    parser.add_argument("--standoffs", default="0.45,0.55,0.65")
    parser.add_argument("--descent-steps", type=int, default=60)
    parser.add_argument("--pregrasp-m", type=float, default=0.12)
    parser.add_argument("--lift-m", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
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
    backend.reset(seed=args.seed)
    T_hand_eef = np.linalg.inv(_body_T(env, "robot0_right_hand")) @ _body_T(env, "gripper0_right_eef")
    print(f"lang={env.get_ep_meta().get('lang')!r}", flush=True)
    print(
        f"hand->eef rotation=\n{np.round(T_hand_eef[:3, :3], 4)}\n"
        f"hand->eef translation={np.round(T_hand_eef[:3, 3], 4).tolist()}",
        flush=True,
    )

    results = []
    for convention in ("naive", "hand"):
        for standoff in (float(value) for value in args.standoffs.split(",") if value.strip()):
            result = _trial(
                env,
                backend,
                adapter,
                object_body=args.object_body,
                standoff=standoff,
                convention=convention,
                descent_steps=args.descent_steps,
                pregrasp_m=args.pregrasp_m,
                lift_m=args.lift_m,
            )
            results.append(result)
            pregrasp = result.get("pregrasp")
            descent = result.get("descent")
            if not isinstance(pregrasp, dict) or not isinstance(descent, dict):
                print(f"{convention:5s} base={standoff:.2f} {result}", flush=True)
                continue
            print(
                f"{convention:5s} base={standoff:.2f} "
                f"pre[eef_err={pregrasp['eef_error_m']:.3f} angle={pregrasp['eef_angle_deg']:.1f} "
                f"moved={pregrasp['object_moved']:.3f} gap={pregrasp['pad_gap']:.3f}] "
                f"desc[eef_err={descent['eef_error_m']:.3f} angle={descent['eef_angle_deg']:.1f} "
                f"moved={descent['object_moved']:.3f} gap={descent['pad_gap']:.3f} "
                f"pad={np.round(descent['pad_center'], 3).tolist()}] "
                f"close[target_N={result['finger_target_N']:.2f} env_N={result['finger_environment_N']:.2f} "
                f"fingers={np.round(result['finger_q'], 3).tolist()}] "
                f"rise={result['object_rise']:+.4f} retained={result['retained']}",
                flush=True,
            )
    retained = [result for result in results if result.get("retained")]
    print(f"retained={len(retained)}/{len(results)}", flush=True)
    backend.close()
    return 0 if retained else 1


if __name__ == "__main__":
    raise SystemExit(main())
