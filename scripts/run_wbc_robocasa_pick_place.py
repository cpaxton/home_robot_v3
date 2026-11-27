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
    parser.add_argument("--base-standoff", type=float, default=0.55)
    parser.add_argument("--teacher", action="store_true", help="use the privileged teacher; no checkpoint")
    parser.add_argument("--teleport-base", action="store_true", help="teacher-only: snap the base to its target")
    parser.add_argument("--grasp-z-offset", type=float, default=0.02, help="wrist target offset below object center")
    parser.add_argument(
        "--grasp-orientation",
        choices=("ready", "topdown"),
        default="ready",
        help="wrist orientation for grasp/place targets",
    )
    parser.add_argument(
        "--grasp-frame",
        choices=("hand", "eef"),
        default="eef",
        help="target the wrist body directly or offset it so the gripper EEF is at the object",
    )
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
        observation, _ = wbc.reset(seed=0)
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
            direction = np.asarray(base_world[:2], dtype=np.float64) - target_xy
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

        if args.grasp_orientation == "topdown":
            from emet.controller.task.tamp.grasp_frames import top_down_grasp_T

            topdown = top_down_grasp_T(object_pos)
            grasp_quat = Rotation.from_matrix(topdown[:3, :3]).as_quat()
        else:
            grasp_quat = tool_quat
        grasp_pos = object_pos + np.array([0.0, 0.0, args.grasp_z_offset])
        eef_offset = np.zeros(3)
        if args.grasp_frame == "eef":
            hand_T = _body_T(env, "robot0_right_hand")
            eef_T = _body_T(env, "gripper0_right_eef")
            eef_in_hand = np.linalg.inv(hand_T) @ eef_T
            eef_offset = Rotation.from_quat(grasp_quat).apply(eef_in_hand[:3, 3])
            grasp_pos = grasp_pos - eef_offset
        finger_axis, finger_center = _finger_axis_center(env)
        lateral = float(np.dot(object_pos - finger_center, finger_axis))
        grasp_pos = grasp_pos + finger_axis * lateral
        print(
            f"grasp_alignment lateral={lateral:.4f} axis={np.round(finger_axis, 3).tolist()} "
            f"object={np.round(object_pos, 3).tolist()}",
            flush=True,
        )
        pregrasp = grasp_pos + np.array([0.0, 0.0, 0.12])
        grasp = grasp_pos
        lift = grasp_pos + np.array([0.0, 0.0, 0.20])
        backend.set_ee_target(_pose(pregrasp, grasp_quat))
        _run_until(wbc, backend, action_source=action, target_tolerance_m=args.tolerance, timeout_s=8.0, label="pregrasp")
        backend.set_ee_target(_pose(grasp, grasp_quat))
        _run_until(wbc, backend, action_source=action, target_tolerance_m=args.tolerance, timeout_s=6.0, label="grasp")
        backend.set_gripper(False)
        _run_until(wbc, backend, action_source=action, target_tolerance_m=0.04, timeout_s=2.0, label="close")
        reading = backend._contact_sensor.read(backend.data)
        print(
            f"close state: fingers={np.round(backend.data.qpos[backend._grip_q], 4).tolist()} "
            f"finger_force={np.round(reading.finger_force, 3).tolist()} "
            f"hand_force={reading.hand_force:.3f} in_contact={reading.contact.in_contact}",
            flush=True,
        )
        _run_lift(
            wbc,
            backend,
            env=env,
            object_body=args.object_body,
            action_source=action,
            start=grasp_pos,
            end=lift,
            quat=grasp_quat,
            speed_m_s=args.lift_speed,
            settle_s=args.lift_settle,
            label="lift",
        )

        # Move the base to the receptacle while holding the arm target.
        place_base = base_target_for(receptacle_pos[:2])
        backend.set_world_base_target(tuple(float(v) for v in place_base))
        backend.set_ee_target(_pose(lift, grasp_quat))
        _run_base(
            wbc, backend, action_source=action, timeout_s=12.0, label="transport", teleport=args.teleport_base
        )

        place = receptacle_pos + np.array([0.0, 0.0, 0.10]) - eef_offset
        backend.set_ee_target(_pose(place, grasp_quat))
        _run_until(wbc, backend, action_source=action, target_tolerance_m=args.tolerance, timeout_s=8.0, label="place")
        backend.set_gripper(True)
        _run_until(wbc, backend, action_source=action, target_tolerance_m=0.05, timeout_s=2.0, label="release")
        backend.set_ee_target(_pose(place + np.array([0.0, 0.0, 0.10]), grasp_quat))
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
