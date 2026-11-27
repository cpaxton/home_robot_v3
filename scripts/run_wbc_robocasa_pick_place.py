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

from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_env import WbcGymEnv
from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend
from emet.rl.manipulation.wbc_training import WbcPPO


def _body_pose(env: Any, body: str) -> tuple[np.ndarray, np.ndarray]:
    model = env.sim.model
    if body not in model.body_names:
        raise ValueError(f"body {body!r} is not in the RoboCasa scene; known={sorted(model.body_names)[:20]}...")
    return (
        np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64).copy(),
        np.asarray(env.sim.data.get_body_xquat(body), dtype=np.float64).copy(),
    )


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
) -> None:
    steps = 0
    good = 0
    limit = int(round(timeout_s / wbc.control_dt))
    target = np.asarray(backend._target_base, dtype=np.float64)
    while steps < limit:
        wbc.step(action_source())
        base = np.asarray(backend._base_pose(), dtype=np.float64)
        error = float(np.linalg.norm(base[:2] - target[:2])) + abs(float(np.arctan2(np.sin(base[2] - target[2]), np.cos(base[2] - target[2]))))
        good = good + 1 if error <= tolerance_m else 0
        steps += 1
        if good >= settle_steps:
            break
    print(f"{label}: steps={steps} base_error={error:.4f} base={np.round(backend._base_pose(), 3).tolist()}", flush=True)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("logs/eval/wbc_multirobot_20261126/pandaomron_robust/best.pt"))
    parser.add_argument("--env-name", default="PickPlaceCounterToSink")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--object-body", default="obj_main")
    parser.add_argument("--receptacle-body", default="sink_main")
    parser.add_argument("--base-standoff", type=float, default=0.65)
    parser.add_argument("--teacher", action="store_true", help="use the privileged teacher; no checkpoint")
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
    wbc = WbcGymEnv(
        backend,
        horizon_s=max(120.0, args.steps * 0.05),
        control_dt=0.05,
        success_hold_s=1e9,
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
        current_pose = backend._observation().arms[0].ee_pose
        _, current_rot = backend._world_pose(current_pose)
        tool_quat = current_rot.as_quat()

        base_world = backend._world_base_pose()
        direction = np.asarray(base_world[:2]) - object_pos[:2]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0])
            norm = 1.0
        base_target_xy = object_pos[:2] + direction / norm * args.base_standoff
        yaw = float(np.arctan2(object_pos[1] - base_target_xy[1], object_pos[0] - base_target_xy[0]))
        backend.set_world_base_target((float(base_target_xy[0]), float(base_target_xy[1]), yaw))
        backend.set_ee_target(_pose(np.asarray(current_pose[:3]) + np.array([0.0, 0.0, 0.0]), tool_quat))
        _run_base(wbc, backend, action_source=action, timeout_s=10.0, label="base_approach")

        pregrasp = object_pos + np.array([0.0, 0.0, 0.12])
        grasp = object_pos + np.array([0.0, 0.0, 0.02])
        lift = object_pos + np.array([0.0, 0.0, 0.20])
        backend.set_ee_target(_pose(pregrasp, tool_quat))
        _run_until(wbc, backend, action_source=action, target_tolerance_m=args.tolerance, timeout_s=8.0, label="pregrasp")
        backend.set_ee_target(_pose(grasp, tool_quat))
        _run_until(wbc, backend, action_source=action, target_tolerance_m=args.tolerance, timeout_s=6.0, label="grasp")
        backend.set_gripper(False)
        _run_until(wbc, backend, action_source=action, target_tolerance_m=0.04, timeout_s=2.0, label="close")
        backend.set_ee_target(_pose(lift, tool_quat))
        _run_until(wbc, backend, action_source=action, target_tolerance_m=args.tolerance, timeout_s=8.0, label="lift")

        # Move the base to the receptacle while holding the arm target.
        direction = np.asarray(base_world[:2]) - receptacle_pos[:2]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0])
            norm = 1.0
        place_base_xy = receptacle_pos[:2] + direction / norm * args.base_standoff
        place_yaw = float(np.arctan2(receptacle_pos[1] - place_base_xy[1], receptacle_pos[0] - place_base_xy[0]))
        backend.set_world_base_target((float(place_base_xy[0]), float(place_base_xy[1]), place_yaw))
        backend.set_ee_target(_pose(lift, tool_quat))
        _run_base(wbc, backend, action_source=action, timeout_s=12.0, label="transport")

        place = receptacle_pos + np.array([0.0, 0.0, 0.12])
        backend.set_ee_target(_pose(place, tool_quat))
        _run_until(wbc, backend, action_source=action, target_tolerance_m=args.tolerance, timeout_s=8.0, label="place")
        backend.set_gripper(True)
        _run_until(wbc, backend, action_source=action, target_tolerance_m=0.05, timeout_s=2.0, label="release")
        backend.set_ee_target(_pose(place + np.array([0.0, 0.0, 0.10]), tool_quat))
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
