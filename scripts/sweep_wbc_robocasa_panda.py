# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Sweep PandaOmron reach targets in one loaded RoboCasa kitchen scene.

The script loads RoboCasa once, samples a joint-space target around the ready
posture for each episode, and runs either the trained policy or the privileged
teacher. With ``--save-data`` it also writes actor-visited observations and
teacher labels for DAgger/BC fine-tuning. This is a reach-distribution sweep,
not task success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_env import WbcGymEnv
from emet.rl.manipulation.wbc_robocasa_panda import RoboCasaPandaBackend
from emet.rl.manipulation.wbc_training import WbcPPO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep reach targets in a loaded RoboCasa scene")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/eval/wbc_multirobot_20261126/pandaomron/best.pt"),
    )
    parser.add_argument("--env-name", default="PickPlaceCounterToCabinet")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--backoff", type=float, default=0.45)
    parser.add_argument("--target-mode", choices=("joint", "offset"), default="joint")
    parser.add_argument("--target-range", type=float, default=0.15)
    parser.add_argument("--offset-range", type=float, default=0.05)
    parser.add_argument("--z-range", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--teacher", action="store_true", help="run the privileged teacher instead of the policy")
    parser.add_argument("--save-data", action="store_true", help="save observations and teacher labels")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
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
    backend = RoboCasaPandaBackend(
        env,
        base_backoff=args.backoff,
        target_mode=args.target_mode,
        target_range=args.target_range,
    )
    wbc = WbcGymEnv(backend, horizon_s=args.horizon, control_dt=0.05, hold_base=True)
    policy = None
    device = None
    if not args.teacher:
        trainer, _ = WbcPPO.load(args.checkpoint, pandaomron_sim_profile())
        policy = trainer.policy
        device = trainer.device
    observations: list[np.ndarray] = []
    teacher_actions: list[np.ndarray] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    episodes: list[dict] = []
    success_count = 0
    try:
        for episode in range(args.episodes):
            observation, _ = wbc.reset(seed=args.seed + episode)
            if args.target_mode == "offset":
                current = backend._observation().arms[0].ee_pose
                rng = np.random.default_rng(args.seed + episode)
                offset = np.r_[
                    rng.uniform(-args.offset_range, args.offset_range, 2),
                    rng.uniform(0.0, args.z_range),
                ]
                local_target = np.r_[np.asarray(current[:3]) + offset, current[3:]]
                backend.set_local_ee_target(tuple(float(v) for v in local_target))
                backend_observation = backend._observation()
                wbc._observation = backend_observation
                observation = wbc.encode(backend_observation)
            target_q = None if backend._target_q is None else backend._target_q.tolist()
            target_ee = None if backend._target_ee is None else list(backend._target_ee)
            done = False
            success = False
            steps = 0
            collisions = 0
            info: dict = {}
            while not done:
                if args.teacher:
                    action = backend.teacher_action()
                else:
                    with torch.no_grad():
                        action = (
                            policy.deterministic_action(
                                torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                            )[0]
                            .cpu()
                            .numpy()
                        )
                if args.save_data:
                    observations.append(np.asarray(observation, dtype=np.float32).copy())
                    teacher_actions.append(np.asarray(backend.teacher_action(), dtype=np.float32))
                    episode_ids.append(episode)
                    step_ids.append(steps)
                observation, _, terminated, truncated, info = wbc.step(action)
                success |= bool(info.get("success", False))
                collisions += int(bool(info.get("collision", False)))
                steps += 1
                done = bool(terminated or truncated)
            terms = info.get("reward_terms", {})
            record = {
                "episode": episode,
                "success": bool(success),
                "steps": steps,
                "collisions": collisions,
                "pos_err": float(terms.get("pos_err", float("nan"))),
                "quat_err": float(terms.get("quat_err", float("nan"))),
                "target_q": target_q,
                "target_ee": target_ee,
                "final_ee": list(backend._observation().arms[0].ee_pose),
            }
            episodes.append(record)
            success_count += int(success)
            print(
                f"episode={episode} success={success} pos={record['pos_err']:.3f} "
                f"quat={record['quat_err']:.3f} steps={steps}",
                flush=True,
            )
    finally:
        wbc.close()

    pos_errors = np.asarray([record["pos_err"] for record in episodes], dtype=np.float64)
    quat_errors = np.asarray([record["quat_err"] for record in episodes], dtype=np.float64)
    summary = {
        "robot": "pandaomron",
        "env": args.env_name,
        "layout": args.layout,
        "style": args.style,
        "episodes": args.episodes,
        "seed": args.seed,
        "teacher": bool(args.teacher),
        "success_rate": success_count / args.episodes,
        "pos_error_mean_m": float(np.nanmean(pos_errors)),
        "pos_error_p95_m": float(np.nanpercentile(pos_errors, 95)),
        "pos_error_max_m": float(np.nanmax(pos_errors)),
        "quat_error_mean_rad": float(np.nanmean(quat_errors)),
        "quat_error_p95_rad": float(np.nanpercentile(quat_errors, 95)),
        "episodes_detail": episodes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sweep.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.save_data:
        np.savez_compressed(
            args.output / "loaded_data.npz",
            observations=np.asarray(observations, dtype=np.float32),
            teacher_actions=np.asarray(teacher_actions, dtype=np.float32),
            episode_ids=np.asarray(episode_ids, dtype=np.int32),
            step_ids=np.asarray(step_ids, dtype=np.int32),
        )
        (args.output / "loaded_data_meta.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "observations": len(observations),
                    "action_size": int(teacher_actions[0].size) if teacher_actions else 0,
                    "scene": {
                        "env": args.env_name,
                        "robot": "PandaOmron",
                        "layout": args.layout,
                        "style": args.style,
                        "base_backoff": args.backoff,
                    },
                    "target_mode": args.target_mode,
                    "target_range": args.target_range,
                    "offset_range": args.offset_range,
                    "z_range": args.z_range,
                    "horizon_s": args.horizon,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    print(
        f"success={summary['success_rate']:.2f} pos_mean={summary['pos_error_mean_m']:.4f} "
        f"pos_p95={summary['pos_error_p95_m']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
