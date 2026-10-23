"""Train, evaluate and replay: python -m emet.rl --help."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def evaluate(policy, config, *, episodes=50, seed=10000, replay_path=None, video_path=None, teacher_approach=False):
    import numpy as np

    from emet.rl.base import MarsBaseEnv
    from emet.rl.control import baseline_action
    from emet.rl.env import MarsReachEnv

    if policy is not None and getattr(policy, "task", config.task) != config.task:
        raise ValueError("policy task does not match evaluation task")
    env_class = MarsBaseEnv if config.task == "base" else MarsReachEnv
    env = env_class(config, render_mode="rgb_array" if video_path else None)
    rows, trace, latencies, teacher_action_errors = [], [], [], []
    video = None
    started = time.perf_counter()
    total_steps = 0
    try:
        if video_path:
            import cv2

            video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))
            if not video.isOpened():
                raise RuntimeError("could not open replay video")
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            minimum = float("inf")
            peak_force, impulse, obstruction_steps = 0.0, 0.0, 0
            any_collision = False
            post_event_peak = 0.0
            smoothness, errors = [], []
            dynamics = []
            for _ in range(round(config.horizon_seconds / 0.05) + 1):
                before = time.perf_counter()
                obs, _ = env.control.observe()
                if config.pose_targets:
                    obs = env._with_pose_targets(obs)
                use_teacher = policy is None or (
                    teacher_approach
                    and getattr(env, "_scenario", None) == "contact_retreat"
                    and env.steps <= env._event_step
                )
                action = (
                    (env.control.baseline_action() if config.task == "base" else baseline_action(env.control))
                    if use_teacher
                    else policy(obs)
                )
                if policy is not None:
                    expert = env.control.baseline_action() if config.task == "base" else baseline_action(env.control)
                    active = (
                        np.r_[np.repeat(env.control.goal.mask[:1], 2), env.control.goal.mask[1:]].astype(bool)
                        if config.task == "coordinated"
                        else np.ones_like(expert, dtype=bool)
                    )
                    teacher_action_errors.extend(np.abs(np.asarray(action)[active] - expert[active]).tolist())
                latencies.append(time.perf_counter() - before)
                obs, reward, terminated, truncated, info = env.step(action)
                total_steps += 1
                if config.task == "base":
                    dynamics.append(info.copy())
                peak_force = max(peak_force, info["peak_contact_force"])
                if info.get("event_count", 0):
                    post_event_peak = max(post_event_peak, info["peak_contact_force"])
                impulse += info["contact_impulse"]
                obstruction_steps += int(info["obstruction"])
                any_collision |= info["collision"]
                minimum = min(minimum, info["clearance"])
                smoothness.append(info["action_change"])
                errors.append([info[k] for k in ("xy_error", "yaw_error", "joint_error")])
                if replay_path:
                    trace.append(
                        {
                            "seed": seed + episode,
                            "time": info["time"],
                            "qpos": env.data.qpos.tolist(),
                            "action": action.tolist(),
                            "goal": env.control.goal.payload(),
                            "metrics": info,
                        }
                    )
                if video is not None:
                    video.write(env.render()[:, :, ::-1])
                if terminated or truncated:
                    break
            rows.append(
                {
                    "seed": seed + episode,
                    **info,
                    "minimum_clearance": minimum,
                    "peak_contact_force": peak_force,
                    "final_contact_force": info["peak_contact_force"],
                    "post_event_peak_contact_force": post_event_peak,
                    "contact_impulse": impulse,
                    "obstruction_steps": obstruction_steps,
                    "collision": any_collision,
                    "mean_action_change": float(np.mean(smoothness)),
                    "mean_tracking_error": np.mean(errors, axis=0).tolist(),
                    **(
                        {
                            "peak_overshoot": max(r["overshoot"] for r in dynamics),
                            "rms_measured_acceleration": np.sqrt(
                                np.mean(np.square([r["measured_acceleration"] for r in dynamics]), axis=0)
                            ).tolist(),
                            "rms_measured_jerk": np.sqrt(
                                np.mean(np.square([r["measured_jerk"] for r in dynamics]), axis=0)
                            ).tolist(),
                            "peak_command_acceleration": np.max(
                                np.abs([r["command_acceleration"] for r in dynamics]), axis=0
                            ).tolist(),
                        }
                        if dynamics
                        else {}
                    ),
                }
            )
    finally:
        env.close()
        if video is not None:
            video.release()
    if replay_path:
        Path(replay_path).write_text(json.dumps(trace, indent=2) + "\n")
    return {
        "config": asdict(config),
        "teacher_approach": teacher_approach,
        "safety_scope": {
            "wheel_lift_evaluated": False,
            "reason": "Planar base constrains vertical motion, roll and pitch; no wheel support dynamics",
            "contact_force_scope": "all robot contacts except base support plane; includes arm-floor contact",
            "force_limit_status": "experimental simulator threshold, not a calibrated damage limit",
        },
        "episodes": rows,
        "success_rate": float(np.mean([r["is_success"] for r in rows])),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        **(
            {"teacher_action_mae": float(np.mean(teacher_action_errors))}
            if teacher_action_errors
            else {}
        ),
        "observation_and_inference_p95_ms": float(np.percentile(latencies, 95) * 1000),
        "steps_per_second": total_steps / (time.perf_counter() - started),
    }


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    from emet.rl.env import EnvConfig, MarsReachEnv
    from emet.rl.policy import PPOPolicy, export_policy

    torch.set_num_threads(1)
    from emet.rl.base import MarsBaseEnv
    from emet.rl.networks import BaseHistoryFeatures, CoordinatedPoseFeatures
    from emet.rl.transformer import HistoryTransformer

    config = EnvConfig(
        task=args.task,
        pose_targets=args.task == "coordinated",
        arm_only=getattr(args, "arm_only", False),
        stage=args.stage,
        contact_training=args.contact_training,
        base_scenario=getattr(args, "base_scenario", "standard"),
        horizon_seconds=getattr(args, "horizon", 10.0),
        terminal_accuracy_weight=getattr(args, "terminal_accuracy_weight", 0.0),
        trajectory_waypoints=getattr(args, "trajectory_waypoints", 1),
    )
    from emet.rl.training import TrainingMonitor, finish_wandb, start_wandb

    for seed in args.seeds:
        directory = Path(args.output) / f"seed-{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "env": asdict(config),
            "seed": seed,
            "envs": args.envs,
            "architecture": args.architecture,
            "resume": args.resume,
            "options": vars(args).copy(),
        }
        run = start_wandb(args, directory, metadata)
        failed = True
        env = None
        try:
            env = make_vec_env(
                MarsBaseEnv if args.task == "base" else MarsReachEnv,
                n_envs=args.envs,
                seed=seed,
                env_kwargs={"config": config},
                vec_env_cls=SubprocVecEnv if args.envs > 1 else DummyVecEnv,
            )
            norm_obs_keys = [
                key
                for key, space in env.observation_space.spaces.items()
                if hasattr(space, "low") and hasattr(space, "high")
            ]
            env = VecNormalize(env, norm_obs=True, norm_reward=False, norm_obs_keys=norm_obs_keys)
            model = PPO(
                "MultiInputPolicy",
                env,
                seed=seed,
                device="cpu",
                verbose=1,
                n_steps=256,
                batch_size=256,
                learning_rate=args.learning_rate if args.learning_rate is not None else 3e-4,
                ent_coef=0.001,
                policy_kwargs={
                    "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                    **(
                        {"features_extractor_class": CoordinatedPoseFeatures}
                        if args.task == "coordinated"
                        else {"features_extractor_class": HistoryTransformer}
                        if args.architecture == "transformer"
                        else {"features_extractor_class": BaseHistoryFeatures}
                        if args.task == "base"
                        else {}
                    ),
                },
            )
            if args.resume:
                import numpy as np

                previous = PPOPolicy(args.resume)
                model = PPO.load(Path(args.resume) / "policy.zip", env=env, device="cpu")
                model.set_random_seed(seed)
                env.training = not previous.manifest.get("normalization_frozen", False)
                env.clip_obs = previous.manifest["clip_obs"]
                env.epsilon = previous.manifest["epsilon"]
                for key, stats in previous.manifest["normalization"].items():
                    env.obs_rms[key].mean = np.asarray(stats["mean"])
                    env.obs_rms[key].var = np.asarray(stats["var"])
                    env.obs_rms[key].count = stats.get("count", 1.0)
            from stable_baselines3.common.utils import get_schedule_fn

            if args.learning_rate is not None:
                model.learning_rate = args.learning_rate
                model.lr_schedule = get_schedule_fn(args.learning_rate)
            if args.clip_range is not None:
                model.clip_range = get_schedule_fn(args.clip_range)
            if getattr(args, "target_kl", None) is not None:
                model.target_kl = args.target_kl
            metadata["ppo"] = {
                key: getattr(model, key)
                for key in (
                    "n_steps",
                    "batch_size",
                    "n_epochs",
                    "gamma",
                    "gae_lambda",
                    "ent_coef",
                    "learning_rate",
                    "target_kl",
                )
            }
            metadata["features_extractor"] = type(model.policy.features_extractor).__name__
            metadata["net_arch"] = model.policy.net_arch
            if run is not None:
                run.config.update(metadata, allow_val_change=True)
            if args.bootstrap_steps:
                from emet.rl.bootstrap import bootstrap_base

                model = bootstrap_base(model, env, args, directory, metadata, run)
            if args.critic_steps:
                from emet.rl.critic import warmup_critic

                warmup_critic(model, env, args.critic_steps, seed=seed, run=run)
            monitor = TrainingMonitor(
                directory,
                config,
                metadata,
                interval=args.eval_every,
                episodes=args.eval_episodes,
                run=run,
                evaluate_initial=bool(args.resume or args.bootstrap_steps),
            )
            model.learn(total_timesteps=args.steps, callback=monitor, reset_num_timesteps=not bool(args.resume))
            export_policy(
                model,
                env,
                directory,
                {**metadata, "steps": model.num_timesteps},
            )
            failed = False
        finally:
            if env is not None:
                env.close()
            finish_wandb(run, directory, failed=failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--task", choices=["base", "reach", "coordinated"], default="base")
    training.add_argument("--output", required=True)
    training.add_argument("--steps", type=int, default=200000)
    training.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    training.add_argument("--envs", type=int, default=4)
    training.add_argument("--stage", type=int, choices=range(5), default=0)
    training.add_argument(
        "--base-scenario", choices=["standard", "mixed", "retarget", "push", "contact_retreat"], default="standard"
    )
    training.add_argument("--arm-only", action="store_true")
    training.add_argument("--horizon", type=float, default=10.0)
    training.add_argument("--terminal-accuracy-weight", type=float, default=0.0)
    training.add_argument(
        "--trajectory-waypoints",
        type=int,
        default=1,
        help="Train/evaluate coordinated episodes as measured sequential pose trajectories",
    )
    training.add_argument("--contact-training", action="store_true", help="Permit light contact; stop on unsafe force")
    training.add_argument("--architecture", choices=["transformer", "mlp"], default="mlp")
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--clip-range", type=float)
    training.add_argument("--target-kl", type=float, help="Early-stop overly large PPO updates")
    training.add_argument("--critic-steps", type=int, default=0)
    training.add_argument(
        "--bootstrap-steps", type=int, default=0, help="Teacher/student samples per initialization round"
    )
    training.add_argument("--bootstrap-rounds", type=int, default=3)
    training.add_argument("--bootstrap-epochs", type=int, default=80)
    training.add_argument("--bootstrap-min-success-rate", type=float, default=0.8)
    training.add_argument("--bootstrap-max-collision-rate", type=float, default=0.05)
    training.add_argument("--bootstrap-max-teacher-action-mae", type=float, default=0.35)
    training.add_argument("--bootstrap-log-std", type=float, default=-3.0)
    training.add_argument("--eval-every", type=int, default=32768)
    training.add_argument("--eval-episodes", type=int, default=20)
    training.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    training.add_argument("--wandb-project", default="emet-mars-control")
    training.add_argument("--wandb-entity")
    training.add_argument("--wandb-name")
    training.add_argument("--resume", help="Continue from an exported bundle, including observation normalization")
    for command in ("evaluate", "replay"):
        p = sub.add_parser(command)
        p.add_argument("--task", choices=["base", "reach", "coordinated"], default="base")
        p.add_argument("--policy", help="Policy bundle directory; omit for deterministic baseline")
        p.add_argument("--contact-training", action="store_true")
        p.add_argument("--stage", type=int, choices=range(5), default=0)
        p.add_argument("--scene", choices=["procedural", "table"], default="procedural")
        p.add_argument("--seed", type=int, default=10000)
        p.add_argument("--episodes", type=int, default=50 if command == "evaluate" else 1)
        p.add_argument(
            "--base-scenario", choices=["standard", "mixed", "retarget", "push", "contact_retreat"], default="standard"
        )
        p.add_argument("--arm-only", action="store_true")
        p.add_argument("--trajectory-waypoints", type=int, default=1)
        p.add_argument("--horizon", type=float, default=10.0)
        p.add_argument("--output", required=True)
        if command == "replay":
            p.add_argument("--video")
    args = parser.parse_args()
    if args.command == "train":
        if min(args.steps, args.envs, args.eval_every, args.eval_episodes) <= 0:
            parser.error("steps, envs and evaluation settings must be positive")
        if (
            args.critic_steps < 0
            or (args.learning_rate is not None and not 0 < args.learning_rate < 1)
            or (args.clip_range is not None and not 0 < args.clip_range < 1)
            or (args.target_kl is not None and not 0 < args.target_kl < 1)
        ):
            parser.error("invalid optimizer or critic settings")
        if args.bootstrap_steps < 0 or min(args.bootstrap_rounds, args.bootstrap_epochs) <= 0:
            parser.error("invalid bootstrap settings")
        if not 0 <= args.bootstrap_min_success_rate <= 1 or not 0 <= args.bootstrap_max_collision_rate <= 1:
            parser.error("bootstrap validation gates must be between zero and one")
        if not 0 <= args.bootstrap_max_teacher_action_mae <= 2:
            parser.error("bootstrap maximum teacher action MAE must be between zero and two")
        if not -10 <= args.bootstrap_log_std <= 0:
            parser.error("bootstrap-log-std must be between -10 and 0")
        if args.bootstrap_steps and (
            args.task not in {"base", "coordinated"}
            or (args.task == "base" and args.architecture != "mlp")
        ):
            parser.error("bootstrap supports the base MLP and coordinated pose policy")
        train(args)
    else:
        if args.episodes <= 0:
            parser.error("episodes must be positive")
        if getattr(args, "video", None):
            os.environ.setdefault("MUJOCO_GL", "egl")
        from emet.rl.env import EnvConfig
        from emet.rl.policy import PPOPolicy

        policy = PPOPolicy(args.policy) if args.policy else None
        report = evaluate(
            policy,
            EnvConfig(
                task=args.task,
                pose_targets=args.task == "coordinated",
                arm_only=args.arm_only,
                scene=args.scene,
                stage=args.stage,
                contact_training=args.contact_training,
                base_scenario=args.base_scenario,
                horizon_seconds=args.horizon,
                trajectory_waypoints=args.trajectory_waypoints,
            ),
            episodes=args.episodes,
            seed=args.seed,
            replay_path=args.output if args.command == "replay" else None,
            video_path=getattr(args, "video", None),
        )
        if args.command == "evaluate":
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
