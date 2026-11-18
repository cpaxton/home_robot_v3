"""Five-output Mars arm diagnostic; fixed wheel/grip requests, unchanged physics.

This isolates learning, not the robot base in physics. The same base/hand goals,
history/preview and force/quiet/tracking guards still apply. Component checkpoints
have no production manipulation manifest and cannot be deployed as whole-body policies.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.command_runtime import task_pose_teacher_action
from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.feedback_policy import FeedbackAction, FeedbackLatent, GoalFeedbackFeatures
from emet.rl.manipulation.hand_audit import hand_tracking_diagnostics
from emet.rl.manipulation.mastery import (
    MasteryEnv,
    load_suite,
    preserve_training_rng,
    read_report,
    save_report,
    select_cases,
)
from emet.rl.manipulation.normalization import CommandVecNormalize
from emet.rl.regression_gate import verify_regressions

ROOT = Path(__file__).resolve().parents[4]
SUITE = ROOT / "configs/benchmarks/mars_tracking_mastery_hand_accuracy_v1.json"
SCOPE = "mars-arm-only-zero-wheel-fixed-grip-v1"


def expand_action(action):
    value = np.asarray(action, dtype=float)
    if value.shape != (5,) or not np.isfinite(value).all() or np.any(abs(value) > 1):
        raise ValueError("five finite normalized arm rates in [-1, 1] required")
    return np.r_[0.0, 0.0, value, 0.0]


class ArmOnlyActions(gym.ActionWrapper):
    def __init__(self, env):
        if env.profile.robot != "innate_mars" or any(c["stage"] != "arm" for c in env.cases):
            raise ValueError("arm-only diagnostic requires unloaded Mars arm cases")
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1, 1, (5,), dtype=np.float32)

    def action(self, action):
        return expand_action(action)


class ArmMean(FeedbackAction):
    def __init__(self, context_width, feedback_width, *, normalization_width=None):
        super().__init__(context_width, feedback_width, 6, normalization_width=normalization_width)

    def forward(self, latent):
        return super().forward(latent)[:, :5]


class ArmPolicy(MultiInputActorCriticPolicy):
    """Reuse the existing learned-feedback actor with exactly five arm outputs."""

    def _build_mlp_extractor(self):
        self.mlp_extractor = FeedbackLatent(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            feedback_width=self.features_extractor.feedback_width,
        )

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = ArmMean(self.mlp_extractor.context_width, self.features_extractor.feedback_width,
                                 normalization_width=self.features_extractor.normalization_width).to(
            self.device
        )
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def evaluate(suite, *, model=None, normalizer=None, teacher=None, population="canonical"):
    before = source_hashes()
    env = MasteryEnv(suite, stages=["arm"], split="all", population=population)
    report = {
        "format": SCOPE,
        "controller": "teacher" if model is None else "ppo",
        "suite_sha256": digest(suite),
        "source_hashes": before,
        "population": population,
        "cases": [],
        "complete": False,
        "hardware_used": False,
        "production_admitted": False,
    }
    paired = {} if teacher is None else {c["id"]: c for c in teacher["cases"]}
    try:
        for case in select_cases(suite, ["arm"], "all", population=population):
            obs, info = env.reset(options={"case_id": case["id"]})
            row = {
                **case,
                "admitted": not info["initialization_failed"],
                "admission": env.mastery_admission,
                "trace": [],
            }
            report["cases"].append(row)
            if row["admitted"]:
                for _ in range(round(env.config.horizon / env.profile.dt) + 1):
                    arm = (
                        task_pose_teacher_action(env.control)[2:-1]
                        if model is None
                        else model.predict(normalizer.normalize_obs(obs), deterministic=True)[0]
                    )
                    full = expand_action(arm)
                    obs, reward, done, truncated, info = env.step(full)
                    row["trace"].append(
                        {
                            "time": info["time"],
                            "action": full.tolist(),
                            "reward": reward,
                            "reward_terms": info.get("reward_terms", {}),
                            "hand": env.control.ee_pose().tolist(),
                            "base": env.control.base_pose().tolist(),
                            "position_error_m": info["ee_position_error"],
                            "orientation_error_rad": info["ee_orientation_error"],
                            "hand_control": hand_tracking_diagnostics(env.control),
                            "code": info["code"],
                        }
                    )
                    if done or truncated:
                        break
            metrics = {
                "hand_position_error_m": info["ee_position_error"],
                "hand_orientation_error_rad": info["ee_orientation_error"],
                "settled_seconds": env.settle_time,
                "hold_drift_m": env.peak_hold_drift,
                "peak_force_N": info["peak_contact_force"],
                "base_linear_jerk": float(env.peak_base_axis_jerk[0]),
                "base_yaw_jerk": float(env.peak_base_axis_jerk[1]),
                "joint_jerk": float(np.max(env.peak_joint_jerk)),
            }
            tracking = bool(
                row["admitted"]
                and info.get("is_success", False)
                and env.settle_time + 1e-8 >= suite["gate"]["terminal_hold_s"]
                and env.peak_hold_drift <= suite["gate"]["hold_drift_m"]
                and not env.path_failed
            )
            failures = [] if tracking else [info["code"], "tracking_or_quiet"]
            if teacher is not None and row["admitted"]:
                ref = paired[case["id"]]
                if any(
                    row["admission"][k] != ref["admission"][k] for k in ["initial_state_sha256", "reference_sha256"]
                ):
                    raise ValueError("component teacher and actor initial state/reference differ")
                floors = {
                    "peak_force_N": "force_floor_N",
                    "base_linear_jerk": "base_linear_jerk_floor_m_s3",
                    "base_yaw_jerk": "base_yaw_jerk_floor_rad_s3",
                    "joint_jerk": "joint_jerk_floor_rad_s3",
                }
                failures.extend(
                    "smoothness/" + k
                    for k, floor in floors.items()
                    if metrics[k] > max(suite["gate"][floor], suite["gate"]["smoothness_ratio"] * ref["metrics"][k])
                )
            row.update(
                metrics=metrics,
                final_info=info,
                tracking_pass=tracking,
                mastered=tracking and not failures,
                failures=failures,
            )
            print("ARM_COMPONENT", case["id"], info["code"], row["mastered"], flush=True)
    finally:
        env.close()
    if source_hashes() != before:
        raise RuntimeError("component sources changed during evaluation")
    report["complete"] = True
    return report


def summary(report):
    return {
        "complete": report["complete"],
        "scope": SCOPE,
        "production_admitted": False,
        "groups": {
            split: {
                "cases": len(rows),
                "tracking_pass": sum(r["tracking_pass"] for r in rows),
                "mastered": sum(r["mastered"] for r in rows),
            }
            for split in ["train", "validation"]
            if (rows := [r for r in report["cases"] if r["split"] == split])
        },
    }


def require_preflight(report, suite):
    expected = {c["id"] for c in select_cases(suite, ["arm"], "all", population="admission")}
    if (
        report.get("format") != SCOPE
        or report.get("controller") != "teacher"
        or not report.get("complete")
        or report.get("source_hashes") != source_hashes()
        or report.get("suite_sha256") != digest(suite)
        or {r["id"] for r in report["cases"]} != expected
        or len(report["cases"]) != len(expected)
        or not all(r["mastered"] for r in report["cases"])
    ):
        raise ValueError("complete current-source arm-only teacher admission required")


def train(args, suite):
    verify_regressions(args.regression_receipt)
    teacher = read_report(args.preflight)
    require_preflight(teacher, suite)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    cfg = ManipulationConfig(**suite["config"])
    env = CommandVecNormalize(
        DummyVecEnv([lambda: ArmOnlyActions(MasteryEnv(suite, stages=["arm"], split="train", population="training"))]),
        config=cfg,
        norm_reward=False,
        clip_obs=20,
    )
    run = None
    try:
        model = PPO(
            ArmPolicy,
            env,
            device="cpu",
            seed=90,
            n_steps=256,
            batch_size=256,
            n_epochs=5,
            learning_rate=1e-4,
            gamma=0.9975,
            target_kl=0.02,
            clip_range=0.2,
            policy_kwargs={
                "features_extractor_class": GoalFeedbackFeatures,
                "features_extractor_kwargs": GoalFeedbackFeatures.kwargs_for(cfg),
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            },
        )
        with torch.no_grad():
            model.policy.log_std.fill_(np.log(0.05))
        (args.output / "component.json").write_text(
            json.dumps(
                {
                    "scope": SCOPE,
                    "source_hashes": sources,
                    "suite": suite,
                    "env": asdict(cfg),
                    "steps": args.steps,
                    "num_envs": 1,
                    "hardware_used": False,
                    "production_admitted": False,
                    "teacher_action_loss": False,
                    "initialization": "scratch",
                    "action_size": 5,
                    "learning_rate": 1e-4,
                    "clip_range": 0.2,
                    "target_kl": 0.02,
                    "initial_action_std": 0.05,
                    "command_smoothness_warmup_steps": 16384,
                    "base_physics": "free; wheel requests zero",
                    "grip_request": "zero offset from executor reference",
                },
                indent=2,
            )
            + "\n"
        )
        if getattr(args, "wandb_mode", "disabled") == "offline":
            import wandb

            run = wandb.init(
                project="emet-mars-control",
                mode="offline",
                dir=str(args.output),
                name="arm-component",
                config=json.loads((args.output / "component.json").read_text()),
            )

        def checkpoint(label):
            if source_hashes() != sources:
                raise RuntimeError("component sources changed")
            dest = args.output / label
            dest.mkdir()
            model.save(dest / "actor.zip")
            env.save(dest / "normalizer.pkl")
            with preserve_training_rng():
                report = evaluate(suite, model=model, normalizer=env, teacher=teacher)
            save_report(dest / "evaluation.json.gz", report)
            (dest / "summary.json").write_text(json.dumps(summary(report), indent=2) + "\n")
            if run:
                run.log(
                    {
                        "steps": model.num_timesteps,
                        **{
                            f"evaluation/{split}/{key}": value
                            for split, group in summary(report)["groups"].items()
                            for key, value in group.items()
                        },
                    }
                )

        class Audit(BaseCallback):
            def __init__(self):
                super().__init__()
                self.reward_sums = {}

            def _on_step(self):
                for key, value in self.locals["infos"][0].get("reward_terms", {}).items():
                    self.reward_sums[key] = self.reward_sums.get(key, 0) + float(value)
                return True

            def _on_rollout_start(self):
                env.env_method("set_command_smoothness_fraction", min(1, self.num_timesteps / 16384))
                if self.num_timesteps and self.num_timesteps % 8192 == 0:
                    checkpoint(f"step-{self.num_timesteps}")

            def _on_rollout_end(self):
                metrics = {
                    "steps": self.num_timesteps,
                    "optimizer_epochs": model._n_updates,
                    "episodes": env.get_attr("case_visits")[0],
                    "action_std": model.policy.log_std.detach().exp().cpu().tolist(),
                    "reward_terms_sum": self.reward_sums,
                }
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                if run:
                    run.log(
                        {
                            "steps": self.num_timesteps,
                            **{f"reward/{k}": v for k, v in self.reward_sums.items()},
                            **{f"action_std/{i}": v for i, v in enumerate(metrics["action_std"])},
                        }
                    )
                self.reward_sums = {}

        checkpoint("step-0")
        model.learn(args.steps, callback=Audit())
        checkpoint("last")
        (args.output / "complete.json").write_text(
            json.dumps({"steps": model.num_timesteps, "production_admitted": False}) + "\n"
        )
    finally:
        env.close()
        if run:
            run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "train"])
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--regression-receipt", type=Path)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--wandb-mode", choices=["disabled", "offline"], default="disabled")
    args = parser.parse_args()
    torch.set_num_threads(1)
    suite = load_suite(args.suite)
    if args.command == "preflight":
        args.output.mkdir(parents=True, exist_ok=False)
        report = evaluate(suite, population="admission")
        save_report(args.output / "report.json.gz", report)
        (args.output / "summary.json").write_text(json.dumps(summary(report), indent=2) + "\n")
    else:
        if not args.preflight or not args.regression_receipt or args.steps <= 0 or args.steps % 256:
            parser.error("fresh preflight, regression receipt and rollout-aligned positive steps required")
        train(args, suite)


if __name__ == "__main__":
    main()
