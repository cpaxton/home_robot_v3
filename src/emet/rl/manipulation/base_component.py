"""Train base exploration/readout while preserving one shared learned arm actor.

The training wrapper supplies frozen learned arm means. Export merges the base
rows back into that same eight-output actor; no component runtime is deployed.
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import StateDependentNoiseDistribution
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.command_policy import ExecutorCommandVecNormalize
from emet.rl.manipulation.feedback_policy import FeedbackActorCriticPolicy, GoalFeedbackFeatures, RestoringYawAction
from emet.rl.manipulation.mastery import (
    MasteryEnv,
    RolloutDiagnostics,
    evaluate_suite,
    load_suite,
    preserve_training_rng,
    read_report,
    save_report,
    summarize,
    validate_preflight,
)
from emet.rl.manipulation.policy import ManipulationPolicy, export
from emet.rl.regression_gate import verify_regressions

ROOT = Path(__file__).resolve().parents[4]
SUITE = ROOT / "configs/benchmarks/mars_tracking_mastery_command_history_v1.json"
SCOPE = "mars-frozen-arm-base-gsde-v1"


class BaseNoise(StateDependentNoiseDistribution):
    """Six announced base-error features plus a small stationary noise floor."""

    def proba_distribution_net(self, latent_dim, log_std_init=-2, latent_sde_dim=None):
        return super().proba_distribution_net(latent_dim, log_std_init, latent_sde_dim=7)


class BaseMean(RestoringYawAction):
    def __init__(self, context_width, feedback_width):
        super().__init__(context_width, feedback_width, 8)

    def forward(self, latent):
        return super().forward(latent)[:, :2]


class BasePolicy(FeedbackActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        if (
            not kwargs.get("use_sde", False)
            or not kwargs.get("full_std", True)
            or kwargs.get("squash_output", False)
            or kwargs.get("use_expln", False)
        ):
            raise ValueError("base component requires full-std unsquashed gSDE")
        super().__init__(*args, **kwargs)
        if (
            self.action_space.shape != (2,)
            or not self.restoring_yaw_feedback
            or self.features_extractor.robot != "innate_mars"
        ):
            raise ValueError("two-output Mars restoring-yaw component required")

    def _build(self, lr_schedule):
        self.action_dist = BaseNoise(2, full_std=True, epsilon=1e-8)
        super()._build(lr_schedule)
        self.action_net = BaseMean(self.mlp_extractor.context_width, self.features_extractor.feedback_width).to(
            self.device
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.mlp_extractor.value_net, self.value_net, self.action_net.turn_gain):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        width = self.features_extractor.feedback_width
        for parameter in (self.action_net.gain.weight, self.action_net.gain.bias):
            parameter.requires_grad_(True)
            mask = torch.zeros_like(parameter).reshape(7, width, -1)
            mask[:2] = 1
            parameter.register_hook(lambda grad, mask=mask.reshape(parameter.shape): grad * mask)
        self.log_std.requires_grad_(True)
        noise_mask = torch.zeros_like(self.log_std)
        noise_mask[[0, 3, 6], 0] = 1
        noise_mask[[1, 2, 4, 5, 6], 1] = 1
        with torch.no_grad():
            self.log_std.fill_(-20)
            self.log_std[noise_mask.bool()] = np.log(0.5)
        self.log_std.register_hook(lambda grad: grad * noise_mask)
        self.optimizer = self.optimizer_class(
            [p for p in self.parameters() if p.requires_grad], lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi):
        errors = latent_pi[:, -self.features_extractor.feedback_width :]
        noise = torch.cat([errors[:, [0, 1, 2, 12, 13, 14]], errors.new_full((len(errors), 1), 0.002)], dim=1)
        return self.action_dist.proba_distribution(self.action_net(latent_pi), self.log_std, noise)


class FrozenArmActions(gym.Wrapper):
    """Only base requests are sampled; the preserved arm uses ordinary inputs."""

    def __init__(self, env, arm):
        super().__init__(env)
        if (
            env.profile.robot != "innate_mars"
            or arm.robot != env.profile.robot
            or arm.manifest["fingerprint"] != env.fingerprint
        ):
            raise ValueError("frozen arm robot/model mismatch")
        self.arm, self.last_observation = arm, None
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    def reset(self, **kwargs):
        self.last_observation, info = self.env.reset(**kwargs)
        return self.last_observation, info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if (
            self.last_observation is None
            or action.shape != (2,)
            or not np.isfinite(action).all()
            or np.any(abs(action) > 1)
        ):
            raise ValueError("reset then provide two finite normalized base requests")
        full = self.arm(self.last_observation).copy()
        full[:2] = action
        self.last_observation, reward, done, truncated, info = self.env.step(full)
        return self.last_observation, reward, done, truncated, info


def initialize(model, normalizer, source, config, fingerprint):
    if (
        source.manifest["environment"] != json.loads(json.dumps(asdict(config)))
        or source.manifest["fingerprint"] != fingerprint
    ):
        raise ValueError("exact source environment/model required")
    if (
        not isinstance(source.model.policy.features_extractor, GoalFeedbackFeatures)
        or source.model.policy.features_extractor.orientation_scale != model.policy.features_extractor.orientation_scale
        or source.model.policy.turn_xy_tolerance != model.policy.turn_xy_tolerance
    ):
        raise ValueError("source feedback units or turn gate differ")
    old, new = source.model.policy.state_dict(), model.policy.state_dict()
    if set(old) != set(new):
        raise ValueError("base component parameter layout differs")
    for key in new:
        if key != "log_std":
            if old[key].shape != new[key].shape:
                raise ValueError("base component parameter shape differs: " + key)
            new[key] = old[key].clone()
    model.policy.load_state_dict(new)
    for key, (mean, var) in source.normalization.items():
        stats = normalizer.obs_rms[key]
        stats.mean, stats.var = mean.copy(), var.copy()
        stats.count = source.manifest["normalization"][key]["count"]
    normalizer.epsilon, normalizer.clip_obs = source.manifest["epsilon"], source.manifest["clip"]
    normalizer.pin_command_features()
    if any(
        not np.array_equal(normalizer.obs_rms[k].mean, mean) or not np.array_equal(normalizer.obs_rms[k].var, var)
        for k, (mean, var) in source.normalization.items()
    ):
        raise ValueError("source normalization is not the executor command-unit contract")
    normalizer.training = False


def verify_frozen(model, source):
    current, original = model.policy.state_dict(), source.model.policy.state_dict()
    for key, value in current.items():
        if key in {"log_std", "value_net.weight", "value_net.bias"} or key.startswith(
            ("mlp_extractor.value_net.", "action_net.turn_gain.")
        ):
            continue
        if key in {"action_net.gain.weight", "action_net.gain.bias"}:
            width = model.policy.features_extractor.feedback_width
            value, expected = value.reshape(7, width, -1)[2:], original[key].reshape(7, width, -1)[2:]
        else:
            expected = original[key]
        if not torch.equal(value, expected):
            raise RuntimeError("frozen arm/context changed: " + key)


def merge(model, normalizer, source, destination, config, fingerprint):
    """Export one ordinary eight-output actor; no component or noise wrapper."""
    verify_frozen(model, source)
    original = {k: v.clone() for k, v in source.model.policy.state_dict().items()}
    original_steps = source.model.num_timesteps
    try:
        full = {k: v.clone() for k, v in original.items()}
        for key, value in model.policy.state_dict().items():
            if key != "log_std":
                full[key] = value.clone()
        source.model.policy.load_state_dict(full)
        source.model.num_timesteps = model.num_timesteps
        export(source.model, normalizer, destination, config, fingerprint, steps=model.num_timesteps)
        path = Path(destination) / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["component_training"] = {
            "scope": SCOPE,
            "optimizer_resume_supported": False,
            "one_runtime_actor": True,
        }
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        source.model.policy.load_state_dict(original)
        source.model.num_timesteps = original_steps
    return ManipulationPolicy(destination, robot=config.robot, fingerprint=fingerprint)


def train(args, suite):
    verify_regressions(args.regression_receipt)
    teacher = read_report(args.preflight)
    validate_preflight(teacher, suite, ["base", "arm"])
    source = ManipulationPolicy(args.source)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    base = MasteryEnv(suite, stages=["base", "arm"], split="train", population="training")
    env = ExecutorCommandVecNormalize(
        DummyVecEnv([lambda: FrozenArmActions(base, source)]), config=base.config, norm_reward=False
    )
    run = None
    sources = source_hashes()
    try:
        model = PPO(
            BasePolicy,
            env,
            device="cpu",
            seed=90,
            n_steps=256,
            batch_size=256,
            n_epochs=5,
            learning_rate=1e-4,
            gamma=0.9975,
            target_kl=0.01,
            clip_range=0.1,
            use_sde=True,
            sde_sample_freq=8,
            policy_kwargs={
                "restoring_yaw_feedback": True,
                "turn_xy_tolerance": base.config.base_xy_tolerance,
                "features_extractor_class": GoalFeedbackFeatures,
                "features_extractor_kwargs": {
                    "robot": "innate_mars",
                    "retention_context": True,
                    "orientation_scale": base.config.feedback_orientation_scale,
                },
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            },
        )
        initialize(model, env, source, base.config, base.fingerprint)
        record = {
            "scope": SCOPE,
            "suite": suite,
            "suite_sha256": digest(suite),
            "source_hashes": sources,
            "source_weights_sha256": source.manifest["weights_sha256"],
            "source_manifest_sha256": digest(source.manifest),
            "teacher_report_sha256": digest(teacher),
            "regression_receipt": json.loads(args.regression_receipt.read_text()),
            "steps": args.steps,
            "seed": 90,
            "frozen_normalization": True,
            "frozen_actor_context_and_arm": True,
            "teacher_action_loss": False,
            "sde_sample_freq": 8,
            "sde_noise_features": "current/future base errors plus constant .002",
            "initial_sde_coefficient_std": 0.5,
            "ppo": {
                "learning_rate": 1e-4,
                "gamma": 0.9975,
                "n_steps": 256,
                "batch_size": 256,
                "n_epochs": 5,
                "clip_range": 0.1,
                "target_kl": 0.01,
            },
            "hardware_used": False,
            "safety_promoted": False,
        }
        (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        if args.wandb_mode == "offline":
            import wandb

            run = wandb.init(
                project="emet-mars-control", mode="offline", dir=str(args.output), name="base-isolation", config=record
            )

        class Audit(BaseCallback):
            def __init__(self):
                super().__init__()
                self.steps = Counter()
                self.starts = Counter()
                self.first = True
                self.rollout = RolloutDiagnostics()
                self.last_update = 0

            def record_update(self):
                if model._n_updates == self.last_update:
                    return
                values = {
                    k: float(v) if np.isfinite(v) else None
                    for k, v in model.logger.name_to_value.items()
                    if k.startswith("train/") and np.isscalar(v)
                }
                with (args.output / "updates.jsonl").open("a") as stream:
                    stream.write(json.dumps({"steps": self.num_timesteps, **values}, allow_nan=False) + "\n")
                if run:
                    run.log({"steps": self.num_timesteps, **values})
                self.last_update = model._n_updates

            def checkpoint(self, label):
                if source_hashes() != sources:
                    raise RuntimeError("training sources changed")
                directory = args.output / label
                directory.mkdir()
                model.save(directory / "component.zip")
                with preserve_training_rng():
                    policy = merge(model, env, source, directory / "actor", base.config, base.fingerprint)
                    report = evaluate_suite(suite, stages=["base", "arm"], policy=policy, teacher_report=teacher)
                save_report(directory / "evaluation.json.gz", report)
                summary = summarize(report)
                (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                if run:
                    run.log(
                        {
                            "steps": model.num_timesteps,
                            **{f"evaluation/{k}/mastered": v["mastered"] for k, v in summary["groups"].items()},
                        }
                    )
                return report

            def _on_step(self):
                info = self.locals["infos"][0]
                cid = info["mastery"]["case_id"]
                self.steps[cid] += 1
                if self.first:
                    self.starts[cid] += 1
                self.first = bool(self.locals["dones"][0])
                self.rollout.add(
                    cid,
                    self.locals["actions"][0],
                    self.locals["clipped_actions"][0],
                    self.locals["rewards"][0],
                    info,
                    self.locals["dones"][0],
                )
                return True

            def _on_rollout_start(self):
                self.record_update()
                if self.num_timesteps and self.num_timesteps % 8192 == 0:
                    self.checkpoint(f"step-{self.num_timesteps}")

            def _on_rollout_end(self):
                verify_frozen(model, source)
                values = {
                    "steps": self.num_timesteps,
                    "sampling_steps": dict(self.steps),
                    "sampling_starts": dict(self.starts),
                    "sde_coefficients": model.policy.log_std.detach().exp().cpu().tolist(),
                    "rollout_cases": self.rollout.record(),
                }
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(values) + "\n")
                if run:
                    run.log({"steps": self.num_timesteps, **{f"sampling/{k}": v for k, v in self.steps.items()}})
                self.rollout = RolloutDiagnostics()

            def _on_training_end(self):
                self.record_update()

        audit = Audit()
        audit.init_callback(model)
        initial = audit.checkpoint("step-0")
        if not all(r["mastered"] for r in initial["cases"] if r["stage"] == "arm"):
            raise RuntimeError("initial merged actor lost canonical arm qualification")
        with preserve_training_rng():
            report = evaluate_suite(
                suite,
                stages=["arm"],
                policy=ManipulationPolicy(args.output / "step-0/actor"),
                teacher_report=teacher,
                population="admission",
            )
        save_report(args.output / "initial-arm-expanded.json.gz", report)
        if len(report["cases"]) != 16 or not all(r["mastered"] for r in report["cases"]):
            raise RuntimeError("initial arm admission failed")
        model.learn(args.steps, callback=audit)
        final = audit.checkpoint("last")
        with preserve_training_rng():
            expanded = evaluate_suite(
                suite,
                stages=["base", "arm"],
                policy=ManipulationPolicy(args.output / "last/actor"),
                teacher_report=teacher,
                population="admission",
            )
        save_report(args.output / "final-expanded.json.gz", expanded)
        (args.output / "complete.json").write_text(
            json.dumps(
                {
                    "steps": model.num_timesteps,
                    "canonical": summarize(final),
                    "expanded": summarize(expanded),
                    "safety_promoted": False,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        env.close()
        if run:
            run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--regression-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--wandb-mode", choices=["disabled", "offline"], default="disabled")
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 256:
        parser.error("positive multiple of256 steps required")
    train(args, load_suite(args.suite))


if __name__ == "__main__":
    main()
