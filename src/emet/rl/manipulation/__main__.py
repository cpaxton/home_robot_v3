"""Train/evaluate physical manipulation and fit separate outcome-risk heads."""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.evaluation import (
    curriculum_ready,
    evaluate,
    paired_comparison,
    target_coverage_ready,
    tracking_precision_ready,
)
from emet.rl.manipulation.policy import ManipulationHistoryFeatures, ManipulationMLPFeatures, ManipulationPolicy, export


class FixedResetSeed(gym.Wrapper):
    """Training-only overfit diagnostic; never wrap held-out evaluation."""

    def __init__(self, env, *, fixed_seed):
        super().__init__(env)
        self.fixed_seed = fixed_seed

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=self.fixed_seed, options=options)


class PhaseLearningRate:
    """Linear decay over additional training steps, including resumed phases."""

    def __init__(self, start, end, previous_steps, additional_steps):
        if not np.isfinite([start, end]).all() or not 0 < end <= start or additional_steps <= 0 or previous_steps < 0:
            raise ValueError("invalid learning-rate decay phase")
        self.start, self.end = float(start), float(end)
        self.initial_progress = additional_steps / (previous_steps + additional_steps)

    def __call__(self, progress_remaining):
        remaining = float(np.clip(progress_remaining / self.initial_progress, 0, 1))
        return self.end + (self.start - self.end) * remaining


def resume_training(directory, env, config, fingerprint, architecture):
    """Restore PPO optimizer and observation statistics after bundle validation."""
    bundle = ManipulationPolicy(directory, robot=config.robot, fingerprint=fingerprint)
    expected = ManipulationHistoryFeatures if architecture == "history" else ManipulationMLPFeatures
    if (
        bundle.action_mode != config.action_mode
        or type(bundle.model.policy.features_extractor) is not expected
        or ("preview" in bundle.normalization) != config.trajectory_preview
        or bundle.manifest.get("pose_only_targets", False) != (config.trajectory_preview or config.tracking_reward)
        or bundle.manifest.get("environment", {}).get("retention_context", False) != config.retention_context
    ):
        raise ValueError("resume policy architecture or control/observation semantics mismatch")
    if set(env.obs_rms) != set(bundle.normalization):
        raise ValueError("resume normalization keys mismatch")
    for key, (mean, var) in bundle.normalization.items():
        if env.obs_rms[key].mean.shape != mean.shape:
            raise ValueError("resume normalization shape mismatch")
        count = bundle.manifest.get("normalization", {}).get(key, {}).get("count", bundle.manifest["count"])
        if not np.isfinite(count) or count <= 0:
            raise ValueError("invalid resume normalization count")
        env.obs_rms[key].mean = mean.copy()
        env.obs_rms[key].var = var.copy()
        env.obs_rms[key].count = float(count)
    env.epsilon, env.clip_obs = bundle.manifest["epsilon"], bundle.manifest["clip"]
    # Loading with the new env rebuilds rollout buffers and discards stale
    # rollout observations while retaining policy/value and optimizer state.
    return PPO.load(Path(directory) / "policy.zip", env=env, device="cpu")


def initialize_retention_context(model, env, directory, config, fingerprint):
    """Warm-start compatible weights with zero influence from the new suffix.

    This deliberately starts a fresh PPO optimizer, unlike exact resume. Robot,
    action, preview and model compatibility checks are never bypassed.
    """
    old = ManipulationPolicy(directory, robot=config.robot, fingerprint=fingerprint)
    if (
        not config.retention_context
        or old.manifest.get("environment", {}).get("retention_context", False)
        or old.action_mode != config.action_mode
        or type(old.model.policy.features_extractor) is not type(model.policy.features_extractor)
        or ("preview" in old.normalization) != config.trajectory_preview
        or old.manifest.get("pose_only_targets", False) != (config.trajectory_preview or config.tracking_reward)
    ):
        raise ValueError("retention warm-start requires matching legacy control/observation semantics")
    old_shape, new_shape = old.mean.shape, env.obs_rms["history"].mean.shape
    if old_shape[0] != new_shape[0] or new_shape[1] != old_shape[1] + 2:
        raise ValueError("retention warm-start requires exactly two appended history fields")
    source = old.model.policy.state_dict()
    target = model.policy.state_dict()
    if source.keys() != target.keys():
        raise ValueError("retention warm-start policy parameters differ")
    for name, value in source.items():
        if value.shape == target[name].shape:
            target[name] = value.clone()
        elif name.endswith("project.weight") and value.shape[1] == old_shape[1]:
            target[name].zero_()
            target[name][:, : old_shape[1]] = value
        elif name in {"mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"}:
            steps = min(4, old_shape[0])
            expanded = torch.zeros_like(target[name])
            expanded[:, : steps * new_shape[1]].reshape(value.shape[0], steps, new_shape[1])[:, :, : old_shape[1]] = (
                value[:, : steps * old_shape[1]].reshape(value.shape[0], steps, old_shape[1])
            )
            expanded[:, steps * new_shape[1] :] = value[:, steps * old_shape[1] :]
            target[name] = expanded
        else:
            raise ValueError(f"unsupported retention warm-start parameter: {name}")
    model.policy.load_state_dict(target)
    for key, (mean, var) in old.normalization.items():
        stats = env.obs_rms[key]
        if key == "history":
            stats.mean.fill(0)
            stats.var.fill(1)
            stats.mean[:, : old_shape[1]], stats.var[:, : old_shape[1]] = mean, var
        else:
            stats.mean, stats.var = mean.copy(), var.copy()
        stats.count = old.manifest.get("normalization", {}).get(key, {}).get("count", old.manifest["count"])
    env.epsilon, env.clip_obs = old.manifest["epsilon"], old.manifest["clip"]
    return {
        "source_weights_sha256": old.manifest["weights_sha256"],
        "optimizer_restored": False,
        "new_context_initial_influence": "zero",
    }


class Monitor(BaseCallback):
    def __init__(
        self,
        directory,
        config,
        fingerprint,
        *,
        interval,
        episodes,
        run=None,
        debug_reset_seed=None,
        held_estimator=None,
    ):
        super().__init__()
        self.directory, self.config, self.fingerprint = directory, config, fingerprint
        self.interval, self.episodes, self.run = interval, episodes, run
        self.randomization_scale = config.randomization_scale
        self.target_scale = config.target_scale
        self.last = 0
        self.best = None
        self.completed = []
        self.debug_reset_seed = debug_reset_seed
        self.held_estimator = held_estimator
        self.episode_counts = Counter()
        self.sample_steps = Counter()
        self.sample_episodes = Counter()
        self.episode_steps = Counter()
        self.task_rewards = defaultdict(list)
        self.reward_terms = defaultdict(list)

    def record(self, values):
        values = {"timesteps": self.num_timesteps, **values}
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(values) + "\n")
        if self.run:
            self.run.log(values)

    def _on_step(self):
        for i, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"], strict=True)):
            scenario = info.get("scenario", "unknown")
            if self.episode_steps[i] == 0:
                self.sample_episodes[scenario] += 1
            self.episode_steps[i] += 1
            self.sample_steps[scenario] += 1
            self.task_rewards[scenario].append(float(self.locals["rewards"][i]))
            for key, value in info.get("reward_terms", {}).items():
                self.reward_terms[f"{scenario}/{key}"].append(float(value))
            label = info.get("held_object_supervision", {})
            if (
                self.held_estimator is not None
                and not info.get("invalid_physics")
                and not info.get("initialization_failed")
            ):
                normalizer = self.model.get_vec_normalize_env()
                if done:
                    observation = normalizer.unnormalize_obs(info["terminal_observation"])
                else:
                    observation = {k: v[i] for k, v in normalizer.get_original_obs().items()}
                # All samples of a vector-slot episode share one split. No
                # adjacent frames from an audit episode enter head training.
                audit = (self.episode_counts[i] + i) % 5 == 4
                self.held_estimator.add(observation["history"], label, audit=audit)
            if done:
                self.completed.append(info)
                self.episode_counts[i] += 1
                self.episode_steps[i] = 0
        return True

    def _on_rollout_end(self):
        values = {
            k: float(v) for k, v in self.model.logger.name_to_value.items() if isinstance(v, (int, float, np.number))
        }
        for scenario in ("reach", "pregrasp", "carry", "place", "disturbance"):
            values[f"sampling/{scenario}/step_fraction"] = self.sample_steps[scenario] / max(
                1, sum(self.sample_steps.values())
            )
            values[f"sampling/{scenario}/episode_fraction"] = self.sample_episodes[scenario] / max(
                1, sum(self.sample_episodes.values())
            )
            if self.task_rewards[scenario]:
                values[f"tasks/{scenario}/reward_per_step"] = float(np.mean(self.task_rewards[scenario]))
        for key, samples in self.reward_terms.items():
            values[f"reward_terms/{key}"] = float(np.mean(samples))
        if self.held_estimator is not None:
            metrics = {**self.held_estimator.fit(), **self.held_estimator.audit_metrics()}
            values.update({f"held_object/{k}": v for k, v in metrics.items()})
        self.sample_steps.clear()
        self.sample_episodes.clear()
        self.task_rewards.clear()
        self.reward_terms.clear()
        if self.completed:
            for key in (
                "is_success",
                "retention_lost",
                "unsafe_contact",
                "initialization_failed",
                "peak_contact_force",
            ):
                values[f"episodes/{key}"] = float(np.mean([r[key] for r in self.completed]))
            self.completed.clear()
        self.record(values)
        if self.num_timesteps - self.last >= self.interval:
            self.checkpoint()

    def checkpoint(self):
        target = self.directory / f"step-{self.num_timesteps}"
        export(
            self.model,
            self.model.get_vec_normalize_env(),
            target,
            self.config,
            self.fingerprint,
            steps=self.num_timesteps,
            held_estimator=self.held_estimator,
        )
        # Checkpoint selection always uses the same full distribution. An easy
        # curriculum stage must not masquerade as full-contract improvement.
        validation_config = replace(
            self.config,
            randomization_scale=1.0,
            target_scale=1.0,
            tracking_precision=4 if self.config.tracking_reward else self.config.tracking_precision,
        )
        actor = ManipulationPolicy(target)
        if self.debug_reset_seed is not None:
            debug = evaluate(actor, config=self.config, episodes=1, seed=self.debug_reset_seed)
            (target / "debug-training-case.json").write_text(json.dumps(debug, indent=2) + "\n")
            self.record(
                {
                    f"debug_training_case/{k}": v
                    for k, v in debug["summary"][self.config.scenario].items()
                    if v is not None
                }
            )
        report = evaluate(actor, config=validation_config, episodes=self.episodes, seed=200000)
        teacher = evaluate(config=validation_config, episodes=self.episodes, seed=200000)
        curriculum = evaluate(
            actor,
            config=replace(self.config, randomization_scale=self.randomization_scale, target_scale=self.target_scale),
            episodes=self.episodes,
            seed=250000,
        )
        (target / "curriculum-validation.json").write_text(json.dumps(curriculum, indent=2) + "\n")
        (target / "teacher-validation.json").write_text(json.dumps(teacher, indent=2) + "\n")
        (target / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        metrics = {
            f"validation/{scenario}/{k}": v
            for scenario, group in report["summary"].items()
            for k, v in group.items()
            if v is not None
        }
        metrics.update(
            {
                f"curriculum_validation/{scenario}/{k}": v
                for scenario, group in curriculum["summary"].items()
                for k, v in group.items()
                if v is not None
            }
        )
        if self.config.base_only_max_distance:
            metrics["curriculum/base_min_distance_m"] = self.config.base_only_min_distance
            metrics["curriculum/base_max_distance_m"] = self.config.base_only_min_distance + self.target_scale * (
                self.config.base_only_max_distance - self.config.base_only_min_distance
            )
        metrics["curriculum/randomization_scale"] = self.randomization_scale
        metrics["curriculum/target_scale"] = self.target_scale
        metrics["curriculum/tracking_precision"] = self.config.tracking_precision
        for scenario, group in teacher["summary"].items():
            metrics[f"teacher/{scenario}/success_rate"] = group["success_rate"]
            metrics[f"paired/{scenario}/success_rate_delta"] = (
                report["summary"][scenario]["success_rate"] - group["success_rate"]
            )
        comparison = paired_comparison(teacher, report)
        (target / "paired-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        metrics["paired/eligible_without_regression"] = comparison["eligible"]
        metrics["paired/common_success_count"] = comparison["paired_success_count"]
        for key in ("teacher_paired_time", "policy_paired_time"):
            if comparison[key] is not None:
                metrics[f"paired/{key}"] = comparison[key]
        self.record(metrics)
        # Advance only after an actual stratified validation gate, not episode
        # count or elapsed training time. Keep validation seeds paired.
        if (
            self.config.tracking_reward
            and self.config.tracking_precision < 4
            and tracking_precision_ready(curriculum["summary"], self.config.tracking_precision, self.episodes)
        ):
            level = self.config.tracking_precision + 1
            self.config = replace(self.config, tracking_precision=level)
            self.model.get_env().env_method("set_tracking_precision", level)
        strict_ready = curriculum_ready(curriculum["summary"], self.episodes)
        coverage_ready = strict_ready or (
            self.config.tracking_reward and target_coverage_ready(curriculum["summary"], self.episodes)
        )
        if coverage_ready:
            self.target_scale = min(1.0, self.target_scale + 0.1)
            self.model.get_env().env_method("set_target_scale", self.target_scale)
        if strict_ready:
            self.randomization_scale = min(1.0, self.randomization_scale + 0.1)
            self.model.get_env().env_method("set_randomization_scale", self.randomization_scale)
        groups = list(report["summary"].values())
        value = (
            min(g["success_rate"] for g in groups),
            float(np.mean([g["success_rate"] for g in groups])),
            -(comparison["policy_paired_time"] or float("inf")),
        )
        if comparison["eligible"] and (self.best is None or value > self.best):
            self.best = value
            export(
                self.model,
                self.model.get_vec_normalize_env(),
                self.directory / "best",
                self.config,
                self.fingerprint,
                steps=self.num_timesteps,
                held_estimator=self.held_estimator,
            )
            (self.directory / "best/validation.json").write_text(json.dumps(report, indent=2) + "\n")
        self.last = self.num_timesteps
        print("Validation", self.num_timesteps, json.dumps(report["summary"]), flush=True)


def training_options(args):
    """Keep imitation optional and use useful exploration for scratch PPO."""
    if args.dagger and args.action_mode != "direct_velocity":
        raise ValueError("DAgger initialization requires direct velocity actions")
    if args.dagger and args.initialization_steps < 5 * args.envs:
        raise ValueError("DAgger needs at least five vector steps")
    std = args.exploration_std
    if std is None:
        std = 0.03 if args.dagger or args.action_mode == "residual" or getattr(args, "initialize_from", None) else 0.3
    if not np.isfinite(std) or not 0 < std <= 1:
        raise ValueError("normalized exploration standard deviation must be in (0, 1]")
    return {
        "initialization": "dagger" if args.dagger else "critic_warmup" if args.action_mode == "residual" else "none",
        "architecture": args.architecture,
        "exploration_std": float(std),
    }


def retention_options(args):
    """Tracking-majority opt-in preset; legacy sampling remains reproducible."""
    from emet.rl.manipulation.retention_training import TRACKING_RETENTION_MIX, validate_mixture

    preset = args.task_mix == "tracking-retention"
    if (preset or args.scenario_weights is not None) and args.scenario != "mixed":
        raise ValueError("task mixtures require --scenario mixed")
    weights = args.scenario_weights if args.scenario_weights is not None else TRACKING_RETENTION_MIX if preset else None
    if weights is not None:
        weights = validate_mixture(weights)
    if preset and weights[0] < 0.5:
        raise ValueError("tracking-retention preset must retain at least 50% ordinary tracking resets")
    context = args.retention_context or preset
    if args.held_object_predictor and not context:
        raise ValueError("held-object predictor requires retention context")
    return {
        "scenario_weights": weights,
        "retention_context": context,
        "empty_grasp_probability": args.empty_grasp_probability
        if args.empty_grasp_probability is not None
        else 0.2
        if preset
        else 0,
        "carry_hold_probability": args.carry_hold_probability
        if args.carry_hold_probability is not None
        else 0.25
        if preset
        else 0,
        "payload_mass_range_kg": args.payload_mass_range_kg
        if args.payload_mass_range_kg is not None
        else (0.02, 0.10)
        if preset and args.payload_mass_kg is None
        else None,
        "payload_com_randomization_m": args.payload_com_randomization_m
        if args.payload_com_randomization_m is not None
        else 0.002
        if preset
        else 0,
    }


def train(args):
    torch.set_num_threads(1)
    options = training_options(args)
    PhaseLearningRate(
        args.learning_rate,
        args.learning_rate if args.final_learning_rate is None else args.final_learning_rate,
        0,
        args.steps,
    )
    if not np.isfinite(args.discount) or not 0 < args.discount < 1:
        raise ValueError("discount must be finite and between zero and one")
    options.update(
        learning_rate=args.learning_rate, final_learning_rate=args.final_learning_rate, discount=args.discount
    )
    if args.resume_from is not None:
        if args.dagger or args.initialization_steps:
            raise ValueError("resume requires zero initialization steps and no DAgger")
        options["resume_from"] = str(args.resume_from.resolve())
    if args.initialize_from is not None:
        if args.resume_from is not None or args.dagger or args.initialization_steps:
            raise ValueError("retention warm-start requires zero initialization steps, no DAgger and no exact resume")
        options["initialize_from"] = str(args.initialize_from.resolve())
    if args.debug_reset_seed is not None:
        if args.scenario != "reach":
            raise ValueError("fixed-case learning audit requires reach")
        options["debug_reset_seed"] = args.debug_reset_seed
    retention = retention_options(args)
    options["task_mix"] = args.task_mix
    options["held_object_predictor"] = (
        args.held_object_predictor if args.held_object_predictor is not None else args.task_mix == "tracking-retention"
    )
    options["held_object_optimizer"] = "separate supervised BCE; no PPO reward or shared gradients"
    config = ManipulationConfig(
        robot=args.robot,
        scenario=args.scenario,
        horizon=args.horizon,
        randomization_scale=args.randomization_scale,
        payload_mass_kg=args.payload_mass_kg,
        payload_asset_set=args.payload_asset_set,
        contact_cone=args.contact_cone,
        contact_impratio=args.contact_impratio,
        balanced_grasp_setup=args.balanced_grasp_setup,
        tamp_carry_trajectories=args.tamp_carry_trajectories,
        carry_target_mode=args.carry_target_mode,
        carry_push_scale=args.carry_push_scale,
        carry_push_probability=args.carry_push_probability,
        grasp_lift_height=args.grasp_lift_height,
        support_half_width=args.support_half_width,
        grasp_pitch_rad=args.grasp_pitch_rad,
        payload_object=args.payload_object,
        initial_heading_range=args.initial_heading_range,
        target_scale=args.target_scale,
        retarget_probability=args.retarget_probability,
        reach_push_probability=args.reach_push_probability,
        certify_reach_paths=args.certify_reach_paths,
        dynamic_references=args.dynamic_references,
        action_mode=args.action_mode,
        base_xy_tolerance=args.base_xy_tolerance,
        base_heading_tolerance=args.base_heading_tolerance,
        base_only_min_distance=args.base_only_min_distance,
        base_only_max_distance=args.base_only_max_distance,
        reach_target_mode=args.reach_target_mode,
        differential_base_reward=args.differential_base_reward,
        hand_position_tolerance=args.hand_position_tolerance,
        hand_orientation_tolerance=args.hand_orientation_tolerance,
        trajectory_preview=args.trajectory_preview,
        tracking_reward=args.tracking_reward,
        tracking_precision=args.tracking_precision,
        endpoint_motion_weight=args.endpoint_motion_weight,
        goal_gated_settling=args.goal_gated_settling,
        **retention,
    )
    # Admit the model before starting workers or a tracking run.
    probe = ManipulationEnv(config)
    fingerprint = probe.fingerprint
    probe.close()
    if config.tamp_carry_trajectories:
        from emet.rl.manipulation.carry_benchmark import canonical_digests, require_training_readiness

        if not args.carry_readiness_report or not args.pick_place_readiness_report:
            raise ValueError("TAMP carry training requires --carry-readiness-report and --pick-place-readiness-report")
        require_training_readiness(
            json.loads(args.carry_readiness_report.read_text()),
            json.loads(args.pick_place_readiness_report.read_text()),
            config,
            fingerprint,
            **canonical_digests(),
        )
        if not args.regression_receipt:
            raise ValueError("TAMP carry training requires a fresh --regression-receipt")
        from emet.rl.regression_gate import verify_regressions

        verify_regressions(args.regression_receipt)
    from emet.rl.manipulation.carry_benchmark import source_hashes as qualification_sources

    for seed in args.seeds:
        directory = Path(args.output) / f"seed-{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "complete.json").exists():
            raise ValueError("refusing to overwrite a completed experiment")
        (directory / "run.json").write_text(
            json.dumps(
                {
                    "env": asdict(config),
                    "seed": seed,
                    "steps": args.steps,
                    "initialization_steps": args.initialization_steps,
                    "training_options": options,
                    "envs": args.envs,
                    "fingerprint": fingerprint,
                    "source_hashes": qualification_sources(),
                    "evaluation_seed": 200000,
                    "eval_every": args.eval_every,
                    "eval_episodes_per_category": args.eval_episodes,
                },
                indent=2,
            )
            + "\n"
        )
        run = None
        if args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project="emet-mars-control",
                mode=args.wandb_mode,
                dir=str(directory),
                name=f"{args.robot}-physical-manipulation-{seed}",
                config={
                    "env": asdict(config),
                    "seed": seed,
                    "steps": args.steps,
                    "initialization_steps": args.initialization_steps,
                    "training_options": options,
                    "fingerprint": fingerprint,
                },
            )
            (directory / "wandb-run.json").write_text(json.dumps({"id": run.id, "url": run.url}) + "\n")
        env = None
        failed = True
        try:
            env = VecNormalize(
                make_vec_env(
                    ManipulationEnv,
                    n_envs=args.envs,
                    seed=seed,
                    env_kwargs={"config": config},
                    wrapper_class=partial(FixedResetSeed, fixed_seed=args.debug_reset_seed)
                    if args.debug_reset_seed is not None
                    else None,
                    vec_env_cls=SubprocVecEnv if args.envs > 1 else DummyVecEnv,
                ),
                norm_reward=False,
                clip_obs=20,
            )
            model = PPO(
                "MultiInputPolicy",
                env,
                device="cpu",
                seed=seed,
                n_steps=256,
                batch_size=256,
                n_epochs=5,
                learning_rate=1e-4,
                target_kl=0.01,
                clip_range=0.1,
                ent_coef=0,
                gamma=args.discount,
                policy_kwargs={
                    "features_extractor_class": ManipulationHistoryFeatures
                    if args.architecture == "history"
                    else ManipulationMLPFeatures,
                    "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                    "log_std_init": -20,
                },
            )
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.zero_()
            if args.resume_from is not None:
                model = resume_training(args.resume_from, env, config, fingerprint, args.architecture)
                model.gamma = model.rollout_buffer.gamma = args.discount
            if args.initialize_from is not None:
                result = initialize_retention_context(model, env, args.initialize_from, config, fingerprint)
                (directory / "retention-warm-start.json").write_text(json.dumps(result, indent=2) + "\n")
            held_estimator = None
            if args.held_object_predictor is True or (
                args.held_object_predictor is None and args.task_mix == "tracking-retention"
            ):
                from emet.rl.manipulation.held_object import HeldObjectEstimator

                if not config.retention_context:
                    raise ValueError("held-object predictor requires --retention-context or tracking-retention mix")
                held_estimator = HeldObjectEstimator(config.robot, env.observation_space["history"].shape, seed=seed)
                if args.resume_from is not None:
                    previous = ManipulationPolicy(args.resume_from).held_estimator
                    if previous is not None and previous.feature_set == held_estimator.feature_set:
                        held_estimator.head.load_state_dict(previous.head.state_dict())
                        held_estimator.class_counts = previous.class_counts.copy()
                        held_estimator.updates = previous.updates
                    elif previous is not None:
                        (directory / "held-object-initialization.json").write_text(
                            json.dumps(
                                {
                                    "reinitialized": True,
                                    "reason": "gripper-local feature contract replaces legacy arm/action inputs",
                                    "previous_feature_set": previous.feature_set,
                                    "feature_set": held_estimator.feature_set,
                                    "ppo_optimizer_restored": True,
                                },
                                indent=2,
                            )
                            + "\n"
                        )
            actor = [
                *model.policy.action_net.parameters(),
                *model.policy.mlp_extractor.policy_net.parameters(),
                model.policy.log_std,
            ]
            for parameter in actor:
                parameter.requires_grad_(False)
            if args.dagger:
                for parameter in actor:
                    parameter.requires_grad_(True)
                from emet.rl.manipulation.imitation import initialize_direct_actor

                result = initialize_direct_actor(model, env, args.initialization_steps)
                (directory / "imitation.json").write_text(json.dumps(result, indent=2) + "\n")
                print("Imitation", result, flush=True)
                if run:
                    run.log(
                        {
                            "imitation/samples": result["samples"],
                            "imitation/final_weighted_mse": result["rounds"][-1]["final_batch_weighted_mse"],
                        }
                    )
            elif config.action_mode == "residual" and args.initialization_steps:
                model.learn(args.initialization_steps)
            for parameter in actor:
                parameter.requires_grad_(True)
            with torch.no_grad():
                model.policy.log_std.fill_(float(np.log(options["exploration_std"])))
            schedule = PhaseLearningRate(
                args.learning_rate,
                args.learning_rate if args.final_learning_rate is None else args.final_learning_rate,
                model.num_timesteps,
                args.steps,
            )
            model.learning_rate = model.lr_schedule = schedule
            monitor = Monitor(
                directory,
                config,
                fingerprint,
                interval=args.eval_every,
                episodes=args.eval_episodes,
                run=run,
                debug_reset_seed=args.debug_reset_seed,
                held_estimator=held_estimator,
            )
            monitor.init_callback(model)
            monitor.num_timesteps = model.num_timesteps
            monitor.checkpoint()
            model.learn(args.steps, callback=monitor, reset_num_timesteps=False)
            monitor.num_timesteps = model.num_timesteps
            if monitor.last != model.num_timesteps:
                monitor.checkpoint()
            export(
                model,
                env,
                directory / "last",
                monitor.config,
                fingerprint,
                steps=model.num_timesteps,
                held_estimator=held_estimator,
            )
            (directory / "complete.json").write_text(
                json.dumps({"steps": model.num_timesteps, "promoted": False}) + "\n"
            )
            failed = False
        finally:
            if env is not None:
                env.close()
            if run is not None:
                # Scalar tracking only. No additional artifact export is authorized.
                run.finish(exit_code=int(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--robot", choices=["stretch", "innate_mars"], default="stretch")
    p.add_argument(
        "--scenario", choices=["mixed", "reach", "pregrasp", "carry", "place", "disturbance"], default="mixed"
    )
    p.add_argument("--horizon", type=float, default=20)
    p.add_argument("--task-mix", choices=["legacy", "tracking-retention"], default="legacy")
    p.add_argument(
        "--scenario-weights",
        type=float,
        nargs=5,
        metavar=("REACH", "PREGRASP", "CARRY", "PLACE", "DISTURBANCE"),
        help="episode probabilities summing to one; tracking-retention requires reach >= .5",
    )
    p.add_argument("--retention-context", action="store_true")
    p.add_argument("--held-object-predictor", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--empty-grasp-probability", type=float, help="closed-empty counterexamples within reach tasks")
    p.add_argument("--carry-hold-probability", type=float, help="stationary load-holding fraction within carry tasks")
    p.add_argument("--payload-mass-range-kg", type=float, nargs=2, metavar=("MIN", "MAX"))
    p.add_argument("--payload-com-randomization-m", type=float)
    p.add_argument("--randomization-scale", type=float, default=0.05)
    p.add_argument("--payload-mass-kg", type=float, help="nominal payload mass before randomization")
    p.add_argument("--payload-asset-set", choices=["training", "native_table"], default="training")
    p.add_argument("--contact-cone", choices=["pyramidal", "elliptic"], default="pyramidal")
    p.add_argument("--contact-impratio", type=float, default=1.0)
    p.add_argument("--balanced-grasp-setup", action="store_true")
    p.add_argument("--tamp-carry-trajectories", action="store_true")
    p.add_argument(
        "--carry-target-mode", choices=["translation", "orientation", "combined", "mixed"], default="combined"
    )
    p.add_argument("--carry-push-scale", type=float, default=0)
    p.add_argument("--carry-push-probability", type=float, default=1)
    p.add_argument("--regression-receipt", type=Path)
    p.add_argument("--carry-readiness-report", type=Path)
    p.add_argument("--pick-place-readiness-report", type=Path)
    p.add_argument("--grasp-lift-height", type=float, default=0.06)
    p.add_argument("--support-half-width", type=float, default=0.012)
    p.add_argument("--grasp-pitch-rad", type=float, default=0.0)
    p.add_argument("--payload-object", choices=["object1", "object2"])
    p.add_argument("--initial-heading-range", type=float, default=float(np.pi))
    p.add_argument("--target-scale", type=float, default=0.25)
    p.add_argument("--debug-reset-seed", type=int, help="overfit one reach case; training diagnostic only")
    p.add_argument(
        "--endpoint-motion-weight", type=float, default=0.0, help="optional bounded endpoint velocity penalty"
    )
    p.add_argument("--retarget-probability", type=float, default=0.25)
    p.add_argument("--reach-push-probability", type=float, default=0.25)
    p.add_argument("--certify-reach-paths", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dynamic-references", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--action-mode", choices=["residual", "direct_velocity"], default="direct_velocity")
    p.add_argument("--base-xy-tolerance", type=float, default=0.05)
    p.add_argument("--base-heading-tolerance", type=float, default=0.04)
    p.add_argument("--base-only-min-distance", type=float, default=0.0)
    p.add_argument("--base-only-max-distance", type=float, default=0.0)
    p.add_argument("--reach-target-mode", choices=["mixed", "base", "hand", "combined"], default="mixed")
    p.add_argument("--differential-base-reward", action="store_true")
    p.add_argument("--goal-gated-settling", action="store_true")
    p.add_argument("--discount", type=float, default=0.99, help="PPO per-control-step discount")
    p.add_argument("--hand-position-tolerance", type=float, default=0.005)
    p.add_argument("--hand-orientation-tolerance", type=float, default=0.05)
    p.add_argument("--trajectory-preview", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tracking-reward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tracking-precision", type=int, choices=range(5), default=0)
    p.add_argument("--dagger", action="store_true", help="optionally initialize the direct actor with DAgger")
    p.add_argument("--architecture", choices=["mlp", "history"], default="mlp")
    p.add_argument(
        "--exploration-std", type=float, help="normalized PPO action noise; default 0.3 scratch / 0.03 initialized"
    )
    p.add_argument("--steps", type=int, default=262144)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--final-learning-rate", type=float, help="linearly decay to this LR over the additional steps")
    p.add_argument("--resume-from", type=Path, help="continue a validated manipulation bundle with its PPO optimizer")
    p.add_argument(
        "--initialize-from",
        type=Path,
        help="warm-start matching legacy actor/critic with two new context inputs; fresh optimizer",
    )
    p.add_argument("--initialization-steps", type=int, default=32768)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-every", type=int, default=32768)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--wandb-mode", choices=["disabled", "online", "offline"], default="disabled")
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--base-heading-tolerance", type=float, default=0.04)
    p.add_argument("--trajectory-preview", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tracking-reward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tracking-precision", type=int, choices=range(5), default=4)
    p.add_argument("--action-mode", choices=["residual", "direct_velocity"], default="direct_velocity")
    p.add_argument("--base-xy-tolerance", type=float, default=0.05)
    p.add_argument("--hand-position-tolerance", type=float, default=0.005)
    p.add_argument("--hand-orientation-tolerance", type=float, default=0.05)
    p.add_argument("--robot", choices=["stretch", "innate_mars"], default="stretch")
    p.add_argument("--policy")
    p.add_argument(
        "--scenario", choices=["mixed", "reach", "pregrasp", "carry", "place", "disturbance"], default="reach"
    )
    p.add_argument("--horizon", type=float, default=20)
    p.add_argument("--certify-reach-paths", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dynamic-references", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--retarget-probability", type=float, default=0.25)
    p.add_argument("--reach-push-probability", type=float, default=0.25)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=300000)
    p.add_argument("--output", required=True)
    p = sub.add_parser("fit-risk")
    p.add_argument("--outcomes", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        if min(args.steps, args.envs, args.eval_every, args.eval_episodes) <= 0 or args.initialization_steps < 0:
            parser.error("positive training settings required")
        try:
            training_options(args)
        except ValueError as error:
            parser.error(str(error))
        train(args)
    elif args.command == "evaluate":
        torch.set_num_threads(1)
        report = evaluate(
            ManipulationPolicy(args.policy) if args.policy else None,
            config=ManipulationConfig(
                robot=args.robot,
                scenario=args.scenario,
                horizon=args.horizon,
                trajectory_preview=args.trajectory_preview,
                tracking_reward=args.tracking_reward,
                tracking_precision=args.tracking_precision,
                action_mode=args.action_mode,
                base_xy_tolerance=args.base_xy_tolerance,
                base_heading_tolerance=args.base_heading_tolerance,
                hand_position_tolerance=args.hand_position_tolerance,
                hand_orientation_tolerance=args.hand_orientation_tolerance,
                initial_heading_range=float(np.pi),
                certify_reach_paths=args.certify_reach_paths,
                dynamic_references=args.dynamic_references,
                retarget_probability=args.retarget_probability,
                reach_push_probability=args.reach_push_probability,
            ),
            episodes=args.episodes,
            seed=args.seed,
        )
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2))
    else:
        from emet.rl.manipulation.risk import fit_risk

        report = json.loads(Path(args.outcomes).read_text())
        rows = [r for r in report["episodes"] if not r["initialization_failed"]]
        first = rows[0]
        result = fit_risk(
            rows,
            args.output,
            robot=first["robot"],
            fingerprint=first["fingerprint"],
            controller_id=first["controller_id"],
        )
        print(json.dumps(result["heads"], indent=2))


if __name__ == "__main__":
    main()
