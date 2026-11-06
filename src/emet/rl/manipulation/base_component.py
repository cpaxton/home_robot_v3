"""Train selected motion parameters of one shared Mars tracking actor.

The default preserves learned arm/context means; whole-body training can adapt
all motion rows and optionally their context network. Export remains one
ordinary eight-output actor; the gripper and executor release authority persist.
"""

import argparse
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import StateDependentNoiseDistribution
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.command_policy import ExecutorCommandVecNormalize
from emet.rl.manipulation.feedback_policy import FeedbackActorCriticPolicy, GoalFeedbackFeatures, RestoringYawAction
from emet.rl.manipulation.mastery import (
    MasteryEnv,
    RolloutDiagnostics,
    TrainingCaseSampler,
    evaluate_suite,
    load_suite,
    preserve_training_rng,
    read_report,
    save_report,
    select_cases,
    summarize,
    validate_preflight,
)
from emet.rl.manipulation.mastery_objective import (
    ContractRewardSchedule,
    TeacherContractReward,
    add_contract_reward_arguments,
    contract_reward_from_args,
)
from emet.rl.manipulation.policy import ManipulationPolicy, export, validate_critic_reward_scale
from emet.rl.regression_gate import verify_regressions

ROOT = Path(__file__).resolve().parents[4]
SUITE = ROOT / "configs/benchmarks/mars_tracking_mastery_command_history_v1.json"
SCOPE = "mars-frozen-arm-base-gsde-v1"


class BaseNoise(StateDependentNoiseDistribution):
    """Six announced base-error features plus a small stationary noise floor."""

    def proba_distribution_net(self, latent_dim, log_std_init=-2, latent_sde_dim=None):
        return super().proba_distribution_net(latent_dim, log_std_init, latent_sde_dim=7)


class BaseMean(RestoringYawAction):
    def __init__(self, context_width, feedback_width, outputs=2):
        super().__init__(context_width, feedback_width, 8)
        self.outputs = outputs

    def forward(self, latent):
        return super().forward(latent)[:, : self.outputs]


class RoleAdam(torch.optim.Adam):
    """Preserve role LR ratios when SB3 updates every group's base LR.

    Group metadata is saved in the ordinary optimizer state. Restoring the base
    rates also keeps SB3 schedules and logged actor LR in their usual units.
    """

    def step(self, closure=None):
        rates = [group["lr"] for group in self.param_groups]
        try:
            for group, rate in zip(self.param_groups, rates, strict=True):
                group["lr"] = rate * group.get("lr_scale", 1.0)
            return super().step(closure)
        finally:
            for group, rate in zip(self.param_groups, rates, strict=True):
                group["lr"] = rate


class BasePolicy(FeedbackActorCriticPolicy):
    motion_outputs = 2
    noise_type = BaseNoise

    def __init__(
        self,
        *args,
        exploration_coefficient_std=0.5,
        critic_lr_multiplier=1.0,
        critic_reward_scale=1.0,
        gain_bias_lr_multiplier=1.0,
        **kwargs,
    ):
        if not np.isfinite(exploration_coefficient_std) or not 0 < exploration_coefficient_std <= 1:
            raise ValueError("exploration coefficient std must be in (0,1]")
        self.exploration_coefficient_std = exploration_coefficient_std
        if not np.isfinite(critic_lr_multiplier) or critic_lr_multiplier <= 0:
            raise ValueError("positive finite critic LR multiplier required")
        self.critic_lr_multiplier = critic_lr_multiplier
        if not np.isfinite(gain_bias_lr_multiplier) or gain_bias_lr_multiplier <= 0:
            raise ValueError("positive finite gain-bias LR multiplier required")
        self.gain_bias_lr_multiplier = gain_bias_lr_multiplier
        self.critic_reward_scale = validate_critic_reward_scale(critic_reward_scale)
        if (
            not kwargs.get("use_sde", False)
            or not kwargs.get("full_std", True)
            or kwargs.get("squash_output", False)
            or kwargs.get("use_expln", False)
        ):
            raise ValueError("base component requires full-std unsquashed gSDE")
        super().__init__(*args, **kwargs)
        if (
            self.action_space.shape != (self.motion_outputs,)
            or not self.restoring_yaw_feedback
            or self.features_extractor.robot != "innate_mars"
        ):
            raise ValueError("Mars restoring-yaw motion component required")

    def _build(self, lr_schedule):
        self.action_dist = self.noise_type(self.motion_outputs, full_std=True, epsilon=1e-8)
        super()._build(lr_schedule)
        self.action_net = BaseMean(
            self.mlp_extractor.context_width, self.features_extractor.feedback_width, outputs=self.motion_outputs
        ).to(self.device)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.mlp_extractor.value_net, self.value_net, self.action_net.turn_gain):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if getattr(self, "train_context", False):
            for parameter in self.mlp_extractor.policy_net.parameters():
                parameter.requires_grad_(True)
        width = self.features_extractor.feedback_width
        for parameter in (self.action_net.gain.weight, self.action_net.gain.bias):
            parameter.requires_grad_(True)
            mask = torch.zeros_like(parameter).reshape(7, width, -1)
            mask[: self.motion_outputs] = 1
            parameter.register_hook(lambda grad, mask=mask.reshape(parameter.shape): grad * mask)
        self.log_std.requires_grad_(True)
        noise_mask, coefficients = self.noise_configuration()
        with torch.no_grad():
            self.log_std.fill_(-20)
            self.log_std[noise_mask.bool()] = coefficients[noise_mask.bool()].log()
        self.log_std.register_hook(lambda grad: grad * noise_mask)
        self._actor_parameter_names = tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad and not self.is_critic(name)
        )
        if self.critic_lr_multiplier == 1 and self.gain_bias_lr_multiplier == 1:
            self.optimizer = self.optimizer_class(
                [p for p in self.parameters() if p.requires_grad], lr=lr_schedule(1), **self.optimizer_kwargs
            )
        else:
            if self.optimizer_class is not torch.optim.Adam:
                raise ValueError("role-scaled base training requires Adam")

            def role(name):
                if self.is_critic(name):
                    return "critic"
                if self.gain_bias_lr_multiplier != 1 and name == "action_net.gain.bias":
                    return "gain_bias"
                return "actor"

            scales = {"actor": 1.0, "critic": self.critic_lr_multiplier}
            if self.gain_bias_lr_multiplier != 1:
                scales["gain_bias"] = self.gain_bias_lr_multiplier
            groups = [
                {
                    "params": [
                        p for name, p in self.named_parameters() if p.requires_grad and role(name) == group_role
                    ],
                    "role": group_role,
                    "lr_scale": scale,
                }
                for group_role, scale in scales.items()
            ]
            self.optimizer = RoleAdam(groups, lr=lr_schedule(1), **self.optimizer_kwargs)

    def noise_configuration(self):
        mask = torch.zeros_like(self.log_std)
        mask[[0, 3, 6], 0] = 1
        mask[[1, 2, 4, 5, 6], 1] = 1
        return mask, torch.full_like(mask, self.exploration_coefficient_std)

    @staticmethod
    def is_critic(name):
        return name.startswith(("mlp_extractor.value_net.", "value_net."))

    def set_actor_trainable(self, enabled):
        """Toggle originally selected motion, context and exploration parameters."""
        for name, parameter in self.named_parameters():
            if name in self._actor_parameter_names:
                parameter.grad = None
                parameter.requires_grad_(enabled)

    def _get_action_dist_from_latent(self, latent_pi):
        errors = latent_pi[:, -self.features_extractor.feedback_width :]
        noise = torch.cat([errors[:, [0, 1, 2, 12, 13, 14]], errors.new_full((len(errors), 1), 0.002)], dim=1)
        return self.action_dist.proba_distribution(self.action_net(latent_pi), self.log_std, noise)


class CriticWarmup(BaseCallback):
    """Fit values on the policy's own noisy rollouts before changing its mean.

    Warmup uses long-return targets (GAE lambda 1); ordinary PPO settings are
    restored at a rollout boundary. No expert actions or runtime filter.
    """

    def __init__(self, steps=0, epochs=20):
        super().__init__()
        if type(steps) is not int or steps < 0 or steps % 256:
            raise ValueError("critic warmup must be a nonnegative multiple of 256")
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("positive integer critic warmup epochs required")
        self.steps, self.epochs = steps, epochs
        self.active = False
        self.actor_unchanged = None

    def _on_training_start(self):
        self.normal_epochs = self.model.n_epochs
        self.normal_lambda = self.model.gae_lambda
        self.actor_before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.policy.named_parameters()
            if name in self.model.policy._actor_parameter_names
        }

    def _on_rollout_start(self):
        warmup = self.num_timesteps < self.steps
        if self.active and not warmup:
            self.verify_actor()
        self.active = warmup
        self.model.policy.set_actor_trainable(not warmup)
        self.model.n_epochs = self.epochs if warmup else self.normal_epochs
        self.model.gae_lambda = 1.0 if warmup else self.normal_lambda
        self.model.rollout_buffer.gae_lambda = self.model.gae_lambda
        self.logger.record("train/critic_warmup", int(warmup))
        self.logger.record("train/critic_lr_multiplier", self.model.policy.critic_lr_multiplier)
        self.logger.record("train/gain_bias_lr_multiplier", self.model.policy.gain_bias_lr_multiplier)

    def verify_actor(self):
        current = dict(self.model.policy.named_parameters())
        self.actor_unchanged = all(torch.equal(current[name], value) for name, value in self.actor_before.items())
        if not self.actor_unchanged:
            raise RuntimeError("actor changed during critic-only warmup")

    def _on_step(self):
        return True

    def _on_training_end(self):
        if self.active:
            self.verify_actor()
        self.model.policy.set_actor_trainable(True)
        self.model.n_epochs, self.model.gae_lambda = self.normal_epochs, self.normal_lambda
        self.model.rollout_buffer.gae_lambda = self.normal_lambda
        self.active = False


class FrozenArmActions(gym.Wrapper):
    """Only base requests are sampled; the preserved arm uses ordinary inputs."""

    def __init__(self, env, arm):
        super().__init__(env)
        native = env.unwrapped
        if (
            native.profile.robot != "innate_mars"
            or arm.robot != native.profile.robot
            or arm.manifest["fingerprint"] != native.fingerprint
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


class TrainingRewardScale(gym.Wrapper):
    """Constant positive units conversion, without clipping or reward reshaping.

    Reward terms and physical evaluation remain in native units. PPO rewards and
    critic predictions use scaled units; explicitly record both in diagnostics.
    """

    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = validate_critic_reward_scale(scale)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return (
            obs,
            reward * self.scale,
            done,
            truncated,
            {
                **info,
                "native_training_reward": float(reward),
                "training_reward_scale": self.scale,
            },
        )


def stage_step_weights(stages, arm_fraction=None, coupled_fraction=None):
    if arm_fraction is None and coupled_fraction is None:
        if set(stages) == {"coupled"}:
            return {"coupled": 1.0}
        if "coupled" in stages:
            raise ValueError("mixed coupled training requires explicit arm/coupled fractions")
        return None
    expected = {"base", "arm"} | ({"coupled"} if coupled_fraction is not None else set())
    fractions = [arm_fraction] + ([] if coupled_fraction is None else [coupled_fraction])
    if set(stages) != expected or any(v is None or not np.isfinite(v) or not 0 < v < 1 for v in fractions):
        raise ValueError("positive arm/coupled fractions require exactly the selected base/arm[/coupled] stages")
    if sum(fractions) >= 1:
        raise ValueError("stage fractions must leave a positive base share")
    return {
        "base": 1 - sum(fractions),
        "arm": arm_fraction,
        **({"coupled": coupled_fraction} if coupled_fraction else {}),
    }


def motion_rollout_size(num_envs, backend):
    """Keep per-worker horizons fixed and count PPO transitions across worlds."""
    if type(num_envs) is not int or not 1 <= num_envs <= 8 or backend not in {"dummy", "subproc"}:
        raise ValueError("1-8 motion environments and dummy/subproc backend required")
    return 256 * num_envs


def make_motion_vec_env(
    suite,
    stages,
    source_directory,
    *,
    train_arm=False,
    num_envs=1,
    backend="subproc",
    stage_weights=None,
    teacher=None,
    objective=None,
    schedule=ContractRewardSchedule(),
    reward_scale=1.0,
):
    """Spawn independent physics and frozen-source inference inside each worker.

    No live MuJoCo pointer, loaded Torch policy or simulator is serialized from
    the parent. Single-worker and dummy execution use the same factory.
    """
    motion_rollout_size(num_envs, backend)
    reward_scale = validate_critic_reward_scale(reward_scale)
    if objective is not None and teacher is None:
        raise ValueError("motion contract objective requires paired teacher evidence")
    source_directory = str(Path(source_directory).resolve())

    def factory():
        torch.set_num_threads(1)
        native = MasteryEnv(
            suite, stages=stages, split="train", population="training", training_stage_weights=stage_weights
        )
        try:
            if stage_weights is not None:
                native.sampler = TrainingCaseSampler(
                    native.cases,
                    stage_weights,
                    sampling="control_steps",
                    primitive_weights=suite.get("training_primitive_weights"),
                )
            source = ManipulationPolicy(source_directory, robot="innate_mars", fingerprint=native.fingerprint)
            wrapped = (
                native
                if objective is None
                else TeacherContractReward(native, teacher, objective, fraction=schedule.fraction(0))
            )
            if reward_scale != 1:
                wrapped = TrainingRewardScale(wrapped, reward_scale)
            if train_arm:
                from emet.rl.manipulation.whole_body import FrozenGripActions

                return FrozenGripActions(wrapped, source)
            return FrozenArmActions(wrapped, source)
        except BaseException:
            native.close()
            raise

    factories = [factory for _ in range(num_envs)]
    return (
        DummyVecEnv(factories)
        if num_envs == 1 or backend == "dummy"
        else SubprocVecEnv(factories, start_method="spawn")
    )


class MotionSamplingDiagnostics:
    """Count starts and control steps for every worker, including auto resets."""

    def __init__(self, num_envs):
        self.steps, self.starts = Counter(), Counter()
        self.first = [True] * num_envs
        self.rollout = RolloutDiagnostics()

    def add(self, actions, clipped, rewards, infos, dones):
        for index, (action, request, reward, info, done) in enumerate(
            zip(actions, clipped, rewards, infos, dones, strict=True)
        ):
            cid = info["mastery"]["case_id"]
            self.steps[cid] += 1
            if self.first[index]:
                self.starts[cid] += 1
            self.first[index] = bool(done)
            self.rollout.add(cid, action, request, reward, info, done)


def initialize(model, normalizer, source, config, fingerprint):
    if (
        source.manifest["environment"] != json.loads(json.dumps(asdict(config)))
        or source.manifest["fingerprint"] != fingerprint
    ):
        raise ValueError("exact source environment/model required")
    if getattr(model.policy, "train_context", False) and torch.count_nonzero(
        source.model.policy.action_net.grip.weight
    ):
        raise ValueError(
            "context adaptation requires a constant gripper head to preserve training/export grip authority"
        )
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
    with torch.no_grad():
        ratio = model.policy.critic_reward_scale / source.critic_reward_scale
        model.policy.value_net.weight.mul_(ratio)
        model.policy.value_net.bias.mul_(ratio)
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
    if getattr(model.policy, "train_context", False) and (
        torch.count_nonzero(model.policy.action_net.grip.weight)
        or torch.count_nonzero(source.model.policy.action_net.grip.weight)
    ):
        raise RuntimeError("context adaptation requires a constant gripper head")
    for key, value in current.items():
        if getattr(model.policy, "train_context", False) and key.startswith("mlp_extractor.policy_net."):
            continue
        if key in {"log_std", "value_net.weight", "value_net.bias"} or key.startswith(
            ("mlp_extractor.value_net.", "action_net.turn_gain.")
        ):
            continue
        if key in {"action_net.gain.weight", "action_net.gain.bias"}:
            width = model.policy.features_extractor.feedback_width
            count = model.policy.motion_outputs
            value, expected = value.reshape(7, width, -1)[count:], original[key].reshape(7, width, -1)[count:]
        else:
            expected = original[key]
        if not torch.equal(value, expected):
            raise RuntimeError("frozen arm/context changed: " + key)


def verify_source_fidelity(report, baseline, *, expected_ids):
    """Permit repair of declared failures only when transfer is physically exact."""
    expected = {c["id"]: c for c in baseline["cases"]}
    if (
        not report.get("complete")
        or not baseline.get("complete")
        or len(expected) != len(baseline["cases"])
        or {c["id"] for c in report["cases"]} != set(expected_ids)
        or not set(expected_ids).issubset(expected)
        or report["suite_sha256"] != baseline["suite_sha256"]
        or report["source_hashes"] != baseline["source_hashes"]
    ):
        raise RuntimeError("source fidelity coverage or source mismatch")
    seen = set()
    for case in report["cases"]:
        cid = case["id"]
        if cid in seen or cid not in expected:
            raise RuntimeError("source fidelity case identity mismatch")
        seen.add(cid)
        original = expected[cid]
        keys = ("admitted", "code", "mastered", "metrics", "failure_reasons")
        identity = ("initial_state_sha256", "reference_sha256")
        trace_keys = ("action", "reward", "hand", "base")
        if (
            any(case[k] != original[k] for k in keys)
            or any(case["admission"][k] != original["admission"][k] for k in identity)
            or len(case["trace"]) != len(original["trace"])
            or any(any(a[k] != b[k] for k in trace_keys) for a, b in zip(case["trace"], original["trace"], strict=True))
        ):
            raise RuntimeError("source transfer changed physical behavior: " + cid)
    if not seen:
        raise RuntimeError("empty source fidelity evaluation")
    return {
        "exact": True,
        "cases": len(seen),
        "source_failures": {
            cid: expected[cid]["failure_reasons"] for cid in sorted(seen) if not expected[cid]["mastered"]
        },
    }


def merge(model, normalizer, source, destination, config, fingerprint):
    """Publish one validated actor bundle; a failed export exposes no partial bundle."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("merged actor destination already exists: " + str(destination))
    verify_frozen(model, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = {k: v.clone() for k, v in source.model.policy.state_dict().items()}
    original_steps = source.model.num_timesteps
    # Stage on the same filesystem: critic units and component provenance must
    # agree with the merged weights before a consumer can see this destination.
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        try:
            full = {k: v.clone() for k, v in original.items()}
            for key, value in model.policy.state_dict().items():
                if key != "log_std":
                    full[key] = value.clone()
            source.model.policy.load_state_dict(full)
            source.model.num_timesteps = model.num_timesteps
            export(source.model, normalizer, staging, config, fingerprint, steps=model.num_timesteps)
            path = staging / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["critic_reward_scale"] = validate_critic_reward_scale(model.policy.critic_reward_scale)
            manifest["component_training"] = {
                "scope": "mars-whole-body-gsde-v1" if model.policy.motion_outputs == 7 else SCOPE,
                "optimizer_resume_supported": False,
                "one_runtime_actor": True,
                "actor_context_trainable": getattr(model.policy, "train_context", False),
            }
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            policy = ManipulationPolicy(staging, robot=config.robot, fingerprint=fingerprint)
        finally:
            source.model.policy.load_state_dict(original)
            source.model.num_timesteps = original_steps
        os.rename(staging, destination)
    return policy


def train(args, suite):
    num_envs, backend = getattr(args, "num_envs", 1), getattr(args, "vec_backend", "subproc")
    rollout_size = motion_rollout_size(num_envs, backend)
    if (
        args.steps <= 0
        or args.steps % rollout_size
        or getattr(args, "evaluation_every", 8192) <= 0
        or getattr(args, "evaluation_every", 8192) % rollout_size
        or getattr(args, "critic_warmup_steps", 0) % rollout_size
    ):
        raise ValueError("steps, evaluation interval and critic warmup must align to 256 * num_envs")
    verify_regressions(args.regression_receipt)
    teacher = read_report(args.preflight)
    stages = getattr(args, "training_stages", ["base", "arm"])
    train_arm = getattr(args, "train_arm", False)
    if getattr(args, "train_context", False) and not train_arm:
        raise ValueError("context adaptation requires joint motion training")
    policy_type = BasePolicy
    if train_arm:
        from emet.rl.manipulation.whole_body import WholeBodyPolicy

        policy_type = WholeBodyPolicy
    evaluation_stages = ["base", "arm"] + (["coupled"] if "coupled" in stages else [])
    validate_preflight(teacher, suite, evaluation_stages)
    source = ManipulationPolicy(args.source)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    evaluation_every = getattr(args, "evaluation_every", 8192)
    objective_config = getattr(args, "contract_reward_config", None)
    schedule = getattr(args, "contract_schedule", ContractRewardSchedule())
    admission_mode = getattr(args, "initial_admission", "strict")
    if admission_mode not in {"strict", "source_fidelity"}:
        raise ValueError("unknown initial admission mode")
    learning_rate = getattr(args, "learning_rate", 1e-4)
    exploration_std = getattr(args, "exploration_coefficient_std", 0.5)
    reward_scale = validate_critic_reward_scale(getattr(args, "reward_scale", 1.0))
    critic_multiplier = getattr(args, "critic_lr_multiplier", 1.0)
    gain_bias_multiplier = getattr(args, "gain_bias_lr_multiplier", 1.0)
    warmup = CriticWarmup(getattr(args, "critic_warmup_steps", 0), getattr(args, "critic_warmup_epochs", 20))
    if warmup.steps >= args.steps:
        raise ValueError("critic warmup must leave at least one actor rollout")
    weights = stage_step_weights(
        stages, getattr(args, "arm_step_fraction", None), getattr(args, "coupled_step_fraction", None)
    )
    vector = make_motion_vec_env(
        suite,
        stages,
        args.source,
        train_arm=train_arm,
        num_envs=num_envs,
        backend=backend,
        stage_weights=weights,
        teacher=teacher,
        objective=objective_config,
        schedule=schedule,
        reward_scale=reward_scale,
    )
    try:
        configs, fingerprints = vector.get_attr("config"), vector.get_attr("fingerprint")
        config, fingerprint = configs[0], fingerprints[0]
        if any(c != config for c in configs) or any(f != fingerprint for f in fingerprints):
            raise ValueError("motion workers disagree on robot configuration/model")
        sampler = vector.get_attr("sampler")[0]
        env = ExecutorCommandVecNormalize(vector, config=config, norm_reward=False)
    except BaseException:
        vector.close()
        raise
    run = None
    sources = source_hashes()
    try:
        model = PPO(
            policy_type,
            env,
            device="cpu",
            seed=90,
            n_steps=256,
            batch_size=256,
            n_epochs=5,
            learning_rate=learning_rate,
            gamma=0.9975,
            target_kl=0.01,
            clip_range=0.1,
            use_sde=True,
            sde_sample_freq=8,
            policy_kwargs={
                "exploration_coefficient_std": exploration_std,
                "critic_lr_multiplier": critic_multiplier,
                "critic_reward_scale": reward_scale,
                "gain_bias_lr_multiplier": gain_bias_multiplier,
                **({"arm_exploration_coefficient_std": args.arm_exploration_coefficient_std} if train_arm else {}),
                **({"train_context": getattr(args, "train_context", False)} if train_arm else {}),
                "restoring_yaw_feedback": True,
                "turn_xy_tolerance": config.base_xy_tolerance,
                "features_extractor_class": GoalFeedbackFeatures,
                "features_extractor_kwargs": {
                    "robot": "innate_mars",
                    "retention_context": True,
                    "orientation_scale": config.feedback_orientation_scale,
                },
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            },
        )
        initialize(model, env, source, config, fingerprint)
        record = {
            "scope": "mars-whole-body-gsde-v1" if train_arm else SCOPE,
            "suite": suite,
            "suite_sha256": digest(suite),
            "source_hashes": sources,
            "source_weights_sha256": source.manifest["weights_sha256"],
            "source_manifest_sha256": digest(source.manifest),
            "teacher_report_sha256": digest(teacher),
            "regression_receipt": json.loads(args.regression_receipt.read_text()),
            "steps": args.steps,
            "training_stages": stages,
            "evaluation_stages": evaluation_stages,
            "training_reward_scale": reward_scale,
            "rollout_reward_terms_units": "native; sampled rewards and critic values use training_reward_scale",
            "training_stage_weights": sampler.weights,
            "effective_stage_probabilities": dict(zip(sampler.groups, sampler.probabilities, strict=True)),
            "evaluation_every": evaluation_every,
            "num_envs": num_envs,
            "vec_backend": "dummy" if num_envs == 1 else backend,
            "rollout_transitions": rollout_size,
            "contract_reward": None if objective_config is None else asdict(objective_config),
            "contract_schedule": None if objective_config is None else asdict(schedule),
            "initial_admission": admission_mode,
            "seed": 90,
            "frozen_normalization": True,
            "frozen_actor_context_and_arm": not train_arm,
            "frozen_actor_context_and_grip": not getattr(args, "train_context", False),
            "frozen_grip_head": True,
            "actor_context_trainable": getattr(args, "train_context", False),
            "arm_trainable": train_arm,
            "arm_exploration_coefficient_std": args.arm_exploration_coefficient_std if train_arm else None,
            "teacher_action_loss": False,
            "sde_sample_freq": 8,
            "sde_noise_features": (
                "masked base/hand errors, free-base hand errors, constant .002"
                if train_arm
                else "current/future base errors plus constant .002"
            ),
            "initial_sde_coefficient_std": exploration_std,
            "ppo": {
                "learning_rate": learning_rate,
                "critic_lr_multiplier": critic_multiplier,
                "gain_bias_lr_multiplier": gain_bias_multiplier,
                "critic_warmup_steps": warmup.steps,
                "critic_warmup_epochs": warmup.epochs,
                "critic_warmup_gae_lambda": 1.0,
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
                project="emet-mars-control", mode="offline", dir=str(args.output), name=args.output.name, config=record
            )

        class Audit(BaseCallback):
            def __init__(self):
                super().__init__()
                self.sampling = MotionSamplingDiagnostics(num_envs)
                self.steps, self.starts = self.sampling.steps, self.sampling.starts
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
                    policy = merge(model, env, source, directory / "actor", config, fingerprint)
                    policy.manifest["training_recipe"] = {
                        "run_sha256": digest(record),
                        "training_stages": stages,
                        "contract_reward": record["contract_reward"],
                        "contract_schedule": record["contract_schedule"],
                        "initial_admission": admission_mode,
                        "critic_lr_multiplier": critic_multiplier,
                        "gain_bias_lr_multiplier": gain_bias_multiplier,
                        "training_reward_scale": reward_scale,
                        "critic_warmup_steps": warmup.steps,
                        "actor_unchanged_during_critic_warmup": warmup.actor_unchanged,
                    }
                    (directory / "actor/manifest.json").write_text(json.dumps(policy.manifest, indent=2) + "\n")
                    report = evaluate_suite(suite, stages=evaluation_stages, policy=policy, teacher_report=teacher)
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
                self.sampling.add(
                    self.locals["actions"],
                    self.locals["clipped_actions"],
                    self.locals["rewards"],
                    self.locals["infos"],
                    self.locals["dones"],
                )
                return True

            def _on_rollout_start(self):
                self.record_update()
                if objective_config is not None:
                    env.env_method("set_reward_fraction", schedule.fraction(self.num_timesteps))
                if self.num_timesteps and (
                    self.num_timesteps % evaluation_every == 0 or self.num_timesteps == warmup.steps
                ):
                    self.checkpoint(f"step-{self.num_timesteps}")

            def _on_rollout_end(self):
                verify_frozen(model, source)
                values = {
                    "steps": self.num_timesteps,
                    "sampling_steps": dict(self.steps),
                    "sampling_starts": dict(self.starts),
                    "sde_coefficients": model.policy.log_std.detach().exp().cpu().tolist(),
                    "base_gain_bias": model.policy.action_net.gain.bias.detach()
                    .reshape(7, model.policy.features_extractor.feedback_width)[:2]
                    .cpu()
                    .tolist(),
                    "motion_gain_bias": model.policy.action_net.gain.bias.detach()
                    .reshape(7, model.policy.features_extractor.feedback_width)[: model.policy.motion_outputs]
                    .cpu()
                    .tolist(),
                    "rollout_cases": self.sampling.rollout.record(),
                    "requested_contract_fraction": schedule.fraction(self.num_timesteps) if objective_config else None,
                    "critic_warmup": warmup.active,
                    "actor_unchanged_during_critic_warmup": warmup.actor_unchanged,
                }
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(values) + "\n")
                if run:
                    run.log({"steps": self.num_timesteps, **{f"sampling/{k}": v for k, v in self.steps.items()}})
                self.sampling.rollout = RolloutDiagnostics()

            def _on_training_end(self):
                self.record_update()

        audit = Audit()
        audit.init_callback(model)
        baseline = None
        if admission_mode == "source_fidelity":
            with preserve_training_rng():
                baseline = evaluate_suite(
                    suite, stages=evaluation_stages, policy=source, teacher_report=teacher, population="admission"
                )
            save_report(args.output / "source-baseline.json.gz", baseline)
            if any(
                c["final_info"].get(flag, False)
                for c in baseline["cases"]
                for flag in ("unsafe_contact", "wheel_lift", "invalid_physics", "retention_lost")
            ):
                raise RuntimeError("source repair requires a physically safe baseline")
        initial = audit.checkpoint("step-0")
        if admission_mode == "strict" and not all(r["mastered"] for r in initial["cases"] if r["stage"] == "arm"):
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
        expected_arm = select_cases(suite, ["arm"], "all", population="admission")
        if len(report["cases"]) != len(expected_arm) or (
            admission_mode == "strict" and not all(r["mastered"] for r in report["cases"])
        ):
            raise RuntimeError("initial arm admission failed")
        if baseline is not None:
            fidelity = {
                "canonical": verify_source_fidelity(
                    initial, baseline, expected_ids={c["id"] for c in select_cases(suite, evaluation_stages, "all")}
                ),
                "expanded_arm": verify_source_fidelity(
                    report,
                    baseline,
                    expected_ids={c["id"] for c in select_cases(suite, ["arm"], "all", population="admission")},
                ),
                "source_weights_sha256": source.manifest["weights_sha256"],
                "source_safety_flags_clear": True,
                "qualification_granted": False,
            }
            (args.output / "initial-fidelity.json").write_text(json.dumps(fidelity, indent=2) + "\n")
        model.learn(args.steps, callback=[warmup, audit])
        final = audit.checkpoint("last")
        with preserve_training_rng():
            expanded = evaluate_suite(
                suite,
                stages=evaluation_stages,
                policy=ManipulationPolicy(args.output / "last/actor"),
                teacher_report=teacher,
                population="admission",
            )
        save_report(args.output / "final-expanded.json.gz", expanded)
        (args.output / "complete.json").write_text(
            json.dumps(
                {
                    "steps": model.num_timesteps,
                    "actor_unchanged_during_critic_warmup": warmup.actor_unchanged,
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
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vec-backend", choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--training-stages", nargs="+", choices=["base", "arm", "coupled"], default=["base", "arm"])
    parser.add_argument("--evaluation-every", type=int, default=8192)
    parser.add_argument("--arm-step-fraction", type=float)
    parser.add_argument("--coupled-step-fraction", type=float)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--train-arm", action="store_true", help="train all seven motion rates; retain executor gripper"
    )
    parser.add_argument("--arm-exploration-coefficient-std", type=float, default=0.03)
    parser.add_argument(
        "--train-context", action="store_true", help="also adapt motion context; requires a constant source grip head"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--exploration-coefficient-std", type=float, default=0.5)
    parser.add_argument("--critic-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--gain-bias-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--critic-warmup-steps", type=int, default=0)
    parser.add_argument("--critic-warmup-epochs", type=int, default=20)
    parser.add_argument("--initial-admission", choices=["strict", "source_fidelity"], default="strict")
    add_contract_reward_arguments(parser)
    parser.add_argument("--wandb-mode", choices=["disabled", "offline"], default="disabled")
    args = parser.parse_args()
    if args.train_context and not args.train_arm:
        parser.error("--train-context requires --train-arm")
    try:
        rollout_size = motion_rollout_size(args.num_envs, args.vec_backend)
    except ValueError as exc:
        parser.error(str(exc))
    if args.steps <= 0 or args.steps % rollout_size:
        parser.error("steps must be a positive multiple of 256 * num_envs")
    if args.evaluation_every <= 0 or args.evaluation_every % rollout_size:
        parser.error("evaluation interval must be a positive multiple of 256 * num_envs")
    if args.critic_warmup_steps % rollout_size:
        parser.error("critic warmup must align to 256 * num_envs")
    if len(set(args.training_stages)) != len(args.training_stages):
        parser.error("duplicate training stages")
    try:
        args.contract_reward_config = contract_reward_from_args(args)
        args.contract_schedule = ContractRewardSchedule(args.contract_warmup_steps, args.contract_ramp_steps)
    except ValueError as exc:
        parser.error(str(exc))
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("positive finite learning rate required")
    if not np.isfinite(args.critic_lr_multiplier) or args.critic_lr_multiplier <= 0:
        parser.error("positive finite critic LR multiplier required")
    if not np.isfinite(args.gain_bias_lr_multiplier) or args.gain_bias_lr_multiplier <= 0:
        parser.error("positive finite gain-bias LR multiplier required")
    try:
        stage_step_weights(args.training_stages, args.arm_step_fraction, args.coupled_step_fraction)
        validate_critic_reward_scale(args.reward_scale)
        CriticWarmup(args.critic_warmup_steps, args.critic_warmup_epochs)
    except ValueError as exc:
        parser.error(str(exc))
    if args.critic_warmup_steps >= args.steps:
        parser.error("critic warmup must leave at least one actor rollout")
    train(args, load_suite(args.suite))


if __name__ == "__main__":
    main()
