# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Minimal PPO trainer for the whole-body torque contract.

This is the first training consumer, not a qualified controller. It trains the
normalized torque action from :mod:`wbc_contract` against the adapter-matched
environment in :mod:`wbc_env`, and writes a checkpoint carrying the full
profile fingerprint so a mismatched embodiment cannot load silently.

Example:
    python -m emet.rl.manipulation.wbc_training --steps 2048 --output runs/wbc
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from emet.controller.manipulation.wbc_contract import SCHEMA_VERSION, EmbodimentProfile


@dataclass(frozen=True)
class WbcPPOConfig:
    hidden: int = 128
    learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    batch_size: int = 64
    rollout_steps: int = 256
    entropy_coefficient: float = 0.001
    bc_kl_coefficient: float = 0.0
    max_grad_norm: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.hidden) is not int or self.hidden <= 0:
            raise ValueError("hidden width must be a positive integer")
        if type(self.epochs) is not int or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size < 2:
            raise ValueError("batch_size must be at least two")
        if type(self.rollout_steps) is not int or self.rollout_steps < 2:
            raise ValueError("rollout_steps must be at least two")
        if not np.isfinite(
            [
                self.learning_rate,
                self.critic_learning_rate,
                self.gamma,
                self.gae_lambda,
                self.clip,
                self.entropy_coefficient,
                self.bc_kl_coefficient,
                self.max_grad_norm,
            ]
        ).all():
            raise ValueError("PPO hyperparameters must be finite")
        if (
            self.learning_rate <= 0
            or self.critic_learning_rate <= 0
            or self.max_grad_norm <= 0
            or self.entropy_coefficient < 0
            or self.bc_kl_coefficient < 0
        ):
            raise ValueError("invalid PPO learning rate/gradient/entropy settings")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or not 0 < self.clip < 1:
            raise ValueError("gamma, lambda, and clip are out of range")


class WbcActorCritic(nn.Module):
    """Gaussian actor with tanh-bounded normalized actions and a value head."""

    def __init__(self, observation_size: int, action_size: int, hidden: int = 128) -> None:
        super().__init__()
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.actor = nn.Sequential(
            nn.Linear(self.observation_size, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.observation_size, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((self.action_size,), -1.0))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(observation), self.critic(observation).squeeze(-1)

    def actor_distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(observation), self.log_std.clamp(-5.0, 0.5).exp())

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        return self.actor_distribution(observation)

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)

    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = distribution.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1), self.value(observation)

    def evaluate(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        clipped = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(clipped)
        log_prob = distribution.log_prob(pre_tanh) - torch.log(1.0 - clipped.pow(2) + 1e-6)
        return log_prob.sum(-1), distribution.entropy().sum(-1), self.value(observation)

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(observation)
        return torch.tanh(mean)


def _expand_layout_weights(state_dict: dict, old_layout, new_layout) -> dict:
    """Move v2 observation columns into the v3 layout, zeroing contact inputs.

    v3 also normalizes the gripper-force channel by a fixed physical scale.
    Rescaling the old column makes ``w_old * force == w_new * (force / scale)``
    for a migrated policy.
    """
    from emet.rl.manipulation.wbc_env import GRIPPER_FORCE_SCALE_N

    expanded: dict = {}
    migrated = 0
    for key, value in state_dict.items():
        if key.endswith("0.weight") and torch.is_tensor(value) and value.ndim == 2:
            if value.shape[1] != old_layout.size:
                raise ValueError(f"{key} has {value.shape[1]} observation columns, expected {old_layout.size}")
            weight = torch.zeros((value.shape[0], new_layout.size), dtype=value.dtype)
            for name, _ in old_layout.fields:
                if name not in new_layout.slices:
                    raise ValueError(f"legacy layout field {name!r} is missing from the new layout")
                scale = GRIPPER_FORCE_SCALE_N if name.endswith(".gripper_force") else 1.0
                weight[:, new_layout.slices[name]] = value[:, old_layout.slices[name]] * scale
            expanded[key] = weight
            migrated += 1
        else:
            expanded[key] = value
    if migrated == 0:
        raise ValueError("v2 checkpoint has no first-layer observation weights to migrate")
    return expanded


class WbcPPO:
    """Single-environment PPO with a profile-checked checkpoint format."""

    format = "emet-wbc-ppo-v1"

    def __init__(
        self,
        policy: WbcActorCritic,
        config: WbcPPOConfig = WbcPPOConfig(),
        *,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.config = config
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": [*self.policy.actor.parameters(), self.policy.log_std],
                    "lr": config.learning_rate,
                },
                {"params": self.policy.critic.parameters(), "lr": config.critic_learning_rate},
            ]
        )
        self.steps_trained = 0
        self.updates = 0
        self.reference_actor: nn.Module | None = None
        self.reference_log_std: torch.Tensor | None = None

    def train(
        self,
        env,
        *,
        total_steps: int,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 0,
        metrics_path: str | Path | None = None,
        checkpoint_metadata: dict | None = None,
    ) -> dict[str, float]:
        """Train for *total_steps* additional steps from the current weights."""
        if type(total_steps) is not int or total_steps < 2:
            raise ValueError("total_steps must be an integer of at least two")
        if type(checkpoint_every) is not int or checkpoint_every < 0:
            raise ValueError("checkpoint_every must be a nonnegative integer")
        if checkpoint_path is not None and checkpoint_every == 0:
            raise ValueError("a checkpoint path requires a positive checkpoint interval")
        torch.manual_seed(self.config.seed + self.steps_trained)
        np.random.seed(self.config.seed + self.steps_trained)
        observation, _ = env.reset(seed=self.config.seed + self.steps_trained)
        history: list[dict[str, float]] = []
        completed = 0
        last_checkpoint = self.steps_trained
        if metrics_path is not None:
            Path(metrics_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        while completed < total_steps:
            remaining = total_steps - completed
            steps = min(self.config.rollout_steps, remaining)
            rollout = {key: [] for key in ("observation", "next_observation", "action", "log_prob", "value", "reward")}
            episode_end = []
            terminated_flags = []
            for _ in range(steps):
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, value = self.policy.act(obs_tensor)
                action_cpu = action[0].detach().cpu()
                next_observation, reward, terminated, truncated, _ = env.step(action_cpu.numpy())
                rollout["observation"].append(observation)
                rollout["next_observation"].append(next_observation)
                rollout["action"].append(action_cpu.numpy())
                rollout["log_prob"].append(float(log_prob[0].detach().cpu()))
                rollout["value"].append(float(value[0].detach().cpu()))
                rollout["reward"].append(float(reward))
                episode_end.append(bool(terminated or truncated))
                terminated_flags.append(bool(terminated))
                observation = next_observation
                completed += 1
                if terminated or truncated:
                    observation, _ = env.reset()
            metrics = self._update(rollout, episode_end, terminated_flags)
            self.steps_trained += steps
            self.updates += 1
            metrics["steps"] = float(self.steps_trained)
            metrics["update"] = float(self.updates)
            history.append(metrics)
            if metrics_path is not None:
                with Path(metrics_path).expanduser().open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(
                f"WBC {self.steps_trained} reward={metrics['mean_reward']:.3f} "
                f"loss={metrics['actor_loss']:.3f} entropy={metrics['entropy']:.3f}",
                flush=True,
            )
            if checkpoint_path is not None and self.steps_trained - last_checkpoint >= checkpoint_every:
                self.save(
                    checkpoint_path,
                    env,
                    metadata={
                        "steps_trained": self.steps_trained,
                        "updates": self.updates,
                        **(checkpoint_metadata or {}),
                    },
                )
                last_checkpoint = self.steps_trained
        if not history:
            raise RuntimeError("PPO produced no update")
        final = history[-1].copy()
        final["mean_reward"] = float(np.mean([entry["mean_reward"] for entry in history]))
        return final

    def pretrain(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> dict[str, float]:
        """Behavior-clone a privileged teacher into the contract actor.

        Only the actor mean is trained; the critic and exploration are left for
        PPO fine-tuning. The teacher may use simulator-only targets because it
        never enters the exported actor or checkpoint.
        """
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.policy.observation_size:
            raise ValueError("teacher observations do not match the actor input")
        if actions.shape != (len(observations), self.policy.action_size):
            raise ValueError("teacher actions do not match the actor output")
        if not np.isfinite(observations).all() or not np.isfinite(actions).all():
            raise ValueError("teacher data contains a non-finite value")
        if len(observations) < 2 or type(epochs) is not int or epochs <= 0:
            raise ValueError("teacher pretraining requires at least two samples and positive epochs")
        if type(batch_size) is not int or batch_size < 2 or learning_rate <= 0:
            raise ValueError("invalid teacher pretraining batch/learning rate")
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(actions, dtype=torch.float32, device=self.device).clamp(-1.0, 1.0)
        optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=learning_rate)
        losses = []
        for _ in range(epochs):
            permutation = torch.randperm(len(obs), device=self.device)
            for start in range(0, len(obs), batch_size):
                index = permutation[start : start + batch_size]
                if len(index) < 2:
                    continue
                prediction = torch.tanh(self.policy.actor(obs[index]))
                loss = ((prediction - target[index]) ** 2).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite teacher pretraining loss")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.config.max_grad_norm)
                optimizer.step()
                losses.append(float(loss.detach()))
        if not losses:
            raise RuntimeError("teacher pretraining produced no update")
        self.reference_actor = copy.deepcopy(self.policy.actor).to(self.device)
        self.reference_actor.eval()
        for parameter in self.reference_actor.parameters():
            parameter.requires_grad_(False)
        self.reference_log_std = self.policy.log_std.detach().clone()
        return {"bc_loss": float(np.mean(losses)), "bc_samples": float(len(observations))}

    def _update(
        self,
        rollout: dict[str, list],
        episode_end: list[bool],
        terminated: list[bool],
    ) -> dict[str, float]:
        observations = torch.as_tensor(np.asarray(rollout["observation"]), dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(
            np.asarray(rollout["next_observation"]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(np.asarray(rollout["action"]), dtype=torch.float32, device=self.device)
        old_log_prob = torch.as_tensor(np.asarray(rollout["log_prob"]), dtype=torch.float32, device=self.device)
        rewards = np.asarray(rollout["reward"], dtype=np.float64)
        with torch.no_grad():
            values = self.policy.value(observations).detach().cpu().numpy().astype(np.float64)
            next_values = self.policy.value(next_observations).detach().cpu().numpy().astype(np.float64)
        next_values[np.asarray(terminated, dtype=bool)] = 0.0
        advantages = np.zeros_like(rewards)
        following = 0.0
        for index in reversed(range(len(rewards))):
            delta = rewards[index] + self.config.gamma * next_values[index] - values[index]
            following = delta + self.config.gamma * self.config.gae_lambda * (not episode_end[index]) * following
            advantages[index] = following
        returns = advantages + values
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std(unbiased=False) + 1e-8)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        actor_losses, value_losses, entropies = [], [], []
        for _ in range(self.config.epochs):
            permutation = torch.randperm(len(actions), device=self.device)
            for start in range(0, len(actions), self.config.batch_size):
                index = permutation[start : start + self.config.batch_size]
                if len(index) < 1:
                    continue
                log_prob, entropy, value = self.policy.evaluate(observations[index], actions[index])
                ratio = (log_prob - old_log_prob[index]).exp()
                surrogate = torch.minimum(
                    ratio * advantage_tensor[index],
                    ratio.clamp(1.0 - self.config.clip, 1.0 + self.config.clip) * advantage_tensor[index],
                )
                actor_loss = -surrogate.mean() - self.config.entropy_coefficient * entropy.mean()
                if self.config.bc_kl_coefficient > 0:
                    if self.reference_actor is None or self.reference_log_std is None:
                        raise RuntimeError("BC KL anchor requires a teacher-pretrained reference actor")
                    with torch.no_grad():
                        reference_mean = self.reference_actor(observations[index])
                    reference = torch.distributions.Normal(reference_mean, self.reference_log_std.exp())
                    current = self.policy.actor_distribution(observations[index])
                    kl = torch.distributions.kl_divergence(current, reference).sum(-1).mean()
                    actor_loss = actor_loss + self.config.bc_kl_coefficient * kl
                value_loss = ((value - returns_tensor[index]) ** 2).mean()
                loss = actor_loss + 0.5 * value_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite whole-body PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                actor_losses.append(float(actor_loss.detach()))
                value_losses.append(float(value_loss.detach()))
                entropies.append(float(entropy.detach().mean()))
        return {
            "mean_reward": float(rewards.mean()),
            "mean_return": float(returns.mean()),
            "actor_loss": float(np.mean(actor_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def save(self, path: str | Path, env, *, metadata: dict | None = None) -> Path:
        """Write a checkpoint plus the profile/layout identity used to load it."""
        from emet.rl.manipulation.wbc_env import OBSERVATION_LAYOUT_VERSION

        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        # Never rewrite the live optimizer state to CPU here: the returned
        # state_dict shares those tensors, and a later step would then fail on
        # a device mismatch. torch.save handles CUDA state; load maps it to CPU.
        optimizer_state = self.optimizer.state_dict()
        payload = {
            "format": self.format,
            "schema_version": SCHEMA_VERSION,
            "observation_layout_version": OBSERVATION_LAYOUT_VERSION,
            "profile": env.profile.to_dict(),
            "profile_fingerprint": env.profile.fingerprint,
            "observation_size": int(env.observation_space.shape[0]),
            "action_size": int(env.action_space.shape[0]),
            "config": asdict(self.config),
            "steps_trained": int(self.steps_trained),
            "updates": int(self.updates),
            "metadata": dict(metadata or {}),
            "state_dict": {key: value.detach().cpu() for key, value in self.policy.state_dict().items()},
            "optimizer_state": optimizer_state,
            "reference_actor_state": None
            if self.reference_actor is None
            else {key: value.detach().cpu() for key, value in self.reference_actor.state_dict().items()},
            "reference_log_std": None if self.reference_log_std is None else self.reference_log_std.detach().cpu(),
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load(cls, path: str | Path, profile: EmbodimentProfile, *, device: str = "cpu") -> tuple[WbcPPO, dict]:
        payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != cls.format:
            raise ValueError("unsupported whole-body policy checkpoint format")
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("whole-body policy schema mismatch")
        from emet.rl.manipulation.wbc_env import OBSERVATION_LAYOUT_VERSION, wbc_observation_layout

        layout_version = int(payload.get("observation_layout_version", -1))
        state_dict = payload["state_dict"]
        optimizer_state = payload.get("optimizer_state")
        reference_state = payload.get("reference_actor_state")
        migrated_from = None
        if layout_version == OBSERVATION_LAYOUT_VERSION:
            observation_size = int(payload["observation_size"])
        elif layout_version == 2 and OBSERVATION_LAYOUT_VERSION == 3:
            old_layout = wbc_observation_layout(profile, include_contact=False)
            new_layout = wbc_observation_layout(profile)
            if int(payload["observation_size"]) != old_layout.size:
                raise ValueError(
                    "v2 checkpoint observation size does not match the legacy layout; "
                    "occupancy-conditioned checkpoints need an explicit migration"
                )
            state_dict = _expand_layout_weights(state_dict, old_layout, new_layout)
            if reference_state is not None:
                reference_state = _expand_layout_weights(reference_state, old_layout, new_layout)
            observation_size = new_layout.size
            optimizer_state = None
            migrated_from = 2
        else:
            raise ValueError("whole-body policy observation layout mismatch")
        if payload.get("profile_fingerprint") != profile.fingerprint:
            raise ValueError("whole-body policy profile fingerprint mismatch")
        if payload.get("profile") != profile.to_dict():
            raise ValueError("whole-body policy profile payload mismatch")
        config = WbcPPOConfig(**payload["config"])
        policy = WbcActorCritic(
            observation_size,
            int(payload["action_size"]),
            hidden=config.hidden,
        )
        policy.load_state_dict(state_dict)
        policy.eval()
        trainer = cls(policy, config, device=device)
        if optimizer_state is not None:
            try:
                trainer.optimizer.load_state_dict(optimizer_state)
            except ValueError:
                # Older checkpoints had one parameter group; weights still load,
                # and the resumed run starts with fresh Adam moments.
                optimizer_state = None
            if optimizer_state is not None:
                for state in trainer.optimizer.state.values():
                    for key, value in list(state.items()):
                        if torch.is_tensor(value):
                            state[key] = value.to(trainer.device)
        if reference_state is not None:
            trainer.reference_actor = copy.deepcopy(policy.actor).to(trainer.device)
            trainer.reference_actor.load_state_dict(reference_state)
            trainer.reference_actor.eval()
            for parameter in trainer.reference_actor.parameters():
                parameter.requires_grad_(False)
            reference_log_std = payload.get("reference_log_std")
            if reference_log_std is None:
                raise ValueError("reference actor checkpoint is missing reference log std")
            trainer.reference_log_std = reference_log_std.to(trainer.device)
        trainer.steps_trained = int(payload.get("steps_trained", 0))
        trainer.updates = int(payload.get("updates", 0))
        metadata = dict(payload.get("metadata") or {})
        if migrated_from is not None:
            metadata["observation_layout_migrated_from"] = migrated_from
        return trainer, metadata


@torch.no_grad()
def evaluate_policy(policy: WbcActorCritic, env, *, episodes: int = 5, seed: int = 0) -> dict[str, float]:
    """Run the deterministic mean policy and report contract-level outcomes."""
    if type(episodes) is not int or episodes < 0:
        raise ValueError("eval episodes must be a nonnegative integer")
    if episodes == 0:
        return {"episodes": 0.0, "mean_return": 0.0, "success_rate": 0.0, "mean_steps": 0.0}
    returns, successes, lengths = [], [], []
    policy.eval()
    device = next(policy.parameters()).device
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total, steps, success = 0.0, 0, False
        done = False
        while not done:
            action = policy.deterministic_action(
                torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            )[0]
            observation, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            total += float(reward)
            steps += 1
            success |= bool(info.get("success", False))
            done = bool(terminated or truncated)
        returns.append(total)
        successes.append(float(success))
        lengths.append(steps)
    return {
        "episodes": float(episodes),
        "mean_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(lengths)),
    }


def collect_teacher_data(env, *, steps: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Roll out a teacher and return contract observations/actions.

    Works for a single ``WbcGymEnv`` or a task-suite environment that exposes
    ``teacher_action`` itself.
    """
    teacher = getattr(env, "teacher_action", None)
    if teacher is None:
        if not hasattr(env.backend, "teacher_action"):
            raise ValueError("backend does not provide a teacher_action")
        teacher = env.backend.teacher_action
    if type(steps) is not int or steps < 2:
        raise ValueError("teacher steps must be at least two")
    observation, _ = env.reset(seed=seed)
    observations, actions = [], []
    for _ in range(steps):
        action = np.asarray(teacher(), dtype=np.float32)
        observations.append(observation)
        actions.append(action)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()
    return np.asarray(observations), np.asarray(actions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a whole-body torque-contract PPO policy in MuJoCo")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-3)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=float, default=6.0)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--target-range", type=float, default=0.35)
    parser.add_argument("--base-goal-probability", type=float, default=0.5)
    parser.add_argument("--base-goal-mode", choices=("forward", "any"), default="forward")
    parser.add_argument("--hold-base", action="store_true", help="ignore base actions for arm-only training")
    parser.add_argument("--collision-force-n", type=float, default=10.0)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path, help="resume from a profile-checked checkpoint")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="write checkpoint.pt every N steps (0 disables periodic snapshots)",
    )
    parser.add_argument(
        "--teacher-steps", type=int, default=0, help="privileged teacher rollout steps for BC warm start"
    )
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument(
        "--bc-kl-coefficient",
        type=float,
        default=0.0,
        help="KL anchor to the BC teacher policy during PPO (0 disables)",
    )
    parser.add_argument("--feedforward", type=Path, help="optional fit FeedforwardModel JSON")
    args = parser.parse_args(argv)
    if args.teacher_steps < 0 or args.teacher_epochs <= 0 or args.teacher_lr <= 0:
        parser.error("teacher steps must be nonnegative, epochs/learning rate positive")
    if args.steps < 2:
        parser.error("--steps must be at least two")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be nonnegative")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but torch.cuda.is_available() is false")
    print(f"WBC device={device}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from emet.rl.manipulation.wbc_env import WbcGymEnv
    from emet.rl.manipulation.wbc_mujoco import MuJoCoWbcBackend

    base_goal_probability = 0.0 if args.hold_base else float(args.base_goal_probability)
    backend = MuJoCoWbcBackend(
        seed=args.seed,
        control_dt=args.control_dt,
        target_range=args.target_range,
        base_goal_probability=base_goal_probability,
        base_goal_mode=args.base_goal_mode,
    )
    feedforward_path = args.feedforward
    if feedforward_path is None and os.environ.get("EMET_WBC_FEEDFORWARD"):
        feedforward_path = Path(os.environ["EMET_WBC_FEEDFORWARD"]).expanduser()
    adapter = None
    if feedforward_path is not None:
        from emet.controller.manipulation.torque_adapter import PositionServoAdapter
        from emet.controller.manipulation.wbc_feedforward import FeedforwardModel, single_calibration

        feedforward = FeedforwardModel.load(feedforward_path)
        single_calibration(feedforward)
        stiffness = {
            joint.name: joint.kp_eff
            for group in backend.profile.joint_groups
            for joint in group.joints
            if joint.kp_eff is not None
        }
        missing = sorted(set(feedforward.joint_bias) - set(stiffness))
        if missing:
            raise ValueError(f"feedforward joints have no profile kp_eff: {missing}")
        adapter = PositionServoAdapter(
            backend.profile,
            gravity_torques=feedforward.torque_mapping(stiffness),
        )
    env = WbcGymEnv(
        backend,
        adapter=adapter,
        horizon_s=args.horizon,
        control_dt=args.control_dt,
        collision_force_n=args.collision_force_n,
        hold_base=args.hold_base,
    )
    contract_config = {
        "target_range": float(args.target_range),
        "base_goal_probability": float(base_goal_probability),
        "base_goal_mode": str(args.base_goal_mode),
        "hold_base": bool(args.hold_base),
        "horizon_s": float(args.horizon),
        "control_dt": float(args.control_dt),
        "success_hold_s": float(env.success_hold_s),
        "collision_force_n": float(args.collision_force_n),
        "scene": str(backend.scene),
        "gripper_mode": str(backend.gripper_mode),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume is not None:
        trainer, resume_metadata = WbcPPO.load(args.resume, backend.profile, device=device)
        policy = trainer.policy
        config = trainer.config
        if trainer.policy.observation_size != env.observation_space.shape[0]:
            raise ValueError("resumed policy observation size does not match the environment")
        if trainer.policy.action_size != env.action_space.shape[0]:
            raise ValueError("resumed policy action size does not match the environment")
        previous_contract = resume_metadata.get("contract")
        if previous_contract is None:
            raise ValueError("resume checkpoint is missing the task contract; refusing an unverified resume")
        if previous_contract != contract_config:
            raise ValueError(f"resumed task contract mismatch: {previous_contract} != {contract_config}")
        print(
            f"WBC resumed {args.resume} steps={trainer.steps_trained} updates={trainer.updates}",
            flush=True,
        )
    else:
        resume_metadata = {}
        config = WbcPPOConfig(
            hidden=args.hidden,
            learning_rate=args.learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            bc_kl_coefficient=args.bc_kl_coefficient,
            rollout_steps=args.rollout_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        policy = WbcActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden=args.hidden)
        trainer = WbcPPO(policy, config, device=device)
    teacher_metrics: dict[str, float] = {}
    bc_evaluation: dict[str, float] | None = None
    if args.resume is None and args.teacher_steps:
        teacher_observations, teacher_actions = collect_teacher_data(
            env,
            steps=args.teacher_steps,
            seed=args.seed,
        )
        teacher_metrics = trainer.pretrain(
            teacher_observations,
            teacher_actions,
            epochs=args.teacher_epochs,
            batch_size=args.batch_size,
            learning_rate=args.teacher_lr,
        )
        print(
            f"WBC teacher pretrain samples={int(teacher_metrics['bc_samples'])} loss={teacher_metrics['bc_loss']:.5f}",
            flush=True,
        )
        bc_evaluation = evaluate_policy(
            trainer.policy,
            env,
            episodes=args.eval_episodes,
            seed=args.seed + 20_000,
        )
        trainer.save(
            args.output / "bc_policy.pt",
            env,
            metadata={
                "stage": "bc",
                "contract": contract_config,
                "teacher": teacher_metrics,
                "evaluation": bc_evaluation,
            },
        )
        print(
            f"WBC BC eval episodes={int(bc_evaluation['episodes'])} "
            f"success={bc_evaluation['success_rate']:.2f} return={bc_evaluation['mean_return']:.3f}",
            flush=True,
        )
    metrics_path = args.output / "metrics.jsonl"
    metrics = trainer.train(
        env,
        total_steps=args.steps,
        checkpoint_path=args.output / "checkpoint.pt" if args.checkpoint_every else None,
        checkpoint_every=args.checkpoint_every,
        metrics_path=metrics_path,
        checkpoint_metadata={"contract": contract_config},
    )
    evaluation = evaluate_policy(policy, env, episodes=args.eval_episodes, seed=args.seed + 10_000)
    checkpoint = trainer.save(
        args.output / "wbc_policy.pt",
        env,
        metadata={
            "robot": env.profile.meta("robot"),
            "model_fingerprint": backend.model_fingerprint,
            "feedforward": None if feedforward_path is None else str(feedforward_path),
            "device": str(trainer.device),
            "contract": contract_config,
            "resumed_from": None if args.resume is None else str(args.resume),
            "resume_metadata": resume_metadata,
            "metrics": metrics,
            "evaluation": evaluation,
            "teacher": teacher_metrics,
            "bc_evaluation": bc_evaluation,
        },
    )
    manifest = {
        "checkpoint": checkpoint.name,
        "profile_fingerprint": env.profile.fingerprint,
        "observation_size": int(env.observation_space.shape[0]),
        "action_size": int(env.action_space.shape[0]),
        "device": str(trainer.device),
        "contract": contract_config,
        "steps_trained": int(trainer.steps_trained),
        "updates": int(trainer.updates),
        "metrics": metrics,
        "evaluation": evaluation,
        "teacher": teacher_metrics,
        "bc_evaluation": bc_evaluation,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    env.close()
    print(
        f"WBC eval episodes={int(evaluation['episodes'])} success={evaluation['success_rate']:.2f} "
        f"return={evaluation['mean_return']:.3f}"
    )
    print(f"WBC checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["WbcActorCritic", "WbcPPO", "WbcPPOConfig", "evaluate_policy", "main"]
