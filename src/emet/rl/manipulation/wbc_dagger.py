# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""DAgger over the whole-body contract subtask suite.

The privileged teacher runs only during collection: it labels the states the
current actor actually visits. The exported actor sees the contract
observation only. Subtasks are arm-only, base-only, and combined base+arm.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from emet.rl.manipulation.wbc_backends import make_wbc_backend
from emet.rl.manipulation.wbc_env import WbcGymEnv
from emet.rl.manipulation.wbc_training import (
    WbcActorCritic,
    WbcPPO,
    WbcPPOConfig,
    collect_teacher_data,
)


@dataclass(frozen=True)
class WbcSubtask:
    """One task distribution in the DAgger suite."""

    name: str
    arm_goal: bool
    base_goal: bool
    hold_base: bool
    target_range: float
    horizon_s: float
    base_goal_mode: str = "forward"
    weight: float = 1.0
    active_arms: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("subtask name must be non-empty")
        if not self.arm_goal and not self.base_goal:
            raise ValueError(f"{self.name} must enable an arm or base goal")
        if self.arm_goal and (not np.isfinite(self.target_range) or self.target_range <= 0):
            raise ValueError(f"{self.name} target_range must be positive")
        if not np.isfinite(self.horizon_s) or self.horizon_s <= 0:
            raise ValueError(f"{self.name} horizon must be positive")
        if self.base_goal_mode not in {"forward", "any"}:
            raise ValueError(f"{self.name} has invalid base_goal_mode")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError(f"{self.name} weight must be positive")
        if self.active_arms is not None:
            if not self.active_arms or any(not str(name) for name in self.active_arms):
                raise ValueError(f"{self.name} active_arms must be non-empty arm names")
            if len(set(self.active_arms)) != len(self.active_arms):
                raise ValueError(f"{self.name} active_arms must be unique")


def default_subtasks(robot: str = "innate_mars") -> tuple[WbcSubtask, ...]:
    """Core suite: arm-only, base-only, and combined base+arm.

    Stretch's underdamped position-servo wrists need a longer reach window;
    the Mars protocol is unchanged so earlier results stay comparable. Robots
    with two arms use single-arm subtasks: a simultaneous ten-plus-DOF reach is
    a much harder imitation target than either arm alone.
    """
    if robot in {"sourccey", "rby1"}:
        return (
            WbcSubtask("arm_left", True, False, True, 0.15, 3.0, "forward", 0.2, ("left",)),
            WbcSubtask("arm_right", True, False, True, 0.15, 3.0, "forward", 0.2, ("right",)),
            WbcSubtask("base", False, True, False, 0.0, 5.0, "forward", 0.2),
            WbcSubtask("combined_left", True, True, False, 0.15, 5.0, "forward", 0.2, ("left",)),
            WbcSubtask("combined_right", True, True, False, 0.15, 5.0, "forward", 0.2, ("right",)),
        )
    long_reach = robot in {"stretch", "pandaomron"}
    return (
        WbcSubtask("arm", True, False, True, 0.15, 5.0 if long_reach else 3.0, "forward", 0.4),
        WbcSubtask("base", False, True, False, 0.0, 5.0, "forward", 0.2),
        WbcSubtask("combined", True, True, False, 0.15, 7.0 if long_reach else 5.0, "forward", 0.4),
    )


class WbcTaskSuiteEnv(gym.Env):
    """Sample one subtask environment per episode and preserve its identity."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        subtasks: tuple[WbcSubtask, ...] | None = None,
        *,
        robot: str = "innate_mars",
        seed: int = 0,
        control_dt: float = 0.05,
        collision_force_n: float = 10.0,
    ) -> None:
        if subtasks is None:
            subtasks = default_subtasks(robot)
        if not subtasks:
            raise ValueError("at least one subtask is required")
        names = [task.name for task in subtasks]
        if len(set(names)) != len(names):
            raise ValueError("duplicate subtask names")
        self.subtasks = tuple(subtasks)
        self.names = tuple(task.name for task in subtasks)
        self.weights = np.asarray([task.weight for task in subtasks], dtype=np.float64)
        self.weights /= self.weights.sum()
        self.envs: dict[str, WbcGymEnv] = {}
        for index, task in enumerate(self.subtasks):
            kwargs: dict[str, Any] = {
                "seed": seed + index,
                "control_dt": control_dt,
                "target_range": task.target_range,
                "arm_goal_probability": float(task.arm_goal),
                "base_goal_probability": float(task.base_goal),
                "base_goal_mode": task.base_goal_mode,
            }
            if task.active_arms is not None:
                kwargs["active_arms"] = task.active_arms
            backend = make_wbc_backend(robot, sim="mujoco", **kwargs)
            self.envs[task.name] = WbcGymEnv(
                backend,
                horizon_s=task.horizon_s,
                control_dt=control_dt,
                collision_force_n=collision_force_n,
                hold_base=task.hold_base,
            )
        first = next(iter(self.envs.values()))
        self.action_space = first.action_space
        self.observation_space = first.observation_space
        if any(
            not np.array_equal(env.action_space.low, self.action_space.low)
            or not np.array_equal(env.action_space.high, self.action_space.high)
            or env.observation_space.shape != self.observation_space.shape
            for env in self.envs.values()
        ):
            raise ValueError("subtask environments must share observation/action contracts")
        self.active_name: str | None = None

    @property
    def active(self) -> WbcGymEnv:
        if self.active_name is None:
            raise RuntimeError("reset the suite before accessing the active environment")
        return self.envs[self.active_name]

    def teacher_action(self) -> np.ndarray:
        return self.active.backend.teacher_action()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        forced = None if options is None else options.get("subtask")
        if forced is not None:
            if forced not in self.envs:
                raise ValueError(f"unknown subtask: {forced!r}")
            name = str(forced)
        else:
            index = int(self.np_random.choice(len(self.names), p=self.weights))
            name = self.names[index]
        self.active_name = name
        observation, info = self.envs[name].reset(seed=seed)
        return observation, {**info, "subtask": name}

    def step(self, action):
        if self.active_name is None:
            raise RuntimeError("reset required")
        observation, reward, terminated, truncated, info = self.active.step(action)
        return observation, reward, terminated, truncated, {**info, "subtask": self.active_name}

    def close(self) -> None:
        for env in self.envs.values():
            env.close()


def _cap_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly subsample an aggregated DAgger buffer to bound BC cost."""
    if len(observations) <= limit:
        return observations, actions
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(observations), size=limit, replace=False))
    return observations[index], actions[index]


def collect_dagger_data(
    env: WbcTaskSuiteEnv,
    policy: WbcActorCritic,
    *,
    steps: int,
    seed: int = 0,
    noise_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Run the actor and label every visited state with the teacher action."""
    if type(steps) is not int or steps < 1:
        raise ValueError("dagger steps must be positive")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be nonnegative")
    device = next(policy.parameters()).device
    rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed)
    observations, actions = [], []
    counts = dict.fromkeys(env.names, 0)
    for _ in range(steps):
        with torch.no_grad():
            action = (
                policy.deterministic_action(
                    torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                )[0]
                .cpu()
                .numpy()
            )
        if noise_std:
            action = np.clip(action + rng.normal(0.0, noise_std, size=action.shape), -1.0, 1.0)
        teacher = np.asarray(env.teacher_action(), dtype=np.float32)
        observations.append(observation)
        actions.append(teacher)
        counts[env.active_name] += 1
        observation, _, terminated, truncated, _ = env.step(action.astype(np.float32))
        if terminated or truncated:
            observation, _ = env.reset()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32), counts


@torch.no_grad()
def evaluate_suite(
    env: WbcTaskSuiteEnv,
    policy: WbcActorCritic | None,
    *,
    episodes_per_task: int = 10,
    seed: int = 0,
    teacher: bool = False,
) -> dict[str, Any]:
    """Evaluate each subtask separately; ``teacher=True`` measures the ceiling."""
    if type(episodes_per_task) is not int or episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    device = None if policy is None else next(policy.parameters()).device
    results: dict[str, Any] = {}
    all_returns, all_success = [], []
    for name in env.names:
        returns, successes, lengths = [], [], []
        for episode in range(episodes_per_task):
            observation, _ = env.reset(seed=seed + episode, options={"subtask": name})
            total, steps, success = 0.0, 0, False
            done = False
            while not done:
                if teacher:
                    action = env.teacher_action()
                else:
                    assert policy is not None
                    action = (
                        policy.deterministic_action(
                            torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                        )[0]
                        .cpu()
                        .numpy()
                    )
                observation, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                steps += 1
                success |= bool(info.get("success", False))
                done = bool(terminated or truncated)
            returns.append(total)
            successes.append(float(success))
            lengths.append(steps)
        results[name] = {
            "episodes": float(episodes_per_task),
            "success_rate": float(np.mean(successes)),
            "mean_return": float(np.mean(returns)),
            "mean_steps": float(np.mean(lengths)),
        }
        all_returns.extend(returns)
        all_success.extend(successes)
    results["aggregate"] = {
        "episodes": float(len(all_returns)),
        "success_rate": float(np.mean(all_success)),
        "mean_return": float(np.mean(all_returns)),
    }
    return results


def dagger_train(
    *,
    output: str | Path,
    robot: str = "innate_mars",
    subtasks: tuple[WbcSubtask, ...] | None = None,
    init: str | Path | None = None,
    iterations: int = 4,
    teacher_steps: int = 2048,
    dagger_steps: int = 1024,
    bc_epochs: int = 200,
    bc_learning_rate: float = 1e-3,
    batch_size: int = 64,
    hidden: int = 128,
    eval_episodes: int = 10,
    max_bc_samples: int = 20000,
    seed: int = 0,
    device: str = "cpu",
    noise_std: float = 0.05,
) -> dict[str, Any]:
    """Teacher warm start, then DAgger iterations over the subtask suite."""
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if type(max_bc_samples) is not int or max_bc_samples < 1:
        raise ValueError("max_bc_samples must be positive")
    if subtasks is None:
        subtasks = default_subtasks(robot)
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    env = WbcTaskSuiteEnv(subtasks, robot=robot, seed=seed)
    config = WbcPPOConfig(
        hidden=hidden,
        batch_size=batch_size,
        bc_kl_coefficient=0.1,
        seed=seed,
    )
    if init is None:
        policy = WbcActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden=hidden)
        trainer = WbcPPO(policy, config, device=device)
    else:
        profile = env.envs[env.names[0]].profile
        trainer, _ = WbcPPO.load(init, profile, device=device)
        policy = trainer.policy
        config = trainer.config
    metrics_path = output / "dagger_metrics.jsonl"
    best_success = -1.0
    best_path = output / "best.pt"
    history: list[dict[str, Any]] = []

    if init is not None:
        initial = evaluate_suite(env, policy, episodes_per_task=eval_episodes, seed=seed + 10_000)
        history.append({"stage": "init_eval", **initial})
        best_success = initial["aggregate"]["success_rate"]
        trainer.save(best_path, env.envs[env.names[0]], metadata={"stage": "init", "evaluation": initial})
        print(
            f"DAgger init_eval aggregate={best_success:.2f}",
            flush=True,
        )

    teacher_evaluation = evaluate_suite(
        env,
        None,
        episodes_per_task=eval_episodes,
        seed=seed + 20_000,
        teacher=True,
    )
    history.append({"stage": "teacher_eval", **teacher_evaluation})
    print(
        f"DAgger teacher_eval aggregate={teacher_evaluation['aggregate']['success_rate']:.2f}",
        flush=True,
    )

    teacher_observations, teacher_actions = collect_teacher_data(env, steps=teacher_steps, seed=seed)
    dataset_observations = [teacher_observations]
    dataset_actions = [teacher_actions]
    bc_observations, bc_actions = _cap_dataset(teacher_observations, teacher_actions, max_bc_samples, seed + 1)
    bc = trainer.pretrain(
        bc_observations,
        bc_actions,
        epochs=bc_epochs,
        batch_size=batch_size,
        learning_rate=bc_learning_rate,
    )
    evaluation = evaluate_suite(env, policy, episodes_per_task=eval_episodes, seed=seed + 10_000)
    history.append({"stage": "teacher_bc", **bc, **evaluation})
    print(
        f"DAgger teacher_bc loss={bc['bc_loss']:.5f} aggregate={evaluation['aggregate']['success_rate']:.2f}",
        flush=True,
    )
    if evaluation["aggregate"]["success_rate"] > best_success:
        best_success = evaluation["aggregate"]["success_rate"]
        trainer.save(best_path, env.envs[env.names[0]], metadata={"stage": "teacher_bc", "evaluation": evaluation})

    for iteration in range(1, iterations + 1):
        observations, actions, counts = collect_dagger_data(
            env,
            policy,
            steps=dagger_steps,
            seed=seed + iteration,
            noise_std=noise_std,
        )
        dataset_observations.append(observations)
        dataset_actions.append(actions)
        all_observations = np.concatenate(dataset_observations, axis=0)
        all_actions = np.concatenate(dataset_actions, axis=0)
        bc_observations, bc_actions = _cap_dataset(all_observations, all_actions, max_bc_samples, seed + iteration)
        bc = trainer.pretrain(
            bc_observations,
            bc_actions,
            epochs=bc_epochs,
            batch_size=batch_size,
            learning_rate=bc_learning_rate,
        )
        evaluation = evaluate_suite(env, policy, episodes_per_task=eval_episodes, seed=seed + 10_000)
        record = {
            "stage": f"dagger_{iteration}",
            "samples": int(len(all_observations)),
            "subtask_counts": counts,
            **bc,
            **evaluation,
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        trainer.save(output / f"policy_iter{iteration}.pt", env.envs[env.names[0]], metadata=record)
        print(
            f"DAgger iter={iteration} samples={record['samples']} loss={bc['bc_loss']:.5f} "
            f"aggregate={evaluation['aggregate']['success_rate']:.2f} counts={counts}",
            flush=True,
        )
        if evaluation["aggregate"]["success_rate"] > best_success:
            best_success = evaluation["aggregate"]["success_rate"]
            trainer.save(best_path, env.envs[env.names[0]], metadata=record)

    final = history[-1]
    trainer.save(output / "final.pt", env.envs[env.names[0]], metadata=final)
    (output / "dagger_report.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    env.close()
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DAgger over the WBC subtask suite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", default="innate_mars")
    parser.add_argument("--init", type=Path, help="profile-checked checkpoint to fine-tune from")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--teacher-steps", type=int, default=2048)
    parser.add_argument("--dagger-steps", type=int, default=1024)
    parser.add_argument("--bc-epochs", type=int, default=200)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-bc-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--subtasks", default=None, help="comma-separated suite; defaults to the robot's full suite")
    args = parser.parse_args(argv)

    tasks = {task.name: task for task in default_subtasks(args.robot)}
    requested = (
        list(tasks) if args.subtasks is None else [name.strip() for name in args.subtasks.split(",") if name.strip()]
    )
    unknown = set(requested) - set(tasks)
    if unknown:
        parser.error(f"unknown subtasks: {sorted(unknown)}")
    subtasks = tuple(tasks[name] for name in requested)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but torch.cuda.is_available() is false")
    print(f"DAgger device={device} robot={args.robot} subtasks={requested}", flush=True)
    dagger_train(
        output=args.output,
        robot=args.robot,
        subtasks=subtasks,
        init=args.init,
        iterations=args.iterations,
        teacher_steps=args.teacher_steps,
        dagger_steps=args.dagger_steps,
        bc_epochs=args.bc_epochs,
        bc_learning_rate=args.bc_learning_rate,
        batch_size=args.batch_size,
        hidden=args.hidden,
        eval_episodes=args.eval_episodes,
        max_bc_samples=args.max_bc_samples,
        seed=args.seed,
        device=device,
        noise_std=args.noise_std,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "WbcSubtask",
    "WbcTaskSuiteEnv",
    "collect_dagger_data",
    "dagger_train",
    "default_subtasks",
    "evaluate_suite",
    "main",
]
