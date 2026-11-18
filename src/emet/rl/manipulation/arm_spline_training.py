"""Bounded PPO distribution ablation: unchanged rewards, random arm splines.

Use fresh regression and teacher receipts. This resumes weights/Adam/critics,
not an old simulator state, and never promotes a checkpoint to hardware.
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.arm_spline_benchmark import DEFAULT, require_preflight, save
from emet.rl.manipulation.arm_splines import MIXTURE, make_training_environment
from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.multi_critic_ppo import MultiCriticPPO, PPOConfig, grouped_gae
from emet.rl.manipulation.reward_config import WholeBodyRewardConfig
from emet.rl.manipulation.whole_body_actor import MultiCriticPolicy
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle, export_actor
from emet.rl.manipulation.whole_body_training import collect_rollout, policy_state_digest
from emet.rl.regression_gate import ROOT, source_state, verify_regressions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def critic_warmup(learner, rollout):
    """Fit the new reset distribution without changing actor or exploration."""
    p, cfg = learner.policy, learner.config
    _, targets = grouped_gae(
        rollout["rewards"],
        rollout["values"],
        rollout["next_values"],
        rollout["terminated"],
        rollout["episode_end"],
        rollout["ticks"],
        gamma=cfg.gamma,
        lam=cfg.gae_lambda,
    )
    observations = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in rollout["observations"].items()}
    targets = torch.from_numpy(targets)
    for _ in range(cfg.critic_epochs):
        for indices in torch.randperm(len(targets)).split(cfg.batch_size):
            loss = (
                0.5
                * ((p.values({k: v[indices] for k, v in observations.items()}) - targets[indices]) ** 2).mean(0).sum()
            )
            learner.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p.critics.parameters(), cfg.max_grad_norm)
            learner.optimizer.step()
    return {"critic_only": True, "value_loss": float(loss.detach())}


def train(args):
    if (
        type(args.steps) is not int
        or args.steps < 2048
        or args.steps % 1024
        or type(args.critic_warmup_steps) is not int
        or not 0 <= args.critic_warmup_steps < args.steps
        or args.critic_warmup_steps % 1024
    ):
        raise ValueError("bounded whole 1024-decision rollouts and a shorter critic warmup required")
    receipt = verify_regressions(args.regression_receipt)
    preflight = require_preflight(args.preflight, args.manifest, args.level)
    state = source_state(ROOT)
    initializer = WholeBodyBundle(args.initializer, robot="innate_mars", fingerprint=preflight["fingerprint"])
    saved = torch.load(args.training_state, map_location="cpu", weights_only=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy = MultiCriticPolicy(
        hidden=initializer.actor.hidden,
        base_rate_feedback=initializer.actor.base_rate_feedback,
        base_navigation_feedback=initializer.actor.base_navigation_feedback,
    )
    cfg = PPOConfig(
        learning_rate=1e-5,
        critic_learning_rate=3e-4,
        epochs=3,
        critic_epochs=8,
        batch_size=128,
        clip=0.05,
        target_kl=0.01,
        entropy_weight=0.0,
    )
    learner = MultiCriticPPO(policy, cfg)
    policy.load_state_dict(saved["policy"], strict=True)
    learner.optimizer.load_state_dict(saved["optimizer"])
    for key, value in initializer.actor.state_dict().items():
        torch.testing.assert_close(policy.actor.state_dict()[key], value, atol=0, rtol=0)
    # Optimizer restore retains moments; this new experiment declares its rates.
    learner.optimizer.param_groups[0]["lr"] = cfg.learning_rate
    learner.optimizer.param_groups[1]["lr"] = cfg.critic_learning_rate
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    reward = WholeBodyRewardConfig()
    metadata = {
        "format": "arm-spline-training-v1",
        "phase": "initializing",
        "complete": False,
        "source_state_digest": digest(state),
        "manifest_sha256": sha(args.manifest),
        "preflight_sha256": sha(args.preflight),
        "regression_receipt_sha256": sha(args.regression_receipt),
        "regression_tests": len(receipt["test_ids"]),
        "initializer_weights_sha256": initializer.manifest["weights_sha256"],
        "training_state_sha256": sha(args.training_state),
        "source_run_decisions": saved["decisions"],
        "environment_state_restored": False,
        "initial_policy_state_sha256": policy_state_digest(policy),
        "seed": args.seed,
        "requested_decisions": args.steps,
        "critic_warmup_decisions": args.critic_warmup_steps,
        "level": args.level,
        "curriculum_advance": "manual; development probes only",
        "training_mix": MIXTURE,
        "reward_configuration": reward.record(),
        "ppo": asdict(cfg),
        "teacher_actions_in_loss": False,
        "actor_inputs": ["history", "preview", "motion"],
        "hardware_used": False,
        "safety_promoted": False,
        "decisions": 0,
    }
    save(output / "run.json", metadata)
    env = None
    run = None
    try:
        if args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project="emet-mars-control",
                name=output.name,
                mode=args.wandb_mode,
                dir=str(output),
                config=metadata,
                settings=wandb.Settings(disable_git=True, save_code=False),
            )
        env = make_training_environment(args.manifest, level=args.level, reward_config=reward)
        native = env.environments["arm_splines"].unwrapped
        if native.fingerprint != preflight["fingerprint"]:
            raise ValueError("training model differs from teacher preflight")
        initial_actor = {k: v.clone() for k, v in policy.actor.state_dict().items()}
        initial_std = policy.log_std.detach().clone()
        observation, _ = env.reset(seed=args.seed)
        ticks = 0

        def on_episode(info):
            row = {
                k: info.get(k)
                for k in (
                    "case_id",
                    "training_family",
                    "code",
                    "primitive_success",
                    "wheel_lift",
                    "unsafe_contact",
                    "spline_peak_path_error",
                    "joint_limit_push_ticks",
                    "spline_rejections",
                )
            }
            with (output / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

        for decisions in range(0, args.steps, 1024):
            rollout, observation = collect_rollout(
                env, policy, observation, 1024, episode_callback=on_episode, gamma=cfg.gamma
            )
            warm = decisions < args.critic_warmup_steps
            if warm:
                stats = critic_warmup(learner, rollout)
                for key, value in policy.actor.state_dict().items():
                    torch.testing.assert_close(value, initial_actor[key], atol=0, rtol=0)
                torch.testing.assert_close(policy.log_std, initial_std, atol=0, rtol=0)
            else:
                fraction = (decisions - args.critic_warmup_steps) / (args.steps - args.critic_warmup_steps)
                learner.optimizer.param_groups[0]["lr"] = cfg.learning_rate * (1 - 0.8 * fraction)
                stats = learner.update(rollout)
            ticks += int(rollout["ticks"].sum())
            row = {
                "decisions": decisions + 1024,
                "control_ticks": ticks,
                "critic_only": warm,
                "mixture": env.ledger(),
                "case_ids": sorted(set(rollout["case_ids"].tolist())),
                **stats,
            }
            with (output / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if run:
                run.log(
                    {
                        "decisions": row["decisions"],
                        "control_ticks": ticks,
                        **{f"reward/{i}": float(v) for i, v in enumerate(rollout["rewards"].mean(0))},
                    },
                    step=row["decisions"],
                )
            metadata.update(phase="training", decisions=row["decisions"], control_ticks=ticks, mixture=env.ledger())
            save(output / "run.json", metadata)
            temporary = output / "training.pt.partial"
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": learner.optimizer.state_dict(),
                    "decisions": row["decisions"],
                    "run": metadata,
                    "environment_state_serialized": False,
                },
                temporary,
            )
            os.replace(temporary, output / "training.pt")
            if row["decisions"] % 16384 == 0 or row["decisions"] == args.steps:
                verify_regressions(args.regression_receipt)
                if source_state(ROOT) != state:
                    raise RuntimeError("source changed during training")
                export_actor(
                    policy.actor,
                    output / f"checkpoint-{row['decisions']}" / "actor",
                    native.config,
                    native.fingerprint,
                    steps=row["decisions"],
                    training=metadata,
                )
            print("PPO", row["decisions"], "ticks", ticks, "warmup", warm, flush=True)
        metadata.update(phase="complete_awaiting_evaluation", complete=True)
        save(output / "run.json", metadata)
    finally:
        if env is not None:
            env.close()
        if run is not None:
            run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--regression-receipt", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=65536)
    parser.add_argument("--critic-warmup-steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
