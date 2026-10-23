"""Refit the compact current-goal/history MLP on an existing teacher dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from emet.rl.__main__ import evaluate
from emet.rl.base import BASE, MarsBaseEnv
from emet.rl.env import EnvConfig
from emet.rl.model import PROXIES
from emet.rl.networks import BaseHistoryFeatures, FixedScaleBaseHistoryFeatures, PrecisionBaseFeatures
from emet.rl.policy import PPOPolicy, export_policy
from emet.rl.training import evaluation_metrics, finish_wandb, start_wandb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--reference-bundle", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--precision-feature", action="store_true")
    p.add_argument(
        "--geometry-normalization", action="store_true", help="Use physical scales for distance and validity features"
    )
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    p.add_argument("--wandb-project", default="emet-mars-control")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-name", default="mars-base-current-goal-refit")
    args = p.parse_args()
    if min(args.epochs, args.eval_every) <= 0:
        p.error("epochs and evaluation interval must be positive")
    torch.set_num_threads(1)
    # Validate reference/model compatibility before accepting collected data.
    reference = PPOPolicy(args.reference_bundle)
    if reference.task != "base":
        raise ValueError("base dataset required")
    if Path(args.dataset).resolve().parent != Path(args.reference_bundle).resolve().parent:
        raise ValueError("dataset and its reference bundle must share the training directory")
    dataset = np.load(args.dataset)
    raw, labels = dataset["history"], dataset["actions"]
    config = EnvConfig(**reference.manifest["training"]["env"])
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "env": reference.manifest["training"]["env"],
        "seed": args.seed,
        "algorithm": "teacher refit",
        "dataset": str(Path(args.dataset).resolve()),
        "examples": len(raw),
        "features": "current goal + motion history",
        "net_arch": [128, 128],
        "precision_feature": args.precision_feature,
        "geometry_normalization": args.geometry_normalization,
        "reference": str(Path(args.reference_bundle).resolve()),
    }
    run = start_wandb(args, directory, metadata)
    if run is not None:
        run.define_metric("*", step_metric="epoch")
    failed = True
    env = VecNormalize(DummyVecEnv([lambda: MarsBaseEnv(config)]), norm_reward=False)
    try:
        if raw.shape[1:] != env.observation_space["history"].shape or labels.shape != (len(raw), 2):
            raise ValueError("dataset layout mismatch")
        if not np.isfinite(raw).all() or not np.isfinite(labels).all():
            raise ValueError("non-finite dataset")
        env.obs_rms["history"].update(raw)
        env.training = False
        rms = env.obs_rms["history"]
        if args.geometry_normalization:
            # Free-space datasets have zero obstacle-feature variance. Their
            # frozen RMS clips every later obstacle to the same extreme value,
            # destroying clearance information. Use fixed physical scales.
            count = len(PROXIES)
            rms.mean[:, 9 : 9 + count] = 1.0
            rms.var[:, 9 : 9 + count] = 0.25**2
            rms.mean[:, 9 + count : 9 + 2 * count] = 0.0
            rms.var[:, 9 + count : 9 + 2 * count] = 1.0
        threshold = (BASE.xy_tolerance * 0.6 - rms.mean[-1, -3]) / np.sqrt(rms.var[-1, -3] + env.epsilon)
        model = PPO(
            "MultiInputPolicy",
            env,
            device="cpu",
            seed=args.seed,
            n_steps=256,
            batch_size=256,
            learning_rate=1e-4,
            ent_coef=0.0001,
            policy_kwargs={
                "features_extractor_class": PrecisionBaseFeatures
                if args.precision_feature
                else (FixedScaleBaseHistoryFeatures if args.geometry_normalization else BaseHistoryFeatures),
                "features_extractor_kwargs": {"normalized_goal_threshold": float(threshold)}
                if args.precision_feature
                else {},
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                "log_std_init": -4.0,
            },
        )
        x = torch.as_tensor(env.normalize_obs({"history": raw})["history"], dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.float32)
        # Precision near the position target matters more than saturated transit commands.
        weights = torch.as_tensor(
            np.where(np.linalg.norm(raw[:, -1, :2], axis=1) < 0.03, 3.0, 1.0), dtype=torch.float32
        )
        optimizer = torch.optim.Adam(model.policy.parameters(), lr=5e-4)
        rng = np.random.default_rng(args.seed)
        best = None
        for epoch in range(1, args.epochs + 1):
            model.policy.set_training_mode(True)
            losses = []
            order = rng.permutation(len(x))
            for offset in range(0, len(x), 1024):
                ids = order[offset : offset + 1024]
                prediction = model.policy.get_distribution({"history": x[ids]}).distribution.mean
                loss = ((prediction - y[ids]).square().mean(dim=1) * weights[ids]).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                model.policy.set_training_mode(False)
                target = directory / f"epoch-{epoch}"
                export_policy(model, env, target, {**metadata, "epochs": epoch})
                report = evaluate(
                    PPOPolicy(target),
                    config,
                    episodes=30,
                    seed=50000,
                    teacher_approach=config.base_scenario != "standard",
                )
                (target / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
                metrics = evaluation_metrics(report)
                row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
                with (directory / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                if run is not None:
                    run.log({k: v for k, v in row.items() if v is not None})
                print(json.dumps(row), flush=True)
                score = (
                    metrics["success_rate"],
                    -metrics["collision_rate"],
                    -metrics["xy_error"] - 0.2 * metrics["yaw_error"],
                )
                if best is None or score > best:
                    best = score
                    export_policy(model, env, directory / "best", {**metadata, "epochs": epoch})
                    (directory / "best" / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        failed = False
    finally:
        env.close()
        finish_wandb(run, directory, failed=failed)


if __name__ == "__main__":
    main()
