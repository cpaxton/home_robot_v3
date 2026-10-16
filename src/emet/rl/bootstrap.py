"""Initialize a base policy from the conventional controller, then refine with PPO.

Teacher labels come from simulator states encountered by a teacher/student
mixture. Validation seeds never enter the dataset.
"""

import json
from pathlib import Path

import numpy as np
import torch

from emet.rl.policy import PPOPolicy, export_policy
from emet.rl.training import evaluation_metrics


def bootstrap_base(model, env, args, directory, metadata, run=None):
    from emet.rl.__main__ import evaluate
    from emet.rl.env import EnvConfig

    if args.task != "base" or args.architecture != "mlp":
        raise ValueError("teacher initialization currently supports only the base MLP")
    metadata = {**metadata, "bootstrap_labels": "desired_velocity"}
    directory = Path(directory)
    rng = np.random.default_rng(args.seeds[0] + 7919)
    inputs, labels = [], []
    best_score = None
    # Optimize the deterministic actor, leaving its critic for RL fine tuning.
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    config = EnvConfig(**metadata["env"])
    total = 0
    for iteration in range(args.bootstrap_rounds):
        obs = env.reset()
        for _ in range(max(1, args.bootstrap_steps // env.num_envs)):
            raw = env.get_original_obs()["history"].copy()
            teacher = np.asarray(env.env_method("expert_action"), dtype=np.float32)
            # Learn desired velocities; the shared adapter enforces acceleration.
            # Regressing already-slewed commands makes copying the previous
            # command an easy near-identity solution that hides goal errors.
            targets = teacher
            inputs.append(raw)
            labels.append(targets.astype(np.float32))
            # Start with demonstrations, then expose the teacher to student errors.
            prediction, _ = model.predict(obs, deterministic=True)
            fraction_teacher = 1.0 if iteration == 0 else 0.5 / iteration
            selected = np.where((rng.random(env.num_envs) < fraction_teacher)[:, None], teacher, prediction)
            obs, _, _, _ = env.step(selected)
            total += env.num_envs
        x = np.concatenate(inputs)
        y = torch.as_tensor(np.concatenate(labels), dtype=torch.float32)
        x = torch.as_tensor(env.normalize_obs({"history": x})["history"], dtype=torch.float32)
        model.policy.set_training_mode(True)
        losses = []
        for _ in range(args.bootstrap_epochs):
            order = rng.permutation(len(x))
            for offset in range(0, len(x), 512):
                indices = order[offset : offset + 512]
                prediction = model.policy.get_distribution({"history": x[indices]}).distribution.mean
                loss = torch.mean((prediction - y[indices]) ** 2)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        # Low exploration for fine tuning a policy that already knows how to move.
        with torch.no_grad():
            model.policy.log_std.fill_(getattr(args, "bootstrap_log_std", -3.0))
        model.policy.set_training_mode(False)
        target = directory / f"bootstrap-{iteration}"
        export_policy(model, env, target, {**metadata, "bootstrap_steps": total, "bootstrap_round": iteration})
        report = evaluate(
            PPOPolicy(target),
            config,
            episodes=args.eval_episodes,
            seed=50000,
            teacher_approach=config.base_scenario != "standard",
        )
        (target / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        metrics = evaluation_metrics(report)
        row = {
            "bootstrap/environment_steps": total,
            "bootstrap/loss": float(np.mean(losses)),
            **{f"bootstrap/validation/{k}": v for k, v in metrics.items() if v is not None},
        }
        with (directory / "bootstrap-metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if run is not None:
            run.log(row)
        print(f"Bootstrap round {iteration}: {json.dumps(row)}", flush=True)
        score = (metrics["success_rate"], -metrics["collision_rate"], -metrics["xy_error"] - 0.2 * metrics["yaw_error"])
        if best_score is None or score > best_score:
            best_score = score
            export_policy(
                model,
                env,
                directory / "bootstrap-best",
                {**metadata, "bootstrap_steps": total, "bootstrap_round": iteration},
            )
            (directory / "bootstrap-best" / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    dataset_path = directory / "demonstrations.tmp.npz"
    np.savez_compressed(dataset_path, history=np.concatenate(inputs), actions=np.concatenate(labels))
    dataset_path.replace(directory / "demonstrations.npz")
    # Restore the best teacher-initialized actor and its normalization together.
    from stable_baselines3 import PPO

    best = PPOPolicy(directory / "bootstrap-best")
    model = PPO.load(directory / "bootstrap-best" / "policy.zip", env=env, device="cpu")
    for key, stats in best.manifest["normalization"].items():
        env.obs_rms[key].mean = np.asarray(stats["mean"])
        env.obs_rms[key].var = np.asarray(stats["var"])
        env.obs_rms[key].count = stats["count"]
    return model
