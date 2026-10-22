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


def _action_loss_mask(observations, task, reference):
    """Only regress commands that the masked goal contract will execute."""
    if task != "coordinated":
        return torch.ones_like(reference)
    goal = torch.as_tensor(observations["goal"], dtype=reference.dtype)
    base = goal[:, 9:10].expand(-1, 2)
    arm = goal[:, 10:15]
    return torch.cat((base, arm), dim=1)


def bootstrap_base(model, env, args, directory, metadata, run=None):
    from emet.rl.__main__ import evaluate
    from emet.rl.env import EnvConfig

    if args.task not in {"base", "coordinated"} or (args.task == "base" and args.architecture != "mlp"):
        raise ValueError("teacher initialization supports base MLP and coordinated pose policies")
    metadata = {**metadata, "bootstrap_labels": "desired_velocity"}
    directory = Path(directory)
    rng = np.random.default_rng(args.seeds[0] + 7919)
    obs = env.reset()
    raw_obs = env.get_original_obs()
    rollout_steps = max(1, args.bootstrap_steps // env.num_envs)
    samples_per_round = rollout_steps * env.num_envs
    total_capacity = samples_per_round * args.bootstrap_rounds
    buffer_paths = []
    input_buffer = {
        key: np.lib.format.open_memmap(
            directory / f".bootstrap-{key}.npy",
            mode="w+",
            dtype=value.dtype,
            shape=(total_capacity, *value.shape[1:]),
        )
        for key, value in raw_obs.items()
    }
    buffer_paths.extend(directory / f".bootstrap-{key}.npy" for key in raw_obs)
    label_path = directory / ".bootstrap-actions.npy"
    buffer_paths.append(label_path)
    label_buffer = np.lib.format.open_memmap(
        label_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_capacity, model.action_space.shape[0]),
    )
    best_score = None
    best_metrics = None
    # Optimize the deterministic actor, leaving its critic for RL fine tuning.
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    config = EnvConfig(**metadata["env"])
    total = 0
    for iteration in range(args.bootstrap_rounds):
        if iteration:
            obs = env.reset()
        # Keep teacher/student choice fixed for a full episode. Switching at
        # every control tick produces a chattering behavior distribution that
        # neither controller will encounter at deployment and needlessly drives
        # the simulator into collision states.
        student_fraction = min(
            0.30,
            0.10 * (iteration + int(bool(getattr(args, "resume", None)))),
        )
        student_rollout = rng.random(env.num_envs) < student_fraction
        for step_index in range(rollout_steps):
            raw = env.get_original_obs()
            teacher = np.asarray(env.env_method("expert_action"), dtype=np.float32)
            # Learn desired velocities; the shared adapter enforces acceleration.
            # Regressing already-slewed commands makes copying the previous
            # command an easy near-identity solution that hides goal errors.
            targets = teacher
            start = iteration * samples_per_round + step_index * env.num_envs
            end = start + env.num_envs
            for key, value in raw.items():
                input_buffer[key][start:end] = value
            label_buffer[start:end] = targets
            # DAgger: query the teacher on student states, but do not randomly
            # splice teacher and student commands within a trajectory.
            prediction, _ = model.predict(obs, deterministic=True)
            selected = np.where(student_rollout[:, None], prediction, teacher)
            obs, _, dones, _ = env.step(selected)
            if iteration > 0 and np.any(dones):
                student_rollout[dones] = rng.random(int(np.sum(dones))) < student_fraction
            total += env.num_envs
        sample_count = total
        raw_inputs = {key: value[:sample_count] for key, value in input_buffer.items()}
        y = torch.as_tensor(label_buffer[:sample_count], dtype=torch.float32)
        # Masks are categorical contract data. Build them before observation
        # normalization, which intentionally standardizes every Box feature.
        action_masks = _action_loss_mask(raw_inputs, args.task, y)
        num_samples = len(y)
        model.policy.set_training_mode(True)
        losses = []
        final_epoch_losses = []
        for _epoch in range(args.bootstrap_epochs):
            order = rng.permutation(num_samples)
            epoch_losses = []
            for offset in range(0, num_samples, 256):
                indices = order[offset : offset + 256]
                raw_batch = {key: value[indices] for key, value in raw_inputs.items()}
                batch = {
                    key: torch.as_tensor(value, dtype=torch.float32)
                    for key, value in env.normalize_obs(raw_batch).items()
                }
                prediction = model.policy.get_distribution(batch).distribution.mean
                mask = action_masks[indices]
                loss = torch.sum((prediction - y[indices]) ** 2 * mask) / torch.clamp(mask.sum(), min=1)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
                optimizer.step()
                value = float(loss.detach())
                losses.append(value)
                epoch_losses.append(value)
            final_epoch_losses = epoch_losses
        # Low exploration for fine tuning a policy that already knows how to move.
        with torch.no_grad():
            model.policy.log_std.fill_(getattr(args, "bootstrap_log_std", -3.0))
        model.policy.set_training_mode(False)
        train_abs_error = []
        with torch.no_grad():
            for offset in range(0, num_samples, 512):
                raw_batch = {key: value[offset : offset + 512] for key, value in raw_inputs.items()}
                batch = {
                    key: torch.as_tensor(value, dtype=torch.float32)
                    for key, value in env.normalize_obs(raw_batch).items()
                }
                prediction = model.policy.get_distribution(batch).distribution.mean
                mask = action_masks[offset : offset + 512]
                train_abs_error.append(
                    (torch.sum(torch.abs(prediction - y[offset : offset + 512]) * mask) / torch.clamp(mask.sum(), min=1)).item()
                )
        train_action_mae = float(np.mean(train_abs_error))
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
            "bootstrap/training_samples": num_samples,
            "bootstrap/student_episode_fraction": student_fraction,
            "bootstrap/loss": float(np.mean(losses)),
            "bootstrap/final_epoch_loss": float(np.mean(final_epoch_losses)),
            "bootstrap/train_action_mae": train_action_mae,
            **{f"bootstrap/validation/{k}": v for k, v in metrics.items() if v is not None},
        }
        with (directory / "bootstrap-metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if run is not None:
            run.log(row)
        print(f"Bootstrap round {iteration}: {json.dumps(row)}", flush=True)
        score = (
            metrics["success_rate"],
            -metrics["collision_rate"],
            -metrics.get("teacher_action_mae", float("inf")),
            -metrics["xy_error"] - 0.2 * metrics["yaw_error"],
        )
        if best_score is None or score > best_score:
            best_score = score
            best_metrics = metrics
            export_policy(
                model,
                env,
                directory / "bootstrap-best",
                {**metadata, "bootstrap_steps": total, "bootstrap_round": iteration},
            )
            (directory / "bootstrap-best" / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    # Keep failed runs inspectable: this data is valuable for diagnosing a
    # distribution mismatch, and the validation gate below still prevents PPO
    # from starting from a bad initialization.
    dataset_path = directory / "demonstrations.tmp.npz"
    training_inputs = {key: value[:total] for key, value in input_buffer.items()}
    np.savez_compressed(dataset_path, **training_inputs, actions=label_buffer[:total])
    dataset_path.replace(directory / "demonstrations.npz")
    for buffer in (*input_buffer.values(), label_buffer):
        buffer.flush()
    del input_buffer, label_buffer
    for path in buffer_paths:
        path.unlink(missing_ok=True)
    if (
        best_metrics["success_rate"] < getattr(args, "bootstrap_min_success_rate", 0.8)
        or best_metrics["collision_rate"] > getattr(args, "bootstrap_max_collision_rate", 0.05)
        or best_metrics.get("teacher_action_mae", float("inf"))
        > getattr(args, "bootstrap_max_teacher_action_mae", 0.35)
    ):
        raise RuntimeError(
            "teacher initialization failed its validation gate: "
            f"success={best_metrics['success_rate']:.3f}, "
            f"collision={best_metrics['collision_rate']:.3f}, "
            f"teacher_action_mae={best_metrics.get('teacher_action_mae', float('inf')):.3f}; "
            "PPO training was not started"
        )
    # Restore the best teacher-initialized actor and its normalization together.
    from stable_baselines3 import PPO

    best = PPOPolicy(directory / "bootstrap-best")
    model = PPO.load(directory / "bootstrap-best" / "policy.zip", env=env, device="cpu")
    for key, stats in best.manifest["normalization"].items():
        env.obs_rms[key].mean = np.asarray(stats["mean"])
        env.obs_rms[key].var = np.asarray(stats["var"])
        env.obs_rms[key].count = stats["count"]
    return model
