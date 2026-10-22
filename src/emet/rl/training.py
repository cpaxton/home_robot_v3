"""Training checkpoints, held-out validation and optional W&B tracking."""

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from emet.rl.policy import PPOPolicy, export_policy


def evaluation_metrics(report):
    rows = report["episodes"]
    result = {k: report[k] for k in ("success_rate", "collision_rate", "observation_and_inference_p95_ms")}
    if "teacher_action_mae" in report:
        result["teacher_action_mae"] = float(report["teacher_action_mae"])
    for key in ("unsafe_contact", "invalid_physics"):
        result[f"{key}_rate"] = float(np.mean([r.get(key, False) for r in rows]))
    result["event_episode_rate"] = float(np.mean([r.get("event_count", 0) > 0 for r in rows]))
    for key in (
        "xy_error",
        "yaw_error",
        "joint_error",
        "mean_joint_error",
        "ee_position_error",
        "ee_orientation_error",
        "peak_contact_force",
        "contact_impulse",
        "peak_overshoot",
    ):
        if key in rows[0]:
            result[key] = float(np.mean([r[key] for r in rows]))
    for key in ("rms_measured_acceleration", "rms_measured_jerk", "peak_command_acceleration"):
        if key in rows[0]:
            values = np.mean([r[key] for r in rows], axis=0)
            result.update({f"{key}/{axis}": float(v) for axis, v in zip(("linear", "angular"), values, strict=True)})
    successes = [r["time"] for r in rows if r["is_success"]]
    result["successful_episode_time"] = float(np.mean(successes)) if successes else None
    for deadline in (3, 5, 10):
        result[f"success_by_{deadline}s"] = float(np.mean([r["is_success"] and r["time"] <= deadline for r in rows]))
    recovery = [r["event_elapsed"] for r in rows if r["is_success"] and r.get("event_elapsed") is not None]
    result["successful_recovery_time"] = float(np.mean(recovery)) if recovery else None
    return result


def checkpoint_score(metrics, *, contact_training=False):
    """Prefer low-severity interaction; contact count is not a damage metric."""
    jerk = sum(metrics.get(f"rms_measured_jerk/{axis}", 0) for axis in ("linear", "angular"))
    return (
        -metrics.get("unsafe_contact_rate", 0),
        -metrics.get("invalid_physics_rate", 0),
        0 if contact_training else -metrics["collision_rate"],
        metrics["success_rate"],
        -metrics.get("peak_contact_force", 0),
        -metrics.get("contact_impulse", 0),
        -jerk,
        -metrics["xy_error"] - 0.2 * metrics["yaw_error"] - metrics.get("joint_error", 0),
    )


class TrainingMonitor(BaseCallback):
    def __init__(self, directory, config, metadata, *, interval=32768, episodes=20, run=None, evaluate_initial=False):
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config, self.metadata = config, metadata
        self.interval, self.episodes, self.run = interval, episodes, run
        self.evaluate_initial = evaluate_initial
        self.best_score = None
        self.last_checkpoint = 0
        self.completed = []

    def _on_training_start(self):
        self.last_checkpoint = self.num_timesteps
        if self.evaluate_initial:
            self.checkpoint()

    def record(self, metrics):
        row = {"timesteps": self.num_timesteps, **metrics}
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if self.run is not None:
            self.run.log(row)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self.completed.append(info)
        return True

    def _on_rollout_end(self):
        metrics = {
            k: float(v)
            for k, v in self.model.logger.name_to_value.items()
            if np.isscalar(v) and isinstance(v, (int, float, np.number))
        }
        if self.completed:
            for key in ("is_success", "xy_error", "yaw_error", "collision", "obstruction"):
                metrics[f"episodes/{key}"] = float(np.mean([r[key] for r in self.completed]))
            metrics["episodes/reward"] = float(np.mean([r["episode"]["r"] for r in self.completed]))
            self.completed.clear()
        self.record(metrics)
        if self.num_timesteps - self.last_checkpoint >= self.interval:
            self.checkpoint()

    def checkpoint(self):
        from emet.rl.__main__ import evaluate

        normalizer = self.model.get_vec_normalize_env()
        metadata = {**self.metadata, "steps": self.num_timesteps}
        target = self.directory / f"step-{self.num_timesteps}"
        export_policy(self.model, normalizer, target, metadata)
        report = evaluate(
            PPOPolicy(target),
            self.config,
            episodes=self.episodes,
            seed=50000,
            teacher_approach=self.config.base_scenario != "standard",
        )
        (target / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        metrics = evaluation_metrics(report)
        self.record({f"validation/{k}": v for k, v in metrics.items() if v is not None})
        score = checkpoint_score(metrics, contact_training=self.config.contact_training)
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            export_policy(self.model, normalizer, self.directory / "best", metadata)
            (self.directory / "best" / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        self.last_checkpoint = self.num_timesteps
        print(f"Validation at {self.num_timesteps}: {json.dumps(metrics)}", flush=True)

    def _on_training_end(self):
        if self.last_checkpoint != self.num_timesteps or self.best_score is None:
            self.checkpoint()


def start_wandb(args, directory, config):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install the rl-tracking extra to enable W&B") from exc
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or Path(directory).name,
        mode=args.wandb_mode,
        dir=str(directory),
        config=config,
        save_code=False,
    )
    run.define_metric("timesteps")
    run.define_metric("*", step_metric="timesteps")
    run.define_metric("bootstrap/*", step_metric="bootstrap/environment_steps")
    run.define_metric("critic_warmup/*", step_metric="critic_warmup/epoch")
    (Path(directory) / "wandb-run.json").write_text(
        json.dumps({"id": run.id, "url": run.url, "mode": args.wandb_mode}) + "\n"
    )
    return run


def finish_wandb(run, directory, *, failed=False):
    if run is None:
        return
    try:
        if not failed and (Path(directory) / "best").exists():
            import wandb

            artifact = wandb.Artifact(f"mars-policy-{run.id}", type="model")
            artifact.add_dir(str(Path(directory) / "best"))
            run.log_artifact(artifact, aliases=["best"])
    finally:
        run.finish(exit_code=1 if failed else 0)
