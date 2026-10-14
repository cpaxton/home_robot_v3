"""Train/evaluate physical manipulation and fit separate outcome-risk heads."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.evaluation import evaluate, score
from emet.rl.manipulation.policy import ManipulationHistoryFeatures, ManipulationPolicy, export


class Monitor(BaseCallback):
    def __init__(self, directory, config, fingerprint, *, interval, episodes, run=None):
        super().__init__()
        self.directory, self.config, self.fingerprint = directory, config, fingerprint
        self.interval, self.episodes, self.run = interval, episodes, run
        self.last = 0
        self.best = None
        self.completed = []

    def record(self, values):
        values = {"timesteps": self.num_timesteps, **values}
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(values) + "\n")
        if self.run:
            self.run.log(values)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self.completed.append(info)
        return True

    def _on_rollout_end(self):
        values = {
            k: float(v) for k, v in self.model.logger.name_to_value.items() if isinstance(v, (int, float, np.number))
        }
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
        )
        report = evaluate(ManipulationPolicy(target), config=self.config, episodes=self.episodes, seed=200000)
        (target / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        metrics = {
            f"validation/{scenario}/{k}": v
            for scenario, group in report["summary"].items()
            for k, v in group.items()
            if v is not None
        }
        self.record(metrics)
        value = score(report["summary"])
        if self.best is None or value > self.best:
            self.best = value
            export(
                self.model,
                self.model.get_vec_normalize_env(),
                self.directory / "best",
                self.config,
                self.fingerprint,
                steps=self.num_timesteps,
            )
            (self.directory / "best/validation.json").write_text(json.dumps(report, indent=2) + "\n")
        self.last = self.num_timesteps
        print("Validation", self.num_timesteps, json.dumps(report["summary"]), flush=True)


def train(args):
    torch.set_num_threads(1)
    config = ManipulationConfig(robot=args.robot, disturbance_curriculum=True)
    # Admit the model before starting workers or a tracking run.
    probe = ManipulationEnv(config)
    fingerprint = probe.fingerprint
    probe.close()
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
                    "envs": args.envs,
                    "fingerprint": fingerprint,
                    "source_hashes": {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(Path(__file__).parent.glob("*.py"))
                    },
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
                policy_kwargs={
                    "features_extractor_class": ManipulationHistoryFeatures,
                    "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                    "log_std_init": -20,
                },
            )
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.zero_()
            actor = [
                *model.policy.action_net.parameters(),
                *model.policy.mlp_extractor.policy_net.parameters(),
                model.policy.log_std,
            ]
            for parameter in actor:
                parameter.requires_grad_(False)
            if args.initialization_steps:
                model.learn(args.initialization_steps)
            for parameter in actor:
                parameter.requires_grad_(True)
            with torch.no_grad():
                model.policy.log_std.fill_(-3.5)
            monitor = Monitor(
                directory, config, fingerprint, interval=args.eval_every, episodes=args.eval_episodes, run=run
            )
            monitor.init_callback(model)
            monitor.num_timesteps = model.num_timesteps
            monitor.checkpoint()
            model.learn(args.steps, callback=monitor, reset_num_timesteps=False)
            monitor.num_timesteps = model.num_timesteps
            if monitor.last != model.num_timesteps:
                monitor.checkpoint()
            export(model, env, directory / "last", config, fingerprint, steps=model.num_timesteps)
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
    p.add_argument("--steps", type=int, default=262144)
    p.add_argument("--initialization-steps", type=int, default=32768)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-every", type=int, default=32768)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--wandb-mode", choices=["disabled", "online", "offline"], default="disabled")
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--robot", choices=["stretch", "innate_mars"], default="stretch")
    p.add_argument("--policy")
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
        train(args)
    elif args.command == "evaluate":
        torch.set_num_threads(1)
        report = evaluate(
            ManipulationPolicy(args.policy) if args.policy else None,
            config=ManipulationConfig(robot=args.robot),
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
