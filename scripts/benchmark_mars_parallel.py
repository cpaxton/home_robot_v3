"""Benchmark native Mars workers and optionally smoke-test PPO; exports no actor.

Run on a compute host, not the unstable workstation. This is throughput/setup
evidence, not a replacement for teacher preflight or qualification gates.
"""

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import numpy as np
    import torch

    from emet.rl.manipulation.carry_benchmark import source_hashes
    from emet.rl.manipulation.env import ManipulationConfig
    from emet.rl.manipulation.mastery import load_suite, make_training_vec_env
    from emet.rl.manipulation.normalization import CommandVecNormalize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "configs/benchmarks/mars_tracking_mastery_restoring_yaw_v1.json"
    )
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--steps-per-env", type=int, default=128)
    parser.add_argument("--ppo-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not all(1 <= n <= 16 for n in args.num_envs) or args.steps_per_env < 1:
        parser.error("fresh output, 1-16 workers and positive steps required")
    torch.set_num_threads(1)
    suite = load_suite(args.suite)
    before = source_hashes()
    result = {
        "format": "emet-mars-native-parallel-probe-v1",
        "hardware_used": False,
        "training_qualified": False,
        "source_hashes": before,
        "config": asdict(ManipulationConfig(**suite["config"])),
        "runs": [],
    }
    for n in args.num_envs:
        start = time.perf_counter()
        env = make_training_vec_env(suite, ["base"], num_envs=n)
        try:
            env.seed(920000)
            obs = env.reset()
            fingerprints = env.get_attr("fingerprint")
            if len(set(fingerprints)) != 1:
                raise RuntimeError("worker model fingerprints differ")
            setup = time.perf_counter() - start
            terminals = Counter()
            start = time.perf_counter()
            for _ in range(args.steps_per_env):
                obs, rewards, dones, infos = env.step(np.zeros((n, *env.action_space.shape), dtype=np.float32))
                if not all(np.isfinite(x).all() for x in obs.values()) or not np.isfinite(rewards).all():
                    raise RuntimeError("non-finite worker output")
                terminals.update(info.get("code", "unknown") for done, info in zip(dones, infos, strict=True) if done)
            elapsed = time.perf_counter() - start
            row = {
                "num_envs": n,
                "backend": "dummy" if n == 1 else "spawn-subproc",
                "setup_seconds": setup,
                "rollout_seconds": elapsed,
                "steps": n * args.steps_per_env,
                "control_steps_per_second": n * args.steps_per_env / elapsed,
                "fingerprint": fingerprints[0],
                "terminal_codes": dict(terminals),
            }
            if args.ppo_smoke:
                from stable_baselines3 import PPO

                from emet.rl.manipulation.feedback_policy import FeedbackActorCriticPolicy, GoalFeedbackFeatures

                cfg = ManipulationConfig(**suite["config"])
                normalized = CommandVecNormalize(env, config=cfg, norm_reward=False, clip_obs=20)
                model = PPO(
                    FeedbackActorCriticPolicy,
                    normalized,
                    device="cpu",
                    seed=90,
                    n_steps=32,
                    batch_size=min(128, n * 32),
                    n_epochs=1,
                    policy_kwargs={
                        "restoring_yaw_feedback": cfg.restoring_yaw_feedback,
                        "turn_xy_tolerance": cfg.base_xy_tolerance,
                        "features_extractor_class": GoalFeedbackFeatures,
                        "features_extractor_kwargs": {
                            "robot": cfg.robot,
                            "retention_context": True,
                            "orientation_scale": cfg.feedback_orientation_scale,
                        },
                        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                    },
                )
                start = time.perf_counter()
                model.learn(total_timesteps=n * 64)
                row["ppo_smoke"] = {
                    "steps": model.num_timesteps,
                    "updates": model._n_updates,
                    "seconds": time.perf_counter() - start,
                    "checkpoint_exported": False,
                }
                if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
                    raise RuntimeError("non-finite PPO parameters")
            result["runs"].append(row)
            print(json.dumps(row), flush=True)
        finally:
            env.close()
    if source_hashes() != before:
        raise RuntimeError("sources changed during parallel probe")
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
