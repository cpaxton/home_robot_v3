# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""PPO fine-tune a carry checkpoint on the payload contact suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_dagger import WbcTaskSuiteEnv, evaluate_suite, subtasks_from_contract
from emet.rl.manipulation.wbc_training import WbcPPO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    profile = pandaomron_sim_profile()
    trainer, metadata = WbcPPO.load(args.checkpoint, profile, device=args.device)
    contract = metadata.get("contract")
    tasks = subtasks_from_contract(contract)
    env = WbcTaskSuiteEnv(tasks, robot="pandaomron", seed=args.seed)
    try:
        before = evaluate_suite(env, trainer.policy, episodes_per_task=args.eval_episodes, seed=args.seed)
        print(
            f"before success={before['aggregate']['success_rate']:.2f} "
            f"track={before['aggregate']['tracking_error_m']:.4f}",
            flush=True,
        )
        metrics = trainer.train(
            env,
            total_steps=args.steps,
            checkpoint_path=args.output / "checkpoint.pt",
            checkpoint_every=max(args.steps // 4, 1),
            metrics_path=args.output / "metrics.jsonl",
            checkpoint_metadata={"stage": "contact_ppo", "contract": contract},
        )
        print("train", json.dumps(metrics, sort_keys=True), flush=True)
        after = evaluate_suite(env, trainer.policy, episodes_per_task=args.eval_episodes, seed=args.seed)
        trainer.save(
            args.output / "final.pt",
            env.envs[env.names[0]],
            metadata={"stage": "contact_ppo", "contract": contract, "evaluation": after},
        )
        print(
            f"after success={after['aggregate']['success_rate']:.2f} "
            f"track={after['aggregate']['tracking_error_m']:.4f}",
            flush=True,
        )
        (args.output / "report.json").write_text(
            json.dumps({"before": before, "train": metrics, "after": after}, indent=2, sort_keys=True) + "\n"
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
