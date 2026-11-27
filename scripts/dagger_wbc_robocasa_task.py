# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Iterative DAgger for the PandaOmron RoboCasa task bridge.

Each round rolls out the current policy, records the actor-visited observations
with the privileged task teacher labels, aggregates the new data with earlier
rounds (and optional successful teacher demonstrations), fine-tunes from the
same starting checkpoint, and evaluates on a fixed task sweep. The policy is
never promoted automatically; the report records success and failure modes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _aggregate(paths: list[Path], output: Path) -> Path:
    arrays: dict[str, list[np.ndarray]] = {}
    episode_offset = 0
    for path in paths:
        payload = np.load(path)
        for key in payload.files:
            values = np.asarray(payload[key])
            if key == "episode_ids":
                values = values.astype(np.int64) + episode_offset
                episode_offset = int(values.max()) + 1
            arrays.setdefault(key, []).append(values)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{key: np.concatenate(values) for key, values in arrays.items()})
    return output


def _run(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)


def _success(path: Path) -> tuple[float, dict]:
    payload = json.loads(path.read_text())
    return float(payload["success_rate"]), payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DAgger against the RoboCasa task teacher")
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-data", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rollout-seed", type=int, default=52000)
    parser.add_argument("--eval-seed", type=int, default=50000)
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--max-move-steps", type=int, default=90)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    sweep = repo / "scripts" / "sweep_wbc_robocasa_tasks.py"
    finetune = repo / "scripts" / "finetune_wbc_task_bc.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    args.output.mkdir(parents=True, exist_ok=True)
    current = args.init_checkpoint
    rollouts: list[Path] = []
    history: list[dict] = []
    for round_index in range(args.rounds):
        round_dir = args.output / f"round_{round_index}"
        rollout_dir = round_dir / "rollout"
        _run(
            [
                sys.executable,
                str(sweep),
                "--checkpoint",
                str(current),
                "--episodes",
                str(args.episodes),
                "--seed",
                str(args.rollout_seed + round_index * 100),
                "--layout",
                str(args.layout),
                "--style",
                str(args.style),
                "--backoff",
                "0.0",
                "--max-move-steps",
                str(args.max_move_steps),
                "--save-data",
                "--output",
                str(rollout_dir),
            ],
            round_dir / "rollout.log",
            env,
        )
        rollout_data = rollout_dir / "loaded_task_data.npz"
        if not rollout_data.exists():
            raise RuntimeError(f"rollout produced no data: {rollout_data}")
        rollouts.append(rollout_data)
        data_paths = [*args.teacher_data, *rollouts]
        aggregated = _aggregate(data_paths, round_dir / "aggregated.npz")
        policy_path = round_dir / "policy.pt"
        _run(
            [
                sys.executable,
                str(finetune),
                "--data",
                *[str(path) for path in [aggregated]],
                "--checkpoint",
                str(args.init_checkpoint),
                "--output",
                str(policy_path),
                "--epochs",
                str(args.epochs),
                "--learning-rate",
                str(args.learning_rate),
                "--device",
                args.device,
                "--seed",
                str(round_index),
            ],
            round_dir / "finetune.log",
            env,
        )
        eval_dir = round_dir / "eval"
        _run(
            [
                sys.executable,
                str(sweep),
                "--checkpoint",
                str(policy_path),
                "--episodes",
                str(args.eval_episodes),
                "--seed",
                str(args.eval_seed),
                "--layout",
                str(args.layout),
                "--style",
                str(args.style),
                "--backoff",
                "0.0",
                "--max-move-steps",
                str(args.max_move_steps),
                "--output",
                str(eval_dir),
            ],
            round_dir / "eval.log",
            env,
        )
        success_rate, eval_payload = _success(eval_dir / "task_sweep.json")
        record = {
            "round": round_index,
            "train_checkpoint": str(current),
            "policy": str(policy_path),
            "samples": int(len(np.load(aggregated)["observations"])),
            "success_rate": success_rate,
            "failure_counts": eval_payload["failure_counts"],
            "episodes": eval_payload["episodes_detail"],
        }
        history.append(record)
        (args.output / "dagger_report.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
        print(
            f"round={round_index} samples={record['samples']} eval_success={success_rate:.2f}",
            flush=True,
        )
        current = policy_path
        if success_rate > 0:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
