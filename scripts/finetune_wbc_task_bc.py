# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Behavior-clone a PandaOmron WBC checkpoint from task DAgger data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_env import OBSERVATION_LAYOUT_VERSION
from emet.rl.manipulation.wbc_training import WbcPPO


def load_task_data(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    episode_offset = 0
    for path in paths:
        with np.load(path) as payload:
            inputs = np.asarray(payload["observations"], dtype=np.float32)
            labels = np.asarray(payload["teacher_actions"], dtype=np.float32)
            if inputs.ndim != 2 or labels.ndim != 2 or not len(inputs) or len(inputs) != len(labels):
                raise ValueError(f"{path} needs one teacher action per observation")
            ids = np.asarray(payload.get("episode_ids", np.zeros(len(inputs))), dtype=np.int64)
            if ids.shape != (len(inputs),) or np.any(ids < 0):
                raise ValueError(f"{path} has invalid episode IDs")
        observations.append(inputs)
        actions.append(labels)
        episodes.append(ids + episode_offset)
        episode_offset = int(episodes[-1].max()) + 1
    return (
        np.concatenate(observations, axis=0),
        np.concatenate(actions, axis=0),
        np.concatenate(episodes, axis=0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a WBC actor on task observations")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    observations, actions, episodes = load_task_data(args.data)
    print(
        f"samples={len(observations)} episodes={len(np.unique(episodes))} "
        f"obs={tuple(observations.shape)} actions={tuple(actions.shape)}",
        flush=True,
    )
    trainer, metadata = WbcPPO.load(args.checkpoint, pandaomron_sim_profile(), device=args.device)
    result = trainer.pretrain(
        observations,
        actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(f"bc_loss={result['bc_loss']:.6f} samples={int(result['bc_samples'])}", flush=True)

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw["observation_layout_version"] = OBSERVATION_LAYOUT_VERSION
    raw["observation_size"] = trainer.policy.observation_size
    raw["state_dict"] = {key: value.detach().cpu() for key, value in trainer.policy.state_dict().items()}
    raw["optimizer_state"] = None
    raw["reference_actor_state"] = None
    raw["reference_log_std"] = None
    raw["metadata"] = {
        **dict(raw.get("metadata", {})),
        "task_bc": {
            "data": [str(path) for path in args.data],
            "samples": int(len(observations)),
            "episodes": int(len(np.unique(episodes))),
            "bc_loss": float(result["bc_loss"]),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "checkpoint": str(args.checkpoint),
            "stage": metadata.get("stage"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw, args.output)
    print(f"saved={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
