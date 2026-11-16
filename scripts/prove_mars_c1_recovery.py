# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Audit original/current C1 code and replay one saved update without simulation.

The replay updates disposable in-memory copies only. It neither trains an
additional controller nor exports an actor; all original artifacts stay intact.
"""

import argparse
import ast
import difflib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.manipulation.multi_critic_ppo import MultiCriticPPO, PPOConfig
from emet.rl.manipulation.pair_recovery import (
    RECIPE,
    ROOT,
    canonical,
    read,
    sha,
    source_equivalence,
    tensor_states_equal,
    verify_recovery,
)
from emet.rl.manipulation.whole_body_actor import MultiCriticPolicy
from emet.rl.manipulation.whole_body_training import arrays_digest


def definition(text, symbol):
    nodes = ast.parse(text).body
    for part in symbol.split("."):
        node = next(n for n in nodes if getattr(n, "name", None) == part)
        nodes = getattr(node, "body", [])
    # The timing repair documents GAE's existing reward contract. A leading
    # docstring changes introspection, not this experiment's executable math.
    if node.body and isinstance(node.body[0], ast.Expr):
        value = node.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            node.body = node.body[1:]
    return ast.dump(node, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--current-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    recipe, reference = read(RECIPE), read(args.reference_run)
    assert sha(args.reference_run) == recipe["reference_run_sha256"]
    assert canonical(reference) == recipe["reference_run_digest"]
    assert sha(Path(__file__)) == recipe["replay_script_sha256"]
    current_sources = source_hashes()
    source_equivalence(reference["source_hashes"], current_sources, recipe)
    args.output.mkdir(parents=True, exist_ok=False)

    symbols = {
        "rl/manipulation/env.py": ["ManipulationEnv"],
        "rl/manipulation/mastery.py": ["MasteryEnv"],
        "rl/manipulation/whole_body_training.py": ["make_environment"],
        "rl/manipulation/multi_critic_ppo.py": ["grouped_gae", "combine_advantages"],
    }
    for name, change in recipe["source_changes"].items():
        old_path, new_path = args.reference_source / "src/emet" / name, ROOT / "src/emet" / name
        before = old_path.read_text() if change["before"] is not None else ""
        after = new_path.read_text() if change["after"] is not None else ""
        if change["before"] is not None:
            assert sha(old_path) == change["before"]
        if change["after"] is not None:
            assert sha(new_path) == change["after"]
        target = args.output / "source-diffs" / (name + ".diff")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), name, name)))
        for symbol in symbols.get(name, []):
            assert definition(before, symbol) == definition(after, symbol), (name, symbol)

    archive = args.reference_run.parent / "first-rollout.npz"
    with np.load(archive, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files if k != "raw_advantage"}
        advantage = data["raw_advantage"].copy()
    first = reference["first_rollout"]
    assert arrays_digest(arrays) == first["arrays_sha256"]
    assert arrays_digest({"raw_gae": advantage}) == first["raw_gae_sha256"]
    rollout = {k: v for k, v in arrays.items() if not k.startswith("obs/")}
    rollout["observations"] = {k[4:]: v for k, v in arrays.items() if k.startswith("obs/")}
    assert len(rollout["ticks"]) == 256 and np.all(rollout["ticks"] == 1)
    initial_path = args.reference_run.parent / "initial/training.pt"
    assert sha(initial_path) == reference["initial_training_state_sha256"]
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)

    # Reconstruct the post-rollout Torch stream through the original saved
    # observations, checking every sampled action, without stepping MuJoCo.
    sampler = MultiCriticPolicy()
    sampler.load_state_dict(initial["policy"])
    torch.set_rng_state(initial["torch_rng"])
    for index, expected in enumerate(rollout["actions"]):
        obs = {k: torch.from_numpy(v[index])[None] for k, v in rollout["observations"].items()}
        with torch.no_grad():
            action = sampler.distribution(obs).sample() * sampler.actor.action_mask(obs)
        np.testing.assert_array_equal(action[0].numpy(), expected)
    optimizer_rng = torch.get_rng_state()

    old_module_path = args.reference_source / "src/emet/rl/manipulation/multi_critic_ppo.py"
    assert sha(old_module_path) == reference["source_hashes"]["rl/manipulation/multi_critic_ppo.py"]
    spec = importlib.util.spec_from_file_location("c1_original_ppo", old_module_path)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    artifacts, states = {}, []
    for label, implementation, config_type in (
        ("reference", original.MultiCriticPPO, original.PPOConfig),
        ("current", MultiCriticPPO, PPOConfig),
    ):
        policy = MultiCriticPolicy()
        policy.load_state_dict(initial["policy"])
        learner = implementation(policy, config_type(**reference["ppo"]))
        learner.optimizer.load_state_dict(initial["optimizer"])
        torch.set_rng_state(optimizer_rng)
        learner.update({**rollout, "reward_discount": reference["ppo"]["gamma"]})
        state = {"policy": policy.state_dict(), "optimizer": learner.optimizer.state_dict()}
        path = args.output / (label + "-post-update.pt")
        torch.save(state, path)
        artifacts[label] = {"path": str(path.resolve()), "sha256": sha(path)}
        states.append(state)
    assert tensor_states_equal(*states), "old/current update changed policy or optimizer numerics"

    proof = {
        "format": "mars-c1-source-recovery-proof-v1",
        "recipe_sha256": sha(RECIPE),
        "reference_run_sha256": sha(args.reference_run),
        "current_sources_digest": canonical(current_sources),
        "reference": {
            "preflight": str(Path(recipe["reference_preflight"]).resolve()),
            "teacher": str(Path(recipe["reference_preflight"]).resolve().parent / "general-teacher.json.gz"),
        },
        "current": {
            "preflight": str(args.current_preflight.resolve()),
            "teacher": str(args.current_preflight.resolve().parent / "general-teacher.json.gz"),
        },
        "optimizer_replay": {"first_rollout": first, **artifacts},
        "replay_script": str(Path(__file__).resolve()),
        "reference_ppo_source": str(old_module_path.resolve()),
        "ast_equal_symbols": symbols,
        "replayed_saved_actions": 256,
        "new_simulator_ticks": 0,
        "new_training_decisions": 0,
        "actor_exported": False,
    }
    path = args.output / "proof.json"
    path.write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    current = {**reference, "source_hashes": current_sources, "preflight_sha256": sha(args.current_preflight)}
    current["ppo"] = {**reference["ppo"], "advantage_normalization": "shared"}
    verify_recovery(reference, current, path, reference_sha256=sha(args.reference_run))
    print(json.dumps({"proof": str(path), "optimizer_update_exact": True, "new_training_decisions": 0}))


if __name__ == "__main__":
    main()
