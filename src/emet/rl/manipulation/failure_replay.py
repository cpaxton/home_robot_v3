"""Training-only neighborhoods of failed, physically initialized trajectories.

The actor receives its existing pose preview and measured history. Replay never
injects a saved object state or makes a task decision at execution time. This
first sampler covers supported-lift failures; public multi-stage TAMP replay is
a separate gate, not a capability claimed by this sampler.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import gymnasium as gym
import numpy as np

from emet.rl.manipulation.env import MIXTURE
from emet.rl.manipulation.retention_training import SCENARIO_NAMES

FORMAT = "mars-lift-failure-neighborhoods-v1"
DEVELOPMENT_START, DEVELOPMENT_END = 33_000_000, 33_500_000


def family_partition(seed):
    """Split parents, not neighboring goals or adjacent rollout frames."""
    return "validation" if int(hashlib.sha256(str(seed).encode()).hexdigest()[:8], 16) % 5 == 0 else "train"


def mine_failures(report, teacher_report, *, goal_jitter_rad=0.01):
    from emet.rl.manipulation.carry_benchmark import digest

    if not report.get("complete") or report.get("controller") != "policy":
        raise ValueError("complete policy development report required")
    if (
        not teacher_report.get("complete")
        or teacher_report.get("controller") != "teacher"
        or teacher_report["config"] != report["config"]
        or teacher_report["source_hashes"] != report["source_hashes"]
    ):
        raise ValueError("matched complete teacher development report required")
    if not np.isfinite(goal_jitter_rad) or not 0 < goal_jitter_rad <= 0.03:
        raise ValueError("goal jitter must be in (0, 0.03] rad")
    config = report["config"]
    if config["robot"] != "innate_mars" or config["scenario"] != "carry" or config["grasp_lift_probability"] != 1:
        raise ValueError("report must contain supported Mars lift trajectories")
    # A dedicated namespace prevents mining the frozen assessment and public
    # regression seeds used by prior experiments. Later corpora need a new split.
    seeds = [c["seed"] for c in report["cases"]]
    if len(set(seeds)) != len(seeds) or any(
        type(s) is not int or not DEVELOPMENT_START <= s < DEVELOPMENT_END for s in seeds
    ):
        raise ValueError("unique development-namespace seeds required; assessment cases cannot be mined")
    families = []
    teacher_cases = {c["seed"]: c for c in teacher_report["cases"]}
    if len(teacher_cases) != len(teacher_report["cases"]) or set(teacher_cases) != set(seeds):
        raise ValueError("teacher development families differ")
    rejected, successes, unverified = [], [], []
    for case in report["cases"]:
        info = case["final_info"]
        if case["initialization_failed"]:
            rejected.append(case["seed"])
        elif info.get("is_success"):
            successes.append(case["seed"])
        else:
            if not case.get("trace") or not case.get("admission", {}).get("accepted"):
                raise ValueError("failed family must have an admitted trajectory and measured rollout")
            teacher = teacher_cases[case["seed"]]
            if teacher["admission"]["initial_state"] != case["admission"]["initial_state"]:
                raise ValueError("teacher and policy initial physics differ")
            if (
                teacher["fingerprint"] != case["fingerprint"]
                or teacher["admission"].get("goal") != case["admission"].get("goal")
                or teacher["admission"].get("trajectory") != case["admission"].get("trajectory")
            ):
                raise ValueError("teacher and policy planned targets differ")
            if teacher["initialization_failed"] or not teacher["final_info"].get("is_success"):
                unverified.append(case["seed"])
                continue
            families.append(
                {"seed": case["seed"], "partition": family_partition(case["seed"]), "failure": info["code"]}
            )
    if {f["partition"] for f in families} != {"train", "validation"}:
        raise ValueError("more development parents required for independent train/validation families")
    fingerprints = {c["fingerprint"] for c in report["cases"]}
    if len(fingerprints) != 1:
        raise ValueError("mixed model fingerprints")
    return {
        "format": FORMAT,
        "source_report_sha256": digest(report),
        "teacher_report_sha256": digest(teacher_report),
        "source_hashes": report["source_hashes"],
        "config": json.loads(json.dumps(config)),
        "fingerprint": fingerprints.pop(),
        "goal_jitter_rad": float(goal_jitter_rad),
        "families": families,
        "rejected_parent_seeds": rejected,
        "successful_parent_seeds": successes,
        "teacher_unverified_parent_seeds": unverified,
        "parent_admission": "matched teacher final success; a debugging subset, not a universal feasibility criterion",
        "admission": "each perturbed target is replanned and checked at reset; rejection is preserved",
        "safety_promoted": False,
    }


def validate_corpus(corpus, config, *, check_sources=True):
    from emet.rl.manipulation.carry_benchmark import source_hashes

    if corpus.get("format") != FORMAT or corpus.get("safety_promoted") is not False:
        raise ValueError("unsupported failure replay corpus")
    if config.robot != "innate_mars" or config.scenario != "mixed" or not config.grasp_lift_probability:
        raise ValueError("failure replay requires the mixed Mars lift curriculum")
    # Only the task selector/subtype/phase mix differs from the forced-lift
    # development probe. Geometry, load distribution, servo and limits agree.
    selection = {
        "scenario",
        "grasp_lift_probability",
        "carry_hold_probability",
        "grasp_lift_phase_weights",
        "carry_push_probability",
    }
    current = json.loads(json.dumps(asdict(config)))
    if {k: v for k, v in current.items() if k not in selection} != {
        k: v for k, v in corpus["config"].items() if k not in selection
    }:
        raise ValueError("failure replay configuration differs from its measured parents")
    jitter = corpus.get("goal_jitter_rad")
    if not isinstance(jitter, (int, float)) or not np.isfinite(jitter) or not 0 < jitter <= 0.03:
        raise ValueError("invalid failure replay jitter")
    families = corpus.get("families", [])
    seeds = [f["seed"] for f in families]
    if (
        not families
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not DEVELOPMENT_START <= s < DEVELOPMENT_END for s in seeds)
        or any(f["partition"] != family_partition(f["seed"]) for f in families)
        or {f["partition"] for f in families} != {"train", "validation"}
    ):
        raise ValueError("invalid or overlapping failure replay families")
    if check_sources and corpus["source_hashes"] != source_hashes():
        raise ValueError("failure replay sources changed; regenerate the development probes")


class FailureReplay(gym.Wrapper):
    """Replace a bounded fraction of carry resets; preserve the outer task mix."""

    def __init__(self, env, *, corpus, carry_fraction=0.5, partition="train"):
        super().__init__(env)
        validate_corpus(corpus, env.unwrapped.config)
        if env.unwrapped.fingerprint != corpus["fingerprint"]:
            raise ValueError("failure replay model mismatch")
        if (
            not np.isfinite(carry_fraction)
            or not 0 < carry_fraction <= 0.75
            or partition not in {"train", "validation"}
        ):
            raise ValueError("invalid carry replay fraction or partition")
        self.config = env.unwrapped.config
        self.corpus, self.carry_fraction, self.partition = corpus, float(carry_fraction), partition
        self.parents = [f for f in corpus["families"] if f["partition"] == partition]
        self.rng = np.random.default_rng()
        self.active = {"selected": False}

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("failure replay owns its bounded reset recipe")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        scenario = str(self.rng.choice(SCENARIO_NAMES, p=self.config.scenario_weights or MIXTURE))
        selected = scenario == "carry" and self.rng.random() < self.carry_fraction
        cfg = replace(self.config, scenario=scenario)
        recipe = None
        self.active = {"selected": bool(selected), "partition": self.partition}
        if selected:
            parent = self.parents[int(self.rng.integers(len(self.parents)))]
            # Keep exact failures as well as nearby targets. Initialization is
            # rerun through the physical grasp procedure; no qpos is replayed.
            delta = self.rng.uniform(-self.corpus["goal_jitter_rad"], self.corpus["goal_jitter_rad"], 5)
            if self.rng.random() < 0.25:
                delta[:] = 0
            seed = parent["seed"]
            recipe = {"lift_goal_delta": delta.tolist()}
            cfg = replace(
                cfg,
                grasp_lift_probability=1,
                carry_hold_probability=0,
                carry_push_probability=0,
                grasp_lift_phase_weights=self.corpus["config"]["grasp_lift_phase_weights"],
            )
            self.active.update(parent_seed=seed, original_failure=parent["failure"], goal_delta_rad=delta.tolist())
        else:
            seed = int(self.rng.integers(1_000_000_000, 2_000_000_000))
        self.env.unwrapped.config = cfg
        observation, info = self.env.reset(seed=seed, options=recipe)
        return observation, {**info, "failure_replay": self.active.copy()}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, {**info, "failure_replay": self.active.copy()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--teacher-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-jitter-rad", type=float, default=0.01)
    args = parser.parse_args()
    corpus = mine_failures(
        json.loads(args.development_report.read_text()),
        json.loads(args.teacher_report.read_text()),
        goal_jitter_rad=args.goal_jitter_rad,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(corpus, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
