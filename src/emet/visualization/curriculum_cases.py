"""Deterministic visual curriculum samples, never an optimizer case source."""

import copy
import json
from pathlib import Path

from emet.rl.manipulation.arm_splines import load_manifest
from emet.rl.manipulation.goal_change_tracking import load_curriculum

ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ("hand", "base", "goal_changes", "arm_splines_l0", "arm_splines_l1", "arm_splines_l2", "tamp", "dense_arm")
TITLES = {
    "hand": "General hand + base tracking",
    "base": "Base position + heading",
    "goal_changes": "Mid-motion goal changes",
    "arm_splines_l0": "Random arm splines | small",
    "arm_splines_l1": "Random arm splines | medium",
    "arm_splines_l2": "Random arm splines | full",
    "tamp": "TAMP unloaded approach / grasp pose",
    "dense_arm": "Larger arm motions | bounded v2 candidate",
}


def cases_for(family):
    if family not in FAMILIES:
        raise ValueError("unknown curriculum family")
    general = json.loads((ROOT / "paper/data/whole_body_control/long_finetune_v1/cases.json").read_text())
    if family in {"hand", "base"}:
        cases = [c for c in general if (c["mode"] == "base_only") == (family == "base")]
        if family == "base":
            # The frozen pool has only eight identities. Show eight additional
            # nearby examples explicitly as visualization-only, not as PPO data.
            for index, original in enumerate(list(cases)):
                case = copy.deepcopy(original)
                case.update(
                    id=f"innate_mars:visualization-base-v1:{index}",
                    seed=480_000_000 + index,
                    split="visualization",
                    travel_m=0.16 if case["travel_m"] >= 0 else -0.16,
                    turn_rad=0.25 if index % 2 == 0 else -0.25,
                )
                cases.append(case)
    elif family == "goal_changes":
        cases = load_curriculum(ROOT / "configs/benchmarks/mars_goal_changes_v1.json")["robots"]["innate_mars"]
    elif family.startswith("arm_splines_") or family == "dense_arm":
        level = 0 if family == "dense_arm" else int(family[-1])
        name = "mars_arm_splines_v2.json" if family == "dense_arm" else "mars_arm_splines_v1.json"
        pool = load_manifest(ROOT / "configs/benchmarks" / name)["cases"]
        cases = [c for c in pool if c["level"] == level and c["split"] == "train"][:8]
        cases += [c for c in pool if c["level"] == level and c["split"] == "audit"][:8]
    else:
        spec = json.loads((ROOT / "configs/benchmarks/mars_whole_body_v2_development.json").read_text())
        cases = [dict(c, split="train") for c in spec["train_cases"]]
        cases += [dict(c, split="audit") for c in spec["audit_cases"]]
        for index in range(10):
            case = copy.deepcopy(cases[index % 6])
            case.update(id=f"visualization:tamp-v1:{index}", seed=480_001_000 + index, split="visualization")
            cases.append(case)
    if len(cases) != 16 or len({(c["id"], c["seed"]) for c in cases}) != 16:
        raise ValueError("sixteen distinct case/seed identities required")
    return copy.deepcopy(cases)
