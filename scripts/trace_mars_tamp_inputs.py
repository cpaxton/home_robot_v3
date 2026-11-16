# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Four fixed native diagnostics: recording parity, actor gap, planner failure.

Raw sensor/physics traces belong in ignored logs, never version control. This
does not train, promote a controller or qualify complete pick/place execution.
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.manipulation.mastery import read_report
from emet.rl.manipulation.policy import load_control_policy
from emet.rl.manipulation.whole_body_observation import pose_rotation
from emet.rl.manipulation.whole_body_training import load_development, primitive_rollout


def summarize_trace(trace):
    """Keep phase-local tracking and command statistics, without raw histories."""
    phases = {}
    for phase in dict.fromkeys(t["phase"] for t in trace["ticks"]):
        ticks = [t for t in trace["ticks"] if t["phase"] == phase]
        position = [float(np.linalg.norm(np.asarray(t["hand"])[:3] - t["hand_goal"][:3])) for t in ticks]
        orientation = [
            float(
                np.linalg.norm(
                    Rotation.from_matrix(pose_rotation(t["hand_goal"]) @ pose_rotation(t["hand"]).T).as_rotvec()
                )
            )
            for t in ticks
        ]
        phases[phase] = {
            "ticks": len(ticks),
            "first_time_s": ticks[0]["time"],
            "last_time_s": ticks[-1]["time"],
            "hand_error_m": {"first": position[0], "last": position[-1], "peak": max(position)},
            "hand_error_rad": {"first": orientation[0], "last": orientation[-1], "peak": max(orientation)},
            "action_sources": dict(Counter(t["action_source"] for t in ticks)),
            "peak_abs_action": np.max(np.abs([t["normalized_action"] for t in ticks]), axis=0).tolist(),
            "last_joint_command": ticks[-1]["commanded_joint_velocity"],
            "last_joint_measured": ticks[-1]["joint_velocity"],
        }
    decisions = trace["decisions"]
    return {
        "outcome": {k: trace["outcome"].get(k) for k in ("status", "code", "error", "executor_result")},
        "stop_confirmed": trace["stop_confirmed"],
        "commands": trace["commands"],
        "phases": phases,
        "decisions": len(decisions),
        "observation_shapes": {k: list(np.asarray(v).shape) for k, v in decisions[0]["observation"].items()}
        if decisions
        else {},
        "observations_finite": all(np.isfinite(v).all() for d in decisions for v in d["observation"].values()),
        "peak_abs_reference_twist": np.max(np.abs([d["observation"]["motion"] for d in decisions]), axis=0).tolist()
        if decisions
        else [],
        "decision_times_increase": all(b["time"] > a["time"] for a, b in zip(decisions, decisions[1:], strict=False)),
    }


def parity_signature(row):
    """Recording must preserve all physical outcomes; exclude wall-clock timing."""
    fields = (
        "success",
        "code",
        "stop_confirmed",
        "control_ticks",
        "decisions",
        "returns",
        "commands",
        "inference_counts",
        "object_translation_m",
        "object_rotation_rad",
        "initial_state_sha256",
        "safety_flags",
    )
    return hashlib.sha256(json.dumps({k: row.get(k) for k in fields}, sort_keys=True).encode()).hexdigest()


def main():
    from validate_mars_current_feedback import LEGACY, PrimitiveBundle, sha, write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    suite, _ = load_development()
    cases = {c["id"]: c for c in suite["train_cases"] + suite["audit_cases"]}
    normal = cases["development:right:50g:1"]
    planning = cases["development:left:50g:2"]
    bundle = load_control_policy(LEGACY)
    if bundle.manifest["weights_sha256"] != "e7b425699510cc635485bdd8f59a5b25692fb165faf248aaecbc2707c0dbb883":
        raise ValueError("frozen retained actor changed")
    policy = PrimitiveBundle(bundle)
    sources = source_hashes()
    summary = {
        "format": "mars-tamp-input-diagnostic-v1",
        "producer_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha(__file__),
        "source_hashes": sources,
        "weights_sha256": bundle.manifest["weights_sha256"],
        "new_ppo_steps": 0,
        "hardware_used": False,
        "qualified": False,
        "groups": {},
        "complete": False,
    }
    try:
        for name, case, actor, record in (
            ("teacher_unrecorded", normal, None, False),
            ("teacher_recorded", normal, None, True),
            ("legacy_recorded", normal, policy, True),
            ("planner_recorded", planning, None, True),
        ):
            directory = args.output / name if record else None
            (row,) = primitive_rollout([case], policy=actor, wheel_feedback_current_velocity=True, trace_dir=directory)
            result = {"row": row, "parity_signature": parity_signature(row)}
            if record:
                archive = directory / "episode-0001.json.gz"
                result.update(trace_sha256=sha(archive), trace=summarize_trace(read_report(archive)))
            summary["groups"][name] = result
            write(args.output / "progress.json", summary)
        summary["recording_preserves_teacher_outcome"] = (
            summary["groups"]["teacher_unrecorded"]["parity_signature"]
            == summary["groups"]["teacher_recorded"]["parity_signature"]
        )
        if not summary["recording_preserves_teacher_outcome"]:
            raise ValueError("recording changed the teacher's physical outcome")
        if sources != source_hashes():
            raise ValueError("runtime changed during diagnostic")
        summary["complete"] = True
    except BaseException as exc:
        summary["error"] = repr(exc)
        raise
    finally:
        write(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
