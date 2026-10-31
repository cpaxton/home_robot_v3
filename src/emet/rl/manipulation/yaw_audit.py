"""Sample nearby signed turns and expose the exact deployed yaw feedback terms.

Development diagnostics only: existing training postures, unchanged native
physics/limits and teacher/actor matched resets. This never replaces canonical
mastery or held-out TAMP qualification, and never repairs an actor's action.
"""

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.mastery import ROOT, evaluate_suite, load_suite, save_report, summarize
from emet.rl.manipulation.observation_layout import history_layout, preview_layout
from emet.rl.manipulation.policy import ManipulationPolicy
from emet.rl.manipulation.profiles import PROFILES
from emet.rl.manipulation.turn_guard import wrapped_yaw


def sample_suite(parent, magnitudes=(0.02, 0.04, 0.06), *, guard=False):
    values = np.asarray(magnitudes, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values <= 0)
        or np.any(values > 0.08)
        or len(set(values)) != len(values)
    ):
        raise ValueError("unique positive nearby yaw magnitudes up to 0.08 rad required")
    suite = deepcopy(parent)
    suite.update(format="mars-yaw-sign-audit-v1", parent_suite_sha256=digest(parent), scope=__doc__)
    suite.pop("training_population", None)
    suite["config"]["base_turn_progress_guard"] = guard
    postures = [c for c in parent["cases"] if c["stage"] == "base" and c["split"] == "train"]
    suite["cases"] = [
        {
            **posture,
            "id": f"train:base:yaw_{direction}:{magnitude:.9g}@posture:{posture['id'].rsplit(':', 1)[-1]}",
            "turn_rad": float(sign * magnitude),
            "travel_m": 0.0,
            "arm_delta": [0.0] * 5,
        }
        for posture in postures
        for magnitude in values
        for direction, sign in [("left", 1), ("right", -1)]
    ]
    return suite


def feedback_terms(policy, observation):
    """Return the unmodified action and its exact pre-tanh yaw contributions."""
    network = policy.model.policy
    head = network.action_net
    if not hasattr(head, "gain") or policy.robot != "innate_mars":
        raise ValueError("Mars gain-feedback policy required for yaw decomposition")
    normalized = policy.normalized_observation(observation)
    with torch.no_grad():
        tensors, _ = network.obs_to_tensor(normalized)
        features = network.extract_features(tensors)
        latent = network.mlp_extractor.forward_actor(features)
        context = latent[:, : getattr(head, "context_width", latent.shape[1] - head.feedback_width)]
        signal = latent[:, -head.feedback_width :]
        gain = head.gain(context).reshape(-1, head.motion_size, head.feedback_width)
        terms = (gain[0, 1] * signal[0] / math.sqrt(head.feedback_width)).cpu().numpy()
        action = head(latent)[0].cpu().numpy()
    np.testing.assert_allclose(action, policy(observation), atol=1e-7, rtol=1e-6)
    p, _ = preview_layout()
    h, _ = history_layout(
        PROFILES[policy.robot], retention_context=policy.manifest["environment"].get("retention_context", False)
    )
    pose = observation["history"][-1, h["base_pose"]]
    target = observation["history"][-1, h["base_goal"]]
    error = wrapped_yaw(float(target[2]) - float(pose[2]))
    observed = float(observation["preview"][1, p["base_error"]][2])
    if abs(wrapped_yaw(error - observed)) > 2e-6:
        raise ValueError("world/preview yaw error disagrees")
    row = {
        "world_yaw_error_rad": error,
        "preview_yaw_error_rad": observed,
        "normalized_yaw_error": float(normalized["preview"][1, p["base_error"]][2]),
        "yaw_action": float(action[1]),
        "yaw_terms": terms.tolist(),
        "yaw_gain_current": float(gain[0, 1, 2]),
        "yaw_gain_preview": float(gain[0, 1, 14]),
        "current_yaw_term": float(terms[2]),
        "preview_yaw_term": float(terms[14]),
        "other_terms": float(terms.sum() - terms[2] - terms[14]),
    }
    if hasattr(head, "turn_gain"):
        with torch.no_grad():
            gains = torch.nn.functional.softplus(head.turn_gain(context))[0].cpu().numpy()
        row.update(
            yaw_terms_scope="general_branch_pre_tanh",
            restoring_gate=float(latent[0, head.context_width]),
            restoring_gains=gains.tolist(),
            restoring_terms=(
                gains
                * np.array([float(signal[0, 2]), float(signal[0, 14]), -float(signal[0, -1])])
                / math.sqrt(head.feedback_width)
            ).tolist(),
        )
    return action, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "configs/benchmarks/mars_tracking_mastery_crossed_v1.json")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[0.02, 0.04, 0.06])
    parser.add_argument("--guard", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    suite = sample_suite(load_suite(args.suite), args.magnitudes, guard=args.guard)
    args.output.mkdir(exist_ok=False, parents=True)
    (args.output / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    teacher = evaluate_suite(suite, stages=["base"])
    save_report(args.output / "teacher.json.gz", teacher)
    policy = ManipulationPolicy(args.policy)

    class Probe:
        robot, manifest = policy.robot, policy.manifest

        def __init__(self):
            self.rows = []

        def __call__(self, observation):
            action, row = feedback_terms(policy, observation)
            self.rows.append(row)
            return action

    probe = Probe()
    report = evaluate_suite(suite, stages=["base"], policy=probe, teacher_report=teacher)
    save_report(args.output / "policy.json.gz", report)
    cursor = 0
    for case in report["cases"]:
        count = len(case["trace"])
        case["feedback_terms"] = probe.rows[cursor : cursor + count]
        cursor += count
    assert cursor == len(probe.rows)
    save_report(args.output / "feedback.json.gz", report)
    summary = {
        "teacher": summarize(teacher),
        "policy": summarize(report),
        "guard": args.guard,
        "hardware_used": False,
        "safety_promoted": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
