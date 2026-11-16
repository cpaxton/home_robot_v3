"""Matched frozen-fixture hold diagnostics, including all six base velocities.

No parameter tuning or threshold changes. Baselines use the same compiled plant
and initial state as the actor. Raw traces retain every deadline/guard failure.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from emet.eval.mars_policy_audit import sources, stationary_trial
from emet.eval.mars_tamp_control import CANONICAL_SUITE
from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.profiles import MARS


def quiet_metrics(trace, *, retention_context, dt=MARS.dt, position_tolerance_m=0.005, orientation_tolerance_rad=0.05):
    if (
        not np.isfinite([dt, position_tolerance_m, orientation_tolerance_rad]).all()
        or min(dt, position_tolerance_m, orientation_tolerance_rad) <= 0
    ):
        raise ValueError("quiet metrics require finite positive time and pose bounds")
    fields, _ = history_layout(MARS, retention_context=retention_context)
    consecutive = longest = quiet = 0
    base, arm = [], []
    for row in trace:
        if row.get("failure") or row.get("action_source") == "fault_stop":
            consecutive = 0
            continue
        observation = row.get("actor_observation")
        if observation is None:
            raise ValueError("raw pre-step observations required for quiet metrics")
        h = np.asarray(observation["history"], dtype=float)[-1]
        if not np.isfinite(h).all():
            raise ValueError("nonfinite hold measurement")
        velocity, joints = h[fields["base_velocity"]], h[fields["proprio/velocities"]]
        base.append(velocity)
        arm.append(joints)
        hand, target = h[fields["hand_pose"]], h[fields["hand_goal"]]
        a, b = hand[3:].reshape(3, 2), target[3:].reshape(3, 2)
        ra = np.column_stack([a, np.cross(a[:, 0], a[:, 1])])
        rb = np.column_stack([b, np.cross(b[:, 0], b[:, 1])])
        angle = np.arccos(np.clip((np.trace(rb.T @ ra) - 1) / 2, -1, 1))
        ok = (
            np.linalg.norm(velocity[:3]) < 0.005
            and np.linalg.norm(velocity[3:]) < 0.02
            and np.max(abs(joints)) < 0.03
            and np.linalg.norm(hand[:3] - target[:3]) <= position_tolerance_m
            and angle <= orientation_tolerance_rad
        )
        quiet += bool(ok)
        consecutive = consecutive + 1 if ok else 0
        longest = max(longest, consecutive)
    v = np.asarray(base)
    return {
        "samples": len(base),
        "position_tolerance_m": position_tolerance_m,
        "orientation_tolerance_rad": orientation_tolerance_rad,
        "quiet_pose_fraction": quiet / max(1, len(base)),
        "longest_quiet_pose_dwell_s": longest * dt,
        "two_second_quiet_pose_hold": longest * dt >= 2 and not any(t.get("failure") for t in trace),
        "base_velocity_rms": np.sqrt(np.mean(v**2, axis=0)).tolist() if len(v) else None,
        "base_velocity_abs_max": np.max(abs(v), axis=0).tolist() if len(v) else None,
        "angular_stop_violation_fraction": float(np.mean(np.linalg.norm(v[:, 3:], axis=1) >= 0.02)) if len(v) else None,
        "linear_stop_violation_fraction": float(np.mean(np.linalg.norm(v[:, :3], axis=1) >= 0.005)) if len(v) else None,
        "arm_velocity_rms": np.sqrt(np.mean(np.asarray(arm) ** 2, axis=0)).tolist() if len(arm) else None,
        "scope": "Raw pre-step measurements; unchanged arm/all-six-base quiet and hand-pose bounds",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["left:20g:1", "right:20g:2"])
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--position-tolerance", type=float, help="Defaults to the checkpoint's hand contract")
    parser.add_argument("--orientation-tolerance", type=float, help="Defaults to the checkpoint's hand contract")
    parser.add_argument(
        "--selectors",
        nargs="+",
        choices=["zero", "explicit_hold", "reference", "checkpoint_policy"],
        default=["zero", "explicit_hold", "reference", "checkpoint_policy"],
    )
    parser.add_argument("--trials", nargs="+", choices=["hold", "nearby", "rotation"], default=["hold"])
    args = parser.parse_args()
    if not np.isfinite(args.duration) or not 0 < args.duration <= 60:
        parser.error("duration must be in (0, 60]")
    args.output.mkdir(parents=True, exist_ok=False)
    catalog = {c["id"]: c for c in json.loads(CANONICAL_SUITE.read_text())["cases"]}
    config = json.loads((args.policy / "manifest.json").read_text())["environment"]
    position_bound = (
        config.get("hand_position_tolerance", 0.005) if args.position_tolerance is None else args.position_tolerance
    )
    angle_bound = (
        config.get("hand_orientation_tolerance", 0.05)
        if args.orientation_tolerance is None
        else args.orientation_tolerance
    )
    report = {"sources": sources(), "rows": [], "complete": False, "safety_promoted": False}
    for case in args.cases:
        for trial in args.trials:
            delta = {"hold": None, "nearby": [0, -0.05, 0, 0, 0], "rotation": [0, 0, 0, 0, 0.12]}[trial]
            for selector in args.selectors:
                if selector == "explicit_hold" and trial != "hold":
                    continue
                row = stationary_trial(
                    catalog[case],
                    args.policy,
                    args.evidence,
                    selector=selector,
                    duration_s=args.duration,
                    joint_delta=delta,
                )
                row.update(case_id=case, trial=trial)
                row["quiet_metrics"] = quiet_metrics(
                    row.get("trace", []),
                    retention_context=config.get("retention_context", False),
                    position_tolerance_m=position_bound,
                    orientation_tolerance_rad=angle_bound,
                )
                report["rows"].append(row)
                print(
                    case,
                    trial,
                    selector,
                    row.get("failure"),
                    row.get("hand_error_at_end_m"),
                    row.get("orientation_error_at_end_rad"),
                    row["quiet_metrics"],
                    flush=True,
                )
    report["complete"] = True
    report["driver_source"] = Path(__file__).read_text()
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    archive = args.output / "report.json.gz"
    archive.write_bytes(gzip.compress(raw, mtime=0))
    summary = {
        **report,
        "rows": [
            {k: v for k, v in r.items() if k not in {"trace", "warmup_trace", "initial_state"}} for r in report["rows"]
        ],
    }
    summary.pop("driver_source")
    summary.update(full_report_archive=str(archive), uncompressed_sha256=hashlib.sha256(raw).hexdigest())
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
