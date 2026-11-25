#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Fit a per-joint feedforward from recorded manipulation failures.

This is the calibration tier of the failure-to-control loop. It reads the
execution sink written by ``failure_record.record_manip_attempt`` and emits a
small JSON model that can be loaded by the kinematic executor
(``EMET_WBC_FEEDFORWARD``) or converted to torques for the WBC adapter.

Examples:
  uv run python scripts/fit_wbc_feedforward.py --embodiment rby1
  uv run python scripts/fit_wbc_feedforward.py \
    --records ~/runs/emet/manip_failures/manip_failures.jsonl \
    --min-samples 3 --out /tmp/rby1_feedforward.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from emet.controller.manipulation.failure_dataset import load_manipulation_attempts, select_attempts
from emet.controller.manipulation.wbc_feedforward import FeedforwardModel, fit_feedforward


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default=os.path.expanduser("~/runs/emet/manip_failures"),
        help="JSONL sink or directory (default: ~/runs/emet/manip_failures)",
    )
    parser.add_argument("--embodiment", default="rby1", help="Robot id to calibrate (default: rby1)")
    parser.add_argument(
        "--out",
        default=os.path.expanduser("~/runs/emet/wbc_feedforward.json"),
        help="Output model path (default: ~/runs/emet/wbc_feedforward.json)",
    )
    parser.add_argument("--window", type=int, default=5, help="Trace samples used for steady state (default: 5)")
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum attempts per joint (default: 2)")
    parser.add_argument("--max-abs-bias", type=float, default=0.75, help="Reject implausible bias (default: 0.75)")
    parser.add_argument("--cluster-distance", type=float, default=0.42, help="Posture cluster radius (default: 0.42)")
    parser.add_argument("--outlier-bias-limit", type=float, default=1.2, help="Reject contaminated attempts (default: 1.2)")
    parser.add_argument(
        "--max-joint-distance",
        type=float,
        default=0.12,
        help="Apply a calibration only within this joint distance (default: 0.12)",
    )
    parser.add_argument(
        "--include-feedforward",
        action="store_true",
        help="Include records produced with a feedforward active (contaminated calibration)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the model without writing it")
    args = parser.parse_args()

    attempts = load_manipulation_attempts(args.records)
    selected = select_attempts(attempts, robot_id=args.embodiment)
    print(f"records={len(attempts)} selected={len(selected)} embodiment={args.embodiment!r}")
    if not selected:
        print("No tracking failures with planned waypoints; collect attempts before fitting.", file=sys.stderr)
        return 2

    try:
        model = fit_feedforward(
            selected,
            embodiment=args.embodiment,
            window=args.window,
            min_samples=args.min_samples,
            max_abs_bias=args.max_abs_bias,
            cluster_distance=args.cluster_distance,
            max_joint_distance=args.max_joint_distance,
            outlier_bias_limit=args.outlier_bias_limit,
            exclude_feedforward=not args.include_feedforward,
        )
    except ValueError as exc:
        print(f"Fit failed: {exc}", file=sys.stderr)
        return 1

    for index, calibration in enumerate(model.calibrations):
        print(f"calibration {index} samples={calibration.samples}")
        print(f"  {'joint':24s} {'reference':>10s} {'bias (rad/m)':>14s}")
        for name in sorted(calibration.joint_bias):
            print(f"  {name:24s} {calibration.reference_q.get(name, 0.0):>10.4f} {calibration.joint_bias[name]:+14.4f}")
    print(f"samples={model.samples} calibrations={len(model.calibrations)}")
    if args.dry_run:
        print("dry-run: not writing", args.out)
        return 0
    path = model.save(args.out)
    print(f"wrote {path}")
    print(f"apply with: EMET_WBC_FEEDFORWARD={path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
