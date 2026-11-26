# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Fit static WBC feedforward from recorded tracking failures.

This is the data-side half of the Mars bench loop. It does not claim a hardware
identification: it extracts a robust per-joint steady-state bias from the
existing manipulation-failure sink. Convert that bias to torque with the
profile's identified ``kp_eff`` and pass it to ``PositionServoAdapter`` as
``gravity_torques``.

Example:
    python -m emet.rl.manipulation.wbc_calibration \
      --records ~/runs/emet/manip_failures/manip_failures.jsonl \
      --embodiment innate_mars --output runs/wbc/feedforward.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emet.controller.manipulation.failure_dataset import load_manipulation_attempts
from emet.controller.manipulation.wbc_feedforward import FeedforwardModel, fit_feedforward


def fit_records(
    records: str | Path,
    *,
    embodiment: str,
    joint_names: tuple[str, ...] = (),
    window: int = 5,
    min_samples: int = 2,
    max_abs_bias: float = 0.75,
    exclude_feedforward: bool = True,
) -> FeedforwardModel:
    """Load a failure sink and fit its static joint bias."""
    attempts = load_manipulation_attempts(records)
    return fit_feedforward(
        attempts,
        embodiment=embodiment,
        joint_names=joint_names,
        window=window,
        min_samples=min_samples,
        max_abs_bias=max_abs_bias,
        exclude_feedforward=exclude_feedforward,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit static whole-body feedforward from tracking failures")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embodiment", default="innate_mars")
    parser.add_argument("--joint", action="append", default=[])
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--max-abs-bias", type=float, default=0.75)
    args = parser.parse_args(argv)
    model = fit_records(
        args.records,
        embodiment=args.embodiment,
        joint_names=tuple(args.joint),
        window=args.window,
        min_samples=args.min_samples,
        max_abs_bias=args.max_abs_bias,
    )
    output = model.save(args.output)
    print(
        f"WBC feedforward: {output} joints={len(model.joint_bias)} samples={model.samples}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["fit_records", "main"]
