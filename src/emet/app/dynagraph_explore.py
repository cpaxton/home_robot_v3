# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Heuristic scripted frontier batches for Dynagraph CLI (no sim coupling)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

ExplorationTermination = Literal[
    "max_iterations",
    "consecutive_failures",
    "timeout",
]


def dynagraph_explore_until_terminated(
    agent: Any,
    *,
    max_iterations: int,
    max_consecutive_failures: int = 3,
    timeout_s: float | None = None,
    log_fn: Callable[[str], Any] | None = None,
) -> tuple[ExplorationTermination, int, int]:
    """
    Call ``agent.run_exploration()`` up to ``max_iterations`` times or until stalled.

    One iteration runs one frontier-based navigation excursion (same as interactive ``explore``).

    Args:
        agent: Dynagraph / Dynamem-like controller exposing ``run_exploration() -> bool``.
        max_iterations: Hard cap on loop iterations (> 0).
        max_consecutive_failures: Stop after this many contiguous ``run_exploration`` failures.
        timeout_s: Optional wall-clock limit (seconds).
        log_fn: Optional callback for human-readable CLI progress).

    Returns:
        Tuple of ``(termination_reason, n_successful, n_iterations_executed)``.
    """
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    def _say(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    streak = 0
    n_ok = 0
    t0 = time.monotonic()
    for i in range(max_iterations):
        if timeout_s is not None and timeout_s >= 0.0 and time.monotonic() - t0 > timeout_s:
            _say(f"Dynagraph explore-loop: stopping on timeout ({timeout_s}s)")
            return "timeout", n_ok, i

        ok = bool(agent.run_exploration())
        if ok:
            n_ok += 1
            streak = 0
            _say(f"Dynagraph explore-loop: step {i + 1}/{max_iterations} ok")
        else:
            streak += 1
            _say(
                f"Dynagraph explore-loop: step {i + 1}/{max_iterations} failed "
                f"({streak}/{max_consecutive_failures} streak)"
            )
            if streak >= max_consecutive_failures:
                _say("Dynagraph explore-loop: stopping after consecutive failures (no frontier / planner blocked).")
                return "consecutive_failures", n_ok, i + 1

    _say(f"Dynagraph explore-loop: reached max_iterations ({max_iterations})")
    return "max_iterations", n_ok, max_iterations
