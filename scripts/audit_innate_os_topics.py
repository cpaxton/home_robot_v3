#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Compare live ROS 2 graph topics/TF frames to innate_mars_bridge expectations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_BRIDGE_SRC = _REPO / "src" / "innate_mars_bridge"


def _import_constants():
    sys.path.insert(0, str(_BRIDGE_SRC))
    from innate_mars_bridge.constants import (  # noqa: PLC0415
        EXPECTED_TF_FRAMES,
        EXPECTED_TOPICS,
        INNATE_OS_GIT_REF,
        INNATE_OS_REPO,
    )

    return EXPECTED_TOPICS, EXPECTED_TF_FRAMES, INNATE_OS_GIT_REF, INNATE_OS_REPO


def _ros2_lines(cmd: list[str], timeout: float = 15.0) -> list[str]:
    try:
        out = subprocess.check_output(cmd, text=True, timeout=timeout, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Failed: {' '.join(cmd)}\n{e}", file=sys.stderr)
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-ros",
        action="store_true",
        help="Print expected constants only (no ros2 CLI)",
    )
    args = parser.parse_args()

    expected_topics, expected_frames, git_ref, repo = _import_constants()
    print(f"Innate-os reference: {repo} @ {git_ref}\n")

    print("Expected topics:")
    for t in expected_topics:
        print(f"  {t}")

    print("\nExpected TF frames (lookup may fail if stack not running):")
    for f in expected_frames:
        print(f"  {f}")

    if args.skip_ros:
        return 0

    print("\n--- Live ROS graph ---")
    topics = _ros2_lines(["ros2", "topic", "list"])
    if not topics:
        print("No topics (is ROS sourced and innate-os / Mars stack running?)")
        return 1

    missing_topics = [t for t in expected_topics if t not in topics]
    extra_mars = sorted(t for t in topics if t.startswith("/mars/"))

    print(f"\nTopics on graph: {len(topics)}")
    if missing_topics:
        print("MISSING expected topics:")
        for t in missing_topics:
            print(f"  - {t}")
    else:
        print("All expected /mars/* and /odom topics present.")

    if extra_mars:
        print("\nOther /mars/* topics:")
        for t in extra_mars:
            print(f"  {t}")

    frames = _ros2_lines(["ros2", "run", "tf2_ros", "tf2_echo", "--help"])
    del frames  # tf2_echo --help just checks tf2_ros package exists

    frame_ok = 0
    frame_fail = 0
    for fr in expected_frames:
        lines = _ros2_lines(
            ["ros2", "run", "tf2_ros", "tf2_echo", "odom", fr, "--flow-style"],
            timeout=3.0,
        )
        if lines and "Failure" not in lines[0] and "Exception" not in lines[0]:
            frame_ok += 1
        else:
            frame_fail += 1
            print(f"TF lookup odom -> {fr}: not available (stack may be idle)")

    print(f"\nTF spot-check: {frame_ok} ok, {frame_fail} unavailable (normal when sim/robot idle)")

    return 0 if not missing_topics else 2


if __name__ == "__main__":
    raise SystemExit(main())
