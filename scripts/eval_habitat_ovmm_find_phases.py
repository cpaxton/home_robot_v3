#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Launch Habitat OVMM find-phase batch eval via ``.venv-habitat/bin/emet-habitat``."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = REPO / "configs" / "ovmm" / "habitat_find_phase_episodes.yaml"
HABITAT_BIN = REPO / ".venv-habitat" / "bin" / "emet-habitat"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Habitat find-phase batch (Phase 2).")
    parser.add_argument("--episodes", default=str(DEFAULT_EPISODES))
    parser.add_argument("--episode-id", action="append", dest="episode_ids")
    parser.add_argument(
        "--backend",
        choices=("dynamem", "graph_eqa", "dynagraph", "ground_truth"),
        default="dynagraph",
    )
    parser.add_argument("--merge-xy-m", type=float, default=None)
    parser.add_argument("--staleness-horizon", type=int, default=None)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--not-rotate", action="store_true")
    parser.add_argument("--output-dir", default=str(REPO / "runs" / "ovmm_habitat"))
    return parser.parse_args()


def main() -> int:
    if not HABITAT_BIN.is_file():
        print(
            "Missing .venv-habitat — run ./scripts/install_habitat.sh and "
            "uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d train",
            file=sys.stderr,
        )
        return 1

    args = _parse_args()
    cmd = [
        str(HABITAT_BIN),
        "run-ovmm-find-batch",
        "--episodes",
        args.episodes,
        "--backend",
        args.backend,
        "--output-dir",
        args.output_dir,
    ]
    if args.merge_xy_m is not None:
        cmd.extend(["--merge-xy-m", str(args.merge_xy_m)])
    if args.staleness_horizon is not None:
        cmd.extend(["--staleness-horizon", str(args.staleness_horizon)])
    if args.cpu_only:
        cmd.append("--cpu-only")
    if args.not_rotate:
        cmd.append("--not-rotate")
    for eid in args.episode_ids or []:
        cmd.extend(["--episode-id", eid])

    print("Running:", " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd, cwd=REPO)


if __name__ == "__main__":
    raise SystemExit(main())
