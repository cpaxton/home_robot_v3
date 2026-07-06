#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Rebuild ``episode_rgb.mp4`` from an existing eval bundle with motion-paced playback."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from emet.eval.episode_video import write_episode_mp4_from_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode_dir",
        type=Path,
        help="Episode bundle directory (contains metadata.jsonl and frames/ or images/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path (default: <episode_dir>/episode_rgb.mp4)",
    )
    parser.add_argument("--fps", type=float, default=6.0, help="Playback FPS")
    parser.add_argument(
        "--no-motion-paced",
        action="store_true",
        help="One frame per metadata row (no motion-weighted repeats)",
    )
    parser.add_argument(
        "--meters-per-frame",
        type=float,
        default=0.25,
        help="Translation per repeated frame when motion-paced",
    )
    parser.add_argument(
        "--radians-per-frame",
        type=float,
        default=math.radians(10.0),
        help="Rotation per repeated frame when motion-paced",
    )
    parser.add_argument(
        "--crossfade-teleport-m",
        type=float,
        default=1.5,
        help="Crossfade when consecutive poses jump at least this far (meters)",
    )
    args = parser.parse_args()

    episode_dir = args.episode_dir.expanduser().resolve()
    out = args.output or (episode_dir / "episode_rgb.mp4")
    path = write_episode_mp4_from_metadata(
        episode_dir,
        fps=args.fps,
        out_path=out,
        motion_paced=not args.no_motion_paced,
        meters_per_repeat=args.meters_per_frame,
        radians_per_repeat=args.radians_per_frame,
        crossfade_teleport_m=args.crossfade_teleport_m,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
