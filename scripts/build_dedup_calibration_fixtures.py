#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
"""Build deterministic noisy calibration JSONL fixtures for offline graph dedup tests.

Outputs (under ``src/test/fixtures/`` by default):
  - ``calibration_frames_stationary_noisy.jsonl`` — repeated views of 2 objects with jitter
  - ``calibration_frames_long_explore_noisy.jsonl`` — long revisit loop over 8 objects
  - ``gt_long_explore_noisy.json`` — GT for the long-explore fixture

Regenerate richer data from sim (optional, needs GPU/sim):
  ``./scripts/run_fusion_calibration_loop.sh innate_mars``
  ``emet run dynagraph --calibration-export …`` + ``scripts/fetch_sim_gt_from_server.py``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "src" / "test" / "fixtures"

STATIONARY_OBJECTS = [
    {
        "id": "apple_main",
        "label": "apple",
        "pos_world": [1.0, -0.5, 0.9],
        "bounds_3d": {
            "min": [0.95, -0.55, 0.85],
            "max": [1.05, -0.45, 0.95],
            "center": [1.0, -0.5, 0.9],
            "size": [0.1, 0.1, 0.1],
        },
    },
    {
        "id": "mug_main",
        "label": "mug",
        "pos_world": [1.2, 0.1, 0.88],
        "bounds_3d": {
            "min": [1.15, 0.05, 0.83],
            "max": [1.25, 0.15, 0.93],
            "center": [1.2, 0.1, 0.88],
            "size": [0.1, 0.1, 0.1],
        },
    },
]

MUG_LABEL_ALIASES = ("mug", "coffee cup", "cup", "mug")

LONG_EXPLORE_OBJECTS = [
    {"id": "obj_a", "label": "apple", "pos_world": [1.0, -0.5, 0.9]},
    {"id": "obj_b", "label": "mug", "pos_world": [1.2, 0.1, 0.88]},
    {"id": "obj_c", "label": "bowl", "pos_world": [2.1, -0.3, 0.85]},
    {"id": "obj_d", "label": "bottle", "pos_world": [2.0, 1.0, 0.92]},
    {"id": "obj_e", "label": "plate", "pos_world": [0.5, 1.2, 0.87]},
    {"id": "obj_f", "label": "pan", "pos_world": [3.0, 0.2, 0.86]},
    {"id": "obj_g", "label": "kettle", "pos_world": [3.2, 1.1, 0.91]},
    {"id": "obj_h", "label": "toaster", "pos_world": [0.8, 2.0, 0.89]},
]


def _bounds_from_center(center: list[float], size: float = 0.1) -> dict[str, list[float]]:
    half = size / 2.0
    c = np.asarray(center, dtype=np.float64)
    mn = (c - half).tolist()
    mx = (c + half).tolist()
    return {"min": mn, "max": mx, "center": c.tolist(), "size": [size, size, size]}


def _jitter_xyz(rng: np.random.Generator, pos: list[float], sigma: float) -> list[float]:
    out = np.asarray(pos, dtype=np.float64) + rng.normal(0.0, sigma, size=3)
    return out.tolist()


def _jitter_bounds(rng: np.random.Generator, bounds: dict[str, list[float]], sigma: float) -> dict[str, list[float]]:
    center = _jitter_xyz(rng, bounds["center"], sigma)
    return _bounds_from_center(center)


def build_stationary_noisy(*, seed: int = 0, n_steps: int = 25) -> list[dict]:
    rng = np.random.default_rng(seed)
    frames: list[dict] = []
    for step in range(1, n_steps + 1):
        detections = []
        for i, obj in enumerate(STATIONARY_OBJECTS):
            label = obj["label"]
            if obj["id"] == "mug_main":
                label = MUG_LABEL_ALIASES[step % len(MUG_LABEL_ALIASES)]
            detections.append(
                {
                    "label": label,
                    "xyz": _jitter_xyz(rng, obj["pos_world"], sigma=0.08),
                    "bounds_3d": _jitter_bounds(rng, obj["bounds_3d"], sigma=0.06),
                    "bbox_xyxy": [300 + i * 10, 200, 340 + i * 10, 240],
                }
            )
        frames.append({"step": step, "detections": detections})
    return frames


def build_long_explore_noisy(*, seed: int = 1, n_steps: int = 48) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    gt_objects = []
    for obj in LONG_EXPLORE_OBJECTS:
        pos = obj["pos_world"]
        gt_objects.append(
            {
                "id": obj["id"],
                "label": obj["label"],
                "label_norm": obj["label"],
                "pos_world": pos,
                "bounds_3d": _bounds_from_center(pos),
            }
        )

    frames: list[dict] = []
    for step in range(1, n_steps + 1):
        obj_idx = (step - 1) % len(LONG_EXPLORE_OBJECTS)
        obj = LONG_EXPLORE_OBJECTS[obj_idx]
        revisit = (step - 1) // len(LONG_EXPLORE_OBJECTS)
        drift_sigma = 0.04 + 0.02 * min(revisit, 4)
        pos = _jitter_xyz(rng, obj["pos_world"], sigma=drift_sigma)
        detections = [
            {
                "label": obj["label"],
                "xyz": pos,
                "bounds_3d": _bounds_from_center(pos),
                "bbox_xyxy": [280 + obj_idx * 8, 190 + revisit * 2, 320 + obj_idx * 8, 230],
            }
        ]
        # Occasional second object in frame (different room corner).
        if step % 5 == 0:
            other_idx = (obj_idx + 3) % len(LONG_EXPLORE_OBJECTS)
            other = LONG_EXPLORE_OBJECTS[other_idx]
            detections.append(
                {
                    "label": other["label"],
                    "xyz": _jitter_xyz(rng, other["pos_world"], sigma=0.05),
                    "bounds_3d": _bounds_from_center(other["pos_world"]),
                }
            )
        frames.append({"step": step, "detections": detections})

    gt = {
        "schema_version": 1,
        "source": "build_dedup_calibration_fixtures",
        "robot": "innate_mars",
        "objects": gt_objects,
    }
    return frames, gt


def write_jsonl(path: Path, frames: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for fr in frames:
            fh.write(json.dumps(fr, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIXTURES,
        help="Directory for generated fixtures",
    )
    args = parser.parse_args()
    out = args.out_dir

    stationary = build_stationary_noisy()
    write_jsonl(out / "calibration_frames_stationary_noisy.jsonl", stationary)

    long_frames, long_gt = build_long_explore_noisy()
    write_jsonl(out / "calibration_frames_long_explore_noisy.jsonl", long_frames)
    (out / "gt_long_explore_noisy.json").write_text(
        json.dumps(long_gt, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {out / 'calibration_frames_stationary_noisy.jsonl'} ({len(stationary)} steps)")
    print(f"Wrote {out / 'calibration_frames_long_explore_noisy.jsonl'} ({len(long_frames)} steps)")
    print(f"Wrote {out / 'gt_long_explore_noisy.json'} ({len(long_gt['objects'])} objects)")


if __name__ == "__main__":
    main()
