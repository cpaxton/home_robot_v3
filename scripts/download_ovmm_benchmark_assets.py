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

"""Download / verify assets for OVMM find-phase benchmarks (home-dir caches)."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OVMM benchmark assets into ~/.cache and ~/runs.")
    parser.add_argument(
        "--benchmark",
        default="configs/ovmm/benchmark.yaml",
        help="Benchmark paths YAML (default: configs/ovmm/benchmark.yaml)",
    )
    parser.add_argument("--habitat-csv", action="store_true", default=True, help="Fetch HM-EQA CSVs")
    parser.add_argument("--no-habitat-csv", action="store_false", dest="habitat_csv")
    parser.add_argument(
        "--habitat-semantics",
        action="store_true",
        help="Download HM3D train semantic annots (~large; skip if scenes already OK)",
    )
    parser.add_argument(
        "--robocasa",
        action="store_true",
        help="Download Robocasa kitchen assets (S1 tier; large)",
    )
    parser.add_argument("--install-habitat", action="store_true", help="Run ./scripts/install_habitat.sh")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only")
    return parser.parse_args()


def _run(cmd: list[str], *, dry_run: bool) -> int:
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=REPO)


def main() -> int:
    from emet.eval.habitat_ovmm_find import load_habitat_ovmm_episodes
    from emet.eval.ovmm_benchmark_config import load_ovmm_benchmark_config
    from emet.habitat.config import hm3d_scene_glb_path, hm3d_scene_semantic_glb_path

    args = _parse_args()
    cfg = load_ovmm_benchmark_config(args.benchmark)
    rc = 0

    print("OVMM benchmark paths:")
    print(f"  sim output:    {cfg.paths.output_dir_sim}")
    print(f"  habitat output:{cfg.paths.output_dir_habitat}")
    print(f"  habitat CSV:   {cfg.paths.habitat_eqa_data}")
    print(f"  HM3D data:     {cfg.paths.hm3d_data}")

    if args.install_habitat:
        rc = max(rc, _run(["./scripts/install_habitat.sh"], dry_run=args.dry_run))

    if args.habitat_csv:
        rc = max(
            rc,
            _run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/download_habitat_eqa_data.py",
                    "--fetch-csv",
                    "--data-dir",
                    str(cfg.paths.habitat_eqa_data),
                ],
                dry_run=args.dry_run,
            ),
        )

    habitat_episodes = load_habitat_ovmm_episodes(cfg.habitat_episodes_yaml)
    hm3d_train = cfg.paths.hm3d_data / "scene_datasets" / "hm3d" / "train"
    missing_scenes: list[str] = []
    missing_sem: list[str] = []
    for ep in habitat_episodes:
        scene = str(ep["scene"])
        glb = hm3d_scene_glb_path(scene, hm3d_train)
        sem = hm3d_scene_semantic_glb_path(scene, hm3d_train)
        print(f"scene {scene}: glb={'OK' if glb.is_file() else 'MISSING'} sem={'OK' if sem.is_file() else 'MISSING'}")
        if not glb.is_file():
            missing_scenes.append(scene)
        if not sem.is_file():
            missing_sem.append(scene)

    if missing_scenes and not args.dry_run:
        print("HM3D train scenes missing — need Matterport tokens for full train download.")
        print("  uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d train")
        rc = 1
    if missing_sem and args.habitat_semantics:
        rc = max(
            rc,
            _run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/download_habitat_eqa_data.py",
                    "--fetch-hm3d-semantics",
                    "train",
                    "--hm3d-data-path",
                    str(cfg.paths.hm3d_data),
                ],
                dry_run=args.dry_run,
            ),
        )
    elif missing_sem:
        print("Semantic .glb missing for some scenes; pass --habitat-semantics to fetch train annots.")

    if args.robocasa:
        rc = max(
            rc,
            _run(["uv", "run", "python", "scripts/download_robocasa_assets.py", "--yes"], dry_run=args.dry_run),
        )

    # Ensure output dirs exist (under home, not git tree)
    for out in (cfg.paths.output_dir_sim, cfg.paths.output_dir_habitat):
        if not args.dry_run:
            out.mkdir(parents=True, exist_ok=True)
        print(f"output dir ready: {out}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
