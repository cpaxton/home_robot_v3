#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Download / document HM-EQA and HM3D assets for the Habitat harness."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from emet.habitat.config import default_habitat_eqa_data_dir

EXPLORE_EQA_QUESTIONS = (
    "https://raw.githubusercontent.com/SaumyaSaxena/explore-eqa_semnav/master/data/questions.csv"
)
EXPLORE_EQA_INIT_POSES = (
    "https://raw.githubusercontent.com/SaumyaSaxena/explore-eqa_semnav/master/data/scene_init_poses.csv"
)


def _print_instructions(data_dir: Path, hm3d_dir: Path) -> None:
    print(
        f"""
Habitat EQA data layout (GraphEQA-compatible)

  HABITAT_EQA_DATA_DIR={data_dir}
    questions.csv
    scene_init_poses.csv
    open-eqa-v0.json   (optional, for OpenEQA subset)

  HM3D_SCENE_DIR={hm3d_dir.parent}/train
    <scene_id>/<scene_id>.basis.glb

HM3D download (requires Hugging Face access):
  https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d

Explore-EQA CSVs (HM-EQA questions):
  {EXPLORE_EQA_QUESTIONS}
  {EXPLORE_EQA_INIT_POSES}

OpenEQA JSON:
  https://github.com/facebookresearch/open-eqa/blob/main/data/open-eqa-v0.json

Quick fetch (CSVs only):
  uv run python scripts/download_habitat_eqa_data.py --fetch-csv
"""
    )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Habitat EQA dataset helper")
    parser.add_argument("--instructions", action="store_true", help="Print download instructions")
    parser.add_argument("--fetch-csv", action="store_true", help="Download HM-EQA questions + init poses CSVs")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    data_dir = args.data_dir or default_habitat_eqa_data_dir()
    hm3d_dir = data_dir.parent / "hm3d" / "train"

    if args.instructions or (not args.fetch_csv):
        _print_instructions(data_dir, hm3d_dir)
        if not args.fetch_csv:
            return 0

    if args.fetch_csv:
        _download(EXPLORE_EQA_QUESTIONS, data_dir / "questions.csv")
        _download(EXPLORE_EQA_INIT_POSES, data_dir / "scene_init_poses.csv")
        print(f"Wrote CSVs under {data_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
