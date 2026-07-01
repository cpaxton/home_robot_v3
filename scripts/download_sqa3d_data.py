#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Download SQA3D annotation JSON from Zenodo."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from emet.benchmarks.sqa3d.config import (
    ZENODO_LOCALIZATION_URL,
    ZENODO_SQA_TASK_URL,
    annotations_json_path,
    balanced_dir,
    default_sqa3d_data_dir,
    questions_json_path,
)
from emet.benchmarks.sqa3d.datasets import load_sqa3d_questions


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _extract_zip(zip_path: Path, dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_root)
    print(f"Extracted -> {dest_root}")


def _print_instructions(data_dir: Path) -> None:
    print(
        f"""
SQA3D data layout

  SQA3D_DATA_DIR={data_dir}
    sqa_task/balanced/v1_balanced_questions_<split>_scannetv2.json
    sqa_task/balanced/v1_balanced_sqa_annotations_<split>_scannetv2.json
    sqa_task/answer_dict.json
    localization_task/balanced/v1_balanced_localization_<split>_scannetv2.json  (optional)

Quick start:
  uv run python scripts/download_sqa3d_data.py --fetch-annotations
  uv run python scripts/download_sqa3d_data.py --fetch-localization
  uv run emet sqa3d info
  uv run emet sqa3d list-questions --split val --limit 5

ScanNet meshes (for embodied replay) are separate — see docs/sqa3d.md.
"""
    )


def _verify_split(data_dir: Path, split: str) -> None:
    qs = load_sqa3d_questions(split, data_dir=data_dir)
    print(f"split={split}: loaded {len(qs)} joined questions")
    if qs:
        q = qs[0]
        print(f"  sample id={q.question_id} scene={q.scene_id} answer={q.primary_answer!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SQA3D benchmark annotations")
    parser.add_argument("--data-dir", type=Path, default=None, help="SQA3D_DATA_DIR target")
    parser.add_argument("--fetch-annotations", action="store_true", help="Download sqa_task.zip")
    parser.add_argument("--fetch-localization", action="store_true", help="Download localization_task.zip")
    parser.add_argument("--verify-split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--instructions", action="store_true", help="Print on-disk layout")
    args = parser.parse_args()

    data_dir = args.data_dir or default_sqa3d_data_dir()
    if args.instructions:
        _print_instructions(data_dir)
        return

    if not args.fetch_annotations and not args.fetch_localization and args.verify_split is None:
        parser.print_help()
        _print_instructions(data_dir)
        return

    with tempfile.TemporaryDirectory(prefix="sqa3d_dl_") as tmp:
        tmp_path = Path(tmp)
        if args.fetch_annotations:
            zpath = tmp_path / "sqa_task.zip"
            _download(ZENODO_SQA_TASK_URL, zpath)
            _extract_zip(zpath, data_dir)
        if args.fetch_localization:
            zpath = tmp_path / "localization_task.zip"
            _download(ZENODO_LOCALIZATION_URL, zpath)
            _extract_zip(zpath, data_dir)

    if args.verify_split:
        _verify_split(data_dir, args.verify_split)
    elif args.fetch_annotations:
        for split in ("train", "val", "test"):
            q_ok = questions_json_path(split, data_dir).is_file()
            a_ok = annotations_json_path(split, data_dir).is_file()
            print(f"  {split}: questions={q_ok} annotations={a_ok}")
        if not balanced_dir(data_dir).is_dir():
            # Zenodo zip may nest one level; fix common layout drift.
            nested = data_dir / "sqa_task" / "sqa_task"
            if nested.is_dir():
                shutil.copytree(nested, data_dir / "sqa_task", dirs_exist_ok=True)
        _verify_split(data_dir, "val")


if __name__ == "__main__":
    main()
