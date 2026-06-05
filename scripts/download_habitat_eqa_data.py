#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Download HM-EQA CSVs and HM3D scenes for the Habitat harness."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from emet.habitat.config import (
    default_habitat_eqa_data_dir,
    default_hm3d_data_path,
    default_hm3d_scene_dir,
    hm3d_scene_glb_path,
)
from emet.habitat.datasets import get_question, load_hmeqa_questions

EXPLORE_EQA_QUESTIONS = (
    "https://raw.githubusercontent.com/SaumyaSaxena/explore-eqa_semnav/master/data/questions.csv"
)
EXPLORE_EQA_INIT_POSES = (
    "https://raw.githubusercontent.com/SaumyaSaxena/explore-eqa_semnav/master/data/scene_init_poses.csv"
)

HM3D_UIDS = {
    "example": "hm3d_example_habitat",
    "minival": "hm3d_minival_habitat_v0.2",
    "train": "hm3d_train_habitat_v0.2",
    "val": "hm3d_val_habitat_v0.2",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _habitat_python() -> Path:
    root = _project_root()
    for candidate in (
        root / ".venv-habitat" / "bin" / "python",
        Path(os.environ.get("HABITAT_PYTHON", "")),
    ):
        if candidate and candidate.is_file():
            return candidate
    return Path(sys.executable)


def _print_instructions(data_dir: Path, hm3d_data: Path, hm3d_train: Path) -> None:
    print(
        f"""
Habitat EQA data layout (GraphEQA-compatible)

  HABITAT_EQA_DATA_DIR={data_dir}
    questions.csv
    scene_init_poses.csv

  HM3D_DATA_PATH={hm3d_data}
    scene_datasets/hm3d/train/<scene_id>/<short_id>.basis.glb
    (e.g. 00004-VqCaAuuoeWk/VqCaAuuoeWk.basis.glb)

  HM3D_SCENE_DIR={hm3d_train}  (default train split root)

HM3D download (API tokens required for train/val/minival — NOT web login):
  Full guide: docs/habitat/data.md#matterport-credentials-hm3d-train--val--minival
  1. Request access: https://matterport.com/partners/meta
  2. Matterport → Settings → Developer Tools → Habitat dataset
  3. export MATTERPORT_USERNAME='<token-id>'
     export MATTERPORT_PASSWORD='<token-secret>'

  Smoke (no auth, ~150MB):
    uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d example

  HM-EQA needs train split (~27GB habitat format):
    uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d train

  Quick dev split (~400MB):
    uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d minival

CSVs:
  uv run python scripts/download_habitat_eqa_data.py --fetch-csv

Verify question 0 scene on disk:
  uv run python scripts/download_habitat_eqa_data.py --verify-question 0
"""
    )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _fetch_hm3d_split(
    split: str,
    *,
    hm3d_data: Path,
    username: str | None,
    password: str | None,
    replace: bool,
) -> int:
    uid = HM3D_UIDS.get(split)
    if uid is None:
        print(f"Unknown HM3D split {split!r}; choose from: {', '.join(HM3D_UIDS)}", file=sys.stderr)
        return 1

    py = _habitat_python()
    chk = subprocess.run([str(py), "-c", "import habitat_sim"], check=False)
    if chk.returncode != 0:
        print("habitat_sim not found. Run ./scripts/install_habitat.sh first.", file=sys.stderr)
        return 1

    user = username or os.environ.get("MATTERPORT_USERNAME", "").strip()
    pwd = password or os.environ.get("MATTERPORT_PASSWORD", "").strip()
    if split != "example" and (not user or not pwd):
        print(
            f"HM3D {split} requires Matterport credentials.\n"
            "Set MATTERPORT_USERNAME and MATTERPORT_PASSWORD (see --instructions).",
            file=sys.stderr,
        )
        return 1

    hm3d_data.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(py),
        "-m",
        "habitat_sim.utils.datasets_download",
        "--uids",
        uid,
        "--data-path",
        str(hm3d_data),
        "--no-replace",
    ]
    if replace:
        cmd = [c for c in cmd if c != "--no-replace"] + ["--replace"]
    if user and pwd:
        cmd.extend(["--username", user, "--password", pwd])

    print(f"Running: {' '.join(cmd[:6])} ... --data-path {hm3d_data} ...")
    return subprocess.call(cmd)


def _verify_question(question_id: int, data_dir: Path, hm3d_train: Path) -> int:
    questions = load_hmeqa_questions(data_dir / "questions.csv")
    q = get_question(questions, question_id=question_id)
    glb = hm3d_scene_glb_path(q.scene, hm3d_train)
    print(f"question_id={question_id} scene={q.scene} floor={q.floor}")
    print(f"expected glb: {glb}")
    if glb.is_file():
        print("status: OK")
        return 0
    print("status: MISSING")
    print(
        "HM-EQA scenes are from the HM3D train split. "
        "Run: uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d train "
        "(requires Matterport credentials; ~27GB)."
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Habitat EQA dataset helper")
    parser.add_argument("--instructions", action="store_true", help="Print download instructions")
    parser.add_argument("--fetch-csv", action="store_true", help="Download HM-EQA questions + init poses CSVs")
    parser.add_argument(
        "--fetch-hm3d",
        choices=sorted(HM3D_UIDS.keys()),
        help="Download HM3D split via habitat_sim (example=minival/train need Matterport creds)",
    )
    parser.add_argument("--hm3d-data-path", type=Path, default=None, help="HM3D datasets_download --data-path")
    parser.add_argument("--data-dir", type=Path, default=None, help="HABITAT_EQA_DATA_DIR for CSVs")
    parser.add_argument("--matterport-username", default=None)
    parser.add_argument("--matterport-password", default=None)
    parser.add_argument("--replace", action="store_true", help="Replace existing HM3D split data")
    parser.add_argument("--verify-question", type=int, default=None, metavar="ID")
    args = parser.parse_args(argv)

    data_dir = args.data_dir or default_habitat_eqa_data_dir()
    hm3d_data = args.hm3d_data_path or default_hm3d_data_path()
    hm3d_train = default_hm3d_scene_dir()

    if args.instructions or (
        not args.fetch_csv and args.fetch_hm3d is None and args.verify_question is None
    ):
        _print_instructions(data_dir, hm3d_data, hm3d_train)
        if not args.fetch_csv and args.fetch_hm3d is None and args.verify_question is None:
            return 0

    if args.fetch_csv:
        _download(EXPLORE_EQA_QUESTIONS, data_dir / "questions.csv")
        _download(EXPLORE_EQA_INIT_POSES, data_dir / "scene_init_poses.csv")
        print(f"Wrote CSVs under {data_dir}")

    if args.fetch_hm3d:
        rc = _fetch_hm3d_split(
            args.fetch_hm3d,
            hm3d_data=hm3d_data,
            username=args.matterport_username,
            password=args.matterport_password,
            replace=args.replace,
        )
        if rc != 0:
            return rc
        print(f"HM3D {args.fetch_hm3d} under {hm3d_data}")
        print(f"Train scenes: {hm3d_train}")

    if args.verify_question is not None:
        return _verify_question(args.verify_question, data_dir, hm3d_train)

    return 0


if __name__ == "__main__":
    sys.exit(main())
