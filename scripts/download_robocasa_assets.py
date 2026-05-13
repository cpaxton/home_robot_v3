#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""
Download Robocasa kitchen assets (~5GB) without importing robocasa.

Robocasa's download_kitchen_assets.py imports robocasa, which asserts numpy 1.23.x.
This project often uses numpy 1.24+. This script uses the same URLs and folder
layout as robocasa/robocasa/scripts/download_kitchen_assets.py but does not
import robocasa, so it runs with the project's Python.

If assets (textures, fixtures, etc.) already exist, prompts: "Re-download? (y/N)"
default N. Use --yes to skip prompts:
- if all required assets exist, skip download
- if only some assets exist, download missing assets without prompting
Use --force to re-download everything even when assets exist.

Usage (from project root):
  python scripts/download_robocasa_assets.py [--yes]
  python scripts/download_robocasa_assets.py --force   # re-download existing
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

# Same URLs and relative paths as robocasa/robocasa/scripts/download_kitchen_assets.py
# (robocasa.__path__[0] + "models/assets/..." -> robocasa_root/robocasa/models/assets/...)
# Optional 4th element: required subpaths that must exist to skip (e.g. fixtures needs "sinks" for kitchen scenes).
ASSETS = [
    (
        "textures",
        "https://utexas.box.com/shared/static/otdsyfjontk17jdp24bkhy2hgalofbh4.zip",
        "models/assets/textures",
        (),
    ),
    (
        "fixtures",
        "https://utexas.box.com/shared/static/pobhbsjyacahg2mx8x4rm5fkz3wlmyzp.zip",
        "models/assets/fixtures",
        ("sinks",),
    ),
    (
        "objaverse",
        "https://utexas.box.com/shared/static/ejt1kc2v5vhae1rl4k5697i4xvpbjcox.zip",
        "models/assets/objects/objaverse",
        (),
    ),
    (
        "generative_textures",
        "https://utexas.box.com/shared/static/gf9nkadvfrowkb9lmkcx58jwt4d6c1g3.zip",
        "models/assets/generative_textures",
        (),
    ),
]


def _asset_dir_exists(folder: Path, required_subpaths: tuple[str, ...]) -> bool:
    if not folder.exists():
        return False
    if next(folder.iterdir(), None) is None:
        return False
    if required_subpaths and not all((folder / sub).exists() for sub in required_subpaths):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Robocasa kitchen assets (~5GB)")
    ap.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    ap.add_argument("--force", action="store_true", help="Re-download even when assets exist (no prompt)")
    ap.add_argument(
        "--robocasa-dir", type=Path, default=None, help="Path to third_party/robocasa (default: auto-detect)"
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    robocasa_root = args.robocasa_dir or (root / "third_party" / "robocasa")
    # Assets go into robocasa package dir: robocasa/robocasa/models/assets/...
    base = robocasa_root / "robocasa"

    if not base.exists():
        print(f"Error: robocasa package not found at {base}", file=sys.stderr)
        return 1

    # Check current asset state
    any_exist = False
    all_exist = True
    for item in ASSETS:
        rel = item[2]
        required_subpaths = item[3] if len(item) > 3 else ()
        folder = base / rel
        exists = _asset_dir_exists(folder, required_subpaths)
        if exists:
            any_exist = True
        else:
            all_exist = False

    force_redownload = args.force
    if all_exist and not force_redownload:
        if not args.yes:
            reply = input("Kitchen assets appear to be present. Re-download? (y/N) ").strip().lower()
            if reply in ("y", "yes"):
                force_redownload = True
            else:
                print("Skipped.")
                return 0
        else:
            print("Kitchen assets already present; skipping (use --force to re-download).")
            return 0
    elif any_exist and not force_redownload and not args.yes:
        reply = input("Some kitchen assets are present. Download missing assets now? (Y/n) ").strip().lower()
        if reply in ("n", "no"):
            print("Skipped.")
            return 0

    if not args.yes and not any_exist:
        print("This will download ~5GB of Robocasa kitchen assets.")
        if input("Proceed? (Y/n) ").strip().lower() in ("n", "no"):
            print("Aborted.")
            return 0

    try:
        import urllib.request
    except ImportError:
        print("Error: urllib.request required (standard library)", file=sys.stderr)
        return 1

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    def download_url(url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if tqdm is not None:
            with tqdm(unit="B", unit_scale=True, miniters=1, desc=path.name) as pbar:

                def reporthook(blocks: int, block_size: int, total: int | None) -> None:
                    if total is not None:
                        pbar.total = total
                    pbar.update(blocks * block_size - pbar.n)

                urllib.request.urlretrieve(url, filename=path, reporthook=reporthook)
        else:
            urllib.request.urlretrieve(url, filename=path)

    for item in ASSETS:
        name = item[0]
        url = item[1]
        rel = item[2]
        required_subpaths = item[3] if len(item) > 3 else ()
        folder = base / rel
        parent = folder.parent
        zip_path = parent / f"{folder.name}.zip"
        skip = not force_redownload and _asset_dir_exists(folder, required_subpaths)
        if skip:
            print(f"Skipping {name} (already exists).")
            continue
        if force_redownload and folder.exists():
            shutil.rmtree(folder)
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
        print(f"Downloading {name}...")
        for attempt in range(3):
            try:
                download_url(url, zip_path)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1}/3 failed: {e}")
        else:
            print(f"Failed to download {name}.", file=sys.stderr)
            return 1
        print("Extracting...")
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(path=parent)
        zip_path.unlink()
        print(f"Done: {name}")

    print("All assets downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
