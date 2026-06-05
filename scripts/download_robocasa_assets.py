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
Download Robocasa kitchen assets without importing robocasa.

Robocasa's download_kitchen_assets.py imports robocasa, which asserts numpy 1.23.x.
This project often uses numpy 1.24+. This script uses the same URLs and folder
layout as robocasa/robocasa/scripts/download_kitchen_assets.py but does not
import robocasa, so it runs with the project's Python.

Downloads (in order):
  - textures, fixtures (base), fixtures_lw (LightWheel registry + meshes), objaverse,
    generative_textures

``fixtures_lw`` is required for ``emet serve robocasa`` — kitchen style YAMLs reference
IDs such as ``Sink025`` that are only registered after the LightWheel pack is extracted.

If assets already exist, prompts: "Re-download? (y/N)" default N.
Use ``--yes`` to skip prompts and download only missing packs.
Use ``--force`` to re-download everything even when assets exist.

Usage (from project root):
  uv run python scripts/download_robocasa_assets.py --yes
  uv run python scripts/download_robocasa_assets.py --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

# (name, url, rel_path_under_robocasa_pkg, check_kind)
# check_kind: "dir_nonempty" | "basic_fixtures" | "lightwheel_registry"
ASSET_SPECS: list[tuple[str, str, str, str]] = [
    (
        "textures",
        "https://utexas.box.com/shared/static/4i85ileasdvstmlln5sbvzptz7keuoy1.zip",
        "models/assets/textures",
        "dir_nonempty",
    ),
    (
        "fixtures",
        "https://utexas.box.com/shared/static/zt9vbo38yb9f1alw9iuahck55hoa65y6.zip",
        "models/assets/fixtures",
        "basic_fixtures",
    ),
    (
        "fixtures_lw",
        "https://utexas.box.com/shared/static/idbncsadpnaz1jfl4i6m8qejawk7p9pi.zip",
        "models/assets/fixtures",
        "lightwheel_registry",
    ),
    (
        "objaverse",
        "https://utexas.box.com/shared/static/03eionyo8fk3a9dsksq9jb8du5lqfw8h.zip",
        "models/assets/objects/objaverse",
        "dir_nonempty",
    ),
    (
        "generative_textures",
        "https://utexas.box.com/shared/static/ebaad09k82tmfmlq6ohdkmrh8izl9vn5.zip",
        "models/assets/generative_textures",
        "dir_nonempty",
    ),
]


def _load_urls_from_box_links(robocasa_pkg: Path) -> dict[str, str]:
    """Map registry name -> direct .zip URL using shipped ``box_links_assets.json``."""
    path = robocasa_pkg / "models" / "assets" / "box_links" / "box_links_assets.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, str] = {}
    key_map = {
        "textures": "textures",
        "fixtures": "fixtures",
        "fixtures_lw": "fixtures_lightwheel",
        "objaverse": "objaverse",
        "generative_textures": "generative_textures",
    }
    for spec_name, box_key in key_map.items():
        shared = raw.get(box_key)
        if not shared:
            continue
        shared_id = str(shared).rstrip("/").split("/")[-1]
        base = str(shared).split("/s/")[0]
        out[spec_name] = f"{base}/shared/static/{shared_id}.zip"
    return out


def _basic_fixtures_present(fixtures_dir: Path) -> bool:
    return (fixtures_dir / "sinks" / "white_sink" / "model.xml").is_file()


def _lightwheel_registry_present(fixtures_dir: Path) -> bool:
    sink_reg = fixtures_dir / "fixture_registry" / "sink.yaml"
    if not sink_reg.is_file():
        return False
    try:
        return "Sink025:" in sink_reg.read_text(encoding="utf-8")
    except OSError:
        return False


def _asset_present(base: Path, rel: str, check_kind: str) -> bool:
    folder = base / rel
    if check_kind == "basic_fixtures":
        return _basic_fixtures_present(folder)
    if check_kind == "lightwheel_registry":
        return _lightwheel_registry_present(folder)
    if not folder.exists() or next(folder.iterdir(), None) is None:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download Robocasa kitchen assets (base + LightWheel fixtures for serve robocasa)"
    )
    ap.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    ap.add_argument("--force", action="store_true", help="Re-download even when assets exist (no prompt)")
    ap.add_argument(
        "--robocasa-dir", type=Path, default=None, help="Path to third_party/robocasa (default: auto-detect)"
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    robocasa_root = args.robocasa_dir or (root / "third_party" / "robocasa")
    base = robocasa_root / "robocasa"

    if not base.exists():
        print(f"Error: robocasa package not found at {base}", file=sys.stderr)
        return 1

    url_overrides = _load_urls_from_box_links(base)
    specs: list[tuple[str, str, str, str]] = []
    for name, default_url, rel, check in ASSET_SPECS:
        url = url_overrides.get(name, default_url)
        specs.append((name, url, rel, check))

    any_exist = False
    all_exist = True
    for name, _url, rel, check in specs:
        if _asset_present(base, rel, check):
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
            print("Kitchen assets already present (base + LightWheel fixtures); skipping.")
            print("Use --force to re-download.")
            return 0
    elif any_exist and not force_redownload and not args.yes:
        reply = input("Some kitchen assets are missing. Download missing packs now? (Y/n) ").strip().lower()
        if reply in ("n", "no"):
            print("Skipped.")
            return 0

    if not args.yes and not any_exist:
        print(
            "This will download Robocasa kitchen assets (~10–15 GB: textures, base fixtures, "
            "LightWheel fixtures_lw, objaverse, generative_textures)."
        )
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

    for name, url, rel, check in specs:
        folder = base / rel
        parent = folder.parent
        zip_path = parent / f"{folder.name}.zip"
        skip = not force_redownload and _asset_present(base, rel, check)
        if skip:
            print(f"Skipping {name} (already present).")
            continue
        if force_redownload and check == "dir_nonempty" and folder.exists():
            shutil.rmtree(folder)
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
        elif force_redownload and check == "basic_fixtures" and not _lightwheel_registry_present(folder):
            # Re-fetch base fixtures only; do not wipe LightWheel registry if already merged.
            if folder.exists() and not _basic_fixtures_present(folder):
                shutil.rmtree(folder)
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
        zip_path.unlink(missing_ok=True)
        print(f"Done: {name}")
        if name == "fixtures_lw":
            try:
                from emet.simulation.robocasa_registry_sync import restore_fixture_registry_from_vcs

                n = restore_fixture_registry_from_vcs(base)
                if n:
                    print(f"Restored {n} fixture_registry YAML file(s) from robocasa git.")
            except Exception as e:
                print(f"Registry restore failed ({e!r}).", file=sys.stderr)
                return 1

    fixtures_dir = base / "models/assets/fixtures"
    try:
        from emet.simulation.robocasa_registry_sync import (
            missing_required_registry_stems,
            restore_fixture_registry_from_vcs,
            sync_lightwheel_registry,
        )

        if missing_required_registry_stems(base):
            n = restore_fixture_registry_from_vcs(base)
            if n:
                print(f"Restored {n} fixture_registry YAML file(s) from robocasa git.")
    except Exception as e:
        print(f"Registry restore failed ({e!r}).", file=sys.stderr)
        return 1

    if _lightwheel_registry_present(fixtures_dir):
        print("LightWheel fixture registry already lists Sink025.")
    elif (fixtures_dir / "sinks" / "Sink025" / "model.xml").is_file():
        print("Syncing LightWheel mesh folders into fixture_registry YAML...")
        try:
            n = sync_lightwheel_registry(base)
            print(f"Added {n} registry entries for LightWheel fixture models.")
        except Exception as e:
            print(f"Registry sync failed ({e!r}).", file=sys.stderr)
            return 1
    if not _lightwheel_registry_present(fixtures_dir):
        print(
            "Warning: LightWheel fixtures still not registered (expected Sink025 in "
            "fixture_registry/sink.yaml). Download fixtures_lw or run sync.",
            file=sys.stderr,
        )
        return 1

    try:
        from emet.simulation.robocasa_assets_check import fixture_registry_layout_ok

        if not fixture_registry_layout_ok(base):
            from emet.simulation.robocasa_registry_sync import missing_required_registry_stems

            missing = ", ".join(missing_required_registry_stems(base))
            print(
                f"Warning: fixture_registry layout incomplete (missing: {missing}). "
                "Restore from git: git -C third_party/robocasa checkout -- "
                "robocasa/models/assets/fixtures/fixture_registry/",
                file=sys.stderr,
            )
            return 1
    except Exception as e:
        print(f"Registry layout check failed ({e!r}).", file=sys.stderr)
        return 1

    try:
        from emet.simulation.robocasa_objaverse_bbox import ensure_objaverse_reg_bbox

        if not ensure_objaverse_reg_bbox(base):
            print(
                "Warning: objaverse reg_bbox processing failed. Run:\n"
                "  uv run python scripts/process_robocasa_objaverse_reg_bbox.py",
                file=sys.stderr,
            )
            return 1
    except Exception as e:
        print(f"Objaverse reg_bbox processing failed ({e!r}).", file=sys.stderr)
        return 1

    print("All required Robocasa kitchen assets downloaded (including LightWheel fixtures).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
