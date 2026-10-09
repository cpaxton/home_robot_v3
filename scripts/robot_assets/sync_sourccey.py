#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Vendor official Sourccey assets from clean local Git checkouts, with hashes.

Usage: python scripts/robot_assets/sync_sourccey.py --simulation CHECKOUT --hardware CHECKOUT
Then: python scripts/robot_assets/assemble_sourccey.py
No upstream Python is executed. Review the resulting Git diff before accepting updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

DEST = Path(__file__).resolve().parents[2] / "src/emet/assets/robot/sourccey/upstream"


def revision(path: Path) -> str:
    if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip():
        raise ValueError(f"Upstream checkout must be clean: {path}")
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def sync(simulation: Path, hardware: Path) -> None:
    revisions = {"simulation": revision(simulation), "hardware": revision(hardware)}
    source = "models/source/SourcceyURDF/SourcceyMkV.urdf"
    if (simulation / source).read_bytes() != (hardware / "URDF/FullBody/SourcceyMkV.urdf").read_bytes():
        raise ValueError("Hardware and simulation full-body URDFs differ; reconcile before updating.")
    # Copy only the source URDF, referenced STLs, calibration metadata, compiled
    # model, and license. Never vendor executable app code or Unity sidecars.
    import xml.etree.ElementTree as ET

    files = {
        source,
        "models/source/SourcceyURDF/README.md",
        "models/source/camera_poses.json",
        "models/source/lidar_config.json",
        "models/sourccey.xml",
        "models/build_info.json",
        "LICENSE",
    }
    for mesh in ET.parse(simulation / source).iter("mesh"):
        path = Path(source).parent / mesh.get("filename")
        if not (simulation / path).resolve().is_relative_to(simulation.resolve()):
            raise ValueError(f"Mesh escapes upstream checkout: {path}")
        files.add(path.as_posix())
    contents = {name: (simulation / name).read_bytes() for name in sorted(files)}
    for name, content in contents.items():
        if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Unresolved Git LFS object: {name}; run git lfs pull upstream")
    if DEST.exists():
        shutil.rmtree(DEST)
    for name, content in contents.items():
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = {
        "repositories": {
            key: {
                "url": f"https://github.com/vulcan-forge/sourccey-{'simulation-mujoco' if key == 'simulation' else 'hardware'}",
                "commit": rev,
            }
            for key, rev in revisions.items()
        },
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()},
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Vendored {len(contents)} files to {DEST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--hardware", required=True, type=Path)
    args = parser.parse_args()
    sync(args.simulation, args.hardware)
