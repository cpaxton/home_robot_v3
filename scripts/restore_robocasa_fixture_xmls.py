# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Restore fixture MJCFs that the downloadable Robocasa fixtures pack omits.

The published ``fixtures`` pack ships ``visuals/model_*.obj`` for accessory
fixtures (knife blocks, utensil racks, …) without their ``model.xml``. The
official Robocasa repository tracks those small XMLs, so fetch them from the
pinned v0.2 commit instead of requiring a local checkout.
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

OFFICIAL_COMMIT = "0d34893"
ARCHIVE_URL = "https://github.com/robocasa/robocasa/archive/{commit}.tar.gz"
FIXTURES_SUFFIX = "/models/assets/fixtures/"


def missing_fixture_xmls(robocasa_pkg: Path) -> list[Path]:
    """Fixture directories with visual meshes but no ``model.xml``."""
    fixtures = robocasa_pkg / "models" / "assets" / "fixtures"
    missing: list[Path] = []
    for visuals in sorted(fixtures.rglob("visuals")):
        if not visuals.is_dir() or not any(visuals.glob("*.obj")):
            continue
        model = visuals.parent / "model.xml"
        if not model.is_file():
            missing.append(model)
    return missing


def restore(robocasa_pkg: Path, *, commit: str = OFFICIAL_COMMIT, dry_run: bool = False) -> int:
    missing = missing_fixture_xmls(robocasa_pkg)
    if not missing:
        print("All fixture MJCFs present.")
        return 0
    print(f"{len(missing)} fixture MJCFs missing; fetching robocasa {commit} archive...")
    if dry_run:
        for path in missing:
            print(path)
        return 0

    with urllib.request.urlopen(ARCHIVE_URL.format(commit=commit), timeout=120) as response:
        archive = response.read()

    fixtures_root = robocasa_pkg / "models" / "assets" / "fixtures"
    wanted = {path.relative_to(fixtures_root).as_posix(): path for path in missing}
    restored = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith("/model.xml"):
                continue
            index = member.name.find(FIXTURES_SUFFIX)
            if index < 0:
                continue
            relative = member.name[index + len(FIXTURES_SUFFIX) :]
            target = wanted.get(relative)
            if target is None:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            target.write_bytes(handle.read())
            restored += 1

    print(f"Restored {restored}/{len(missing)} fixture MJCFs from robocasa {commit}.")
    return 0 if restored else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocasa-dir", type=Path, required=True, help="Path to third_party/robocasa")
    parser.add_argument("--commit", default=OFFICIAL_COMMIT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pkg = args.robocasa_dir / "robocasa"
    if not pkg.is_dir():
        print(f"robocasa package not found under {args.robocasa_dir}", file=sys.stderr)
        return 2
    return restore(pkg, commit=args.commit, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
