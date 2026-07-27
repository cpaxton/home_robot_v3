#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Insert a license header into Python files that do not already have one.

Pre-commit hook. Unlike a path allowlist, this decides from file *content*: any file
whose head already carries a license marker (legacy Hello Robot headers, vendored SPDX
notices, upstream attributions) is left untouched, and everything else gets the project
header. New files therefore cannot be stamped with the wrong copyright holder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# A license marker anywhere in the head means the file already declares provenance.
LICENSE_MARKERS = ("copyright", "spdx-license-identifier")
SCAN_LINES = 15


def build_header(license_file: Path) -> list[str]:
    """Read the license text and return it as Python comment lines."""
    raw = license_file.read_text(encoding="utf-8").strip("\n").split("\n")
    return [f"# {line}".rstrip() for line in raw]


def has_license(text: str) -> bool:
    head = text.split("\n")[:SCAN_LINES]
    return any(marker in line.lower() for line in head for marker in LICENSE_MARKERS)


def insert_header(path: Path, header: list[str]) -> bool:
    """Insert *header* into *path*. Return True when the file was modified."""
    text = path.read_text(encoding="utf-8")
    if not text.strip() or has_license(text):
        return False

    lines = text.split("\n")
    # Keep shebang and encoding declarations first; both must stay on the top lines.
    start = 0
    if lines and lines[0].startswith("#!"):
        start = 1
    if start < len(lines) and lines[start].startswith("#") and "coding" in lines[start]:
        start += 1

    path.write_text("\n".join(lines[:start] + header + [""] + lines[start:]), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--license-file", required=True, type=Path)
    parser.add_argument("filenames", nargs="*", type=Path)
    args = parser.parse_args(argv)

    header = build_header(args.license_file)
    modified = [p for p in args.filenames if p.is_file() and insert_header(p, header)]
    for path in modified:
        print(f"Added license header to {path}")
    return 1 if modified else 0


if __name__ == "__main__":
    sys.exit(main())
