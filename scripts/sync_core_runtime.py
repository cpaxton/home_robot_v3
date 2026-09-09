#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Refresh the lightweight robot package's shared command runtime, or check parity.

Canonical sources live in src/emet/core. The standalone emet-core distribution
must carry identical runtime code without importing the workstation package.
"""

import argparse
from pathlib import Path

RUNTIME_FILES = ("server.py", "command_tracker.py", "command_runtime.py", "navigation_result.py", "command_client.py")


def sync(root: Path, *, check: bool) -> list[str]:
    different = []
    for name in RUNTIME_FILES:
        source = root / "src/emet/core" / name
        destination = root / "src/emet_core/emet/core" / name
        contents = source.read_bytes()
        if not destination.exists() or destination.read_bytes() != contents:
            different.append(name)
            if not check:
                destination.write_bytes(contents)
    return different


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = sync(Path(__file__).resolve().parents[1], check=args.check)
    if changed:
        print(("Out of date: " if args.check else "Updated: ") + ", ".join(changed))
    raise SystemExit(1 if args.check and changed else 0)
