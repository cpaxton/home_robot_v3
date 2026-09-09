# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Sync the deployable ZMQ subset from canonical src/emet sources.

Run after editing these modules; --check is suitable for CI. Deployments use
emet_core first on PYTHONPATH, so intentional behavioral divergence is forbidden.
"""

import argparse
import shutil
from pathlib import Path

FILES = ("core/zmq_obs_codec.py", "core/zmq_server_env.py", "utils/compression.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "src"
    mismatches = []
    for name in FILES:
        source = root / "emet" / name
        target = root / "emet_core/emet" / name
        if source.read_bytes() != target.read_bytes():
            mismatches.append(name)
            if not args.check:
                shutil.copyfile(source, target)
    if args.check and mismatches:
        parser.exit(1, "Run scripts/sync_zmq_runtime.py: " + ", ".join(mismatches) + "\n")


if __name__ == "__main__":
    main()
