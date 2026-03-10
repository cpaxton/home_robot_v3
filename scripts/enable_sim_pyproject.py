#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Restore sim extra and robosuite/robocasa path sources in pyproject.toml.

Run this after scripts/install_simulation.sh has cloned third_party/robocasa
and third_party/robosuite. Then run: uv lock && uv sync -e sim

Usage:
  python scripts/enable_sim_pyproject.py
  # or: uv run python scripts/enable_sim_pyproject.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

SIM_BLOCK = """# sim: empty by default so uv sync works without third_party/robocasa. Run: scripts/enable_sim_pyproject.py after install_simulation.sh
sim = []"""

SIM_BLOCK_FULL = """sim = [
    "mujoco>=3.3.0",  # Align with upstream robosuite; 3.2.6 was for older Stretch compat
    "hello-robot-stretch-urdf",
    "grpcio",
    "click>=8.1.8",
    "inputs>=0.5",
    "robosuite",   # From third_party (clone with: emet install sim)
    "robocasa",    # From third_party (clone with: emet install sim)
]"""

SOURCES_COMMENTED = """# Uncomment after running scripts/install_simulation.sh, then run scripts/enable_sim_pyproject.py
# robosuite = { path = "third_party/robosuite", editable = true }
# robocasa = { path = "third_party/robocasa", editable = true }"""

SOURCES_ACTIVE = """robosuite = { path = "third_party/robosuite", editable = true }
robocasa = { path = "third_party/robocasa", editable = true }"""


def main() -> None:
    if not (ROOT / "third_party" / "robocasa").is_dir() or not (ROOT / "third_party" / "robosuite").is_dir():
        raise SystemExit(
            "third_party/robocasa or third_party/robosuite missing. Run: ./scripts/install_simulation.sh first"
        )

    text = PYPROJECT.read_text()

    if SIM_BLOCK not in text:
        if "robocasa" in text and "robosuite" in text and "sim = [" in text:
            print("pyproject.toml already has sim enabled.")
            return
        raise SystemExit("pyproject.toml format changed; cannot apply enable_sim.")

    text = text.replace(SIM_BLOCK, SIM_BLOCK_FULL, 1)
    text = text.replace(SOURCES_COMMENTED, SOURCES_ACTIVE, 1)
    PYPROJECT.write_text(text)
    print("Enabled sim in pyproject.toml. Run: uv lock && uv sync -e sim")


if __name__ == "__main__":
    main()
