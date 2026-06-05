#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Add reg_bbox geoms and MJCF default classes to Robocasa objaverse object models."""

from __future__ import annotations

import sys

from emet.simulation.robocasa_objaverse_bbox import (
    ensure_objaverse_mjcf_defaults,
    ensure_objaverse_reg_bbox,
    objaverse_dir,
    objaverse_reg_bbox_present,
)


def main() -> int:
    obj_dir = objaverse_dir()
    if not obj_dir.is_dir():
        print(
            "Objaverse directory missing. Download assets first:\n"
            "  uv run python scripts/download_robocasa_assets.py --yes",
            file=sys.stderr,
        )
        return 1

    if not ensure_objaverse_reg_bbox():
        print(
            "Failed to process objaverse models. See logs above.",
            file=sys.stderr,
        )
        return 1

    if not objaverse_reg_bbox_present():
        n = ensure_objaverse_mjcf_defaults(obj_dir)
        if n:
            print(f"Added MJCF default classes to {n} model.xml file(s).")
        if not objaverse_reg_bbox_present():
            print("Objaverse post-process incomplete.", file=sys.stderr)
            return 1

    print("Objaverse models ready for Robocasa object sampling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
