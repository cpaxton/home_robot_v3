# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Measure MolmoSpaces spawn hints from a merged MJCF and update ``molmospaces_spawn.json``.

Offline maintainer tool (not used at serve time). Runtime spawn reads the JSON via
:mod:`emet.simulation.molmospaces_spawn_metadata` and falls back to heuristics when absent.

Example::

    # Merge scene + robot, then measure spawn fields for that robot
    emet molmospaces merge-scene --scene ithor --robot rby1 \\
      -o /tmp/ithor_rby1.xml --install-if-missing
    emet molmospaces write-spawn-metadata --robot rby1 --mjcf /tmp/ithor_rby1.xml

    # Same via module (no Click)
    uv run python -m emet.app.write_molmospaces_spawn_metadata \\
      --robot rby1 --mjcf /tmp/ithor_rby1.xml

See ``docs/molmospaces_spawn_metadata.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco

from emet.simulation import molmospaces_spawn
from emet.simulation.molmospaces_spawn_metadata import METADATA_FILENAME, molmospaces_spawn_metadata_path


def compute_spawn_metadata_from_mjcf(
    mjcf_path: Path | str,
    *,
    robot_key: str,
    base_body_name: str = "base_link",
    seed_xy: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    """Kinematic floor settle at *seed_xy*; return JSON-ready spawn hints.

    Requires a merged scene+robot MJCF with a walkable floor and a free-floating *base_body_name*.
    """
    path = Path(mjcf_path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if bid < 0:
        raise ValueError(f"base body {base_body_name!r} not found in {path}")

    x, y = float(seed_xy[0]), float(seed_xy[1])
    if not molmospaces_spawn.write_freejoint_base_xyzw(
        model,
        data,
        x=x,
        y=y,
        z=0.5,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        base_body_name=base_body_name,
    ):
        raise RuntimeError(f"failed to place free joint on {base_body_name!r}")

    if not molmospaces_spawn.resettle_free_base_z_at_current_xy_preserving_yaw(
        model,
        data,
        base_body_name=base_body_name,
        robot_key=robot_key,
    ):
        raise RuntimeError("floor Z settle failed (no walkable floor or collision)")

    mujoco.mj_forward(model, data)
    z_floor = molmospaces_spawn.walkable_floor_z_at_xy(model, data, x, y, exclude_body_id=bid)
    if z_floor is None:
        raise RuntimeError("no walkable floor under base")

    base_z = float(data.xpos[bid, 2])
    rb = molmospaces_spawn._bodies_descending_from(model, bid)  # noqa: SLF001
    zb = molmospaces_spawn._min_robot_collision_geom_bottom_z(model, data, rb)  # noqa: SLF001
    foot_clearance = float(zb - z_floor) if zb is not None else None

    out: dict[str, Any] = {
        "schema_version": 1,
        "notes": f"Measured from {path.name} via emet.app.write_molmospaces_spawn_metadata",
        "molmospaces_nominal_base_height_above_floor_m": round(base_z - float(z_floor), 4),
    }
    if foot_clearance is not None:
        out["molmospaces_target_foot_clearance_above_floor_m"] = round(foot_clearance, 4)
    return out


def write_molmospaces_spawn_metadata(
    robot_key: str,
    mjcf_path: Path | str,
    *,
    output_path: Path | None = None,
    merge_existing: bool = True,
    base_body_name: str = "base_link",
    seed_xy: tuple[float, float] = (0.0, 0.0),
) -> Path:
    """Write or update ``molmospaces_spawn.json`` next to the robot MJCF (or *output_path*)."""
    measured = compute_spawn_metadata_from_mjcf(
        mjcf_path,
        robot_key=robot_key,
        base_body_name=base_body_name,
        seed_xy=seed_xy,
    )
    dest = output_path
    if dest is None:
        meta_path = molmospaces_spawn_metadata_path(robot_key)
        if meta_path is None:
            raise ValueError(
                f"No vendored MJCF for robot {robot_key!r}; cannot choose default {METADATA_FILENAME!r} path."
            )
        dest = meta_path
    dest = Path(dest)
    if merge_existing and dest.is_file():
        try:
            prev = json.loads(dest.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                prev.update(measured)
                measured = prev
        except (OSError, json.JSONDecodeError):
            pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
    return dest


def run_write(
    robot_key: str,
    mjcf_path: Path | str,
    *,
    output_path: Path | None = None,
    merge_existing: bool = True,
    base_body_name: str = "base_link",
    seed_x: float = 0.0,
    seed_y: float = 0.0,
) -> Path:
    """CLI/programmatic entry: measure and write spawn metadata."""
    return write_molmospaces_spawn_metadata(
        robot_key,
        mjcf_path,
        output_path=output_path,
        merge_existing=merge_existing,
        base_body_name=base_body_name,
        seed_xy=(float(seed_x), float(seed_y)),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure MolmoSpaces spawn hints from a merged MJCF and update molmospaces_spawn.json.",
        epilog=(
            "Default output: <robot_mjcf_dir>/molmospaces_spawn.json (e.g. galaxea_r1/ for rby1). "
            "See docs/molmospaces_spawn_metadata.md."
        ),
    )
    p.add_argument(
        "--robot",
        required=True,
        help="Robot id (must match the robot merged into the MJCF, e.g. rby1, stretch, innate_mars).",
    )
    p.add_argument(
        "--mjcf",
        required=True,
        type=str,
        help="Path to merged scene+robot MJCF (from emet molmospaces merge-scene or emet serve --scene ithor).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help=f"Override output JSON path (default: next to vendored MJCF as {METADATA_FILENAME}).",
    )
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="Replace the JSON file instead of merging keys into an existing file.",
    )
    p.add_argument(
        "--base-body",
        default="base_link",
        help="Free-joint base body name in the merged MJCF (default: base_link).",
    )
    p.add_argument(
        "--seed-x",
        type=float,
        default=0.0,
        help="World X (m) for the measurement pose before Z settle (default: 0).",
    )
    p.add_argument(
        "--seed-y",
        type=float,
        default=0.0,
        help="World Y (m) for the measurement pose before Z settle (default: 0).",
    )
    args = p.parse_args()
    out = run_write(
        str(args.robot).strip(),
        Path(args.mjcf),
        output_path=Path(args.output) if str(args.output).strip() else None,
        merge_existing=not bool(args.no_merge),
        base_body_name=str(args.base_body).strip() or "base_link",
        seed_x=float(args.seed_x),
        seed_y=float(args.seed_y),
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
