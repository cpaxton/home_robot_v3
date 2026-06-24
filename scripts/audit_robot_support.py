#!/usr/bin/env python3
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
# Maintainer audit: registry vs MJCF vs MolmoSpaces vs RoboCasa vs data tooling.

"""Print robot support matrix for emet sim + learning stacks.

Run from repo root::

    uv run python scripts/audit_robot_support.py
    uv run python scripts/audit_robot_support.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _spawn_json_path(robot_key: str) -> Path | None:
    from emet.utils.assets import get_robot_mjcf_path

    mjcf = get_robot_mjcf_path(robot_key)
    if mjcf is None:
        return None
    candidate = mjcf.parent / "molmospaces_spawn.json"
    return candidate if candidate.is_file() else None


_GALAXEA_KEYS = frozenset({"rby1", "rby1m", "galaxea_r1", "galaxear1", "rb_y1", "rby_1"})
_STRIP_PLACEHOLDER = frozenset(
    {"stretch", "hello_stretch", "hellostretch", "innate_mars", "xlerobot", "xlerobot_dual", *_GALAXEA_KEYS}
)


def _robocasa_mode(robot_key: str) -> str:
    from emet.robots import get_robot_spec

    key = robot_key.lower().replace("-", "_")
    if key in _STRIP_PLACEHOLDER:
        try:
            spec = get_robot_spec(key)
        except NotImplementedError:
            return "strip-replace (stub backend)"
        if spec is not None and spec.planar_base_joint_names:
            return "strip-replace (planar base)"
        return "strip-replace (freejoint or stretch stack)"
    return f"robosuite-native ({key})"


def _zmq_stack(robot_key: str) -> str:
    from emet.robots import get_robot_spec

    key = robot_key.lower().replace("-", "_")
    if key in ("stretch", "hello_stretch", "hellostretch"):
        return "MujocoZmqServer (StretchZmqClient)"
    try:
        spec = get_robot_spec(key)
    except NotImplementedError:
        return "n/a (stub)"
    if spec is None:
        return "unknown"
    if spec.sim_uses_stretch_mujoco_zmq:
        return "MujocoZmqServer"
    return "RobosuiteZmqServer + GenericZmqClient"


def _backend_status(robot_key: str) -> str:
    from emet.robots import get_robot_spec

    try:
        spec = get_robot_spec(robot_key)
    except NotImplementedError:
        return "stub"
    if spec is None:
        return "missing"
    return "ok"


def _test_hits(robot_key: str) -> list[str]:
    key = robot_key.lower().replace("-", "_")
    hits: list[str] = []
    test_root = REPO / "src" / "test"
    if not test_root.is_dir():
        return hits
    for path in test_root.rglob("*.py"):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if key in text or robot_key in text:
            rel = path.relative_to(REPO).as_posix()
            if rel not in hits:
                hits.append(rel)
    return sorted(hits)[:8]


def collect_rows() -> list[dict[str, object]]:
    from emet.robots import ROBOT_REGISTRY
    from emet.simulation.molmospaces_config import MOLMOSPACES_ROBOT_IDS
    from emet.utils.assets import get_robot_mjcf_path

    keys: set[str] = set(ROBOT_REGISTRY.keys())
    keys.add("stretch")
    keys.update(MOLMOSPACES_ROBOT_IDS)

    rows: list[dict[str, object]] = []
    for key in sorted(keys):
        norm = key.lower().replace("-", "_")
        mjcf = get_robot_mjcf_path(norm)
        spawn = _spawn_json_path(norm)
        rows.append(
            {
                "robot_key": norm,
                "in_registry": norm in ROBOT_REGISTRY or norm == "stretch",
                "molmo_listed": norm in MOLMOSPACES_ROBOT_IDS,
                "mjcf_path": str(mjcf) if mjcf else None,
                "mjcf_exists": bool(mjcf and mjcf.is_file()),
                "spawn_json": str(spawn) if spawn else None,
                "backend": _backend_status(norm),
                "robocasa": _robocasa_mode(norm),
                "zmq_stack": _zmq_stack(norm),
                "test_files_sample": _test_hits(norm),
            }
        )
    return rows


def print_table(rows: list[dict[str, object]]) -> None:
    print(f"{'robot':<16} {'mjcf':<6} {'backend':<8} {'molmo':<6} {'robocasa':<28} zmq")
    print("-" * 96)
    for row in rows:
        mjcf = "yes" if row["mjcf_exists"] else "no"
        molmo = "yes" if row["molmo_listed"] else "no"
        robocasa = str(row["robocasa"])[:28]
        print(f"{row['robot_key']:<16} {mjcf:<6} {row['backend']:<8} {molmo:<6} {robocasa:<28} {row['zmq_stack']}")
    print()
    print("Molmo-native IDs without vendored MJCF cannot use emet merge-scene until assets are added.")
    print("See docs/robots/supported_robots.md for the full capability matrix.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit emet robot support.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()
    rows = collect_rows()
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
