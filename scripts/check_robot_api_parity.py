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
# This source code is licensed under the LICENSE file in the
# root directory of this source tree.

"""
Compare embodied-stack ``self.robot`` call patterns against :class:`GenericZmqClient`.

Run from repo root: ``uv run python scripts/check_robot_api_parity.py`` (exit 0; prints notes).

**Remaining risks** (not all are attributes on the client; some are code-path-specific):
* ``arm_to``, full IK, ``get_ee_pose``, ``blocking_spin``, ``_rerun`` — Stretch or instance-memory
  only; see source comments in :class:`GenericZmqClient`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "emet" / "controller"
FILES = [
    SRC / "controller_dynamem.py",
    SRC / "task" / "dynamem" / "dynamem_task.py",
    SRC / "controller_graph_eqa.py",
]

ROBOT_RE = "self.robot"


def _robot_method_name(func: ast.AST) -> str | None:
    """Return ``my_method`` for ``self.robot.my_method(...)``; else None."""
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr is None:
        return None
    base = func.value
    if not isinstance(base, ast.Attribute) or base.attr != "robot":
        return None
    if not isinstance(base.value, ast.Name) or base.value.id != "self":
        return None
    return str(func.attr)


def _attr_calls_from_file(path: Path) -> set[str]:
    out: set[str] = set()
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        name = _robot_method_name(n.func)
        if name is not None:
            out.add(name)
    return out


def main() -> int:
    all_calls: set[str] = set()
    for p in FILES:
        c = _attr_calls_from_file(p)
        all_calls |= c
        print(f"--- {p.relative_to(REPO)} ---\n" + " ".join(sorted(c)) + "\n")

    from emet.controller.generic_zmq_client import GenericZmqClient

    missing: list[str] = []
    for name in sorted(all_calls):
        if not hasattr(GenericZmqClient, name):
            missing.append(name)
    if missing:
        print("--- GenericZmqClient missing (may still be inherited/optional) ---")
        print(" ".join(missing))
    else:
        print("--- All dynamem+graph call-site names are present on GenericZmqClient ---")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main())
