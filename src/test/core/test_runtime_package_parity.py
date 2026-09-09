# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import runpy
from pathlib import Path


def test_robot_package_carries_identical_command_runtime():
    root = Path(__file__).resolve().parents[3]
    module = runpy.run_path(str(root / "scripts/sync_core_runtime.py"))
    assert module["sync"](root, check=True) == [], "Run python scripts/sync_core_runtime.py"
