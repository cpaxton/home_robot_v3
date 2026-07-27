# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# Ensure src is on sys.path so emet is importable when running pytest from project root.
import os
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def pytest_collection_modifyitems(config, items) -> None:
    """Skip @pytest.mark.sim when RUN_SIM_TESTS=0 (``emet test --no-sim``)."""
    if RUN_SIM_TESTS:
        return
    skip = pytest.mark.skip(reason="RUN_SIM_TESTS=0")
    for item in items:
        if "sim" in item.keywords:
            item.add_marker(skip)
