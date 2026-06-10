# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Ensure ``innate_mars_bridge`` is importable in pytest (ROS package under src/)."""

import sys
from pathlib import Path

_BRIDGE_SRC = Path(__file__).resolve().parents[2] / "innate_mars_bridge"
if _BRIDGE_SRC.is_dir() and str(_BRIDGE_SRC) not in sys.path:
    sys.path.insert(0, str(_BRIDGE_SRC))
