# Copyright (c) Hello Robot, Inc.
# Ensure src is on sys.path so emet is importable when running pytest from project root.
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
