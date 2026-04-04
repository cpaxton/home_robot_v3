# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# Ensures Robocasa kitchen assets exist before running simulation tests (downloads if missing).

import subprocess
import sys
from pathlib import Path

import pytest

# Project root: src/test/simulation/conftest.py -> ../../..
_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TEST_DIR.parent.parent.parent
_ROBOCASA_ASSETS_FIXTURES = _PROJECT_ROOT / "third_party" / "robocasa" / "robocasa" / "models" / "assets" / "fixtures"
# After fixtures zip download, sinks/white_sink exists (style001 can use Sink025 alias -> white_sink).
_SENTINEL_FILE = _ROBOCASA_ASSETS_FIXTURES / "sinks" / "white_sink" / "model.xml"
_ASSETS_OK = _SENTINEL_FILE.exists()


def _ensure_robocasa_assets() -> None:
    """Download Robocasa kitchen assets if missing. Skip simulation tests if robocasa not cloned."""
    robocasa_root = _PROJECT_ROOT / "third_party" / "robocasa"
    if not (robocasa_root / "robocasa").exists():
        pytest.skip("third_party/robocasa not found. Run: emet install sim")
    if _ASSETS_OK:
        return
    script = _PROJECT_ROOT / "scripts" / "download_robocasa_assets.py"
    if not script.exists():
        pytest.fail("Robocasa assets missing and scripts/download_robocasa_assets.py not found. Run: emet install sim")
    result = subprocess.run(
        [sys.executable, str(script), "--yes", "--robocasa-dir", str(robocasa_root)],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Failed to download Robocasa assets (exit {result.returncode}). "
            f"Run: python scripts/download_robocasa_assets.py --yes\n{result.stderr or result.stdout}"
        )
    if not _SENTINEL_FILE.exists():
        pytest.fail(
            f"Robocasa assets download completed but {_SENTINEL_FILE} still missing. "
            "Run: python scripts/download_robocasa_assets.py --yes"
        )


@pytest.fixture(scope="session", autouse=True)
def ensure_robocasa_assets():
    """Ensure Robocasa kitchen assets exist; download if missing so simulation tests can pass."""
    _ensure_robocasa_assets()
