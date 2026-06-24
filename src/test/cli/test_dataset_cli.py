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

import subprocess
import sys


def test_dataset_molmobot_inspect_help():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "dataset", "molmobot", "inspect", "--help"],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[3]),
    )
    assert result.returncode == 0
    assert "path" in result.stdout.lower() or "PATH" in result.stdout


def test_dataset_group_help():
    result = subprocess.run(
        [sys.executable, "-m", "emet.cli", "dataset", "--help"],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[3]),
    )
    assert result.returncode == 0
    assert "molmobot" in result.stdout
