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

import os
import socket
from pathlib import Path

import pytest
from click.testing import CliRunner

from emet.app.capture_sim_dataset_episode import capture_sim_episode


def _port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.skipif(os.environ.get("RUN_SIM_DATASET_CAPTURE") != "1", reason="set RUN_SIM_DATASET_CAPTURE=1 to run")
@pytest.mark.timeout(180)
def test_capture_no_server_when_sim_on_default_port(tmp_path: Path) -> None:
    if not _port_open("127.0.0.1", 4401):
        pytest.skip("no MuJoCo ZMQ server on 127.0.0.1:4401")

    from emet.app.robot_cli import discover_zmq_server_robot_id

    rid = discover_zmq_server_robot_id("127.0.0.1", port_offset=0, timeout=3.0)
    if not rid:
        pytest.skip("could not read emet_robot_id from ZMQ")

    out = tmp_path / "ep0"
    runner = CliRunner()
    result = runner.invoke(
        capture_sim_episode,
        [
            "--output-dir",
            str(out),
            "--no-server",
            "--rotate-only",
            "--robot",
            rid,
            "--steps",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "ground_truth.json").is_file()
    assert (out / "dataset_manifest.json").is_file()
    assert (out / "dynamem").is_dir()
