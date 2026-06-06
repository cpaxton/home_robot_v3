# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import os

import pytest

from emet.habitat.wrapper_config import build_habitat_wrapper_command


@pytest.mark.skipif(os.environ.get("RUN_HABITAT_TESTS") != "1", reason="Set RUN_HABITAT_TESTS=1")
def test_habitat_wrapper_info_smoke():
    import subprocess

    cmd = build_habitat_wrapper_command(["info"])
    if cmd is None:
        pytest.skip("Install ./scripts/install_habitat.sh for wrapper smoke")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0, r.stderr or r.stdout
