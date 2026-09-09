# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Importing eval harness / affinity must not load OVMM or MuJoCo."""

from __future__ import annotations

import subprocess
import sys


def test_eval_harness_import_does_not_load_ovmm_or_mujoco() -> None:
    code = (
        "import sys\n"
        "import emet.eval\n"
        "import emet.eval.harness\n"
        "assert 'emet.eval.ovmm_find_phase' not in sys.modules\n"
        "assert 'mujoco' not in sys.modules\n"
        "from emet.eval import ovmm_find_phase\n"
        "assert hasattr(ovmm_find_phase, 'score_find_object')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
