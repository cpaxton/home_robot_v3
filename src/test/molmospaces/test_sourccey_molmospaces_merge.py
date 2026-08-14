# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Sourccey MolmoSpaces merge smoke: merged iTHOR + sourccey loads with robot joints/cameras.

Run (requires the MolmoSpaces venv + assets):
    RUN_MOLMOSPACES_TESTS=1 uv run emet test src/test/molmospaces/test_sourccey_molmospaces_merge.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _SRC_ROOT.parent  # repo root (merge-scene must run from here, not src/)
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


def _truthy(env: str) -> bool:
    return os.environ.get(env, "").strip().lower() in ("1", "true", "yes", "on")


_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _skip_reason() -> str | None:
    if not RUN_SIM_TESTS:
        return "RUN_SIM_TESTS=0"
    if not _truthy("RUN_MOLMOSPACES_TESTS"):
        return "RUN_MOLMOSPACES_TESTS=1 required (MolmoSpaces assets + wrapper)"
    return None


_SKIP = _skip_reason()


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "skipped")
@pytest.mark.timeout(300)
def test_sourccey_molmospaces_merge_loads():
    """Merge an iTHOR scene with the vendored Sourccey MJCF; the result must load in MuJoCo
    with the robot's planar-base + arm joints and all 4 cameras present."""
    import mujoco
    import numpy as np

    from emet.simulation.molmospaces_config import (
        ensure_molmo_asset_layout_symlinks,
        ensure_molmospaces_assets_dir_env,
        validate_molmospaces_robot,
    )
    from emet.utils.assets import get_robot_mjcf_path

    env = os.environ.copy()
    ensure_molmospaces_assets_dir_env(env)
    ensure_molmo_asset_layout_symlinks()
    assert validate_molmospaces_robot("sourccey") == "sourccey"
    robot_mjcf = get_robot_mjcf_path("sourccey")
    assert robot_mjcf is not None and robot_mjcf.is_file()

    # Run merge-scene in-process via the MolmoSpaces venv python (the same runner
    # ``emet-molmospaces merge-scene`` uses). Calling the exe as a subprocess can pick
    # up a stale ``robot_size_*.xml`` under pytest's env, so shell out to a one-liner
    # that invokes ``run_merge_scene`` directly. The merged wrapper is written into the
    # robot dir so the robot's relative ``meshes/`` resolve against it.
    venv_py = Path(
        os.environ.get(
            "MOLMOSPACES_PYTHON",
            _REPO_ROOT / ".venv-molmospaces" / "bin" / "python",
        )
    )
    out = robot_mjcf.parent / "molmospaces_merged_selftest.xml"
    script = (
        "import os, sys\n"
        f"os.environ.setdefault('MLSPACES_ASSETS_DIR', {os.environ.get('MLSPACES_ASSETS_DIR', '')!r})\n"
        f"os.environ.setdefault('MLSPACES_CACHE_DIR', {os.environ.get('MLSPACES_CACHE_DIR', '')!r})\n"
        "import emet_molmospaces.runner as r\n"
        f"rc = r.run_merge_scene('ithor', 'train', 0, 'sourccey', {str(out)!r})\n"
        "sys.exit(rc)\n"
    )
    try:
        r = subprocess.run(
            [str(venv_py), "-c", script],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert r.returncode == 0, f"merge-scene failed: {r.stdout}\n{r.stderr}"
        assert out.is_file()

        m = mujoco.MjModel.from_xml_path(str(out))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for _ in range(20):
            mujoco.mj_step(m, d)
        assert np.isfinite(d.qacc).all()

        for j in ("base_x", "base_y", "base_yaw", "lift", "left_shoulder_pan", "right_elbow_flex", "left_gripper"):
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0, f"missing joint {j} in merged scene"
        cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
        for c in ("front_left", "front_right", "wrist_left", "wrist_right"):
            assert c in cams, f"missing camera {c} in merged scene"
    finally:
        if out.exists():
            out.unlink()
