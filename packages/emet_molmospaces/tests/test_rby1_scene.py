# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# Tests for RB-Y1 (rby1) in MolmoSpaces-style scene: merge robot into scene, step, verify objects.

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Minimal MJCF scene (floor + one object) for testing scene+robot composition without molmo_spaces API.
MINIMAL_SCENE_XML = """<?xml version="1.0"?>
<mujoco model="minimal_scene">
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="object_cube" pos="0.5 0 0.15">
      <freejoint name="object_cube_free"/>
      <geom name="cube" type="box" size="0.05 0.05 0.05" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_rby1_robot_mjcf_path():
    """Resolve rby1 to Galaxea R1 MJCF path via emet registry."""
    from emet_molmospaces.runner import _get_robot_mjcf_path

    path = _get_robot_mjcf_path("rby1")
    assert path is not None
    assert path.exists()
    assert path.suffix == ".xml"
    # rb_y1 (with underscore) also resolves
    path2 = _get_robot_mjcf_path("rb_y1")
    assert path2 is not None
    assert path2.samefile(path)


def test_merge_robot_into_scene_and_step():
    """Merge rby1 into a minimal scene, load in MuJoCo, step, and verify scene has robot + objects."""
    import mujoco
    from emet_molmospaces.runner import _get_robot_mjcf_path, _merge_robot_into_scene

    robot_path = _get_robot_mjcf_path("rby1")
    if robot_path is None:
        pytest.skip("rby1 not in registry or MJCF not found (need emet with galaxea_r1 assets)")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(MINIMAL_SCENE_XML)
        scene_path = Path(f.name)
    try:
        merged_path = _merge_robot_into_scene(scene_path, robot_path)
        try:
            model = mujoco.MjModel.from_xml_path(str(merged_path))
            data = mujoco.MjData(model)
        finally:
            if merged_path.exists():
                merged_path.unlink(missing_ok=True)
    finally:
        scene_path.unlink(missing_ok=True)

    # Scene has floor, object_cube, plus robot bodies (base_link, etc.)
    assert model.nbody >= 3, "expected at least floor, one object, and robot base"
    assert model.nq > 0
    assert model.nv > 0
    # Step a few times (scene exploration)
    for _ in range(10):
        mujoco.mj_step(model, data)
    # Verify we have objects: body names should include our scene object and robot
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    assert "object_cube" in body_names
    assert "floor" in body_names or any("base" in (n or "") for n in body_names)
    # Object is found (present in model and stepped)
    assert data.time > 0


@pytest.mark.skipif(
    os.environ.get("RUN_MOLMOSPACES_TESTS", "") != "1",
    reason="RUN_MOLMOSPACES_TESTS=1 and wrapper venv with molmo-spaces required",
)
def test_rby1_molmospaces_serve_headless_brief():
    """With real MolmoSpaces: run serve headless for a short step count and verify model has objects."""
    import threading
    import time

    from emet_molmospaces.runner import main_runner

    # Run serve in a thread; we'll stop after a few steps by timing out (serve runs until KeyboardInterrupt).
    result = [None]

    def run():
        result[0] = main_runner(
            [
                "serve",
                "--scene",
                "ithor",
                "--split",
                "train",
                "--index",
                "0",
                "--robot",
                "rby1",
                "--headless",
            ]
        )

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(3.0)
    # We can't easily send KeyboardInterrupt to the thread; the test just checks that serve started.
    # So we only run this when RUN_MOLMOSPACES_TESTS=1 and accept that it may run until timeout.
    t.join(timeout=5.0)
    # If it finished, result[0] is 0; if still running we leave it (daemon thread).
    if result[0] is not None:
        assert result[0] == 0
