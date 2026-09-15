# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared EQA navigation must not initialize another simulator's GL backend."""

import os
import subprocess
import sys
from pathlib import Path


def test_eqa_runner_imports_without_mujoco():
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root / "packages/emet_habitat")])
    code = """
import importlib.abc
import sys
class NoMujoco(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mujoco' or fullname.startswith('mujoco.'):
            raise AssertionError('EQA startup must not import MuJoCo')
sys.meta_path.insert(0, NoMujoco())
import emet_habitat.runner
from emet.motion.base_goal_rank import navigable_neighbors
assert list(navigable_neighbors((0, 0), lambda p: p == (1, 0))) == [(1, 0)]
"""
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_mujoco_fk_still_available_on_demand():
    import mujoco

    from emet.motion.voxel_arm_collision import fk_link_xy_samples

    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="tip" pos="1 2 3"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    assert fk_link_xy_samples(model, data, ["tip"]) == [(1.0, 2.0)]
