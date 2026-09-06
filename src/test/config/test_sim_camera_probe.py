# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("flip_rows", [False, True])
def test_image_up_respects_intrinsic_pixel_transform(flip_rows):
    probe = runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/probe_rby1_camera.py"))
    pose = np.eye(4)
    pose[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    intrinsics = np.diag([400.0, 400.0, 1.0])
    if flip_rows:
        intrinsics[1, 1] *= -1
    assert probe["image_up_world_z"](pose, intrinsics) == pytest.approx(-1.0 if flip_rows else 1.0)
