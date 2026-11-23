# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.motion.local_geometry_distance import cell_separations
from emet.motion.observed_solids import WorldSolid
from emet.motion.placement_geometry import box_corners


@pytest.mark.parametrize('convex', [False, True])
def test_native_local_distance_preserves_pose_and_mesh_compiler_transform(convex):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    pose[:3, 3] = [.2, -.3, .4]
    if convex:
        shape = WorldSolid('convex', (), pose, vertices=box_corners([[-.01, -.02, -.03], [.01, .02, .03]]),
                           material_calibration_id='test')
    else:
        shape = WorldSolid('box', (.01, .02, .03), pose)
    distances = cell_separations(shape, np.array([[.23, -.3, .4], [.2, -.3, .4]]), resolution=.005)
    assert distances[0] == pytest.approx(.0075, abs=1e-6)
    assert distances[1] < 0
