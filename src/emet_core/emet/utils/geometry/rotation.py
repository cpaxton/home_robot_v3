# Copyright (c) Hello Robot, Inc. All rights reserved.
# Slim rotation utils for emet-core (no torch).

import numpy as np
from scipy.spatial.transform import Rotation


def get_rotation_from_xyz(x_axis, y_axis, z_axis):
    """Build scipy Rotation from columns (x, y, z axes)."""
    rotation = np.zeros((3, 3))
    rotation[:, 0] = x_axis
    rotation[:, 1] = y_axis
    rotation[:, 2] = z_axis
    return Rotation.from_matrix(rotation)
