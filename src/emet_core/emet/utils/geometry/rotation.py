# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

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
