# Copyright (c) Allen Institute for AI (MolmoSpaces). Apache-2.0.
# Vendored from molmo_spaces/utils/linalg_utils.py (subset).

from __future__ import annotations

from functools import wraps

import numpy as np


def inverse_homogeneous_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 matrix.")
    rotation_matrix = matrix[0:3, 0:3]
    translation_vector = matrix[0:3, 3]
    inverse_rotation = np.transpose(rotation_matrix)
    inverse_translation = -np.dot(inverse_rotation, translation_vector)
    inverse_matrix = np.identity(4)
    inverse_matrix[0:3, 0:3] = inverse_rotation
    inverse_matrix[0:3, 3] = inverse_translation
    return inverse_matrix


def single_or_batch(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        idx = 1 if len(args) > 0 and hasattr(args[0], "__dict__") else 0
        x = np.asarray(args[idx])
        not_batch = x.ndim == 1
        if not_batch:
            x = x.reshape(1, -1)
        ret = func(*args[:idx], x, *args[idx + 1 :], **kwargs)
        return ret[0] if not_batch else ret

    return wrapper


@single_or_batch
def homogenize(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2
    return np.hstack([x, np.ones((x.shape[0], 1))])
