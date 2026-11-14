"""Torch geometry for feedback features; no solver or teacher action inputs."""

import math

import torch


def rotation_matrix(columns):
    """Reconstruct SO(3) from the row-major first two columns in preview_v1."""
    columns = columns.reshape(*columns.shape[:-1], 3, 2)
    first, second = columns.unbind(dim=-1)
    return torch.stack([first, second, torch.linalg.cross(first, second)], dim=-1)


def rotation_vector(matrix):
    """Principal SO(3) logarithm, including small angles and half turns.

    Select the largest quaternion component so the conversion never divides
    by a small trace. The sign at exactly pi is necessarily ambiguous.
    """
    r00, r11, r22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
    r01, r02, r12 = matrix[..., 0, 1], matrix[..., 0, 2], matrix[..., 1, 2]
    r10, r20, r21 = matrix[..., 1, 0], matrix[..., 2, 0], matrix[..., 2, 1]
    squared = torch.stack([1 + r00 + r11 + r22, 1 + r00 - r11 - r22, 1 - r00 + r11 - r22, 1 - r00 - r11 + r22], dim=-1)
    # Keep even unselected candidates differentiable at identity/half turns.
    magnitude = squared.clamp_min(1e-8).sqrt()
    candidates = torch.stack(
        [
            torch.stack([squared[..., 0], r21 - r12, r02 - r20, r10 - r01], dim=-1),
            torch.stack([r21 - r12, squared[..., 1], r10 + r01, r02 + r20], dim=-1),
            torch.stack([r02 - r20, r10 + r01, squared[..., 2], r12 + r21], dim=-1),
            torch.stack([r10 - r01, r20 + r02, r21 + r12, squared[..., 3]], dim=-1),
        ],
        dim=-2,
    ) / (2 * magnitude[..., :, None])
    index = magnitude.argmax(dim=-1)
    quaternion = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 4)).squeeze(-2)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    angle = 2 * torch.atan2(torch.linalg.vector_norm(quaternion[..., 1:], dim=-1), quaternion[..., 0])
    return quaternion[..., 1:] / (0.5 * torch.sinc(angle[..., None] / (2 * math.pi)))
