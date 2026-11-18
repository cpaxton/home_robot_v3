# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import numpy as np
import pytest

from emet.utils.pinhole_intrinsics import chain_pinhole_K_pixel_ops, pinhole_K_from_opengl_projection


def perspective(width=640, height=480):
    scale = 1 / np.tan(np.deg2rad(60))
    return np.array([[scale, 0, 0, 0], [0, scale * width / height, 0, 0], [0, 0, -1.01, -0.2], [0, 0, -1, 0]])


@pytest.mark.parametrize("shift", [(0, 0), (0.1, -0.05)])
def test_intrinsics_project_the_same_rays_as_opengl_raster_pixel_centers(shift):
    projection = perspective()
    projection[0, 2], projection[1, 2] = shift
    k = pinhole_K_from_opengl_projection(projection, 640, 480)
    gl = np.array([[-0.8, 0.4, -2, 1], [0.6, -0.8, -3, 1], [0, 0, -1, 1]])
    clip = gl @ projection.T
    ndc = clip[:, :2] / clip[:, 3:4]
    raster = (ndc * [1, -1] + 1) * [320, 240] - 0.5
    cv = gl[:, :3] * [1, -1, -1]
    pixels = cv @ k.T
    np.testing.assert_allclose(pixels[:, :2] / pixels[:, 2:3], raster)


def test_centered_projection_is_invariant_under_vertical_pixel_flip():
    k = pinhole_K_from_opengl_projection(perspective(), 640, 480)
    np.testing.assert_allclose(k[:2, 2], [319.5, 239.5])
    flipped, _, _ = chain_pinhole_K_pixel_ops(k, 480, 640, ["flipud"])
    assert flipped[1, 2] == k[1, 2]


def test_native_integer_viewport_unprojection_y_flip_encloses_raster_centres():
    projection = perspective()
    k = pinhole_K_from_opengl_projection(projection, 640, 480)

    def native_ray(u, v):
        # RenderCamera::unproject uses (u, height-v-1), then NDC.
        ndc = 2 * np.array([u, 480 - v - 1]) / [640, 480] - 1
        return np.array([ndc[0] / projection[0, 0], ndc[1] / projection[1, 1], -1])

    for u, v in ((0, 1), (319, 239), (638, 478)):
        midpoint = (native_ray(u, v) + native_ray(u + 1, v - 1)) / 2
        np.testing.assert_allclose(midpoint * [1, -1, -1], np.linalg.inv(k) @ [u, v, 1])


@pytest.mark.parametrize("bad", [np.eye(4), np.zeros((3, 3)), perspective().T, perspective() * np.nan])
def test_invalid_or_orthographic_projection_cannot_masquerade_as_pinhole(bad):
    with pytest.raises(ValueError, match="perspective"):
        pinhole_K_from_opengl_projection(bad, 640, 480)
