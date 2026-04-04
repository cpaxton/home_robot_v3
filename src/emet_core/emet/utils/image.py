# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
# Slim image utils for emet-core (no torch): Camera, scale_camera_matrix, adjust_gamma.


import cv2
import numpy as np


def compute_pinhole_K(width, height, fov_degrees) -> np.ndarray:
    """Create a simple pinhole camera given minimal information only. Fov is in degrees."""
    horizontal_fov_rad = np.radians(fov_degrees)
    h_focal_length = width / (2 * np.tan(horizontal_fov_rad / 2))
    v_focal_length = width / (2 * np.tan(horizontal_fov_rad / 2) * float(height) / width)
    principal_point_x = (width - 1.0) / 2
    principal_point_y = (height - 1.0) / 2
    K = np.array([[v_focal_length, 0, principal_point_x], [0, h_focal_length, principal_point_y], [0, 0, 1]])
    return K


class Camera:
    """Simple pinhole camera model (numpy/cv2 only)."""

    @staticmethod
    def from_width_height_fov(
        width: float,
        height: float,
        fov_degrees: float,
        near_val: float = 0.1,
        far_val: float = 4.0,
    ):
        horizontal_fov_rad = np.radians(fov_degrees)
        h_focal_length = width / (2 * np.tan(horizontal_fov_rad / 2))
        v_focal_length = width / (2 * np.tan(horizontal_fov_rad / 2) * float(height) / width)
        principal_point_x = (width - 1.0) / 2
        principal_point_y = (height - 1.0) / 2
        return Camera(
            (0, 0, 0),
            (0, 0, 0, 1),
            height,
            width,
            v_focal_length,
            h_focal_length,
            principal_point_x,
            principal_point_y,
            near_val,
            far_val,
            np.eye(4),
            None,
            None,
            horizontal_fov_rad,
        )

    @staticmethod
    def from_K(K: np.ndarray, width: float, height: float):
        assert K.shape == (3, 3)
        return Camera(
            (0, 0, 0),
            (0, 0, 0, 1),
            height,
            width,
            K[0, 0],
            K[1, 1],
            K[0, 2],
            K[1, 2],
            0,
            5,
            np.eye(4),
            None,
            None,
            None,
        )

    def __init__(
        self,
        pos,
        orn,
        height,
        width,
        fx,
        fy,
        px,
        py,
        near_val,
        far_val,
        pose_matrix,
        proj_matrix,
        view_matrix,
        fov,
        *args,
        **kwargs,
    ):
        self.pos = pos
        self.orn = orn
        self.height = height
        self.width = width
        self.px = px
        self.py = py
        self.fov = fov
        self.near_val = near_val
        self.far_val = far_val
        self.fx = fx
        self.fy = fy
        self.pose_matrix = pose_matrix
        self.proj_matrix = proj_matrix
        self.view_matrix = view_matrix
        self.max_depth = far_val
        self.K = np.array([[self.fy, 0, self.px], [0, self.fy, self.py], [0, 0, 1]])

    def to_dict(self):
        return {
            "pos": self.pos,
            "orn": self.orn,
            "height": self.height,
            "width": self.width,
            "near_val": self.near_val,
            "far_val": self.far_val,
            "proj_matrix": self.proj_matrix,
            "view_matrix": self.view_matrix,
            "max_depth": self.max_depth,
            "pose_matrix": self.pose_matrix,
            "px": self.px,
            "py": self.py,
            "fx": self.fx,
            "fy": self.fy,
            "fov": self.fov,
        }

    def get_pose(self):
        return self.pose_matrix.copy()

    def depth_to_xyz(self, depth, data_type: type = np.float16):
        indices = np.indices((self.height, self.width), dtype=np.float32).transpose(1, 2, 0)
        z = depth
        x = (indices[:, :, 1] - self.px) * (z / self.fx)
        y = (indices[:, :, 0] - self.py) * (z / self.fy)
        xyz = np.stack([x, y, z], axis=-1).astype(data_type)
        return xyz

    def fix_depth(self, depth):
        depth = np.asarray(depth).copy()
        depth[depth > self.far_val] = 0
        depth[depth < self.near_val] = 0
        return depth


def camera_xyz_to_global_xyz(camera_xyz, camera_pose):
    height, width, _ = camera_xyz.shape
    camera_xyz_flat = camera_xyz.reshape(-1, 3)
    ones = np.ones((camera_xyz_flat.shape[0], 1))
    camera_homogeneous = np.hstack((camera_xyz_flat, ones))
    global_homogeneous = (camera_pose @ camera_homogeneous.T).T
    global_xyz_flat = global_homogeneous[:, :3] / global_homogeneous[:, 3:4]
    return global_xyz_flat.reshape(height, width, 3)


def adjust_gamma(image: np.ndarray, gamma: float = 1.0):
    """Gamma correction using a lookup table."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def scale_camera_matrix(K: np.ndarray, scale_factor: float) -> np.ndarray:
    """Modify camera matrix K when shrinking an image by scale_factor (0 < scale_factor <= 1)."""
    if not 0 < scale_factor <= 1:
        raise ValueError("Scale factor must be between 0 and 1")
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_factor
    K_scaled[1, 1] *= scale_factor
    K_scaled[0, 2] *= scale_factor
    K_scaled[1, 2] *= scale_factor
    return K_scaled
