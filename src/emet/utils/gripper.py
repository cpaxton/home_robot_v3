from __future__ import annotations

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.
from collections.abc import Sequence

import cv2
import numpy as np


def get_gripper_aruco_detector() -> cv2.aruco.ArucoDetector:
    """Create an aruco detector preconfigured for the gripper AR markers to make it easier to track them."""
    aruco_parameters = cv2.aruco.DetectorParameters()
    aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
    return aruco_detector


def detect_aruco_markers(
    image: np.ndarray, aruco_detector: cv2.aruco.ArucoDetector
) -> tuple[Sequence[np.ndarray], np.ndarray]:
    """Detect AR markers in an image."""
    corners, ids, _ = aruco_detector.detectMarkers(image)
    return corners, ids


class GripperArucoDetector:
    def __init__(self):
        self.aruco_detector = get_gripper_aruco_detector()

    def detect_aruco_markers(self, image: np.ndarray) -> tuple[Sequence[np.ndarray], np.ndarray]:
        """Detect AR markers in an image.

        Args:
            image: The image to detect markers in.

        Returns:
            A tuple of two numpy arrays. The first array contains the corners of the detected markers, and the second
            array contains the IDs of the detected markers.
        """
        return detect_aruco_markers(image, self.aruco_detector)

    def detect_and_draw_aruco_markers(self, image: np.ndarray) -> tuple[Sequence[np.ndarray], np.ndarray, np.ndarray]:
        """Detect AR markers in an image and draw them.

        Args:
            image: The image to detect markers in.

        Returns:
            A tuple of two numpy arrays. The first array contains the corners of the detected markers, and the second
            array contains the IDs of the detected markers.
        """
        corners, ids = self.detect_aruco_markers(image)
        image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        return corners, ids, image

    def detect_aruco_centers(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect AR markers in an image and return their centers.

        Args:
            image: The image to detect markers in.

        Returns:
            A tuple of two numpy arrays. The first array contains the centers of the detected markers, and the second
            array contains the IDs of the detected markers.
        """
        corners, ids = self.detect_aruco_markers(image)
        centers = np.array([np.mean(c, axis=1) for c in corners])
        return centers, ids

    def detect_center(self, image: np.ndarray) -> np.ndarray | None:
        """Get the center of the first detected AR marker in an image.

        Args:
            image: The image to detect the marker in.

        Returns:
            center: 2D array, The center point between the two finger AR markers.
        """
        centers, _ = self.detect_aruco_centers(image)
        if len(centers) < 2:
            return None
        center = (centers[0] + centers[1]) / 2
        return center[0]


def measure_observed_aperture(servo, target_points, detector):
    """Return finger-marker span and observed object extent along that axis.

    Marker centers are not the inner jaw surfaces. A caller must retain a
    conservative margin for finger thickness and unobserved object geometry.
    Never infer a metric opening from an uncalibrated gripper command value.
    """
    centers, ids = detector.detect_aruco_centers(servo.ee_rgb)
    if ids is None or servo.ee_depth is None or servo.ee_camera_K is None or servo.ee_camera_pose is None:
        raise ValueError("Calibrated finger-marker RGB-D required")
    ids = np.asarray(ids).reshape(-1)
    centers = np.asarray(centers).reshape(-1, 2)
    K = np.asarray(servo.ee_camera_K)
    pose = np.asarray(servo.ee_camera_pose)
    if K.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(K).all() or not np.isfinite(pose).all():
        raise ValueError("Invalid aperture calibration")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid aperture focal length")
    height, width = servo.ee_depth.shape
    fingers = []
    for marker in (200, 201):  # Stretch's left/right finger marker IDs.
        selected = np.flatnonzero(ids == marker)
        if len(selected) != 1:
            raise ValueError("Both distinct finger markers are required")
        x, y = centers[selected[0]]
        if not np.isfinite([x, y]).all() or not (2 <= x < width - 2 and 2 <= y < height - 2):
            raise ValueError("Finger marker clipped at image edge")
        ix, iy = int(round(x)), int(round(y))
        depth = servo.ee_depth[iy - 2 : iy + 3, ix - 2 : ix + 3]
        depth = depth[np.isfinite(depth) & (depth > 0)]
        if len(depth) < 10:
            raise ValueError("Insufficient depth on finger marker")
        z = float(np.median(depth))
        fingers.append([(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z])
    fingers = np.asarray(fingers)
    span = float(np.linalg.norm(fingers[1] - fingers[0]))
    if not 0.02 < span < 0.3:
        raise ValueError("Implausible observed finger-marker span")
    points = np.asarray(target_points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Finite object geometry required for aperture")
    camera_points = (points - pose[:3, 3]) @ pose[:3, :3]
    if camera_points[:, 2].min() - fingers[:, 2].max() < 0.06:
        raise ValueError("Object is too close to adjust aperture safely")
    axis = (fingers[1] - fingers[0]) / span
    extent = float(np.ptp(camera_points @ axis))
    if extent <= 0:
        raise ValueError("Object has no measured extent along the finger axis")
    return span, extent
