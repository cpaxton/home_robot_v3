# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""ScanNet posed RGB-D from ``.sens`` recordings."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from emet.benchmarks.sqa3d.scannet.pose import agent_pose_matrix
from emet.benchmarks.sqa3d.scannet.sensor_data import ScanNetSensorData
from emet.benchmarks.sqa3d.scannet.simulator import ScanNetFrame


_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def scannet_camera_to_opencv_world_to_camera(camera_to_world: np.ndarray) -> np.ndarray:
    """ScanNet OpenGL camera-to-world → OpenCV world-to-camera (4×4)."""
    w2c_gl = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    return _GL_TO_CV @ w2c_gl


def scannet_camera_to_opencv_camera_to_world(camera_to_world: np.ndarray) -> np.ndarray:
    """ScanNet OpenGL camera-to-world → emet OpenCV camera-to-world (4×4)."""
    c2w_gl = np.asarray(camera_to_world, dtype=np.float64)
    return c2w_gl @ _GL_TO_CV


def target_camera_center(
    position: np.ndarray,
    quat_xyzw: np.ndarray,
    *,
    sensor_height: float,
    camera_tilt_deg: float,
) -> np.ndarray:
    cam_offset = np.eye(4, dtype=np.float64)
    cam_offset[2, 3] = float(sensor_height)
    tilt = np.eye(4, dtype=np.float64)
    tilt[:3, :3] = R.from_euler("y", np.deg2rad(camera_tilt_deg)).as_matrix()
    T = agent_pose_matrix(position, quat_xyzw) @ cam_offset @ tilt
    return T[:3, 3]


def target_camera_forward_xy(
    position: np.ndarray,
    quat_xyzw: np.ndarray,
    *,
    sensor_height: float,
    camera_tilt_deg: float,
) -> np.ndarray:
    cam_offset = np.eye(4, dtype=np.float64)
    cam_offset[2, 3] = float(sensor_height)
    tilt = np.eye(4, dtype=np.float64)
    tilt[:3, :3] = R.from_euler("y", np.deg2rad(camera_tilt_deg)).as_matrix()
    T = agent_pose_matrix(position, quat_xyzw) @ cam_offset @ tilt
    fwd = np.asarray(T[:3, 0], dtype=np.float64)
    fwd[2] = 0.0
    norm = float(np.linalg.norm(fwd[:2]))
    if norm < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float64)
    fwd[:2] /= norm
    return fwd[:2]


def _camera_forward_xy(camera_to_world: np.ndarray) -> np.ndarray:
    # ScanNet OpenGL camera looks down -Z in camera frame → world -R[:,2].
    fwd = -np.asarray(camera_to_world[:3, 2], dtype=np.float64)
    fwd[2] = 0.0
    norm = float(np.linalg.norm(fwd[:2]))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float64)
    fwd[:2] /= norm
    return fwd[:2]


def _valid_sens_frame_mask(frames: list) -> np.ndarray:
    mask = np.ones(len(frames), dtype=bool)
    for i, frame in enumerate(frames):
        c2w = np.asarray(frame.camera_to_world, dtype=np.float64)
        if not np.isfinite(c2w).all():
            mask[i] = False
            continue
        if not np.isfinite(c2w[:3, 3]).all():
            mask[i] = False
    return mask


def _resize_rgb_depth_intrinsics(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_w, out_h = int(width), int(height)
    in_h, in_w = rgb.shape[:2]
    if in_w == out_w and in_h == out_h:
        return rgb, depth, intrinsics
    rgb_out = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    depth_out = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    k = np.asarray(intrinsics, dtype=np.float64).copy()
    sx = out_w / float(in_w)
    sy = out_h / float(in_h)
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] *= sx
    k[1, 2] *= sy
    return rgb_out, depth_out, k


def _yaw_distance_rad(a: np.ndarray, b: np.ndarray) -> float:
    cross = float(a[0] * b[1] - a[1] * b[0])
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return abs(float(np.arctan2(cross, dot)))


class ScanNetSensLoader:
    """Lazy posed RGB-D lookup from a ScanNet ``.sens`` file."""

    def __init__(self, sens_path: Path | str):
        self._path = Path(sens_path)
        self._data = ScanNetSensorData.load(self._path)
        self._valid_mask = _valid_sens_frame_mask(self._data.frames)
        if not np.any(self._valid_mask):
            raise ValueError(f"No valid posed frames in {self._path}")
        self._cam_centers = np.stack([f.camera_to_world[:3, 3] for f in self._data.frames], axis=0)
        self._cam_forward_xy = np.stack([_camera_forward_xy(f.camera_to_world) for f in self._data.frames], axis=0)

    @property
    def n_frames(self) -> int:
        return len(self._data.frames)

    @property
    def depth_shift(self) -> float:
        return float(self._data.depth_shift)

    def nearest_frame_index(
        self,
        position: np.ndarray,
        quat_xyzw: np.ndarray,
        *,
        sensor_height: float,
        camera_tilt_deg: float,
        xy_radius_m: float | None = None,
    ) -> int:
        target = target_camera_center(position, quat_xyzw, sensor_height=sensor_height, camera_tilt_deg=camera_tilt_deg)
        target_fwd = target_camera_forward_xy(
            position, quat_xyzw, sensor_height=sensor_height, camera_tilt_deg=camera_tilt_deg
        )
        deltas = self._cam_centers - target.reshape(1, 3)
        pos_err = np.linalg.norm(deltas, axis=1)
        if xy_radius_m is not None:
            xy_err = np.linalg.norm(deltas[:, :2], axis=1)
            mask = xy_err <= float(xy_radius_m)
            if np.any(mask):
                pos_err = np.where(mask, pos_err, np.inf)
        yaw_err = np.array(
            [_yaw_distance_rad(self._cam_forward_xy[i], target_fwd) for i in range(len(self._data.frames))],
            dtype=np.float64,
        )
        score = pos_err + 0.35 * yaw_err
        score[~self._valid_mask] = np.inf
        score = np.where(np.isfinite(score), score, np.inf)
        valid_idx = np.flatnonzero(self._valid_mask)
        return int(valid_idx[int(np.argmin(score[valid_idx]))])

    def frame_at_index(
        self,
        index: int,
        *,
        position: np.ndarray,
        quat_xyzw: np.ndarray,
        output_size: tuple[int, int] | None = None,
    ) -> ScanNetFrame:
        index = int(index)
        if not self._valid_mask[index]:
            raise ValueError(f"Invalid ScanNet sens frame index {index} in {self._path}")
        frame = self._data.frames[index]
        rgb = frame.decompress_color()
        depth_u16 = frame.decompress_depth()
        if depth_u16.shape != rgb.shape[:2]:
            depth_u16 = cv2.resize(
                depth_u16,
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        depth_m = depth_u16.astype(np.float32) / max(float(frame.depth_shift), 1e-6)
        K = np.asarray(self._data.intrinsic_color[:3, :3], dtype=np.float64)
        if output_size is not None:
            rgb, depth_m, K = _resize_rgb_depth_intrinsics(
                rgb,
                depth_m,
                K,
                width=output_size[0],
                height=output_size[1],
            )
        return ScanNetFrame(
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=depth_m,
            position=np.asarray(position, dtype=np.float64).copy(),
            quat_xyzw=np.asarray(quat_xyzw, dtype=np.float64).copy(),
            intrinsics=K,
            camera_to_world=np.asarray(frame.camera_to_world, dtype=np.float64),
            sens_frame_index=int(index),
            replay_source="sens",
        )

    def frame_for_agent_pose(
        self,
        position: np.ndarray,
        quat_xyzw: np.ndarray,
        *,
        sensor_height: float,
        camera_tilt_deg: float,
        xy_radius_m: float | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> ScanNetFrame:
        idx = self.nearest_frame_index(
            position,
            quat_xyzw,
            sensor_height=sensor_height,
            camera_tilt_deg=camera_tilt_deg,
            xy_radius_m=xy_radius_m,
        )
        return self.frame_at_index(
            idx,
            position=position,
            quat_xyzw=quat_xyzw,
            output_size=output_size,
        )

    def multiview_rgb_at_agent(
        self,
        position: np.ndarray,
        quat_xyzw: np.ndarray,
        *,
        sensor_height: float,
        camera_tilt_deg: float,
        n_views: int,
        xy_radius_m: float = 1.5,
    ) -> list[np.ndarray]:
        """Pick ``n_views`` real frames near agent XY with evenly spaced yaw."""
        n_views = max(1, int(n_views))
        base_yaw = float(
            np.arctan2(
                target_camera_forward_xy(position, quat_xyzw, sensor_height=sensor_height, camera_tilt_deg=camera_tilt_deg)[1],
                target_camera_forward_xy(position, quat_xyzw, sensor_height=sensor_height, camera_tilt_deg=camera_tilt_deg)[0],
            )
        )
        xy = np.asarray(position[:2], dtype=np.float64)
        xy_dists = np.linalg.norm(self._cam_centers[:, :2] - xy.reshape(1, 2), axis=1)
        pool = np.where((xy_dists <= float(xy_radius_m)) & self._valid_mask)[0]
        if pool.size == 0:
            pool = np.flatnonzero(self._valid_mask)

        chosen: list[int] = []
        for i in range(n_views):
            target_yaw = base_yaw + (2.0 * np.pi * i) / float(n_views)
            target_fwd = np.array([np.cos(target_yaw), np.sin(target_yaw)], dtype=np.float64)
            best = int(pool[0])
            best_score = float("inf")
            for idx in pool:
                if idx in chosen:
                    continue
                yaw_err = _yaw_distance_rad(self._cam_forward_xy[idx], target_fwd)
                score = float(xy_dists[idx]) + 0.5 * yaw_err
                if not np.isfinite(score):
                    continue
                if score < best_score:
                    best_score = score
                    best = int(idx)
            chosen.append(best)
        return [self.frame_at_index(i, position=position, quat_xyzw=quat_xyzw).rgb for i in chosen]
