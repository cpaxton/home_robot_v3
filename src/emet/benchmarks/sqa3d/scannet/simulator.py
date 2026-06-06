# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Open3D offscreen RGB-D simulator over ScanNet meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from emet.benchmarks.sqa3d.datasets import SQA3DQuestion
from emet.benchmarks.sqa3d.scannet.mesh import load_scannet_mesh
from emet.benchmarks.sqa3d.scannet.pose import agent_pose_matrix, apply_forward, apply_turn


@dataclass
class ScanNetFrame:
    rgb: np.ndarray
    depth: np.ndarray
    position: np.ndarray
    quat_xyzw: np.ndarray
    intrinsics: np.ndarray


class ScanNetEQASimulator:
    """RGB-D rendering from ScanNet ``_vh_clean_2.ply`` at SQA3D agent poses."""

    def __init__(
        self,
        scene_id: str,
        *,
        scannet_root: Path | None = None,
        sensor_height: float = 1.5,
        image_width: int = 640,
        image_height: int = 480,
        hfov_deg: float = 90.0,
        camera_tilt_deg: float = -30.0,
        turn_deg: float = 10.0,
        forward_m: float = 0.25,
    ):
        self.scene_id = scene_id
        self._sensor_height = float(sensor_height)
        self._camera_tilt_deg = float(camera_tilt_deg)
        self._turn_rad = np.deg2rad(turn_deg)
        self._forward_m = float(forward_m)
        self._width = int(image_width)
        self._height = int(image_height)
        self._hfov = np.deg2rad(hfov_deg)
        self._step_count = 0

        self._mesh = load_scannet_mesh(scene_id, scannet_root)
        self._renderer = o3d.visualization.rendering.OffscreenRenderer(self._width, self._height)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self._renderer.scene.set_background([0.05, 0.05, 0.08, 1.0])
        self._renderer.scene.add_geometry("scannet_mesh", self._mesh, mat)

        self._position = np.zeros(3, dtype=np.float64)
        self._quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    @property
    def intrinsics(self) -> np.ndarray:
        fx = 0.5 * self._width / np.tan(self._hfov / 2.0)
        fy = fx
        cx = self._width / 2.0
        cy = self._height / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    @property
    def sensor_height(self) -> float:
        return self._sensor_height

    @property
    def camera_tilt_deg(self) -> float:
        return self._camera_tilt_deg

    def set_sqa3d_pose(self, question: SQA3DQuestion) -> None:
        self._position = np.asarray(question.position, dtype=np.float64)
        self._quat_xyzw = np.asarray(question.rotation_xyzw, dtype=np.float64)
        self._render_camera()

    def set_pose(
        self,
        position: tuple[float, float, float] | np.ndarray,
        quat_xyzw: tuple[float, float, float, float] | np.ndarray,
    ) -> None:
        self._position = np.asarray(position, dtype=np.float64)
        self._quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
        self._render_camera()

    def _camera_eye_center_up(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        body = agent_pose_matrix(self._position, self._quat_xyzw)
        cam_offset = np.eye(4, dtype=np.float64)
        cam_offset[2, 3] = self._sensor_height
        tilt = np.eye(4, dtype=np.float64)
        tilt[:3, :3] = R.from_euler("y", np.deg2rad(self._camera_tilt_deg)).as_matrix()
        T = body @ cam_offset @ tilt
        eye = T[:3, 3]
        forward = T[:3, 0]
        up = T[:3, 2]
        center = eye + forward * 2.0
        return eye, center, up

    def _render_camera(self) -> None:
        eye, center, up = self._camera_eye_center_up()
        self._renderer.setup_camera(
            float(np.rad2deg(self._hfov)),
            center.tolist(),
            eye.tolist(),
            up.tolist(),
        )

    def get_frame(self) -> ScanNetFrame:
        rgb = np.asarray(self._renderer.render_to_image())
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        depth = np.asarray(self._renderer.render_to_depth_image(z_in_view_space=True))
        return ScanNetFrame(
            rgb=rgb,
            depth=depth,
            position=self._position.copy(),
            quat_xyzw=self._quat_xyzw.copy(),
            intrinsics=self.intrinsics,
        )

    def step(self, action: str) -> ScanNetFrame:
        if action == "turn_left":
            self._quat_xyzw = apply_turn(self._quat_xyzw, self._turn_rad)
        elif action == "turn_right":
            self._quat_xyzw = apply_turn(self._quat_xyzw, -self._turn_rad)
        elif action == "move_forward":
            self._position = apply_forward(self._position, self._quat_xyzw, self._forward_m)
        else:
            raise ValueError(f"Unknown action {action!r}")
        self._step_count += 1
        self._render_camera()
        return self.get_frame()

    @property
    def step_count(self) -> int:
        return self._step_count

    def close(self) -> None:
        self._renderer = None  # type: ignore[assignment]

    @classmethod
    def from_scene_id(cls, scene_id: str, scannet_root: Path | None = None, **kwargs) -> ScanNetEQASimulator:
        return cls(scene_id, scannet_root=scannet_root, **kwargs)
