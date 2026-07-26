# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""ScanNet replay: posed ``.sens`` RGB-D with Open3D mesh fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from emet.benchmarks.sqa3d.datasets import SQA3DQuestion
from emet.benchmarks.sqa3d.scannet.config import scene_sens_path
from emet.benchmarks.sqa3d.scannet.mesh import load_scannet_mesh
from emet.benchmarks.sqa3d.scannet.pose import agent_pose_matrix, apply_forward, apply_turn
from emet.benchmarks.sqa3d.scannet.render import configure_scene_lighting, mesh_material

ScanNetReplayMode = Literal["auto", "sens", "mesh"]


@dataclass
class ScanNetFrame:
    rgb: np.ndarray
    depth: np.ndarray
    position: np.ndarray
    quat_xyzw: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray | None = None
    sens_frame_index: int | None = None
    sens_match_xy_m: float | None = None
    replay_source: str = "mesh"


class ScanNetEQASimulator:
    """RGB-D rendering from ScanNet ``_vh_clean_2.ply`` at SQA3D agent poses."""

    def __init__(
        self,
        scene_id: str,
        *,
        scannet_root: Path | None = None,
        sensor_height: float = 1.5,
        image_width: int = 960,
        image_height: int = 720,
        hfov_deg: float = 79.0,
        camera_tilt_deg: float = -15.0,
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
        configure_scene_lighting(self._renderer)
        self._renderer.scene.add_geometry("scannet_mesh", self._mesh, mesh_material(self._mesh))

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

    def capture_rotate_views(self, n_views: int = 12) -> list[np.ndarray]:
        """RGB frames at the current XY position, evenly spaced in yaw (returns to start pose)."""
        n_views = max(1, int(n_views))
        saved_pos = self._position.copy()
        saved_quat = self._quat_xyzw.copy()
        frames: list[np.ndarray] = []
        frames.append(self.get_frame().rgb.copy())
        if n_views > 1:
            step_rad = 2.0 * np.pi / float(n_views)
            for _ in range(n_views - 1):
                self._quat_xyzw = apply_turn(self._quat_xyzw, step_rad)
                self._render_camera()
                frames.append(self.get_frame().rgb.copy())
        self._position = saved_pos
        self._quat_xyzw = saved_quat
        self._render_camera()
        return frames

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


class ScanNetReplaySimulator:
    """Posed ScanNet ``.sens`` RGB-D near the agent; mesh Open3D when navigated away."""

    def __init__(
        self,
        scene_id: str,
        *,
        scannet_root: Path | None = None,
        replay_mode: ScanNetReplayMode = "auto",
        sens_xy_radius_m: float = 0.75,
        sens_match_max_xy_m: float = 0.75,
        **mesh_kwargs,
    ):
        self.scene_id = scene_id
        self._replay_mode = replay_mode
        self._sens_xy_radius_m = float(sens_xy_radius_m)
        self._sens_match_max_xy_m = float(sens_match_max_xy_m)
        # Validate .sens before Open3D OffscreenRenderer — missing GL/EGL SIGSEGVs
        # the interpreter instead of raising, which kills the whole pytest process.
        sens_path = scene_sens_path(scene_id, scannet_root)
        if replay_mode == "sens" and not sens_path.is_file():
            raise FileNotFoundError(
                f"ScanNet .sens not found for {scene_id}: {sens_path}\n"
                "Download: uv run python scripts/download_scannet_data.py --accept-tos "
                f"--scene {scene_id} --with-sens"
            )
        self._mesh = ScanNetEQASimulator(scene_id, scannet_root=scannet_root, **mesh_kwargs)
        self._sens = None
        if replay_mode in ("auto", "sens") and sens_path.is_file():
            from emet.benchmarks.sqa3d.scannet.sens import ScanNetSensLoader

            self._sens = ScanNetSensLoader(sens_path)
        self._anchor_xy: np.ndarray | None = None
        self._anchor_replay_backend = "mesh"
        self._anchor_sens_frame_index: int | None = None
        self._anchor_sens_match_xy_m: float | None = None

    @property
    def replay_backend(self) -> str:
        return self._anchor_replay_backend

    @property
    def anchor_sens_frame_index(self) -> int | None:
        return self._anchor_sens_frame_index

    @property
    def anchor_sens_match_xy_m(self) -> float | None:
        return self._anchor_sens_match_xy_m

    @property
    def sensor_height(self) -> float:
        return self._mesh.sensor_height

    @property
    def camera_tilt_deg(self) -> float:
        return self._mesh.camera_tilt_deg

    @property
    def step_count(self) -> int:
        return self._mesh.step_count

    def _sens_match_at_current_pose(self):
        if self._sens is None:
            return None
        return self._sens.nearest_frame_match(
            self._mesh._position,
            self._mesh._quat_xyzw,
            sensor_height=self._mesh.sensor_height,
            camera_tilt_deg=self._mesh.camera_tilt_deg,
            xy_radius_m=self._sens_xy_radius_m,
        )

    def _sens_usable(self, match) -> bool:
        if match is None:
            return False
        if self._replay_mode == "auto" and match.xy_m > self._sens_match_max_xy_m:
            return False
        return True

    def _prefer_sens(self) -> bool:
        if self._sens is None or self._replay_mode == "mesh":
            return False
        if self._anchor_xy is not None:
            pos = self._mesh._position
            xy_dist = float(np.linalg.norm(pos[:2] - self._anchor_xy))
            if xy_dist > self._sens_xy_radius_m:
                return False
        return self._sens_usable(self._sens_match_at_current_pose())

    def _update_anchor_replay_info(self) -> None:
        if self._sens is None or self._replay_mode == "mesh":
            self._anchor_replay_backend = "mesh"
            self._anchor_sens_frame_index = None
            self._anchor_sens_match_xy_m = None
            return
        match = self._sens_match_at_current_pose()
        if match is None or not self._sens_usable(match):
            self._anchor_replay_backend = "mesh"
            self._anchor_sens_frame_index = None
            self._anchor_sens_match_xy_m = float(match.xy_m) if match is not None else None
            return
        self._anchor_replay_backend = "sens"
        self._anchor_sens_frame_index = int(match.frame_index)
        self._anchor_sens_match_xy_m = float(match.xy_m)

    def set_sqa3d_pose(self, question: SQA3DQuestion) -> None:
        self._mesh.set_sqa3d_pose(question)
        self._anchor_xy = np.asarray(question.position[:2], dtype=np.float64)
        self._update_anchor_replay_info()

    def set_pose(
        self,
        position: tuple[float, float, float] | np.ndarray,
        quat_xyzw: tuple[float, float, float, float] | np.ndarray,
    ) -> None:
        self._mesh.set_pose(position, quat_xyzw)

    def get_frame(self) -> ScanNetFrame:
        if self._prefer_sens():
            frame = self._sens.frame_for_agent_pose(
                self._mesh._position,
                self._mesh._quat_xyzw,
                sensor_height=self._mesh.sensor_height,
                camera_tilt_deg=self._mesh.camera_tilt_deg,
                xy_radius_m=self._sens_xy_radius_m,
                output_size=(self._mesh._width, self._mesh._height),
            )
            frame.replay_source = "sens"
            return frame
        frame = self._mesh.get_frame()
        frame.replay_source = "mesh"
        return frame

    def capture_rotate_views(self, n_views: int = 12) -> list[np.ndarray]:
        if self._prefer_sens():
            return self._sens.multiview_rgb_at_agent(
                self._mesh._position,
                self._mesh._quat_xyzw,
                sensor_height=self._mesh.sensor_height,
                camera_tilt_deg=self._mesh.camera_tilt_deg,
                n_views=n_views,
                xy_radius_m=max(self._sens_xy_radius_m, 1.5),
            )
        return self._mesh.capture_rotate_views(n_views=n_views)

    def step(self, action: str) -> ScanNetFrame:
        self._mesh.step(action)
        return self.get_frame()

    def close(self) -> None:
        self._mesh.close()


def create_scannet_simulator(
    scene_id: str,
    *,
    scannet_root: Path | None = None,
    replay_mode: ScanNetReplayMode = "auto",
    **kwargs,
) -> ScanNetReplaySimulator:
    return ScanNetReplaySimulator(scene_id, scannet_root=scannet_root, replay_mode=replay_mode, **kwargs)
