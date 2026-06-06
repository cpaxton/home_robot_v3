# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Minimal Habitat-Sim environment for HM-EQA episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emet.habitat.config import default_hm3d_scene_dir, hm3d_scene_glb_path, hm3d_scene_navmesh_path
from emet.habitat.datasets import SceneInitPose
from emet.habitat.hm3d_semantics import (
    Hm3dSemanticLabeler,
    hm3d_annotated_scene_dataset_config,
    hm3d_semantic_glb_for_basis,
)


@dataclass
class HabitatFrame:
    rgb: np.ndarray
    depth: np.ndarray
    agent_state: object
    intrinsics: np.ndarray
    semantic: np.ndarray | None = None


class HabitatEQASimulator:
    """RGB-D (+ optional HM3D semantic) Habitat-Sim wrapper."""

    def __init__(
        self,
        scene_glb: Path,
        *,
        sensor_height: float = 1.5,
        image_width: int = 640,
        image_height: int = 480,
        hfov_deg: float = 120.0,
        camera_tilt_deg: float = -30.0,
        use_hm3d_semantics: bool | None = None,
        scene_id: str | None = None,
        hm3d_train_root: Path | None = None,
    ):
        import habitat_sim
        import habitat_sim.agent
        import magnum as mn

        if not scene_glb.is_file():
            raise FileNotFoundError(f"HM3D scene not found: {scene_glb}")

        self._scene_id = scene_id or scene_glb.parent.name
        train_root = hm3d_train_root or scene_glb.parent.parent
        hm3d_data_root = train_root.parent.parent.parent
        split = train_root.name
        has_semantic_assets = hm3d_semantic_glb_for_basis(scene_glb).is_file()
        want_semantics = has_semantic_assets if use_hm3d_semantics is None else bool(use_hm3d_semantics)
        annotated_cfg = hm3d_annotated_scene_dataset_config(hm3d_data_root, split=split)
        self._use_semantics = want_semantics and annotated_cfg is not None

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_glb)
        sim_cfg.enable_physics = False
        if self._use_semantics and annotated_cfg is not None:
            sim_cfg.scene_dataset_config_file = str(annotated_cfg)

        sensor_pos = mn.Vector3(0.0, sensor_height, 0.0)

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [image_height, image_width]
        rgb_spec.position = sensor_pos
        rgb_spec.hfov = hfov_deg

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [image_height, image_width]
        depth_spec.position = sensor_pos
        depth_spec.hfov = hfov_deg

        sensor_specs = [rgb_spec, depth_spec]
        if self._use_semantics:
            sem_spec = habitat_sim.CameraSensorSpec()
            sem_spec.uuid = "semantic_sensor"
            sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            sem_spec.resolution = [image_height, image_width]
            sem_spec.position = sensor_pos
            sem_spec.hfov = hfov_deg
            sensor_specs.append(sem_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self._sim = habitat_sim.Simulator(cfg)
        self._agent = self._sim.initialize_agent(0)
        self._load_navmesh(scene_glb)
        self._mn = mn
        self._habitat_sim = habitat_sim
        self._width = image_width
        self._height = image_height
        self._hfov = np.deg2rad(hfov_deg)
        self._sensor_height = sensor_height
        self._camera_tilt_deg = camera_tilt_deg
        self._step_count = 0
        self.semantic_labeler: Hm3dSemanticLabeler | None = None
        if self._use_semantics:
            self.semantic_labeler = Hm3dSemanticLabeler.from_semantic_scene(self._sim.semantic_scene)

    def _load_navmesh(self, scene_glb: Path) -> None:
        navmesh = scene_glb.parent / f"{scene_glb.stem}.navmesh"
        if not navmesh.is_file():
            navmesh = hm3d_scene_navmesh_path(self._scene_id)
        if navmesh.is_file():
            self._sim.pathfinder.load_nav_mesh(str(navmesh))

    @classmethod
    def from_scene_id(
        cls,
        scene_id: str,
        hm3d_root: Path | None = None,
        **kwargs,
    ) -> HabitatEQASimulator:
        train_root = hm3d_root or default_hm3d_scene_dir()
        glb = hm3d_scene_glb_path(scene_id, train_root)
        return cls(glb, scene_id=scene_id, hm3d_train_root=train_root, **kwargs)

    @property
    def uses_hm3d_semantics(self) -> bool:
        return self._use_semantics and self.semantic_labeler is not None

    @property
    def intrinsics(self) -> np.ndarray:
        fx = 0.5 * self._width / np.tan(self._hfov / 2.0)
        fy = fx
        cx = self._width / 2.0
        cy = self._height / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def set_init_pose(self, pose: SceneInitPose) -> None:
        """Place agent at Explore-EQA ``(init_x, init_y, init_z)`` with ``init_angle`` yaw."""
        import habitat_sim.utils.common as hsim_utils

        state = self._agent.get_state()
        position = np.array([pose.x, pose.y, pose.z], dtype=np.float32)
        if self._sim.pathfinder.is_loaded:
            snapped = self._sim.pathfinder.snap_point(position)
            if np.isfinite(snapped).all():
                position = np.asarray(snapped, dtype=np.float32)
        state.position = position
        camera_tilt = np.deg2rad(self._camera_tilt_deg)
        state.rotation = hsim_utils.quat_from_angle_axis(
            float(pose.heading),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ) * hsim_utils.quat_from_angle_axis(
            float(camera_tilt),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        self._agent.set_state(state)

    def find_path_to_xy(self, goal_x: float, goal_z: float) -> np.ndarray | None:
        """Return navmesh shortest-path waypoints in Habitat coords, or None."""
        if not self._sim.pathfinder.is_loaded:
            return None
        import habitat_sim

        start = np.array(self._agent.get_state().position, dtype=np.float32)
        end = np.array([goal_x, start[1], goal_z], dtype=np.float32)
        path = habitat_sim.nav.ShortestPath()
        path.requested_start = start
        path.requested_end = end
        if not self._sim.pathfinder.find_path(path) or len(path.points) < 2:
            return None
        return np.asarray(path.points, dtype=np.float64)

    def _frame_from_obs(self, obs: dict) -> HabitatFrame:
        rgb = obs["color_sensor"][..., :3]
        depth = obs["depth_sensor"]
        semantic = obs.get("semantic_sensor")
        if semantic is not None:
            semantic = np.asarray(semantic)
        return HabitatFrame(
            rgb=rgb,
            depth=depth,
            agent_state=self._agent.get_state(),
            intrinsics=self.intrinsics,
            semantic=semantic,
        )

    def step(self, action: str) -> HabitatFrame:
        self._sim.step(action)
        self._step_count += 1
        return self._frame_from_obs(self._sim.get_sensor_observations())

    def get_frame(self) -> HabitatFrame:
        return self._frame_from_obs(self._sim.get_sensor_observations())

    @property
    def step_count(self) -> int:
        return self._step_count

    def close(self) -> None:
        self._sim.close()
