# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Minimal Habitat-Sim environment for HM-EQA episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emet.habitat.config import hm3d_scene_glb_path
from emet.habitat.datasets import SceneInitPose


@dataclass
class HabitatFrame:
    rgb: np.ndarray
    depth: np.ndarray
    agent_state: object
    intrinsics: np.ndarray


class HabitatEQASimulator:
    """RGB-D Habitat-Sim wrapper with locobot-style navigation state."""

    def __init__(
        self,
        scene_glb: Path,
        *,
        sensor_height: float = 1.25,
        image_width: int = 640,
        image_height: int = 480,
        hfov_deg: float = 90.0,
    ):
        import habitat_sim
        import habitat_sim.agent
        import magnum as mn

        if not scene_glb.is_file():
            raise FileNotFoundError(f"HM3D scene not found: {scene_glb}")

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_glb)
        sim_cfg.enable_physics = False

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [image_height, image_width]
        rgb_spec.position = [0.0, sensor_height, 0.0]
        rgb_spec.hfov = hfov_deg

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [image_height, image_width]
        depth_spec.position = [0.0, sensor_height, 0.0]
        depth_spec.hfov = hfov_deg

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
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
        self._mn = mn
        self._habitat_sim = habitat_sim
        self._width = image_width
        self._height = image_height
        self._hfov = np.deg2rad(hfov_deg)
        self._sensor_height = sensor_height
        self._step_count = 0

    @classmethod
    def from_scene_id(
        cls,
        scene_id: str,
        hm3d_root: Path | None = None,
        **kwargs,
    ) -> HabitatEQASimulator:
        glb = hm3d_scene_glb_path(scene_id, hm3d_root)
        return cls(glb, **kwargs)

    @property
    def intrinsics(self) -> np.ndarray:
        fx = 0.5 * self._width / np.tan(self._hfov / 2.0)
        fy = fx
        cx = self._width / 2.0
        cy = self._height / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def set_init_pose(self, pose: SceneInitPose) -> None:
        state = self._agent.get_state()
        state.position = np.array([pose.x, self._sensor_height, pose.y], dtype=np.float32)
        # Habitat agent forward is -Z; heading about Y
        q = self._mn.Quaternion.rotation(
            self._mn.Deg(-np.rad2deg(pose.heading)),
            np.array([0.0, 1.0, 0.0]),
        )
        state.rotation = np.quaternion(q.scalar, *q.vector)
        self._agent.set_state(state)

    def step(self, action: str) -> HabitatFrame:
        self._sim.step(action)
        obs = self._sim.get_sensor_observations()
        self._step_count += 1
        rgb = obs["color_sensor"][..., :3]
        depth = obs["depth_sensor"]
        return HabitatFrame(
            rgb=rgb,
            depth=depth,
            agent_state=self._agent.get_state(),
            intrinsics=self.intrinsics,
        )

    def get_frame(self) -> HabitatFrame:
        obs = self._sim.get_sensor_observations()
        return HabitatFrame(
            rgb=obs["color_sensor"][..., :3],
            depth=obs["depth_sensor"],
            agent_state=self._agent.get_state(),
            intrinsics=self.intrinsics,
        )

    @property
    def step_count(self) -> int:
        return self._step_count

    def close(self) -> None:
        self._sim.close()
