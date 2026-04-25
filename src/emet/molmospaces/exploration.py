# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# MolmoSpaces (or any ZMQ MuJoCo) exploration loop with episode recording.

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from emet.core.interfaces import ContinuousNavigationAction, Observations
from emet.core.parameters import Parameters
from emet.core.robot import AbstractRobotClient
from emet.memory.graph_eqa import GraphEQAMemory, SensorGraphBuilder, format_scene_graph_pretty
from emet.memory.graph_eqa.graph_memory import labels_are_semantic_graph_hypothesis
from emet.memory.graph_eqa.sensor_graph_builder import world_xyz_median_from_depth
from emet.molmospaces.episode_writer import MolmoEpisodeWriter

logger = logging.getLogger(__name__)


def _anchor_xyz_from_obs(obs: Observations) -> np.ndarray:
    if obs.camera_pose is not None and obs.depth is not None and obs.camera_K is not None:
        try:
            return world_xyz_median_from_depth(obs)
        except Exception:
            pass
    if obs.camera_pose is not None:
        return np.asarray(obs.camera_pose[:3, 3], dtype=np.float64)
    g = np.asarray(obs.gps, dtype=np.float64).reshape(-1)
    th = float(np.asarray(obs.compass, dtype=np.float64).reshape(-1)[0])
    if g.size >= 2:
        return np.array([float(g[0]), float(g[1]), 0.5 * np.sin(th)], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


class MolmoExploreSession:
    """
    Connect to a running MuJoCo ZMQ server, capture posed RGB, and optionally random-walk.

    Expects ``robot.get_observation()`` (e.g. ``GenericZmqClient`` / ``StretchZmqClient``).
    """

    def __init__(
        self,
        robot: AbstractRobotClient,
        writer: MolmoEpisodeWriter,
        *,
        goal_xy_bounds: tuple[float, float, float, float] = (-4.0, 4.0, -4.0, 4.0),
        navigate_every: int = 5,
        nav_timeout: float = 90.0,
        graph_memory: GraphEQAMemory | None = None,
        sensor_builder: SensorGraphBuilder | None = None,
    ) -> None:
        self.robot = robot
        self.writer = writer
        self._xmin, self._xmax, self._ymin, self._ymax = goal_xy_bounds
        self.navigate_every = max(1, int(navigate_every))
        self.nav_timeout = float(nav_timeout)
        self._graph_memory = graph_memory
        self._sensor_builder = sensor_builder
        self._step_idx = 0
        self.navigation_goal_timeouts = 0

    def _maybe_update_graph(self, obs: Observations) -> None:
        if self._graph_memory is None or self._sensor_builder is None:
            return
        try:
            labels, desc = self._sensor_builder.labels_and_description_from_observation(obs)
        except Exception:
            return
        xyz = _anchor_xyz_from_obs(obs)
        try:
            bp = np.asarray(self.robot.get_base_pose(), dtype=np.float64).reshape(-1)
            base_xyz = (
                np.array([float(bp[0]), float(bp[1]), float(bp[2]) if bp.size >= 3 else 0.0], dtype=np.float64)
                if bp.size >= 2
                else None
            )
        except Exception:
            base_xyz = None
        if labels_are_semantic_graph_hypothesis(labels):
            self._graph_memory.add_observation(obs.rgb, xyz, labels, description=desc)
        else:
            self._graph_memory.record_navigation_sample(obs.rgb, xyz, base_xyz=base_xyz)

    def _random_goal_xyt(self) -> np.ndarray:
        x = random.uniform(self._xmin, self._xmax)
        y = random.uniform(self._ymin, self._ymax)
        theta = random.uniform(-np.pi, np.pi)
        return np.array([x, y, theta], dtype=np.float64)

    def _navigate_if_due(self) -> None:
        if self._step_idx % self.navigate_every != 0 or self._step_idx == 0:
            return
        try:
            self.robot.switch_to_navigation_mode()
        except Exception:
            pass
        goal = self._random_goal_xyt()
        out: bool | None = True
        need_fallback = False
        try:
            out = self.robot.move_base_to(
                ContinuousNavigationAction(goal),
                relative=False,
                blocking=True,
                timeout=self.nav_timeout,
            )
            if out is False:
                need_fallback = True
        except Exception:
            need_fallback = True
            out = None
        if need_fallback:
            try:
                out = self.robot.move_base_to(goal, relative=False, blocking=True, timeout=self.nav_timeout)
            except Exception:
                out = None
        if out is False:
            self.navigation_goal_timeouts += 1
            logger.debug(
                "Random navigation goal timed out (count=%s); continuing capture.",
                self.navigation_goal_timeouts,
            )

    def run(
        self,
        *,
        steps: int,
        capture_hz: float,
        sleep_after_nav: float = 0.5,
    ) -> None:
        """Run ``steps`` capture iterations; rate-limit captures with ``capture_hz``."""
        dt = 1.0 / max(0.1, float(capture_hz))
        if hasattr(self.robot, "move_to_nav_posture"):
            try:
                self.robot.move_to_nav_posture()
            except Exception:
                pass
        if hasattr(self.robot, "set_velocity"):
            try:
                self.robot.set_velocity(v=30.0, w=20.0)
            except Exception:
                pass

        for _ in range(int(steps)):
            t0 = time.monotonic()
            self._navigate_if_due()
            time.sleep(sleep_after_nav)

            obs = self._get_observation_safe()
            if obs is None:
                time.sleep(dt)
                self._step_idx += 1
                continue

            self.writer.write_frame(obs)
            self._maybe_update_graph(obs)
            self._step_idx += 1

            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def _get_observation_safe(self) -> Observations | None:
        go = getattr(self.robot, "get_observation", None)
        if not callable(go):
            return None
        try:
            out = go()
        except Exception:
            return None
        if not isinstance(out, Observations):
            return None
        return out

    def save_graph_report(self, path: Path | str) -> None:
        if self._graph_memory is None:
            return
        p = Path(path)
        p.write_text(format_scene_graph_pretty(self._graph_memory, title="Scene graph (explore)"), encoding="utf-8")


def build_graph_sidecar(
    parameters: Parameters | None,
    *,
    cpu_only: bool,
    device: str = "cuda",
    perception_client: Any | None = None,
) -> tuple[GraphEQAMemory, SensorGraphBuilder]:
    mem = GraphEQAMemory(parameters=parameters, defer_llm_clients=True)
    builder = SensorGraphBuilder(
        perception_client=perception_client,
        use_voxel_fallback=True,
        device=device,
        cpu_only=cpu_only,
        parameters=parameters,
    )
    return mem, builder
