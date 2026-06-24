# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Dataclasses for MolmoBot / MolmoSpaces trajectory H5 layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MolmoCameraCalib:
    name: str
    cam2world_gl: list[list[float]] | None = None
    intrinsic_cv: list[list[float]] | None = None


@dataclass
class MolmoBotStep:
    index: int
    qpos: dict[str, float]
    action_joint_pos: dict[str, float] | None = None
    commanded_action: dict[str, Any] | None = None
    camera_video_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class MolmoBotEpisode:
    traj_key: str
    h5_path: str
    obs_scene: dict[str, Any]
    steps: list[MolmoBotStep]
    rewards: list[float] = field(default_factory=list)
    success: list[bool] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.steps)

    def trimmed_actions(self) -> list[MolmoBotStep]:
        """Drop dummy first action and terminal state per upstream MolmoSpaces docs."""
        if len(self.steps) <= 2:
            return []
        return self.steps[1:-1]
