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
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Base abstractions for robot backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emet.controller.emotes.backend import EmoteBackend
    from emet.core.robot import AbstractRobotClient
    from emet.motion.robot import RobotModel
    from emet.simulation.mujoco_stationary_control import MujocoStationaryControl

from emet.robots.footprint import Footprint


@dataclass
class RobotSpawnSpec:
    """Optional MolmoSpaces / merged-scene spawn tuning (see ``molmospaces_spawn.json`` next to MJCF)."""

    molmospaces_target_foot_clearance_above_floor_m: float | None = None
    molmospaces_nominal_base_height_above_floor_m: float | None = None
    requires_floating_base_spawn_settle: bool = False


@dataclass
class RobotSpec:
    """Declarative config for a robot (DOF, cameras, URDF path, etc.)."""

    name: str
    dof: int
    joint_names: list[str]
    camera_names: list[str]
    urdf_path: str | None
    footprint: Footprint
    mjcf_path: str | None = None
    actuator_names: list[str] = field(default_factory=list)
    base_link_name: str = "base_link"
    # --- Optional install / app hints (names match pyproject [project.optional-dependencies]) ---
    optional_uv_extras: tuple[str, ...] = ()
    """Extras to ``uv sync --extra …`` for typical full-stack use when something is not in core deps."""
    dynamem_depth_source_hint: str | None = None
    """When hardware omits depth on ZMQ, DynaMem usually needs this ``depth_source`` (see dynav YAML)."""
    robosuite_rgb_depth_ops: tuple[str, ...] = ()
    """MuJoCo RGB/depth post-steps for :class:`RobosuiteZmqServer` (``flipud``, ``rot90_cw``). Intrinsics are chained."""
    spawn: RobotSpawnSpec | None = None
    """Spawn / placement hints for MolmoSpaces merge and similar sims (optional)."""
    sim_uses_stretch_mujoco_zmq: bool = False
    """True when ``emet serve mujoco`` uses :class:`MujocoZmqServer` (Stretch table sim), not RobosuiteZmqServer."""


def format_uv_sync_extras_hint(spec: RobotSpec) -> str | None:
    """One-line ``uv sync`` reminder for :attr:`RobotSpec.optional_uv_extras`, or None if unset."""
    if not spec.optional_uv_extras:
        return None
    parts = " ".join(f"--extra {x}" for x in spec.optional_uv_extras)
    return f"uv sync {parts}"


def format_robot_runtime_notes(spec: RobotSpec) -> str | None:
    """Human-readable notes (extras + DynaMem depth hint) for docs or CLI; None if nothing declared."""
    chunks: list[str] = []
    u = format_uv_sync_extras_hint(spec)
    if u:
        chunks.append(u)
    if spec.dynamem_depth_source_hint:
        chunks.append(
            f"DynaMem without hardware depth: depth_source={spec.dynamem_depth_source_hint!r} "
            f"(e.g. config dynav_{spec.name}.yaml when shipped)"
        )
    return " — ".join(chunks) if chunks else None


class RobotBackend(ABC):
    """Base class for robot-specific logic."""

    @abstractmethod
    def get_spec(self) -> RobotSpec:
        """Return the robot specification."""
        ...

    @abstractmethod
    def create_client(self, robot_ip: str, **kwargs: Any) -> "AbstractRobotClient":
        """Create a client for communicating with the robot (real or simulated)."""
        ...

    @abstractmethod
    def create_model(self, **kwargs: Any) -> "RobotModel":
        """Create a kinematic/planning model of the robot."""
        ...

    def get_emote_backend(self) -> "EmoteBackend":
        """Emote/gesture implementation for this robot (Stretch motion vs speech-only, etc.)."""
        from emet.controller.emotes.backend import GenericEmoteBackend

        return GenericEmoteBackend(self.get_spec().name)

    def create_mujoco_stationary_control(self) -> "MujocoStationaryControl | None":
        """Optional MuJoCo ``ctrl``/hold policy for :class:`emet.simulation.robosuite_server.RobosuiteZmqServer`.

        Return ``None`` to use :class:`emet.simulation.mujoco_stationary_control.DefaultMujocoStationaryControl`.
        """
        return None
