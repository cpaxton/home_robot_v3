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
    from emet.motion.arm_manip_profile import ArmManipProfile
    from emet.motion.robot import RobotModel
    from emet.simulation.mujoco_stationary_control import MujocoStationaryControl

from emet.robots.footprint import Footprint


@dataclass(frozen=True)
class ArmChain:
    """Declarative arm kinematics for a robot (used by :class:`ArmManipProfile`).

    Names refer to the robot's **MJCF** joints/bodies (the kinematic model the
    executor drives), not the high-level ``RobotSpec.joint_names`` API names.
    ``actuator_names`` are the subset of ``RobotSpec.actuator_names`` that drive the
    arm; when empty, the arm joints are used directly. ``home_arm_q`` overrides the
    MJCF qpos0 home when non-empty.
    """

    joint_names: tuple[str, ...]
    ee_body: str
    actuator_names: tuple[str, ...] = ()
    link_bodies: tuple[str, ...] = ()
    gripper_bodies: tuple[str, ...] = ()
    home_arm_q: tuple[float, ...] = ()
    base_freejoint_name: str = "base_freejoint"


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
    # Curated arm kinematics (MJCF joint/body names) for kinematic pick/place. When set,
    # :meth:`ArmManipProfile.for_robot` uses it before MJCF auto-discovery. Optional: the
    # backend may instead implement :meth:`RobotBackend.build_arm_manip_profile`.
    arm_chain: "ArmChain | None" = None
    # Per-arm chains for bimanual robots; ``arm_chains[arm]`` wins over ``arm_chain``.
    arm_chains: "dict[str, ArmChain]" = field(default_factory=dict)
    advertise_kinematic_manip: bool = False
    """If True, :class:`~emet.simulation.robosuite_server.RobosuiteZmqServer` advertises
    ``capabilities.kinematic_manip`` when an :class:`~emet.motion.arm_manip_profile.ArmManipProfile`
    also resolves. Default False so offline IK discovery (xlerobot, franka, …) does not
    switch live DynaMem/OVMM pick/place from teleport to latch."""
    # --- Optional install / app hints (names match pyproject [project.optional-dependencies]) ---
    optional_uv_extras: tuple[str, ...] = ()
    """Extras to ``uv sync --extra …`` for typical full-stack use when something is not in core deps."""
    dynamem_depth_source_hint: str | None = None
    """When hardware omits depth on ZMQ, DynaMem usually needs this ``depth_source`` (see dynav YAML)."""
    default_dynav_config: str | None = None
    """Optional packaged basename under ``emet/config/`` when ``emet run dynamem`` uses the global
    default ``--dynav-config dynav_config.yaml``. Most robots omit this and share ``dynav_config.yaml``.
    """
    dynav_parameter_overrides: dict[str, Any] = field(default_factory=dict)
    """Optional ``dynav_config.yaml`` key overrides applied in ``run_dynagraph`` / ``run_dynamem`` after load."""
    planar_base_joint_names: tuple[str, str, str] | None = None
    """If set, MuJoCo nav uses three scalar joints (slide, slide, hinge yaw) as velocity targets instead
    of a ``base_link`` free joint. Slide axes live on the parent body of the first slide (e.g. ``base_root``);
    world `(x, y, yaw)` is converted to joint values when that body has a non-identity Robocasa merge pose.
    Names must match the MJCF and :class:`RobosuiteZmqServer` SE(2) commands."""
    robosuite_rgb_depth_ops: tuple[str, ...] = ()
    """MuJoCo RGB/depth post-steps for :class:`RobosuiteZmqServer` (``flipud``, ``rot90_cw``). Intrinsics are chained."""
    spawn: RobotSpawnSpec | None = None
    """Spawn / placement hints for MolmoSpaces merge and similar sims (optional)."""
    sim_uses_stretch_mujoco_zmq: bool = False
    """True when ``emet serve mujoco`` uses :class:`MujocoZmqServer` (Stretch table sim), not RobosuiteZmqServer."""
    planar_spawn_xy_extra_margin_m: float = 0.0
    """Added to Robocasa planar spawn ``footprint_xy_margin_m`` (erodes walkable XY clip). Use when
    :attr:`footprint` is base-sized but appendages extend horizontally (typical mobile manipulators)."""
    planar_spawn_clip_edge_pad_m: float | None = None
    """Minimum XY distance from :func:`~emet.simulation.scene_base_spawn.collision_scene_xy_clip_rect`
    edges for the base link. If ``None``, uses ``0.22 + 0.5 * planar_spawn_xy_extra_margin_m``."""
    planar_spawn_clip_guard_body_name: str | None = None
    """Legacy single guard body. Prefer :attr:`planar_spawn_clip_guard_body_names`; if that tuple is
    empty and this is set, spawn uses a one-element guard list."""
    planar_spawn_clip_guard_body_names: tuple[str, ...] = ()
    """Bodies whose world XY must lie inside the scene walkable clip (same pad). Use for EE + mid-arm when
    meshes are visual-only."""
    planar_spawn_clip_guard_pad_m: float = 0.18
    """XY inset inside the scene clip for each :attr:`planar_spawn_clip_guard_body_names` body (meters)."""
    planar_spawn_robocasa_first_clearance_m: float | None = None
    """If set (Robocasa planar spawn only), the first :func:`~emet.simulation.scene_base_spawn.find_planar_base_xyt`
    contact-distance pass uses at least this many meters of ``worst`` before relaxing (see clearance ladder there)."""


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
            f"(e.g. ``--dynav-config dynav_innate_mars.yaml`` for innate_mars; or a packaged dynav_<robot>.yaml)"
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

    def build_arm_manip_profile(self, arm: str = "left") -> "ArmManipProfile | None":
        """Optional code fallback for building an :class:`ArmManipProfile`.

        Declarative ``RobotSpec.arm_chain`` is preferred; implement this only when a
        robot needs custom logic (e.g. derived actuator mapping). Return ``None`` to
        fall through to MJCF auto-discovery.
        """
        return None
