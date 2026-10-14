"""Robot-specific manipulation contracts; never reinterpret reaching weights."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ManipulationProfile:
    robot: str
    joint_groups: tuple[tuple[str, ...], ...]
    actuators: tuple[str, ...]
    speed: tuple[float, ...]
    acceleration: tuple[float, ...]
    gripper_joint: str
    gripper_actuator: str
    gripper_open: float
    gripper_closed: float
    gripper_speed: float
    ee_body: str
    finger_bodies: tuple[str, str]
    wheel_bodies: tuple[str, ...]
    base_body: str
    version: int = 1
    dt: float = 0.05
    history_steps: int = 20
    grip_fraction: float = 0.05
    grip_rate_fraction: float = 0.25
    unsafe_force: float = 30.0  # Experimental simulator ceiling, not a damage rating.

    @property
    def arm_size(self):
        return len(self.joint_groups)

    def contract(self):
        return {
            **asdict(self),
            "action": "arm_velocity_residual_and_grip_offset",
            "base": "stationary",
            "actor_inputs": "proprioception_reference_history_only",
            "release": "executor_only",
        }


MARS = ManipulationProfile(
    "innate_mars",
    tuple((f"joint{i}",) for i in range(1, 6)),
    tuple(f"joint{i}" for i in range(1, 6)),
    (0.6,) * 5,
    (1.2,) * 5,
    "joint6",
    "joint6",
    -0.3491,
    0.3491,
    0.6,
    "ee_link",
    ("link61", "link62"),
    (),
    "base_link",
)
STRETCH = ManipulationProfile(
    "stretch",
    (
        ("joint_lift",),
        tuple(f"joint_arm_l{i}" for i in (3, 2, 1, 0)),
        ("joint_wrist_yaw",),
        ("joint_wrist_pitch",),
        ("joint_wrist_roll",),
    ),
    ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll"),
    (0.10, 0.10, 0.6, 0.6, 0.6),
    (0.2, 0.2, 1.2, 1.2, 1.2),
    "joint_gripper_slide",
    "gripper",
    0.04,
    (-0.3 + 0.376) / 0.936 * 0.06 - 0.02,
    0.02,
    "link_grasp_center",
    ("rubber_tip_left", "rubber_tip_right"),
    ("link_left_wheel", "link_right_wheel"),
    "base_link",
)
PROFILES = {p.robot: p for p in (MARS, STRETCH)}


class ModelAdmissionError(ValueError):
    """Required physical evidence is missing; do not report a passing safety gate."""


def require_finite(value, shape):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"expected finite array of shape {shape}")
    return value
