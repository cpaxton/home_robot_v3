"""Robot-specific control limits; independent of Gym, torch, ROS and transport."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ControlProfile:
    robot: str = "innate_mars"
    arm_joints: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4", "joint5")
    base_joints: tuple[str, ...] = ("base_x", "base_y", "base_yaw")
    held_joints: tuple[str, ...] = ("joint6", "joint_head")
    control_dt: float = 0.05
    physics_dt: float = 0.002
    max_v: float = 0.30
    max_w: float = 0.80
    max_joint_speed: float = 0.60
    obstacle_range: float = 1.0
    xy_tolerance: float = 0.05
    yaw_tolerance: float = 0.0872664626
    joint_tolerance: float = 0.05
    settle_seconds: float = 0.25
    history_steps: int = 20
    max_target_lead: float = 0.08
    obstruction_seconds: float = 0.20
    arm_kp: float = 80.0
    arm_kv: float = 4.0
    arm_effort_limit: float = 2.0  # Pinned vendor URDF rating; hardware calibration still needed.
    version: int = 4

    def metadata(self):
        return asdict(self)


MARS = ControlProfile()
