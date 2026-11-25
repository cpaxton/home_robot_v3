"""Serialized training reward settings, independent of actuator/safety contracts."""

from dataclasses import asdict, dataclass, fields
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class WholeBodyRewardConfig:
    # Defaults reproduce the original RewardMeter. Fine scales are resolved
    # from the actual environment hand contract, never from curriculum level.
    position_scale_m: float = 0.01
    rotation_scale_rad: float = 0.1
    fine_pose_weight: float = 0.0
    hand_pose_weight: float = 1.0
    hand_velocity_weight: float = 0.5
    base_pose_weight: float = 1.0
    base_velocity_weight: float = 0.5
    hand_linear_velocity_scale_m_s: float = 0.02
    hand_angular_velocity_scale_rad_s: float = 0.2
    base_position_scale_m: float = 0.02
    base_heading_scale_rad: float = 0.05
    base_linear_velocity_scale_m_s: float = 0.02
    base_angular_velocity_scale_rad_s: float = 0.1
    pose_gated_velocity: bool = False
    progress_weight: float = 0.0
    stability_scale: float = 1.0
    jerk_weight: float = 0.01
    effort_weight: float = 0.001
    command_delta_weight: float = 0.01
    force_free_N: float = 3.0
    force_scale_N: float = 3.0
    force_weight: float = 0.1
    mars_folded_pose_weight: float = 0.0
    guard_penalty: float = 10.0
    success_bonus: float = 2.0
    failure_penalty: float = 2.0
    tamp_failure_penalty: float = 10.0

    def __post_init__(self):
        if type(self.pose_gated_velocity) is not bool:
            raise ValueError("pose-gated velocity must be boolean")
        positive = {f.name for f in fields(self) if "scale" in f.name and f.name != "stability_scale"}
        positive |= {"guard_penalty"}
        for field in fields(self):
            if field.name == "pose_gated_velocity":
                continue
            value = getattr(self, field.name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError("finite real reward settings required: " + field.name)
            if value < 0 or (field.name in positive and value == 0):
                raise ValueError("invalid reward setting: " + field.name)
            object.__setattr__(self, field.name, float(value))
        if self.fine_pose_weight > 1 or self.stability_scale > 1:
            raise ValueError("fine pose weight and stability curriculum must be in [0,1]")
        if self.hand_pose_weight <= 0 or self.base_pose_weight <= 0:
            raise ValueError("hand and base pose rewards must retain positive weight")

    @classmethod
    def contract_tracking(cls):
        """Experimental fine hand priority; no policy-performance claim."""
        return cls(
            position_scale_m=0.02,
            rotation_scale_rad=0.2,
            fine_pose_weight=0.75,
            hand_pose_weight=4.0,
            pose_gated_velocity=True,
            progress_weight=0.1,
        )

    def record(self):
        values = asdict(self)
        if self.mars_folded_pose_weight == 0:
            del values["mars_folded_pose_weight"]
        return {"format": "whole_body_reward_v1", **values}

    @classmethod
    def from_record(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        expected = {"format", *(f.name for f in fields(cls))}
        if not isinstance(value, dict) or set(value) not in (expected, expected - {"mars_folded_pose_weight"}):
            raise ValueError("complete versioned reward configuration required")
        if value["format"] != "whole_body_reward_v1":
            raise ValueError("unsupported reward configuration format")
        return cls(**{k: v for k, v in value.items() if k != "format"})


def hand_tracking_terms(config, position_error, rotation_error, velocity_error, position_tolerance, rotation_tolerance):
    """Pure terms used by native and spawned wrappers; no actor/critic state."""
    velocity_error = np.asarray(velocity_error, dtype=float)
    values = np.asarray([position_error, rotation_error, position_tolerance, rotation_tolerance], dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values[:2] < 0)
        or np.any(values[2:] <= 0)
        or velocity_error.shape != (6,)
        or not np.isfinite(velocity_error).all()
    ):
        raise ValueError("finite pose errors, positive tolerances and a six-dimensional twist required")
    coarse = np.exp(-position_error / config.position_scale_m - rotation_error / config.rotation_scale_rad)
    fine = np.exp(-position_error / position_tolerance - rotation_error / rotation_tolerance)
    pose = (1 - config.fine_pose_weight) * coarse + config.fine_pose_weight * fine
    gate = fine if config.pose_gated_velocity else 1.0
    velocity = np.exp(
        -np.sum((velocity_error[:3] / config.hand_linear_velocity_scale_m_s) ** 2)
        - np.sum((velocity_error[3:] / config.hand_angular_velocity_scale_rad_s) ** 2)
    )
    return float(config.hand_pose_weight * pose), float(config.hand_velocity_weight * gate * velocity), float(gate)


def mars_folded_pose_cost(joints_rad):
    """Unitless exposure to the measured Mars shoulder/elbow fold region.

    The two smooth ramps are provisional geometry priors, not force estimates.
    The cost can be applied to measured states or summed over candidate paths.
    """
    q = np.asarray(joints_rad, dtype=float)
    if q.ndim == 0 or q.shape[-1] != 5 or not np.isfinite(q).all():
        raise ValueError("expected finite five-axis Mars joint positions")
    shoulder = np.clip((-q[..., 1] - np.deg2rad(45)) / np.deg2rad(25), 0.0, 1.0)
    elbow = np.clip((q[..., 2] - np.deg2rad(70)) / np.deg2rad(20), 0.0, 1.0)
    return shoulder**2 * (3 - 2 * shoulder) * elbow**2 * (3 - 2 * elbow)
