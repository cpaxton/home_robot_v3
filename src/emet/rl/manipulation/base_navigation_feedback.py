"""Opt-in base-only feedback structure, with learned magnitudes and native limits.

This is an architectural prior, not a stability certificate or a teacher action.
Active hand commands keep the original coupled actor. Only fully constrained
base-only commands use the sign/symmetry constraints and declared hold region.
"""

import math
from dataclasses import dataclass
from numbers import Real

import torch


@dataclass(frozen=True)
class BaseNavigationFeedback:
    xy_region_m: float = 0.05
    yaw_region_rad: float = math.radians(5)
    reference_tracking: bool = False
    heading_gain_scale: float = 1.0

    def __post_init__(self):
        if type(self.reference_tracking) is not bool:
            raise ValueError("reference_tracking must be a boolean")
        if (
            isinstance(self.heading_gain_scale, bool)
            or not isinstance(self.heading_gain_scale, Real)
            or not math.isfinite(self.heading_gain_scale)
            or self.heading_gain_scale <= 0
        ):
            raise ValueError("heading_gain_scale must be positive and finite")
        object.__setattr__(self, "heading_gain_scale", float(self.heading_gain_scale))
        for name, maximum in (("xy_region_m", 1.0), ("yaw_region_rad", math.pi)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError("base navigation regions must be positive finite distances/angles")
            object.__setattr__(self, name, float(value))

    def record(self):
        result = {
            "format": "signed_base_navigation_v1",
            "xy_region_m": self.xy_region_m,
            "yaw_region_rad": self.yaw_region_rad,
        }
        if self.reference_tracking or self.heading_gain_scale != 1.0:
            result.update(
                format="signed_base_navigation_v2", reference_tracking=self.reference_tracking,
                heading_gain_scale=self.heading_gain_scale,
            )
        return result

    def validate_tolerances(self, xy, yaw):
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0 for v in (xy, yaw)) or (
            0.5 * self.xy_region_m > xy or 0.5 * self.yaw_region_rad > yaw
        ):
            raise ValueError("base navigation inner hold region exceeds valid declared environment tolerances")

    @classmethod
    def from_record(cls, value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("invalid base navigation feedback contract")
        keys = {"format", "xy_region_m", "yaw_region_rad"}
        if value.get("format") == "signed_base_navigation_v1" and set(value) == keys:
            return cls(value["xy_region_m"], value["yaw_region_rad"])
        if (
            value.get("format") == "signed_base_navigation_v2"
            and set(value) == keys | {"reference_tracking", "heading_gain_scale"}
        ):
            result = cls(
                value["xy_region_m"], value["yaw_region_rad"],
                reference_tracking=value["reference_tracking"], heading_gain_scale=value["heading_gain_scale"],
            )
            if result.record() == value:
                return result
        raise ValueError("invalid base navigation feedback contract")


def navigation_feedback_gain(gain, preview, motion, eligible, fields, config):
    """Retain learned feedback sizes while imposing base-only sign conventions.

    Inputs use SensorEncoder units (base errors / [0.1m, 0.1m, 0.5rad]). The
    native driver still applies wheel speed/acceleration and stop guards. A
    stationary, constant preview uses half the declared acceptance radius as
    its inner hold region, leaving margin to enter the actual acceptance band.
    The versioned reference-tracking mode keeps the XY region active throughout
    motion and reserves announced-twist feedforward for the exact unit mapping
    in the actor; learned gains supply pose correction, not velocity rescaling.
    No output smoothing, task labels, joint witness or teacher is involved.
    """
    task_width, motion_start = 3 + 9, gain.shape[-1] - 9
    error = fields["base_error"]
    x = preview[:, 4, error][:, 0]
    direction = torch.where(motion[:, 0].abs() > 1e-5, motion[:, 0], x).sign()
    direction = torch.where(direction == 0, 1.0, direction)
    proposed = torch.zeros_like(gain[:, :2])

    def magnitude(value):
        # Unlike abs()'s zero derivative at zero, this one-sided convention
        # lets a newly zero-initialized gain head learn positive magnitudes.
        return torch.where(value < 0, -value, value)

    if not config.reference_tracking:
        proposed[:, 0, motion_start] = magnitude(gain[:, 0, motion_start])
        proposed[:, 1, motion_start + 2] = magnitude(gain[:, 1, motion_start + 2])
    constant = (preview[:, 1:, error] == preview[:, 1:2, error]).all(dim=(1, 2))
    stationary = (motion[:, :3] == 0).all(-1) & constant
    for row, offset in ((1, 0), (4, task_width)):
        value = preview[:, row, error]
        xy = (1 - 0.5 * config.xy_region_m / 0.1 / value[:, :2].norm(dim=-1).clamp_min(1e-8)).clamp_min(0)
        yaw = (1 - 0.5 * config.yaw_region_rad / 0.5 / value[:, 2].abs().clamp_min(1e-8)).clamp_min(0)
        xy_factor = xy if config.reference_tracking else torch.where(stationary, xy, 1.0)
        yaw_factor = torch.where(stationary, yaw, 1.0)
        proposed[:, 0, offset] = magnitude(gain[:, 0, offset]) * xy_factor
        proposed[:, 1, offset + 1] = magnitude(gain[:, 1, offset + 1]) * direction * xy_factor
        proposed[:, 1, offset + 2] = magnitude(gain[:, 1, offset + 2]) * yaw_factor * config.heading_gain_scale
    base = torch.where(eligible[:, None, None], proposed, gain[:, :2])
    return torch.cat([base, gain[:, 2:]], dim=1)
