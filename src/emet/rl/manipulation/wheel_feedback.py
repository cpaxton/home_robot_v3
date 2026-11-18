"""Versioned native feedback settings; independent of the learned action map."""

from numbers import Real

import numpy as np


def validate_gain(value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not 0 <= value <= 2
    ):
        raise ValueError("wheel feedback gain must be finite and in [0, 2]")
    return float(value)


def validate_base_only_gains(value):
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("base-only wheel feedback requires a linear/yaw gain pair")
    return tuple(validate_gain(v) for v in value)


def configure_wheel_feedback(control, config):
    """Apply declared settings without touching targets, history or stop latches."""
    control.base_velocity_feedback_gain = config.wheel_feedback_gain
    control.base_only_velocity_feedback_gains = config.wheel_feedback_base_only_gains
    control.base_velocity_feedback_current = config.wheel_feedback_current_velocity


def execution_record(config):
    """Bind the opt-in law and its physical request boundary to the weights."""
    if config.wheel_feedback_base_only_gains is None:
        return None
    return {
        "format": "mars_masked_wheel_feedback_v1",
        "scope": "all_base_axes_active_and_hand_inactive",
        "fallback_gain": config.wheel_feedback_gain,
        "base_only_gains": list(config.wheel_feedback_base_only_gains),
        "current_com_velocity": config.wheel_feedback_current_velocity,
        "physical_wheel_slew": config.physical_wheel_slew,
        "wheel_jerk_limits": None if config.wheel_jerk_limits is None else list(config.wheel_jerk_limits),
    }
