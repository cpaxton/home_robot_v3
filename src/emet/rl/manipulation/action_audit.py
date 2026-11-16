"""Group recorded action statistics using the actual velocity-command layout."""

import numpy as np


def grouped_action_rms(per_action_rms, contract):
    base_size = 2 if contract["wheel_bodies"] else 0
    arm_size = len(contract["joint_groups"])
    values = np.asarray(per_action_rms, dtype=float)
    if values.shape != (base_size + arm_size + 1,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("finite nonnegative RMS for every base/arm/grip action required")
    return {
        "base_rms": float(np.sqrt(np.mean(values[:base_size] ** 2))) if base_size else 0.0,
        "arm_rms": float(np.sqrt(np.mean(values[base_size : base_size + arm_size] ** 2))),
        "grip_rms": float(values[-1]),
    }
