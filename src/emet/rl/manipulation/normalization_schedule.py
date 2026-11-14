"""Training-only jaw scaling curriculum; exports ordinary frozen bundle statistics.

Keep means, other channels and running counts fixed. Geometric interpolation of
standard deviations bounds the relative scale change between rollout boundaries.
No measurement, action, reward or executor guard is changed here.
"""

from numbers import Real

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from emet.rl.manipulation.observation_layout import history_layout


def validate_grip_floor(value, profile):
    """Zero disables the curriculum; a positive floor is in joint/slide units."""
    travel = abs(profile.gripper_open - profile.gripper_closed)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= travel:
        raise ValueError("grip normalization floor must be finite and within the physical jaw travel")
    return float(value)


class GripNormalizationSchedule(BaseCallback):
    def __init__(self, normalizer, *, minimum_std, warmup_steps, ramp_steps):
        super().__init__()
        self.normalizer = normalizer
        self.minimum_std = validate_grip_floor(minimum_std, normalizer.profile)
        if self.minimum_std <= np.sqrt(normalizer.epsilon):
            raise ValueError("grip normalization floor must exceed normalization epsilon scale")
        if type(warmup_steps) is not int or warmup_steps < 0 or type(ramp_steps) is not int or ramp_steps <= 0:
            raise ValueError("grip normalization requires nonnegative warmup and positive integer ramp")
        stats = normalizer.obs_rms["history"]
        fields, width = history_layout(normalizer.profile, retention_context=normalizer.retention_context)
        if stats.var.shape[-1] != width or normalizer.training:
            raise ValueError("grip normalization requires frozen running statistics with the declared layout")
        self.fields = tuple("proprio/" + n for n in ("grip_position", "grip_error", "grip_reference"))
        self.columns = [fields[name].start for name in self.fields]
        self.initial_var, self.initial_mean, self.initial_count = stats.var.copy(), stats.mean.copy(), stats.count
        self.warmup_steps, self.ramp_steps, self.fraction = warmup_steps, ramp_steps, 0.0

    def record(self):
        return {
            "minimum_std": self.minimum_std,
            "units": "profile gripper joint/slide coordinates",
            "warmup_steps": self.warmup_steps,
            "ramp_steps": self.ramp_steps,
            "fraction": self.fraction,
            "fields": list(self.fields),
            "means_and_other_channels_fixed": True,
            "interpolation": "geometric_standard_deviation_at_rollout_boundaries",
        }

    def apply(self, steps):
        if type(steps) is not int or steps < 0:
            raise ValueError("nonnegative integer normalization step required")
        stats = self.normalizer.obs_rms["history"]
        if (
            self.normalizer.training
            or not np.array_equal(stats.mean, self.initial_mean)
            or stats.count != self.initial_count
        ):
            raise RuntimeError("grip normalization requires frozen means and running counts")
        fraction = float(np.clip((steps - self.warmup_steps) / self.ramp_steps, 0, 1))
        changed = fraction != self.fraction
        variance = self.initial_var.copy()
        if fraction:
            epsilon = self.normalizer.epsilon
            before = np.sqrt(variance[:, self.columns] + epsilon)
            after = np.maximum(before, self.minimum_std)
            sigma = before * np.exp(fraction * np.log(after / before))
            variance[:, self.columns] = np.maximum(0, sigma**2 - epsilon)
        stats.var = variance
        self.fraction = fraction
        return changed

    def _on_rollout_start(self):
        if self.apply(int(self.model.num_timesteps)):
            # PPO otherwise retains the previous rollout's normalization of
            # this same measured state. Refresh its cached input, without a reset.
            original = self.normalizer.get_original_obs()
            if original is None or self.model._last_obs is None:
                raise RuntimeError("normalization curriculum requires an initialized measured observation")
            self.model._last_obs = self.normalizer.normalize_obs(original)
        self.logger.record("normalization/grip_fraction", self.fraction)

    def _on_step(self):
        return True
