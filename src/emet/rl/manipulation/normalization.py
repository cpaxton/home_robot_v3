"""Portable normalization with categorical command features kept in native units.

Identity mean/variance are exported in ordinary bundle statistics. Runtime
loading needs no special preprocessing or change to existing checkpoint inputs.
"""

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from emet.rl.manipulation.observation_layout import history_layout, preview_layout
from emet.rl.manipulation.profiles import PROFILES
from emet.rl.pose_targets import BASE_MASK_COLUMNS, HAND_MASK_COLUMN, TARGET_TOKEN_WIDTH


def categorical_columns(profile, *, retention_context):
    fields, width = history_layout(profile, retention_context=retention_context)
    history = np.zeros(width, dtype=bool)
    for name in ("phase", "base_mask", "hand_mask", "target_mask", "retain_object_command"):
        if name in fields:
            history[fields[name]] = True
    tokens = fields["target_tokens"].start
    history[tokens + BASE_MASK_COLUMNS.start : tokens + BASE_MASK_COLUMNS.stop] = True
    history[tokens + TARGET_TOKEN_WIDTH + HAND_MASK_COLUMN] = True
    preview_fields, preview_width = preview_layout()
    preview = np.zeros(preview_width, dtype=bool)
    preview[preview_fields["base_mask"]] = True
    preview[preview_fields["hand_mask"]] = True
    return {"history": history, "preview": preview}


class CommandVecNormalize(VecNormalize):
    """Running statistics for continuous fields, identity for command categories."""

    def __init__(self, venv, *, config, **kwargs):
        self.profile = PROFILES[config.robot]
        self.feedback_policy = config.feedback_policy
        self.retention_context = config.retention_context
        self.fixed_columns = categorical_columns(PROFILES[config.robot], retention_context=config.retention_context)
        super().__init__(venv, **kwargs)
        self.pin_command_features()

    def pin_command_features(self):
        if not 0 < self.epsilon < 1:
            raise ValueError("categorical normalization requires epsilon in (0, 1)")
        for name, columns in self.fixed_columns.items():
            if name in self.obs_rms:
                stats = self.obs_rms[name]
                if stats.mean.shape[-1] != len(columns):
                    raise ValueError("categorical normalization layout mismatch")
                stats.mean[..., columns] = 0
                stats.var[..., columns] = 1 - self.epsilon
        if self.feedback_policy:
            fields, _ = history_layout(self.profile, retention_context=self.retention_context)
            stats = self.obs_rms["history"]
            scales = {
                "proprio/velocities": np.asarray(self.profile.speed),
                "base_velocity": np.array(
                    [max(self.profile.base_speed, 0.01)] * 3 + [max(self.profile.base_yaw_speed, 0.1)] * 3
                ),
            }
            for name, scale in scales.items():
                if np.any(scale**2 <= self.epsilon):
                    raise ValueError("normalization epsilon must be smaller than every fixed physical variance")
                stats.mean[..., fields[name]] = 0
                stats.var[..., fields[name]] = scale**2 - self.epsilon
            preview, _ = preview_layout()
            stats = self.obs_rms["preview"]
            for name, scale in {
                "base_error": np.array([0.1, 0.1, 0.5]),
                "hand_error": np.array([0.02] * 3 + [1.0] * 6),
            }.items():
                if np.any(scale**2 <= self.epsilon):
                    raise ValueError("normalization epsilon must be smaller than every fixed physical variance")
                stats.mean[..., preview[name]] = 0
                stats.var[..., preview[name]] = scale**2 - self.epsilon

    def normalize_obs(self, obs):
        # VecNormalize updates RMS immediately before this call. Pin only the
        # categorical columns; continuous statistics retain their measured count.
        self.pin_command_features()
        return super().normalize_obs(obs)
