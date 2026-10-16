"""Compact base features: current task context plus motion/command history."""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BaseHistoryFeatures(BaseFeaturesExtractor):
    """Keep old goals from blurring the current target during reactive updates.

    The full current token supplies goal and obstacle context. History supplies
    measured body velocities and applied commands, the signals needed to infer
    response delay and tracking problems. This is a fixed selector, not a transformer.
    """

    def __init__(self, observation_space):
        steps, width = observation_space["history"].shape
        super().__init__(observation_space, width + steps * 5)

    def forward(self, observations):
        history = observations["history"]
        return torch.cat([history[:, -1], history[:, :, 4:9].flatten(start_dim=1)], dim=1)


class FixedScaleBaseHistoryFeatures(BaseHistoryFeatures):
    """History features whose observation scales must survive export/resume."""

    fixed_normalization = True


class PrecisionBaseFeatures(FixedScaleBaseHistoryFeatures):
    """Explicit position-arrival flag for the final heading/settling phase.

    The threshold is stored in normalized observation units. Its normalizer must
    remain frozen during subsequent PPO training; export/resume enforce that.
    The flag is an input, not a restriction on the policy's actions.
    """

    fixed_normalization = True

    def __init__(self, observation_space, normalized_goal_threshold):
        super().__init__(observation_space)
        self._features_dim += 1
        self.register_buffer("goal_threshold", torch.tensor(float(normalized_goal_threshold)))

    def forward(self, observations):
        features = super().forward(observations)
        arrived = (observations["history"][:, -1, -3] <= self.goal_threshold).float()[:, None]
        return torch.cat([features, arrived], dim=1)
