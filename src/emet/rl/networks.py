"""Compact base features: current task context plus motion/command history."""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


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


class CoordinatedPoseFeatures(BaseFeaturesExtractor):
    """History transformer fused with masked, permutation-invariant pose targets."""

    def __init__(self, observation_space, features_dim=96):
        super().__init__(observation_space, features_dim)
        steps, width = observation_space["history"].shape
        self.history_dim = 64
        self.project = nn.Linear(width, self.history_dim)
        self.position = nn.Parameter(torch.zeros(1, steps, self.history_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            self.history_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.register_buffer("causal_mask", torch.triu(torch.ones(steps, steps, dtype=torch.bool), diagonal=1))
        self.target_encoder = nn.Sequential(nn.Linear(10, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(self.history_dim + 64, features_dim), nn.Tanh())

    def forward(self, observations):
        history = observations["history"]
        history = self.history_encoder(
            self.project(history) + self.position,
            mask=self.causal_mask,
        )[:, -1]
        mask = observations["target_mask"].bool()
        encoded = self.target_encoder(observations["target_tokens"])
        encoded = encoded * mask.unsqueeze(-1)
        pooled = encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.fusion(torch.cat([history, pooled], dim=1))
