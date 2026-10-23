"""Small causal history encoder for PPO; no recurrent state outside observations."""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class HistoryTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        steps, width = observation_space["history"].shape
        self.project = nn.Linear(width, features_dim)
        self.position = nn.Parameter(torch.zeros(1, steps, features_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            features_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.register_buffer("causal_mask", torch.triu(torch.ones(steps, steps, dtype=torch.bool), diagonal=1))

    def forward(self, observations):
        tokens = self.project(observations["history"]) + self.position
        return self.encoder(tokens, mask=self.causal_mask)[:, -1]
