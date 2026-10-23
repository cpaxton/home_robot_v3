"""Manipulation bundles are incompatible with reaching/base policy bundles."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from emet.rl.manipulation.profiles import PROFILES


class ManipulationMLPFeatures(BaseFeaturesExtractor):
    """A small fixed history for a plain feed-forward controller."""

    def __init__(self, observation_space):
        steps, width = observation_space["history"].shape
        self.steps = min(4, steps)
        self.has_preview = "preview" in observation_space.spaces
        preview_size = int(np.prod(observation_space["preview"].shape)) if self.has_preview else 0
        super().__init__(observation_space, self.steps * width + preview_size)

    def forward(self, obs):
        features = obs["history"][:, -self.steps :].flatten(start_dim=1)
        return torch.cat([features, obs["preview"].flatten(start_dim=1)], dim=1) if self.has_preview else features


class ManipulationHistoryFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        steps, width = observation_space["history"].shape
        self.has_preview = "preview" in observation_space.spaces
        preview_size = int(np.prod(observation_space["preview"].shape)) if self.has_preview else 0
        super().__init__(observation_space, 128 + preview_size)
        self.project = nn.Linear(width, 128)
        self.position = nn.Parameter(torch.zeros(1, steps, 128))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.register_buffer("causal_mask", torch.triu(torch.ones(steps, steps, dtype=torch.bool), diagonal=1))

    def forward(self, obs):
        tokens = self.project(obs["history"]) + self.position
        features = self.encoder(tokens, mask=self.causal_mask)[:, -1]
        return torch.cat([features, obs["preview"].flatten(start_dim=1)], dim=1) if self.has_preview else features


def export(model, normalizer, directory, env_config, fingerprint, *, steps):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "policy.zip")
    stats = normalizer.obs_rms["history"]
    manifest = {
        "format": "emet-manipulation-v1",
        "robot": env_config.robot,
        "contract": PROFILES[env_config.robot].contract(env_config.action_mode),
        "fingerprint": fingerprint,
        "steps": steps,
        "mean": stats.mean.tolist(),
        "var": stats.var.tolist(),
        "count": float(stats.count),
        "epsilon": normalizer.epsilon,
        "clip": normalizer.clip_obs,
        "weights_sha256": hashlib.sha256((directory / "policy.zip").read_bytes()).hexdigest(),
        "safety_promoted": False,
        "actor_ground_truth_inputs": False,
        "environment": asdict(env_config),
        "training_reward": (
            "differential_base_tracking_v7"
            if env_config.differential_base_reward
            else "coupled_task_pose_tracking_endpoint_ramp_v6"
            if env_config.endpoint_motion_weight
            else "coupled_task_pose_tracking_and_hold_v5"
        )
        if env_config.tracking_reward
        else "active_pose_tracking_and_settling_v3",
        "normalization": {
            key: {"mean": value.mean.tolist(), "var": value.var.tolist(), "count": float(value.count)}
            for key, value in normalizer.obs_rms.items()
        },
        "action_mode": env_config.action_mode,
        "feature_extractor": type(model.policy.features_extractor).__name__,
        "pose_only_targets": env_config.tracking_reward or env_config.trajectory_preview,
    }
    if env_config.trajectory_preview:
        from emet.rl.manipulation.tracking import preview_contract

        manifest["preview_contract"] = preview_contract()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


class ManipulationPolicy:
    def __init__(self, directory, *, robot=None, fingerprint=None):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        m = self.manifest
        if m.get("format") != "emet-manipulation-v1" or m.get("robot") not in PROFILES:
            raise ValueError("not a supported manipulation policy")
        self.robot = m["robot"]
        self.action_mode = m.get("action_mode", "residual")
        if self.action_mode not in {"residual", "direct_velocity"}:
            raise ValueError("unsupported policy action semantics")
        if robot is not None and robot != self.robot:
            raise ValueError("robot mismatch")
        if m["contract"] != json.loads(json.dumps(PROFILES[self.robot].contract(self.action_mode))):
            raise ValueError("manipulation contract mismatch")
        if fingerprint is not None and fingerprint != m["fingerprint"]:
            raise ValueError("manipulation model mismatch")
        if hashlib.sha256((directory / "policy.zip").read_bytes()).hexdigest() != m["weights_sha256"]:
            raise ValueError("policy checksum mismatch")
        normalization = m.get("normalization", {"history": {"mean": m["mean"], "var": m["var"]}})
        if set(normalization) not in ({"history"}, {"history", "preview"}):
            raise ValueError("unsupported actor inputs")
        self.normalization = {
            key: (np.asarray(value["mean"]), np.asarray(value["var"])) for key, value in normalization.items()
        }
        self.mean, self.var = self.normalization["history"]
        for mean, var in self.normalization.values():
            if mean.shape != var.shape or not np.isfinite(mean).all() or not np.isfinite(var).all() or np.any(var < 0):
                raise ValueError("invalid normalization")
        if not np.isfinite([m["epsilon"], m["clip"]]).all() or min(m["epsilon"], m["clip"]) <= 0:
            raise ValueError("invalid normalization")
        if "preview" in self.normalization:
            from emet.rl.manipulation.tracking import PREVIEW_OFFSETS, PREVIEW_WIDTH, preview_contract

            if m.get("preview_contract") != preview_contract() or self.normalization["preview"][0].shape != (
                len(PREVIEW_OFFSETS),
                PREVIEW_WIDTH,
            ):
                raise ValueError("preview contract mismatch")

        self.model = PPO.load(directory / "policy.zip", device="cpu")
        if (
            self.model.action_space.shape != (PROFILES[self.robot].action_size,)
            or set(self.model.observation_space.spaces) != set(self.normalization)
            or any(
                self.model.observation_space[key].shape != mean.shape for key, (mean, _) in self.normalization.items()
            )
        ):
            raise ValueError("manipulation policy layout mismatch")

    def __call__(self, observation):
        if set(observation) != set(self.normalization):
            raise ValueError("unexpected actor inputs")
        normalized = {}
        for key, (mean, var) in self.normalization.items():
            values = np.asarray(observation[key])
            if values.shape != mean.shape or not np.isfinite(values).all():
                raise ValueError("invalid actor observation")
            normalized[key] = np.clip(
                (values - mean) / np.sqrt(var + self.manifest["epsilon"]), -self.manifest["clip"], self.manifest["clip"]
            ).astype(np.float32)
        return self.model.predict(normalized, deterministic=True)[0]
