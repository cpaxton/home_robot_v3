"""Manipulation bundles are incompatible with reaching/base policy bundles."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from emet.rl.manipulation.profiles import PROFILES


class ManipulationHistoryFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        history, width = observation_space["history"].shape
        super().__init__(observation_space, width + history * 15)

    def forward(self, obs):
        history = obs["history"]
        return torch.cat([history[:, -1], history[:, :, :15].flatten(1)], dim=1)


def export(model, normalizer, directory, env_config, fingerprint, *, steps):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "policy.zip")
    stats = normalizer.obs_rms["history"]
    manifest = {
        "format": "emet-manipulation-v1",
        "robot": env_config.robot,
        "contract": PROFILES[env_config.robot].contract(),
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
        "training_reward": "physical_running_cost_integral_v2",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


class ManipulationPolicy:
    def __init__(self, directory, *, robot=None, fingerprint=None):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        m = self.manifest
        if m.get("format") != "emet-manipulation-v1" or m.get("robot") not in PROFILES:
            raise ValueError("not a supported manipulation policy")
        self.robot = m["robot"]
        if robot is not None and robot != self.robot:
            raise ValueError("robot mismatch")
        if m["contract"] != json.loads(json.dumps(PROFILES[self.robot].contract())):
            raise ValueError("manipulation contract mismatch")
        if fingerprint is not None and fingerprint != m["fingerprint"]:
            raise ValueError("manipulation model mismatch")
        if hashlib.sha256((directory / "policy.zip").read_bytes()).hexdigest() != m["weights_sha256"]:
            raise ValueError("policy checksum mismatch")
        self.mean, self.var = np.asarray(m["mean"]), np.asarray(m["var"])
        if (
            not np.isfinite([*self.mean.ravel(), *self.var.ravel(), m["epsilon"], m["clip"]]).all()
            or np.any(self.var < 0)
            or min(m["epsilon"], m["clip"]) <= 0
        ):
            raise ValueError("invalid normalization")
        self.model = PPO.load(directory / "policy.zip", device="cpu")
        if (
            self.model.action_space.shape != (PROFILES[self.robot].arm_size + 1,)
            or self.model.observation_space["history"].shape != self.mean.shape
        ):
            raise ValueError("manipulation policy layout mismatch")

    def __call__(self, observation):
        if set(observation) != {"history"}:
            raise ValueError("unexpected actor inputs")
        values = np.asarray(observation["history"])
        if values.shape != self.mean.shape or not np.isfinite(values).all():
            raise ValueError("invalid actor observation")
        normalized = np.clip(
            (values - self.mean) / np.sqrt(self.var + self.manifest["epsilon"]),
            -self.manifest["clip"],
            self.manifest["clip"],
        ).astype(np.float32)
        return self.model.predict({"history": normalized}, deterministic=True)[0]
