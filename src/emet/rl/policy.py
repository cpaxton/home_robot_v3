"""Portable policy bundle: weights, normalization and a validated control contract."""

import hashlib
import json
from pathlib import Path

import numpy as np

from emet.rl.model import PROXIES, model_fingerprint
from emet.rl.profile import MARS


def export_policy(model, normalizer, directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "policy.zip")
    normalization = {
        key: {"mean": rms.mean.tolist(), "var": rms.var.tolist(), "count": float(rms.count)}
        for key, rms in normalizer.obs_rms.items()
    }
    task = config.get("env", {}).get("task", "reach")
    from emet.rl.base import BASE, BASE_CONTRACT

    manifest = {
        "task": task,
        "base_contract": BASE_CONTRACT if task == "base" else None,
        "normalization_frozen": bool(getattr(model.policy.features_extractor, "fixed_normalization", False)),
        "version": 1,
        "robot": MARS.robot,
        "profile": (BASE if task == "base" else MARS).metadata(),
        "model_fingerprint": model_fingerprint(),
        "normalization": normalization,
        "clip_obs": normalizer.clip_obs,
        "epsilon": normalizer.epsilon,
        "training": config,
        "weights_sha256": hashlib.sha256((directory / "policy.zip").read_bytes()).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


class PPOPolicy:
    def __init__(self, directory):
        # Optional dependencies must never be imported by emet-core.
        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(1)
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        m = self.manifest
        from emet.rl.base import BASE, BASE_CONTRACT

        self.task = m.get("task", "reach")
        profile = BASE if self.task == "base" else MARS
        if self.task not in {"base", "reach"} or (self.task == "base" and m.get("base_contract") != BASE_CONTRACT):
            raise ValueError("policy task/control contract mismatch")
        if (
            m.get("version") != 1
            or m.get("robot") != MARS.robot
            or m.get("profile") != json.loads(json.dumps(profile.metadata()))
            or m.get("model_fingerprint") != model_fingerprint()
        ):
            raise ValueError("policy bundle does not match the Mars control profile/model")
        if hashlib.sha256((directory / "policy.zip").read_bytes()).hexdigest() != m["weights_sha256"]:
            raise ValueError("policy weights checksum mismatch")
        shapes = {
            "state": (25,),
            "goal": (15,),
            "obstacles": (len(PROXIES) * 5,),
            "poses": (26,),
            "history": (MARS.history_steps, 66 + len(PROXIES) * 5),
        }
        if self.task == "base":
            shapes = {"history": (BASE.history_steps, BASE_CONTRACT["history_width"])}
        if set(m.get("normalization", {})) != set(shapes):
            raise ValueError("policy normalization layout mismatch")
        if not np.isfinite(m["epsilon"]) or m["epsilon"] <= 0 or not np.isfinite(m["clip_obs"]) or m["clip_obs"] <= 0:
            raise ValueError("invalid policy normalization parameters")
        for key, shape in shapes.items():
            stats = m["normalization"][key]
            mean, var = np.asarray(stats["mean"]), np.asarray(stats["var"])
            if (
                mean.shape != shape
                or var.shape != shape
                or not np.isfinite(mean).all()
                or not np.isfinite(var).all()
                or np.any(var < 0)
            ):
                raise ValueError("invalid policy normalization statistics")
        self.model = PPO.load(directory / "policy.zip", device="cpu")
        if self.model.action_space.shape != ((2,) if self.task == "base" else (7,)):
            raise ValueError("policy action layout mismatch")
        if {k: v.shape for k, v in self.model.observation_space.spaces.items()} != shapes:
            raise ValueError("policy observation layout mismatch")

    def __call__(self, observation):
        m = self.manifest
        arm_only = m.get("training", {}).get("env", {}).get("arm_only", False)
        if arm_only and observation["goal"][9] != 0:
            raise ValueError("arm-only policy cannot execute a base goal")
        normalized = {}
        if set(observation) != set(m["normalization"]):
            raise ValueError("policy observation layout mismatch")
        for key, value in observation.items():
            stats = m["normalization"][key]
            mean, var = np.asarray(stats["mean"]), np.asarray(stats["var"])
            if mean.shape != value.shape or var.shape != value.shape or not np.isfinite(value).all():
                raise ValueError("invalid policy observation")
            normalized[key] = np.clip(
                (value - mean) / np.sqrt(var + m["epsilon"]), -m["clip_obs"], m["clip_obs"]
            ).astype(np.float32)
        action, _ = self.model.predict(normalized, deterministic=True)
        if arm_only:
            action[:2] = 0
        return action
