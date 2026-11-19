"""Calibrated execution-risk estimates, not geometric infeasibility certificates.

The PPO critic is deliberately not exposed as a feasibility oracle. These heads
learn explicit outcomes for one robot/model/controller and abstain on unsupported
labels, out-of-distribution queries, or inadequate calibration evidence.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

HEADS = ("success", "retention_lost", "unsafe_contact", "wheel_lift")
SCHEMA = "emet-trajectory-risk-v2"


def trajectory_features(history, trajectory, duration, *, profile=None):
    """Robot-specific proprioception only; phase/task/preview columns are excluded."""
    from emet.rl.manipulation.observation_layout import proprioceptive_layout
    from emet.rl.manipulation.profiles import PROFILES

    profile = PROFILES["innate_mars"] if profile is None else profile
    fields, width = proprioceptive_layout(profile)
    history, trajectory = np.asarray(history, dtype=float), np.asarray(trajectory, dtype=float)
    if history.shape != (profile.history_steps, width) or trajectory.ndim != 2 or trajectory.shape[1] != profile.arm_size or len(trajectory) < 1:
        raise ValueError("risk inputs require manipulation history and five-coordinate arm trajectory")
    if (
        not np.isfinite(history).all()
        or not np.isfinite(trajectory).all()
        or not np.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("nonfinite trajectory or invalid duration")
    delta = trajectory - history[-1, fields["positions"]]
    segments = np.diff(np.vstack([history[-1, fields["positions"]], trajectory]), axis=0)
    motion = history[:, fields["velocities"].start : fields["base_command"].stop]
    return np.r_[
        history[-1],
        motion.mean(axis=0),
        motion.std(axis=0),
        np.max(abs(motion), axis=0),
        delta[-1],
        delta.mean(axis=0),
        np.max(abs(delta), axis=0),
        np.sum(abs(segments), axis=0),
        np.max(abs(segments), axis=0),
        len(trajectory),
        duration,
    ].astype(np.float32)


def _network(width):
    return nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, len(HEADS)))


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def _calibration(labels, probabilities):
    brier = float(np.mean((labels - probabilities) ** 2))
    ece = 0.0
    bins = np.minimum((probabilities * 10).astype(int), 9)
    for group in range(10):
        selected = bins == group
        if selected.any():
            ece += selected.mean() * abs(float(labels[selected].mean() - probabilities[selected].mean()))
    return {
        "brier": brier,
        "ece": float(ece),
        "samples": len(labels),
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
    }


def fit_risk(rows, directory, *, robot, fingerprint, controller_id, seed=0, epochs=200):
    if len(rows) < 100:
        raise ValueError("at least 100 independent outcome episodes required")
    if any(r.get("split") == "frozen_regression" or r.get("initialization_failed") for r in rows):
        raise ValueError("frozen cases and failed initializations cannot train executable-trajectory risk")
    if any(
        (r.get("robot"), r.get("fingerprint"), r.get("controller_id")) != (robot, fingerprint, controller_id)
        for r in rows
    ):
        raise ValueError("risk outcomes must match one robot/model/controller")
    ids = [r["episode_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate episodes would leak across calibration splits")
    x = np.asarray([r["risk_features"] for r in rows], dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("invalid risk features")
    y = np.array(
        [[r["is_success"], r["retention_lost"], r["unsafe_contact"], r.get("wheel_lift")] for r in rows], dtype=float
    )
    if np.any(np.isfinite(y) & ~np.isin(y, [0, 1])):
        raise ValueError("outcome labels must be binary or unknown")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    # Temperature fitting and acceptance must not reuse the same outcomes.
    # Keep an independent audit partition, including for unsupported heads.
    train, calibration, audit = np.split(order, [int(0.6 * len(x)), int(0.8 * len(x))])
    mean, std = x[train].mean(0), x[train].std(0).clip(min=1e-3)
    z = (x - mean) / std
    models, validation, audit_predictions = [], [], []
    torch.set_num_threads(1)
    for member in range(3):
        torch.manual_seed(seed + member)
        model = _network(x.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.001)
        sample = rng.choice(train, len(train), replace=True)
        inputs = torch.from_numpy(z[sample])
        valid = torch.from_numpy(np.isfinite(y[sample]).astype(np.float32))
        labels = torch.from_numpy(np.nan_to_num(y[sample]).astype(np.float32))
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(inputs), labels, reduction="none")
            loss = (loss * valid).sum() / valid.sum().clamp(min=1)
            loss.backward()
            optimizer.step()
        models.append(model.state_dict())
        with torch.no_grad():
            validation.append(model(torch.from_numpy(z[calibration])).numpy())
            audit_predictions.append(model(torch.from_numpy(z[audit])).numpy())
    logits = np.stack(validation)
    audit_logits = np.stack(audit_predictions)
    heads = {}
    for index, name in enumerate(HEADS):
        valid = np.isfinite(y[calibration, index])
        labels = y[calibration, index][valid]
        train_labels = y[train, index][np.isfinite(y[train, index])]
        audit_valid = np.isfinite(y[audit, index])
        audit_labels = y[audit, index][audit_valid]
        counts = [np.sum(partition == value) for partition in (train_labels, labels, audit_labels) for value in (0, 1)]
        if min(counts) < 20:
            heads[name] = {
                "supported": False,
                "reason": "insufficient positive/negative training, calibration or audit evidence",
            }
            continue
        candidates = np.geomspace(0.25, 8, 40)
        losses = []
        for temperature in candidates:
            prob = _sigmoid(logits[:, valid, index] / temperature).mean(0).clip(1e-6, 1 - 1e-6)
            losses.append(-np.mean(labels * np.log(prob) + (1 - labels) * np.log(1 - prob)))
        temperature = float(candidates[np.argmin(losses)])
        probabilities = _sigmoid(audit_logits[:, audit_valid, index] / temperature).mean(0)
        metrics = _calibration(audit_labels, probabilities)
        metrics["constant_brier"] = float(np.mean((audit_labels - train_labels.mean()) ** 2))
        supported = metrics["ece"] <= 0.1 and metrics["brier"] <= metrics["constant_brier"]
        heads[name] = {
            "supported": bool(supported),
            "temperature": temperature,
            "metrics_split": "independent_audit",
            **metrics,
            "reason": None if supported else "calibration gate failed",
        }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(models, directory / "models.pt")
    manifest = {
        "format": SCHEMA,
        "robot": robot,
        "fingerprint": fingerprint,
        "controller_id": controller_id,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": x[train].min(0).tolist(),
        "max": x[train].max(0).tolist(),
        "heads": heads,
        "training_episodes": [ids[i] for i in train],
        "calibration_episodes": [ids[i] for i in calibration],
        "audit_episodes": [ids[i] for i in audit],
        "weights_sha256": hashlib.sha256((directory / "models.pt").read_bytes()).hexdigest(),
        "scope": "advisory calibrated outcome estimates; not a safety or infeasibility certificate",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class TrajectoryRiskPredictor:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        m = self.manifest
        if (
            m["format"] != SCHEMA
            or hashlib.sha256((directory / "models.pt").read_bytes()).hexdigest() != m["weights_sha256"]
        ):
            raise ValueError("risk model contract/checksum mismatch")
        self.models = []
        for weights in torch.load(directory / "models.pt", map_location="cpu", weights_only=True):
            model = _network(len(m["mean"]))
            model.load_state_dict(weights)
            model.eval()
            self.models.append(model)

    def predict(self, features, *, robot, fingerprint, controller_id, geometry_valid=None):
        m = self.manifest
        empty = {
            "status": "unknown",
            "probabilities": dict.fromkeys(HEADS),
            "unavailable_heads": dict.fromkeys(HEADS, "not evaluated"),
            "safety_certificate": False,
            "geometric_infeasibility": False,
        }
        if geometry_valid is False:
            return {
                **empty,
                "status": "geometry_rejected",
                "geometric_infeasibility": True,
                "reason": "caller geometric validation rejected this path; not a learned conclusion",
            }
        if (robot, fingerprint, controller_id) != (m["robot"], m["fingerprint"], m["controller_id"]):
            return {**empty, "reason": "robot/model/controller outside calibration scope"}
        x = np.asarray(features, dtype=np.float32)
        lo, hi = np.asarray(m["min"]), np.asarray(m["max"])
        if x.shape != lo.shape or not np.isfinite(x).all():
            raise ValueError("invalid trajectory features")
        margin = 0.1 * (hi - lo) + 0.001
        if np.any((x < lo - margin) | (x > hi + margin)):
            return {**empty, "reason": "out_of_distribution"}
        z = torch.from_numpy(((x - np.asarray(m["mean"])) / np.asarray(m["std"])).astype(np.float32)[None])
        with torch.no_grad():
            logits = np.stack([model(z).numpy()[0] for model in self.models])
        probabilities, disagreement, unavailable = {}, {}, {}
        for index, name in enumerate(HEADS):
            head = m["heads"][name]
            if not head["supported"]:
                probabilities[name] = None
                disagreement[name] = None
                unavailable[name] = head.get("reason") or "unsupported calibration"
                continue
            p = _sigmoid(logits[:, index] / head["temperature"])
            probabilities[name] = float(p.mean()) if p.std() <= 0.1 else None
            disagreement[name] = float(p.std())
            if probabilities[name] is None:
                unavailable[name] = "ensemble disagreement exceeds 0.1"
        unknown = geometry_valid is not True or any(v is None for v in probabilities.values())
        return {
            **empty,
            "status": "unknown" if unknown else "estimated",
            "probabilities": probabilities,
            "unavailable_heads": unavailable,
            "ensemble_disagreement": disagreement,
            "reason": "missing geometry/calibration evidence" if unknown else None,
        }
