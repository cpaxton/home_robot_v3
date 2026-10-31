"""Advisory object-in-hand estimator, trained separately from PPO rewards.

Only encoder/command history is consumed. Simulator contacts supply masked
labels through info, never actor inputs. Episode-disjoint audit data is excluded
from optimizer updates; its metrics are diagnostics, not safety calibration.
"""

import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from emet.rl.manipulation.observation_layout import proprioceptive_layout
from emet.rl.manipulation.profiles import PROFILES


class HeldObjectEstimator:
    def __init__(self, robot, history_shape, *, seed=0, capacity=4096, feature_set="gripper"):
        self.robot = robot
        self.history_shape = tuple(history_shape)
        self.steps = min(10, self.history_shape[0])
        fields, _ = proprioceptive_layout(PROFILES[robot])
        if feature_set not in {"gripper", "legacy_proprioception"}:
            raise ValueError("unknown held-object feature set")
        self.feature_set = feature_set
        names = ("grip_position", "grip_error", "grip_reference")
        if feature_set == "legacy_proprioception":
            names = ("velocities", "target_error", *names, "previous_action")
        self.columns = np.array(
            [i for name in names for i in range(fields[name].start, fields[name].stop)]
            + [self.history_shape[1] - 2, self.history_shape[1] - 1]
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = nn.Sequential(nn.Linear(self.steps * len(self.columns), 64), nn.Tanh(), nn.Linear(64, 1))
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=1e-3)
        self.rng = np.random.default_rng(seed)
        self.training = deque(maxlen=capacity)
        self.audit = deque(maxlen=max(128, capacity // 4))
        self.class_counts = [0, 0]
        self.updates = 0

    def features(self, history):
        values = np.asarray(history, dtype=np.float32)
        if values.shape != self.history_shape or not np.isfinite(values).all():
            raise ValueError("invalid held-object history")
        # Fixed clipping, independent of evolving VecNormalize statistics.
        return np.clip(values[-self.steps :, self.columns], -10, 10).reshape(-1).copy()

    def add(self, history, supervision, *, audit=False):
        if not supervision.get("valid", False):
            return
        target = float(supervision["target"])
        if target not in (0.0, 1.0):
            raise ValueError("held-object target must be binary")
        features = self.features(history)
        (self.audit if audit else self.training).append((features, target))
        if not audit:
            self.class_counts[int(target)] += 1

    def fit(self, *, batches=8, batch_size=128):
        labels = np.array([s[1] for s in self.training])
        positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        if min(len(positives), len(negatives)) < 8:
            return {}
        # Balance the auxiliary classifier, not PPO task selection or rewards.
        # Deliberately retain closed-empty counterexamples; easy open hands
        # otherwise let a small head mistake the closure command for occupancy.
        closed_empty = np.array([i for i in negatives if self.training[i][0][-2] > 0.5], dtype=int)
        losses = []
        for _ in range(batches):
            positive_count = max(1, batch_size // 2)
            hard_count = batch_size // 4 if len(closed_empty) else 0
            indices = np.r_[
                self.rng.choice(positives, positive_count),
                self.rng.choice(closed_empty, hard_count) if hard_count else np.array([], dtype=int),
                self.rng.choice(negatives, batch_size - positive_count - hard_count),
            ]
            samples = [self.training[int(i)] for i in indices]
            x = torch.as_tensor(np.stack([s[0] for s in samples]))
            y = torch.as_tensor([s[1] for s in samples], dtype=torch.float32)
            loss = nn.functional.binary_cross_entropy_with_logits(self.head(x).squeeze(-1), y)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.head.parameters(), 1)
            self.optimizer.step()
            self.updates += 1
            losses.append(float(loss.detach()))
        return {"loss": float(np.mean(losses)), "updates": self.updates}

    def estimate(self, history):
        with torch.no_grad():
            probability = float(torch.sigmoid(self.head(torch.from_numpy(self.features(history)))).item())
        return {
            "probability": probability,
            "available": self.updates > 0 and min(self.class_counts) >= 16,
            "calibrated": False,
            "advisory_only": True,
        }

    def audit_metrics(self):
        result = {
            "train_negative_samples": self.class_counts[0],
            "train_positive_samples": self.class_counts[1],
            "audit_samples": len(self.audit),
        }
        if not self.audit:
            return result
        with torch.no_grad():
            x = torch.as_tensor(np.stack([s[0] for s in self.audit]))
            probabilities = torch.sigmoid(self.head(x)).squeeze(-1).numpy()
        labels = np.array([s[1] for s in self.audit])
        result["audit_brier"] = float(np.mean((probabilities - labels) ** 2))
        result["audit_positive_samples"] = int(labels.sum())
        result["audit_negative_samples"] = int((labels == 0).sum())
        if labels.any():
            result["audit_held_recall"] = float(np.mean(probabilities[labels == 1] >= 0.5))
        if (labels == 0).any():
            result["audit_empty_false_positive_rate"] = float(np.mean(probabilities[labels == 0] >= 0.5))
        return result

    def save(self, directory):
        directory = Path(directory)
        path = directory / "held_object.npz"
        np.savez_compressed(path, **{k: v.detach().numpy() for k, v in self.head.state_dict().items()})
        manifest = {
            "format": "emet-held-object-v2" if self.feature_set == "gripper" else "emet-held-object-v1",
            "feature_set": self.feature_set,
            "robot": self.robot,
            "history_shape": self.history_shape,
            "columns": self.columns.tolist(),
            "class_counts": self.class_counts,
            "updates": self.updates,
            "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "calibrated": False,
            "advisory_only": True,
            "label": "debounced bilateral payload contact; ambiguous transitions masked",
            "audit_split": "whole training episodes, never used by head optimizer",
            "minibatches": "half held, half empty; reserve a quarter for retention-command empty examples when available",
            "metrics": self.audit_metrics(),
        }
        (directory / "held_object.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return hashlib.sha256((directory / "held_object.json").read_bytes()).hexdigest()

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "held_object.json").read_text())
        if manifest.get("format") not in {"emet-held-object-v1", "emet-held-object-v2"} or not manifest.get(
            "advisory_only"
        ):
            raise ValueError("unsupported held-object estimator")
        if hashlib.sha256((directory / "held_object.npz").read_bytes()).hexdigest() != manifest["weights_sha256"]:
            raise ValueError("held-object estimator checksum mismatch")
        model = cls(
            manifest["robot"],
            manifest["history_shape"],
            feature_set=("gripper" if manifest["format"] == "emet-held-object-v2" else "legacy_proprioception"),
        )
        if manifest["columns"] != model.columns.tolist():
            raise ValueError("held-object observation semantics mismatch")
        with np.load(directory / "held_object.npz", allow_pickle=False) as weights:
            model.head.load_state_dict({k: torch.from_numpy(weights[k].copy()) for k in weights.files})
        if not all(torch.isfinite(p).all() for p in model.head.parameters()):
            raise ValueError("invalid held-object weights")
        model.class_counts, model.updates = manifest["class_counts"], manifest["updates"]
        return model
