# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Embodiment feedforward identified from recorded tracking failures.

The first tier of the failure-to-control loop is calibration, not gain tuning:
for every joint we measure the steady-state offset between the commanded final
posture and the observed pose. Postures matter — a torso bias fitted at a deep
reach over-corrects a shallower lift — so schema 2 stores a *table* of
calibrations keyed by the commanded joint posture. A joint's bias is applied
only from the nearest calibrated posture within ``max_joint_distance``.

The same table can be converted to torques for the WBC
``PositionServoAdapter`` when the effective stiffness is known. Torque output
is an explicit conversion, never stored ambiguously.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from emet.controller.manipulation.failure_dataset import ManipulationAttempt, select_attempts

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FeedforwardCalibration:
    """One measured posture and its steady-state joint bias."""

    reference_q: dict[str, float]
    joint_bias: dict[str, float]
    samples: int = 0

    def to_dict(self) -> dict:
        return {
            "reference_q": {name: float(value) for name, value in sorted(self.reference_q.items())},
            "joint_bias": {name: float(value) for name, value in sorted(self.joint_bias.items())},
            "samples": int(self.samples),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeedforwardCalibration:
        reference = data.get("reference_q") or {}
        bias = data.get("joint_bias") or {}
        if not isinstance(bias, dict) or not bias:
            raise ValueError("calibration has no joint_bias")
        cleaned = {str(name): float(value) for name, value in bias.items()}
        if not all(math.isfinite(value) for value in cleaned.values()):
            raise ValueError("calibration contains non-finite bias")
        return cls(
            reference_q={str(k): float(v) for k, v in reference.items()},
            joint_bias=cleaned,
            samples=int(data.get("samples") or 0),
        )


@dataclass(frozen=True)
class FeedforwardModel:
    """Posture-conditioned per-joint feedforward for one embodiment."""

    embodiment: str
    calibrations: tuple[FeedforwardCalibration, ...]
    samples: int = 0
    source: str = ""
    schema: int = SCHEMA_VERSION
    max_joint_distance: float = 0.20
    metadata: dict[str, float] = field(default_factory=dict)

    @property
    def joint_bias(self) -> dict[str, float]:
        """First calibration (compatibility for single-posture callers)."""
        return dict(self.calibrations[0].joint_bias) if self.calibrations else {}

    @property
    def reference_q(self) -> dict[str, float]:
        return dict(self.calibrations[0].reference_q) if self.calibrations else {}

    def bias_vector(self, joint_names: Iterable[str]) -> np.ndarray:
        return np.asarray([float(self.joint_bias.get(str(name), 0.0)) for name in joint_names], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "schema": int(self.schema),
            "embodiment": self.embodiment,
            "calibrations": [calibration.to_dict() for calibration in self.calibrations],
            "samples": int(self.samples),
            "source": self.source,
            "max_joint_distance": float(self.max_joint_distance),
            "metadata": {key: float(value) for key, value in self.metadata.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeedforwardModel:
        if not isinstance(data, dict):
            raise ValueError("feedforward model must be an object")
        schema = int(data.get("schema", -1))
        if schema == 1:
            calibration = FeedforwardCalibration.from_dict(data)
            calibrations = (calibration,)
        elif schema == 2:
            raw = data.get("calibrations")
            if not isinstance(raw, list) or not raw:
                raise ValueError("feedforward model has no calibrations")
            calibrations = tuple(FeedforwardCalibration.from_dict(item) for item in raw)
        else:
            raise ValueError(f"unsupported feedforward schema: {data!r}")
        return cls(
            embodiment=str(data.get("embodiment") or ""),
            calibrations=calibrations,
            samples=int(data.get("samples") or sum(cal.samples for cal in calibrations)),
            source=str(data.get("source") or ""),
            max_joint_distance=float(data.get("max_joint_distance", 0.20)),
            metadata={str(k): float(v) for k, v in (data.get("metadata") or {}).items()},
        )

    def save(self, path: str | Path) -> Path:
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> FeedforwardModel:
        return cls.from_dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))

    def torque_mapping(self, stiffness: dict[str, float], *, calibration: int = 0) -> dict[str, float]:
        """Convert one calibration's joint-unit bias to feedforward torques."""
        if not 0 <= calibration < len(self.calibrations):
            raise IndexError(f"calibration index {calibration} out of range")
        bias = self.calibrations[calibration].joint_bias
        return {name: float(value) * float(stiffness[name]) for name, value in bias.items() if name in stiffness}


def _cluster_attempts(
    selected: list[ManipulationAttempt],
    *,
    window: int,
    cluster_distance: float,
) -> list[list[ManipulationAttempt]]:
    """Greedy posture clusters by final commanded joint vector."""
    clusters: list[list[ManipulationAttempt]] = []
    centroids: list[np.ndarray] = []
    joints: list[str] = []
    for attempt in selected:
        target = attempt.final_target_q()
        if target is None:
            continue
        names = list(attempt.joint_names)
        if not joints:
            joints = names
        if names != joints:
            continue
        vector = np.asarray(target, dtype=np.float64)
        if not np.isfinite(vector).all():
            continue
        best_index, best_distance = None, float("inf")
        for index, centroid in enumerate(centroids):
            distance = float(np.linalg.norm(vector - centroid))
            if distance < best_distance:
                best_index, best_distance = index, distance
        if best_index is not None and best_distance <= float(cluster_distance):
            clusters[best_index].append(attempt)
            centroids[best_index] = np.median(
                np.asarray([a.final_target_q() for a in clusters[best_index]], dtype=np.float64), axis=0
            )
        else:
            clusters.append([attempt])
            centroids.append(vector)
    return clusters


def fit_feedforward(
    attempts: Iterable[ManipulationAttempt],
    *,
    embodiment: str = "",
    joint_names: Iterable[str] = (),
    window: int = 5,
    min_samples: int = 2,
    max_abs_bias: float = 0.75,
    exclude_feedforward: bool = True,
    cluster_distance: float = 0.42,
    max_joint_distance: float = 0.20,
    outlier_bias_limit: float = 1.2,
) -> FeedforwardModel:
    """Fit a table of per-posture biases from clean tracking failures.

    A posture cluster is emitted only when it has ``min_samples`` valid
    target/observed pairs and its per-joint median is finite and within
    ``max_abs_bias``. This fails closed instead of baking one outlier into
    every command.
    """
    selected = select_attempts(
        attempts, robot_id=embodiment or "", exclude_feedforward=exclude_feedforward
    )
    # An attempt with an extreme per-joint offset is a contaminated run (e.g. an
    # overcorrecting controller), not a posture to learn from.
    clean: list[ManipulationAttempt] = []
    for attempt in selected:
        bias = attempt.joint_bias(window=window)
        if not bias:
            continue
        if any(abs(float(value)) > float(outlier_bias_limit) for value in bias.values()):
            continue
        clean.append(attempt)
    requested = [str(name) for name in joint_names]
    clusters = _cluster_attempts(clean, window=window, cluster_distance=cluster_distance)
    calibrations: list[FeedforwardCalibration] = []
    usable = 0
    for cluster in clusters:
        if len(cluster) < max(1, int(min_samples)):
            continue
        usable += len(cluster)
        samples: dict[str, list[float]] = {}
        targets: dict[str, list[float]] = {}
        for attempt in cluster:
            bias = attempt.joint_bias(window=window)
            target = attempt.final_target_q()
            if not bias or target is None:
                continue
            for index, name in enumerate(attempt.joint_names):
                if requested and name not in requested:
                    continue
                samples.setdefault(name, []).append(float(bias[name]))
                targets.setdefault(name, []).append(float(target[index]))
        fitted: dict[str, float] = {}
        reference: dict[str, float] = {}
        for name, values in samples.items():
            if len(values) < max(1, int(min_samples)):
                continue
            median = float(np.median(np.asarray(values, dtype=np.float64)))
            if not math.isfinite(median) or abs(median) > float(max_abs_bias):
                continue
            fitted[name] = median
            reference[name] = float(np.median(np.asarray(targets.get(name) or [0.0], dtype=np.float64)))
        if fitted:
            calibrations.append(
                FeedforwardCalibration(reference_q=reference, joint_bias=fitted, samples=len(cluster))
            )
    if not calibrations:
        raise ValueError(
            "no posture had enough valid tracking attempts "
            f"(selected={len(selected)}, clean={len(clean)}, clusters={len(clusters)}, "
            f"min_samples={min_samples})"
        )
    return FeedforwardModel(
        embodiment=str(embodiment),
        calibrations=tuple(calibrations),
        samples=usable,
        source="manip_failures",
        max_joint_distance=float(max_joint_distance),
        metadata={
            "window": float(window),
            "min_samples": float(min_samples),
            "max_abs_bias": float(max_abs_bias),
            "cluster_distance": float(cluster_distance),
        },
    )


def apply_bias(
    final_q: Iterable[float],
    joint_names: Iterable[str],
    model: FeedforwardModel | None,
    *,
    lower: Iterable[float] | None = None,
    upper: Iterable[float] | None = None,
) -> np.ndarray:
    """Add the nearest calibrated posture's bias, clipped to joint limits.

    The lookup is per joint: a new posture only inherits a bias from a
    calibration whose reference is within ``max_joint_distance``. This is what
    keeps a deep pregrasp correction from leaking into a shallower lift.
    """
    q = np.asarray(list(final_q), dtype=np.float64).reshape(-1)
    names = [str(name) for name in joint_names]
    if q.size != len(names):
        raise ValueError(f"final_q has {q.size} values for {len(names)} joints")
    if model is None:
        return q
    for index, name in enumerate(names):
        nearest: tuple[float, float] | None = None
        for calibration in model.calibrations:
            reference = calibration.reference_q.get(name)
            if reference is None:
                continue
            distance = abs(float(q[index]) - float(reference))
            if distance > model.max_joint_distance:
                continue
            if nearest is None or distance < nearest[0]:
                nearest = (distance, float(calibration.joint_bias.get(name, 0.0)))
        if nearest is not None:
            q[index] += nearest[1]
    if lower is not None or upper is not None:
        lo = np.asarray(list(lower) if lower is not None else [-math.inf] * q.size, dtype=np.float64)
        hi = np.asarray(list(upper) if upper is not None else [math.inf] * q.size, dtype=np.float64)
        if lo.size != q.size or hi.size != q.size:
            raise ValueError("joint bounds must match final_q")
        q = np.clip(q, lo, hi)
    return q
