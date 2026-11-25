# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Embodiment feedforward identified from recorded tracking failures.

The first tier of the failure-to-control loop is calibration, not gain tuning:
for every joint we measure the steady-state offset between the commanded final
posture and the observed pose. A robust median over attempts gives a static
gravity/load feedforward that is added to the commanded posture. The same model
can be converted to torques for the WBC ``PositionServoAdapter`` when the
effective stiffness is known.

Schema version 1 is a per-joint bias in joint units (rad or m). Torque output
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

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FeedforwardModel:
    """Per-joint static feedforward for one embodiment."""

    embodiment: str
    joint_bias: dict[str, float]
    samples: int = 0
    source: str = ""
    schema: int = SCHEMA_VERSION
    metadata: dict[str, float] = field(default_factory=dict)

    def bias_vector(self, joint_names: Iterable[str]) -> np.ndarray:
        return np.asarray([float(self.joint_bias.get(str(name), 0.0)) for name in joint_names], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "schema": int(self.schema),
            "embodiment": self.embodiment,
            "joint_bias": {name: float(value) for name, value in sorted(self.joint_bias.items())},
            "samples": int(self.samples),
            "source": self.source,
            "metadata": {key: float(value) for key, value in self.metadata.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeedforwardModel:
        if not isinstance(data, dict) or int(data.get("schema", -1)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported feedforward schema: {data!r}")
        bias = data.get("joint_bias")
        if not isinstance(bias, dict) or not bias:
            raise ValueError("feedforward model has no joint_bias")
        cleaned = {str(name): float(value) for name, value in bias.items()}
        if not all(math.isfinite(value) for value in cleaned.values()):
            raise ValueError("feedforward model contains non-finite bias")
        return cls(
            embodiment=str(data.get("embodiment") or ""),
            joint_bias=cleaned,
            samples=int(data.get("samples") or 0),
            source=str(data.get("source") or ""),
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

    def torque_mapping(self, stiffness: dict[str, float]) -> dict[str, float]:
        """Convert the joint-unit bias to feedforward torques (N*m)."""
        return {name: float(bias) * float(stiffness[name]) for name, bias in self.joint_bias.items() if name in stiffness}


def fit_feedforward(
    attempts: Iterable[ManipulationAttempt],
    *,
    embodiment: str = "",
    joint_names: Iterable[str] = (),
    window: int = 5,
    min_samples: int = 2,
    max_abs_bias: float = 0.75,
) -> FeedforwardModel:
    """Fit a robust per-joint bias from tracking failures.

    A joint is emitted only when at least ``min_samples`` attempts contain a
    valid target/observed pair and the median is finite and within
    ``max_abs_bias``. This fails closed on a single bad record instead of
    baking an outlier into every command.
    """
    selected = select_attempts(attempts, robot_id=embodiment or "")
    requested = [str(name) for name in joint_names]
    samples: dict[str, list[float]] = {name: [] for name in requested}
    usable = 0
    for attempt in selected:
        bias = attempt.joint_bias(window=window)
        if not bias:
            continue
        usable += 1
        for name, value in bias.items():
            if requested and name not in samples:
                continue
            samples.setdefault(name, []).append(float(value))

    fitted: dict[str, float] = {}
    for name, values in samples.items():
        if len(values) < max(1, int(min_samples)):
            continue
        median = float(np.median(np.asarray(values, dtype=np.float64)))
        if not math.isfinite(median) or abs(median) > float(max_abs_bias):
            continue
        fitted[name] = median
    if not fitted:
        raise ValueError(
            "no joint had enough valid tracking attempts "
            f"(selected={len(selected)}, usable={usable}, min_samples={min_samples})"
        )
    return FeedforwardModel(
        embodiment=str(embodiment),
        joint_bias=fitted,
        samples=usable,
        source="manip_failures",
        metadata={"window": float(window), "min_samples": float(min_samples), "max_abs_bias": float(max_abs_bias)},
    )


def apply_bias(
    final_q: Iterable[float],
    joint_names: Iterable[str],
    model: FeedforwardModel | None,
    *,
    lower: Iterable[float] | None = None,
    upper: Iterable[float] | None = None,
) -> np.ndarray:
    """Add a feedforward bias to a final posture, clipped to joint limits."""
    q = np.asarray(list(final_q), dtype=np.float64).reshape(-1)
    names = [str(name) for name in joint_names]
    if q.size != len(names):
        raise ValueError(f"final_q has {q.size} values for {len(names)} joints")
    if model is None:
        return q
    q = q + model.bias_vector(names)
    if lower is not None or upper is not None:
        lo = np.asarray(list(lower) if lower is not None else [-math.inf] * q.size, dtype=np.float64)
        hi = np.asarray(list(upper) if upper is not None else [math.inf] * q.size, dtype=np.float64)
        if lo.size != q.size or hi.size != q.size:
            raise ValueError("joint bounds must match final_q")
        q = np.clip(q, lo, hi)
    return q
