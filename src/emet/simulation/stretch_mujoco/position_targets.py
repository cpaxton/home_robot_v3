# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Rate-limited position references, separate from measured joint completion."""

import math


class PositionTargets:
    def __init__(self, model, data, rates):
        self.model = model
        self.data = data
        self.rates = dict(rates)
        if any(not math.isfinite(rate) or rate <= 0 for rate in self.rates.values()):
            raise ValueError("Position-reference rates must be finite and positive")
        self.pending = {}

    def set(self, name, target):
        if not math.isfinite(target):
            raise ValueError("Position target must be finite")
        actuator = self.model.actuator(name)
        if actuator.ctrllimited[0]:
            target = max(float(actuator.ctrlrange[0]), min(float(actuator.ctrlrange[1]), target))
        if name in self.rates:
            self.pending[name] = target
        else:
            self.data.actuator(name).ctrl = target

    def step(self):
        # Local physics-time state: no manager RPC, wall-clock dependence, or
        # command resend reset. This limits references, not measured velocities.
        for name, target in self.pending.items():
            actuator = self.data.actuator(name)
            current = float(actuator.ctrl[0])
            step = self.rates[name] * self.model.opt.timestep
            actuator.ctrl = current + max(-step, min(step, target - current))

    def reset(self):
        """Explicit keyframe/reset commands supersede all pending motion."""
        self.pending.clear()
