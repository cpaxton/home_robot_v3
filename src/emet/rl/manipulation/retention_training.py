"""Opt-in task balance and simulator-only supervision for grasp retention.

Intent is a command, not evidence of an object. Contact labels never enter the
actor observation, authorize release, or override physical stop conditions.
"""

from dataclasses import dataclass

import numpy as np

SCENARIO_NAMES = ("reach", "pregrasp", "carry", "place", "disturbance")
TRACKING_RETENTION_MIX = (0.70, 0.10, 0.12, 0.05, 0.03)
GRASP_CONTEXT_CONTRACT = {
    "version": "retention-intent-v1",
    "history_suffix": ["retain_object_command", "gripper_velocity_over_speed_limit"],
    "object_ground_truth": False,
}


def validate_mixture(weights):
    values = np.asarray(weights, dtype=float)
    if values.shape != (len(SCENARIO_NAMES),) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("scenario weights require five finite nonnegative values")
    if not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("scenario weights must sum to one")
    return tuple(float(v) for v in values)


@dataclass
class HeldObjectLabel:
    """Debounced bilateral grasp vs clearly empty hand; other states abstain.

    This labels contact occupancy, not force closure, payload capacity or future
    retention. One-finger contacts and ambiguous transitions are not false negatives.
    """

    contact_time: float = 0.0
    empty_time: float = 0.0
    target: float = 0.0
    valid: bool = False

    def update(self, *, bilateral, clearly_empty, dt, physics_valid=True):
        if not physics_valid:
            self.contact_time = self.empty_time = 0.0
            self.valid = False
            return
        self.contact_time = self.contact_time + dt if bilateral else 0.0
        self.empty_time = self.empty_time + dt if clearly_empty and not bilateral else 0.0
        self.valid = self.contact_time >= 0.1 - 1e-8 or self.empty_time >= 0.1 - 1e-8
        self.target = float(self.contact_time >= 0.1 - 1e-8)

    def record(self):
        return {"target": self.target, "valid": self.valid, "source": "simulator_contact_supervision"}
