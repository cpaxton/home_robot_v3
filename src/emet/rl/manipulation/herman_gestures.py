"""Small FK-witnessed gestures for offline Herman rehearsal, never motor commands."""

import numpy as np

from emet.rl.manipulation.jerk_reference import compile_joint_path
from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory


def wave_case():
    return {
        "id": "herman-wave-v1:wrist_wave",
        "seed": 430_010_000,
        "split": "audit",
        "mode": "coupled",
        "travel_m": 0.0,
        "turn_rad": 0.0,
        "arm_delta": [0.0] * 5,
        "base_mask": [True] * 3,
        "gesture": "wrist_wave",
        "initial_kind": "synthetic_geometry_admitted_staging_proposal",
        "hardware_executable": False,
    }


class HermanWaveReference(TaskPoseTrajectory):
    """Hold, tiny wrist roll to either side, return, then hold the original pose.

    The base pose and gripper reference stay fixed. A joint-space FK witness is
    private geometry; the actor receives only the normal SE(2)/SE(3) preview.
    Derivative limits describe the reference, not measured policy motion or a
    calibrated hardware envelope. No physical staging path is supplied here.
    """

    def __init__(self, control):
        if control.profile.robot != "innate_mars":
            raise ValueError("Herman wave requires Mars")
        self.fk = TaskFrameFK(control.model, control.data, control)
        self.start_base = control.base_pose().copy()
        self.fixed_hand = None
        self.initial_hold_s = 2.0
        initial = control.positions().copy()
        delta = np.array([0, 0, 0, 0, 0.06])
        points = np.array([initial, initial + delta, initial, initial - delta, initial])
        if np.any(points < control.limits[:, 0] + 0.002) or np.any(points > control.limits[:, 1] - 0.002):
            raise ValueError("wave witness exceeds joint margin; do not clamp")
        p = control.profile
        self.path, self.certificate = compile_joint_path(
            points,
            np.minimum(p.speed, 0.03),
            np.minimum(p.acceleration, 0.06),
            np.minimum(4 * np.asarray(p.acceleration), 0.24),
        )
        self.certificate.update(
            {
                "gesture": "wrist_wave",
                "amplitude_rad": 0.06,
                "base_pose_fixed": True,
                "gripper_target_fixed": True,
                "initial_hold_s": self.initial_hold_s,
                "hardware_executable": False,
            }
        )
        duration = self.initial_hold_s + self.path.end
        times = np.r_[np.arange(0, duration, p.dt), duration]
        hands = [self.fk(*self.coordinates(t)) for t in times]
        super().__init__(
            times,
            np.tile(self.start_base, (len(times), 1)),
            hands,
            np.full(len(times), control.grip_reference),
            base_mask=[True] * 3,
            hand_mask=True,
        )

    def coordinates(self, time):
        joints = self.path.sample(float(np.clip(time - self.initial_hold_s, 0, self.path.end)))[0]
        return self.start_base.copy(), joints.copy()

    def sample(self, time):
        base, joints = self.coordinates(time)
        return base, self.fk(base, joints), float(self.gripper[0])

    def record(self):
        return {
            **super().record(),
            "format": "herman-wave-reference-v1",
            "certificate": self.certificate,
            "joint_witness": self.path.record(),
            "fk_anchor_qpos": self.fk.qpos.tolist(),
        }


def wave_tracking_gate(trace, end):
    """Require tracking through the wave, not merely returning to its endpoint."""
    samples = [s for s in trace if 2.0 - 1e-8 <= s["time_s"] <= end + 1e-8]
    complete = bool(trace and trace[-1]["time_s"] >= end - 1e-8 and samples)
    errors = np.array([[s["position_error_m"], s["rotation_error_rad"]] for s in samples])
    finite = bool(len(samples) and np.isfinite(errors).all() and np.all(errors >= 0))
    peak = errors.max(axis=0).tolist() if finite else [None, None]
    return {
        "complete_wave_observed": complete,
        "peak_position_error_m": peak[0],
        "peak_orientation_error_rad": peak[1],
        "position_tolerance_m": 0.003,
        "orientation_tolerance_rad": 0.02,
        "passed": bool(complete and finite and np.all(np.asarray(peak) <= [0.003, 0.02])),
    }
