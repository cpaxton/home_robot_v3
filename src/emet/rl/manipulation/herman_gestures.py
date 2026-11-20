"""Small FK-witnessed gestures for offline Herman rehearsal, never motor commands."""

import numpy as np

from emet.rl.manipulation.jerk_reference import compile_joint_path
from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory


def wave_case(gesture="elbow_wrist_wave"):
    if gesture not in {"wrist_wave", "elbow_wrist_wave", "normal_wave"}:
        raise ValueError("unknown wave gesture")
    version, seed = {
        "wrist_wave": (1, 430_010_000),
        "elbow_wrist_wave": (2, 430_020_000),
        "normal_wave": (3, 430_030_000),
    }[gesture]
    return {
        "id": f"herman-wave-v{version}:{gesture}",
        "seed": seed,
        "split": "audit",
        "mode": "coupled",
        "travel_m": 0.0,
        "turn_rad": 0.0,
        "arm_delta": [0.0] * 5,
        "base_mask": [True] * 3,
        "gesture": gesture,
        "initial_kind": "synthetic_geometry_admitted_staging_proposal",
        "hardware_executable": False,
    }


class HermanWaveReference(TaskPoseTrajectory):
    """Raise the forearm, wave the elbow and wrist, then return and hold.

    The base pose and gripper reference stay fixed. A joint-space FK witness is
    private geometry; the actor receives only the normal SE(2)/SE(3) preview.
    Derivative limits describe the reference, not measured policy motion or a
    calibrated hardware envelope. No physical staging path is supplied here.
    """

    def __init__(self, control, *, gesture="elbow_wrist_wave"):
        if control.profile.robot != "innate_mars":
            raise ValueError("Herman wave requires Mars")
        self.fk = TaskFrameFK(control.model, control.data, control)
        self.start_base = control.base_pose().copy()
        self.fixed_hand = None
        self.initial_hold_s = 2.0
        initial = control.positions().copy()
        if gesture == "wrist_wave":
            delta = np.array([0, 0, 0, 0, 0.06])
            points = np.array([initial, initial + delta, initial, initial - delta, initial])
            speed, acceleration, jerk = 0.03, 0.06, 0.24
        elif gesture == "elbow_wrist_wave":
            raised = initial + [0, 0.12, -0.45, 0.33, 0]
            left = raised + [0, 0, 0.20, -0.12, 0.30]
            right = raised + [0, 0, -0.20, 0.12, -0.30]
            points = np.array([initial, raised, left, right, left, raised, initial])
            speed, acceleration, jerk = 0.25, 0.5, 2.0
        elif gesture == "normal_wave":
            self.initial_hold_s = 0.2
            raised = initial + [0, 0.035, -0.12, 0.085, 0.26]
            points = np.array([initial, raised, initial])
            speed, acceleration, jerk = 0.6, 1.5, 6.0
        else:
            raise ValueError("unknown wave gesture")
        if np.any(points < control.limits[:, 0] + 0.002) or np.any(points > control.limits[:, 1] - 0.002):
            raise ValueError("wave witness exceeds joint margin; do not clamp")
        p = control.profile
        self.path, self.certificate = compile_joint_path(
            points,
            np.minimum(p.speed, speed),
            np.minimum(p.acceleration, acceleration),
            np.minimum(4 * np.asarray(p.acceleration), jerk),
        )
        self.certificate.update(
            {
                "gesture": gesture,
                "joint_names": [group[0] for group in p.joint_groups],
                "joint_waypoints_rad": points.tolist(),
                "joint_span_rad": np.ptp(points, axis=0).tolist(),
                "test_speed_limit_rad_s": speed,
                "test_acceleration_limit_rad_s2": acceleration,
                "test_jerk_limit_rad_s3": jerk,
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


def wave_tracking_gate(trace, end, *, initial_hold_s=2.0):
    """Require tracking through the wave, not merely returning to its endpoint."""

    def reference_time(sample):
        return sample.get("reference_time_s", sample["time_s"])

    samples = [s for s in trace if initial_hold_s - 1e-8 <= reference_time(s) <= end + 1e-8]
    complete = bool(trace and reference_time(trace[-1]) >= end - 1e-8 and samples)
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
