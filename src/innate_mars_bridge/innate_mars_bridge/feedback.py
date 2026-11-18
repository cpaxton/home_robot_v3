"""Source-timed Mars feedback, independent of ROS and command authority.

Structural validity is deliberately separate from verified units/bus acquisition.
A fresh publisher can still report fabricated zeros; this store cannot certify it.
"""

import copy
import math
import threading
import time

import numpy as np


class FeedbackStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.samples = {}
        self.sequence = {"arm": 0, "odom": 0}

    def _save(self, kind, stamp_ns, payload, error=None):
        with self.lock:
            self.sequence[kind] += 1
            self.samples[kind] = {
                "sequence": self.sequence[kind],
                "source_stamp_ns": int(stamp_ns),
                "received_monotonic": time.monotonic(),
                "structurally_valid": error is None,
                "error": error,
                **payload,
            }

    def arm(self, names, positions, velocities, efforts, stamp_ns):
        required = [f"joint{i}" for i in range(1, 7)]
        try:
            if len(names) != len(set(names)) or len(names) != len(positions) or len(names) != len(velocities):
                raise ValueError("complete unique joint names, positions and velocities required")
            indexes = [names.index(name) for name in required]
            q, dq = np.asarray(positions, float)[indexes], np.asarray(velocities, float)[indexes]
            if stamp_ns <= 0 or not np.isfinite(np.r_[q, dq]).all():
                raise ValueError("finite arm state and positive source stamp required")
            effort = np.asarray(efforts, float)[indexes] if len(efforts) == len(names) else None
            if effort is not None and not np.isfinite(effort).all():
                effort = None
            self._save(
                "arm",
                stamp_ns,
                {
                    "names": required,
                    "position": q.tolist(),
                    "velocity": dq.tolist(),
                    "effort_proxy": None if effort is None else effort.tolist(),
                },
            )
        except (ValueError, IndexError, TypeError) as exc:
            self._save("arm", stamp_ns, {}, str(exc))

    def odom(self, position, quaternion_xyzw, twist, stamp_ns, frame_id, child_frame_id):
        try:
            p, q, v = np.asarray(position, float), np.asarray(quaternion_xyzw, float), np.asarray(twist, float)
            if p.shape != (3,) or q.shape != (4,) or v.shape != (6,) or not np.isfinite(np.r_[p, q, v]).all():
                raise ValueError("finite complete odometry pose and six-component twist required")
            if stamp_ns <= 0 or not frame_id or not child_frame_id or abs(np.linalg.norm(q) - 1) > 1e-3:
                raise ValueError("source stamp, frames and unit quaternion required")
            x, y, z, w = q
            rotation = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            world_linear, world_angular = rotation @ v[:3], rotation @ v[3:]
            self._save(
                "odom",
                stamp_ns,
                {
                    "frame_id": frame_id,
                    "child_frame_id": child_frame_id,
                    "base_pose": [float(p[0]), float(p[1]), math.atan2(rotation[1, 0], rotation[0, 0])],
                    "position": p.tolist(),
                    "quaternion_xyzw": q.tolist(),
                    "twist_child": v.tolist(),
                    "base_velocity_world": [float(world_linear[0]), float(world_linear[1]), float(world_angular[2])],
                },
            )
        except (ValueError, TypeError) as exc:
            self._save("odom", stamp_ns, {}, str(exc))

    def snapshot(self, ros_now_ns, *, max_age_s=0.2):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("positive finite maximum age required")
        with self.lock:
            samples = copy.deepcopy(self.samples)
        now = time.monotonic()
        for sample in samples.values():
            age = (ros_now_ns - sample["source_stamp_ns"]) / 1e9
            sample["source_age_s"] = age
            sample["arrival_age_s"] = now - sample.pop("received_monotonic")
            sample["fresh"] = bool(
                sample["structurally_valid"] and 0 <= age <= max_age_s and 0 <= sample["arrival_age_s"] <= max_age_s
            )
        return {
            "format": "mars-source-feedback-v1",
            "clock_domain": "robot_ros",
            "samples": samples,
            "complete_and_fresh": all(k in samples and samples[k]["fresh"] for k in ("arm", "odom")),
            "units_verified": False,
            "motor_read_validity_verified": False,
            "effort_units": "driver_current_or_load_proxy_unverified",
        }
