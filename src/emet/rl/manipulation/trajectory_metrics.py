"""Within-trajectory motion/spread, distinct from a dataset's reset-pose cloud."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def reference_is_moving(motion):
    """Useful-speed diagnostic, on the already-masked world reference twist."""
    motion = np.asarray(motion, float)
    if motion.shape != (9,) or not np.isfinite(motion).all():
        raise ValueError("finite masked nine-dimensional reference twist required")
    return bool(
        np.linalg.norm(motion[:2]) > 0.001
        or abs(motion[2]) > 0.005
        or np.linalg.norm(motion[3:6]) > 0.001
        or np.linalg.norm(motion[6:]) > 0.005
    )


def rotations(hand):
    columns = hand[:, 3:].reshape(-1, 3, 2)
    result = np.concatenate([columns, np.cross(columns[:, :, 0], columns[:, :, 1])[:, :, None]], axis=2)
    if not np.allclose(np.transpose(result, (0, 2, 1)) @ result, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("valid hand rotation columns required")
    return result


def pose_spread(position, rotation, dt):
    steps = np.linalg.norm(np.diff(position, axis=0), axis=1)
    angles = Rotation.from_matrix(rotation[1:] @ np.transpose(rotation[:-1], (0, 2, 1))).magnitude()
    offset = np.linalg.norm(position - position[0], axis=1)
    orientation = Rotation.from_matrix(rotation @ rotation[0].T).magnitude()
    centered = position - position.mean(0)
    return (
        {
            "maximum_excursion_m": float(offset.max()),
            "path_length_m": float(steps.sum()),
            "endpoint_displacement_m": float(offset[-1]),
            "axis_span_m": np.ptp(position, axis=0).tolist(),
            "position_covariance_eigenvalues_m2": np.linalg.eigvalsh(centered.T @ centered / len(position)).tolist(),
            "rotation_excursion_rad": float(orientation.max()),
            "rotation_travel_rad": float(angles.sum()),
            "peak_linear_speed_m_s": float((steps / dt).max()),
            "mean_linear_speed_m_s": float(steps.sum() / dt.sum()),
            "peak_angular_speed_rad_s": float((angles / dt).max()),
        },
        steps,
        angles,
    )


def trajectory_spread(times, hand, base, *, hand_mask=True, base_mask=(True, True, True), joints=None):
    """Time-weighted motion statistics; local hand uses the supplied SE(2) frame.

    Base-only hand sweeps cannot count as active hand motion. Local hand spread
    removes planar base translation/yaw; it is not a full measured SE(3) torso
    transform. Closed paths retain excursion/path length despite zero endpoint.
    Numerical-motion and useful-speed duty fractions are reported separately.
    """
    times, hand, base = (np.asarray(x, float) for x in (times, hand, base))
    n = len(times)
    mask = np.asarray(base_mask)
    if (
        times.shape != (n,)
        or n < 2
        or hand.shape != (n, 9)
        or base.shape != (n, 3)
        or not all(np.isfinite(x).all() for x in (times, hand, base))
        or np.any(np.diff(times) <= 0)
        or type(hand_mask) is not bool
        or mask.shape != (3,)
        or mask.dtype != np.dtype(bool)
    ):
        raise ValueError("finite increasing times, complete poses and boolean masks required")
    dt = np.diff(times)
    r = rotations(hand)
    rb = Rotation.from_euler("z", base[:, 2]).as_matrix()
    local = np.einsum("nji,nj->ni", rb, hand[:, :3] - np.column_stack([base[:, :2], np.zeros(n)]))
    relative = np.einsum("nji,njk->nik", rb, r)
    world, dp, dr = pose_spread(hand[:, :3], r, dt)
    body, _, _ = pose_spread(local, relative, dt)
    delta = np.diff(base, axis=0)
    delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
    delta *= mask
    db, dy = np.linalg.norm(delta[:, :2], axis=1), abs(delta[:, 2])
    numerical = (db > 1e-6) | (dy > 1e-6)
    useful = (db / dt > 0.001) | (dy / dt > 0.005)
    if hand_mask:
        numerical |= (dp > 1e-6) | (dr > 1e-6)
        useful |= (dp / dt > 0.001) | (dr / dt > 0.005)
    result = {
        "format": "trajectory-spread-v1",
        "duration_s": float(dt.sum()),
        "hand_active": hand_mask,
        "base_mask": mask.tolist(),
        "world_hand": world if hand_mask else None,
        "base_relative_hand": body if hand_mask else None,
        "base": {
            "path_length_m": float(db.sum()),
            "yaw_travel_rad": float(dy.sum()),
            "maximum_xy_excursion_m": float(np.linalg.norm((base - base[0])[:, :2] * mask[:2], axis=1).max()),
        },
        "reference_motion_s": float(dt[numerical].sum()),
        "reference_motion_fraction": float(dt[numerical].sum() / dt.sum()),
        "useful_speed_motion_s": float(dt[useful].sum()),
        "useful_speed_motion_fraction": float(dt[useful].sum() / dt.sum()),
        "stationary_reference_s": float(dt[~numerical].sum()),
        "speed_thresholds": {"position_m_s": 0.001, "rotation_rad_s": 0.005},
        "numerical_step_thresholds": {"position_m": 1e-6, "rotation_rad": 1e-6},
    }
    if joints is not None:
        joints = np.asarray(joints, float)
        if joints.ndim != 2 or joints.shape[0] != n or not np.isfinite(joints).all():
            raise ValueError("one finite joint vector per timestamp required")
        result["joint_span_rad"] = np.ptp(joints, axis=0).tolist()
        result["joint_travel_rad"] = np.abs(np.diff(joints, axis=0)).sum(0).tolist()
    return result


@dataclass(frozen=True)
class SpreadRequirements:
    minimum_local_position_excursion_m: float = 0.05
    minimum_rotation_excursion_rad: float = 0.35
    mode: str = "either"

    def __post_init__(self):
        values = [self.minimum_local_position_excursion_m, self.minimum_rotation_excursion_rad]
        if (
            any(isinstance(x, (bool, np.bool_)) for x in values)
            or not np.isfinite(values).all()
            or min(values) <= 0
            or values[0] > 0.3
            or values[1] > np.pi
            or self.mode not in {"either", "both"}
        ):
            raise ValueError("bounded positive pose-spread requirements and either/both mode required")

    def accepts(self, metrics):
        if not metrics["hand_active"]:
            return False
        local = metrics["base_relative_hand"]
        conditions = (
            local["maximum_excursion_m"] >= self.minimum_local_position_excursion_m,
            local["rotation_excursion_rad"] >= self.minimum_rotation_excursion_rad,
        )
        return bool(any(conditions) if self.mode == "either" else all(conditions))

    def record(self):
        return asdict(self)
