"""Sensor-only motion commands and separately named simulator critic features."""

import numpy as np
from scipy.spatial.transform import Rotation

ACTOR_KEYS = ("history", "preview", "motion")
PRIVILEGED_WIDTH = 34
PRIVILEGED_FIELDS = {
    "payload_mass_kg": (0, 1),
    "payload_pose_in_hand": (1, 7),
    "finger_contact_magnitudes": (7, 13),
    "applied_ee_wrench_world": (13, 19),
    "payload_support_normal_N": (19, 20),
    "wheel_normal_loads_N": (20, 24),
    "payload_friction": (24, 27),
    "retention_intent": (27, 28),
    "base_projected_gravity_xy": (28, 30),
    "hand_pose_errors": (30, 32),
    "elapsed_and_remaining_fraction": (32, 34),
}


def pose_rotation(pose):
    columns = np.asarray(pose)[3:].reshape(3, 2)
    return np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])


def reference_velocity(control, interval=0.001):
    """World SE(2)/SE(3) twist of the announced, actually retimed reference.

    Only the reference sampler is queried. Its private FK may run; physics is
    never stepped or edited. Static pose commands have zero feedforward.
    """
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("positive finite reference derivative interval required")
    if control.preview_sampler is None:
        return np.zeros(9, dtype=np.float32)
    now = float(control.data.time)
    a = control.preview_sampler(now - interval)
    b = control.preview_sampler(now + interval)
    base = (np.asarray(b[0]) - a[0]) / (2 * interval)
    base[2] = np.arctan2(np.sin(b[0][2] - a[0][2]), np.cos(b[0][2] - a[0][2])) / (2 * interval)
    linear = (np.asarray(b[1])[:3] - a[1][:3]) / (2 * interval)
    angular = Rotation.from_matrix(pose_rotation(b[1]) @ pose_rotation(a[1]).T).as_rotvec() / (2 * interval)
    value = np.r_[base * control.base_mask, np.r_[linear, angular] * control.hand_mask]
    if not np.isfinite(value).all():
        raise ValueError("nonfinite announced reference velocity")
    return value.astype(np.float32)


def actor_observation(control, observation=None):
    result = control.observe() if observation is None else observation
    return {k: np.asarray(result[k], dtype=np.float32).copy() for k in ("history", "preview")} | {
        "motion": reference_velocity(control)
    }


def privileged_state(env, elapsed=0.0, horizon=12.0):
    """Training critic only; never passed to an actor or exported in a bundle."""
    from emet.rl.contact import solved_contact_force
    from emet.rl.manipulation.safety import relative_pose

    c, m, d = env.control, env.model, env.data
    relative, rotation = relative_pose(d, c.ee, env.payload)
    fingers = [m.body(n).id for n in c.profile.finger_bodies]
    wheels = [m.body(n).id for n in c.profile.wheel_bodies]
    finger_load, wheel_load, support = np.zeros((2, 3)), np.zeros(4), 0.0
    table = m.body("table").id
    geom_bodies = np.asarray(m.geom_bodyid).copy()
    for i, contact in enumerate(d.contact[: d.ncon]):
        geoms = np.asarray(contact.geom).copy()
        if np.any(geoms < 0) or np.any(geoms >= m.ngeom):
            raise ValueError("invalid active critic contact geometry")
        bodies = set(geom_bodies[geoms])
        normal, linear_bound = solved_contact_force(m, d, i)
        for index, body in enumerate(fingers):
            if body in bodies:
                # N, N (tangential L1 bound), and contact count; no mixed N/Nm.
                finger_load[index] += [normal, max(0.0, linear_bound - normal), float(normal > 0.01)]
        for index, body in enumerate(wheels[:4]):
            if body in bodies:
                wheel_load[index] += normal
        if bodies == {env.payload, table}:
            support += normal
    current = c.ee_pose()
    error = [
        np.linalg.norm(c.hand_goal[:3] - current[:3]),
        np.linalg.norm(Rotation.from_matrix(pose_rotation(c.hand_goal) @ pose_rotation(current).T).as_rotvec()),
    ]
    value = np.r_[
        float(m.body_mass[env.payload]),
        relative,
        Rotation.from_matrix(rotation).as_rotvec(),
        finger_load.ravel(),
        d.xfrc_applied[c.ee].copy(),
        support,
        wheel_load,
        m.geom_friction[int(m.body_geomadr[env.payload])].copy(),
        float(c.retain_object),
        (d.xmat[c.base].reshape(3, 3).T @ [0.0, 0.0, -1.0])[:2],
        error,
        elapsed / horizon,
        max(0, 1 - elapsed / horizon),
    ]
    if value.shape != (PRIVILEGED_WIDTH,) or not np.isfinite(value).all():
        raise ValueError("invalid privileged critic state")
    return value.astype(np.float32)
