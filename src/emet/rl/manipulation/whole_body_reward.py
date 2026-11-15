"""Explicit three-group tracking rewards; independent hard guards still stop motion."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.whole_body_observation import pose_rotation

REWARD_GROUPS = ("tracking", "stability", "contact_safety")


def current_contact_force(env):
    """Current maximum body-pair load, excluding normal wheel/floor support.

    A read-only equivalent of the monitor's force calculation: do not update
    its running maxima/impulse twice or charge an old peak every future tick.
    """
    from emet.rl.contact import solved_contact_force

    monitor, model, data = env.monitor, env.model, env.data
    bodies = np.asarray(model.geom_bodyid).copy()
    pairs = {}
    for i, contact in enumerate(data.contact[: data.ncon]):
        a, b = map(int, np.asarray(contact.geom).copy())
        if not (0 <= a < model.ngeom and 0 <= b < model.ngeom):
            raise ValueError("invalid active contact geometry")
        if a not in monitor.robot_geoms and b not in monitor.robot_geoms and env.payload not in (bodies[a], bodies[b]):
            continue
        if (a in monitor.support_geoms and b in monitor.floor) or (b in monitor.support_geoms and a in monitor.floor):
            continue
        pair = tuple(sorted((int(bodies[a]), int(bodies[b]))))
        pairs[pair] = pairs.get(pair, 0.0) + solved_contact_force(model, data, i)[1]
    return max(pairs.values(), default=0.0)


class RewardMeter:
    def __init__(self, env):
        self.env = env
        c = env.control
        self.velocity, self.acceleration = c.velocities().copy(), np.zeros(c.profile.arm_size)
        jp, jr = np.zeros((3, env.model.nv)), np.zeros((3, env.model.nv))
        mujoco.mj_jacBody(env.model, env.data, jp, jr, c.base)
        self.base_velocity = np.r_[(jp @ env.data.qvel.copy())[:2], (jr @ env.data.qvel.copy())[2]]
        self.base_acceleration = np.zeros(3)
        self.action = c.last_action.copy()
        self.last_terms = {}

    def tick(self, action, motion):
        e, c = self.env, self.env.control
        p, dt = c.profile, c.profile.dt
        hand = c.ee_pose()
        pe = float(np.linalg.norm(c.hand_goal[:3] - hand[:3]))
        re = float(np.linalg.norm(Rotation.from_matrix(pose_rotation(c.hand_goal) @ pose_rotation(hand).T).as_rotvec()))
        jp, jr = np.zeros((3, e.model.nv)), np.zeros((3, e.model.nv))
        mujoco.mj_jacBody(e.model, e.data, jp, jr, c.ee)
        twist = np.r_[jp @ e.data.qvel.copy(), jr @ e.data.qvel.copy()]
        velocity_error = twist - motion[3:]
        hand_score = (
            (
                np.exp(-pe / 0.01 - re / 0.1)
                + 0.5 * np.exp(-np.sum((velocity_error[:3] / 0.02) ** 2) - np.sum((velocity_error[3:] / 0.2) ** 2))
            )
            if c.hand_mask
            else 0.0
        )
        delta = c.base_goal - c.base_pose()
        delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
        outside = max(0, np.linalg.norm(delta[:2] * c.base_mask[:2]) - e.config.base_xy_tolerance)
        base_score = np.exp(-outside / 0.02 - abs(delta[2] * c.base_mask[2]) / 0.05) if c.base_mask.any() else 0.0
        velocity = c.velocities().copy()
        acceleration = (velocity - self.velocity) / dt
        jerk = (acceleration - self.acceleration) / dt
        mujoco.mj_jacBody(e.model, e.data, jp, jr, c.base)
        base_velocity = np.r_[(jp @ e.data.qvel.copy())[:2], (jr @ e.data.qvel.copy())[2]]
        base_velocity_error = (base_velocity - motion[:3]) * c.base_mask
        if c.base_mask.any():
            base_score += 0.5 * np.exp(-np.sum((base_velocity_error / [0.02, 0.02, 0.1]) ** 2))
        base_acceleration = (base_velocity - self.base_velocity) / dt
        base_jerk = (base_acceleration - self.base_acceleration) / dt
        action = np.asarray(action, dtype=float)
        scaled_jerk = np.r_[jerk / (4 * np.asarray(p.acceleration)), base_jerk / [1.0, 1.0, 4.0]]
        jerk_cost = np.sum(np.sqrt(1 + scaled_jerk**2) - 1)
        command_cost = np.sum((action[:-1] - self.action[:-1]) ** 2)
        effort = np.sum(action[:-1] ** 2)
        force = current_contact_force(e)
        force_cost = (max(0, force - 3.0) / 3.0) ** 2
        bad = bool(
            e.monitor.wheel_lift or e._physics_invalid or force > p.unsafe_force or getattr(e, "retention_lost", False)
        )
        groups = np.array(
            [
                dt * (hand_score + base_score),
                -dt * (0.01 * jerk_cost + 0.001 * effort) - 0.01 * command_cost,
                -dt * 0.1 * force_cost - 10.0 * bad,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(groups).all():
            raise ValueError("nonfinite whole-body reward")
        self.velocity, self.acceleration = velocity, acceleration
        self.base_velocity, self.base_acceleration, self.action = base_velocity, base_acceleration, action.copy()
        self.last_terms = {
            "hand_pose_score": float(hand_score),
            "base_pose_score": float(base_score),
            "linear_velocity_error_m_s": float(np.linalg.norm(velocity_error[:3])),
            "angular_velocity_error_rad_s": float(np.linalg.norm(velocity_error[3:])),
            "base_velocity_error": base_velocity_error.tolist(),
            "jerk_cost": float(jerk_cost),
            "command_cost": float(command_cost),
            "force_excess_cost": float(force_cost),
            "guard_failed": bad,
        }
        return groups
