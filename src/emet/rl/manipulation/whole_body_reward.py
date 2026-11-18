"""Explicit three-group tracking rewards; independent hard guards still stop motion."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.reward_config import WholeBodyRewardConfig, hand_tracking_terms
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
    # Copy the active scalar contact arrays once, exactly as the critic path
    # does. Never construct individual C-backed contact objects: that slice/
    # iterator has crashed in long mixed rollouts, and this reward meter runs
    # every physical control tick rather than once per decision.
    contacts = data.contact
    count = min(int(data.ncon), len(contacts))
    geom1 = contacts.geom1[:count].copy()
    geom2 = contacts.geom2[:count].copy()
    addresses = contacts.efc_address[:count].copy()
    dimensions = contacts.dim[:count].copy()
    excludes = contacts.exclude[:count].copy()
    friction = contacts.friction[:count].copy()
    forces = np.asarray(data.efc_force).copy()
    pairs = {}
    for i in range(count):
        a, b = int(geom1[i]), int(geom2[i])
        if not (0 <= a < model.ngeom and 0 <= b < model.ngeom):
            raise ValueError("invalid active contact geometry")
        if a not in monitor.robot_geoms and b not in monitor.robot_geoms and env.payload not in (bodies[a], bodies[b]):
            continue
        if (a in monitor.support_geoms and b in monitor.floor) or (b in monitor.support_geoms and a in monitor.floor):
            continue
        pair = tuple(sorted((int(bodies[a]), int(bodies[b]))))
        _, bound = solved_contact_force(
            model,
            data,
            i,
            efc_address=addresses[i],
            dim=dimensions[i],
            exclude=excludes[i],
            efc_force=forces,
            friction=friction[i],
        )
        pairs[pair] = pairs.get(pair, 0.0) + bound
    return max(pairs.values(), default=0.0)


class RewardMeter:
    def __init__(self, env, *, base_heading_region=False, config=None):
        self.env = env
        self.config = WholeBodyRewardConfig.from_record(config)
        self.base_heading_region = base_heading_region
        self.before_hand = None
        c = env.control
        self.velocity, self.acceleration = c.velocities().copy(), np.zeros(c.profile.arm_size)
        jp, jr = np.zeros((3, env.model.nv)), np.zeros((3, env.model.nv))
        mujoco.mj_jacBody(env.model, env.data, jp, jr, c.base)
        self.base_velocity = np.r_[(jp @ env.data.qvel.copy())[:2], (jr @ env.data.qvel.copy())[2]]
        self.base_acceleration = np.zeros(3)
        self.action = c.last_action.copy()
        self.last_terms = {}

    def record(self):
        return {
            "configuration": self.config.record(),
            "hand_position_tolerance_m": self.env.config.hand_position_tolerance,
            "hand_rotation_tolerance_rad": self.env.config.hand_orientation_tolerance,
            "base_heading_region": self.base_heading_region,
            "progress": "measured_pose_change_against_same_poststep_active_goal",
            "episode_contract": "finite_tracking_deadline_v1",
        }

    def before_step(self):
        """Capture measured pose, without advancing goals/history/physics.

        Progress compares this pose and the next measured pose against the same
        newly active goal. Goal motion alone therefore cannot earn progress.
        """
        self.before_hand = self.env.control.ee_pose().copy()
        self.before_hand_active = bool(self.env.control.hand_mask)

    def terminal_bonus(self, success):
        return self.config.success_bonus if success else -self.config.failure_penalty

    def tick(self, action, motion):
        e, c = self.env, self.env.control
        p, dt = c.profile, c.profile.dt
        settings = self.config
        hand = c.ee_pose()
        pe = float(np.linalg.norm(c.hand_goal[:3] - hand[:3]))
        re = float(np.linalg.norm(Rotation.from_matrix(pose_rotation(c.hand_goal) @ pose_rotation(hand).T).as_rotvec()))
        jp, jr = np.zeros((3, e.model.nv)), np.zeros((3, e.model.nv))
        mujoco.mj_jacBody(e.model, e.data, jp, jr, c.ee)
        twist = np.r_[jp @ e.data.qvel.copy(), jr @ e.data.qvel.copy()]
        velocity_error = twist - motion[3:]
        hand_pose_score, hand_velocity_score, pose_gate = hand_tracking_terms(
            settings,
            pe,
            re,
            velocity_error,
            e.config.hand_position_tolerance,
            e.config.hand_orientation_tolerance,
        )
        hand_score = hand_pose_score + hand_velocity_score if c.hand_mask else 0.0
        progress = 0.0
        if settings.progress_weight:
            if self.before_hand is None:
                raise ValueError("progress reward requires before_step for each physical tick")
            if c.hand_mask and self.before_hand_active:
                before_pe = np.linalg.norm(c.hand_goal[:3] - self.before_hand[:3])
                before_re = Rotation.from_matrix(
                    pose_rotation(c.hand_goal) @ pose_rotation(self.before_hand).T
                ).magnitude()

                def potential(position, rotation):
                    return np.log1p(position / e.config.hand_position_tolerance) + np.log1p(
                        rotation / e.config.hand_orientation_tolerance
                    )

                progress = settings.progress_weight * (potential(before_pe, before_re) - potential(pe, re))
        self.before_hand = None
        delta = c.base_goal - c.base_pose()
        delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
        outside = max(0, np.linalg.norm(delta[:2] * c.base_mask[:2]) - e.config.base_xy_tolerance)
        heading_error = abs(delta[2] * c.base_mask[2])
        if self.base_heading_region:
            heading_error = max(0, heading_error - e.config.base_heading_tolerance)
        base_score = (
            settings.base_pose_weight
            * np.exp(-outside / settings.base_position_scale_m - heading_error / settings.base_heading_scale_rad)
            if c.base_mask.any()
            else 0.0
        )
        velocity = c.velocities().copy()
        acceleration = (velocity - self.velocity) / dt
        jerk = (acceleration - self.acceleration) / dt
        mujoco.mj_jacBody(e.model, e.data, jp, jr, c.base)
        base_velocity = np.r_[(jp @ e.data.qvel.copy())[:2], (jr @ e.data.qvel.copy())[2]]
        base_velocity_error = (base_velocity - motion[:3]) * c.base_mask
        if c.base_mask.any():
            base_gate = pose_gate if settings.pose_gated_velocity and c.hand_mask else 1.0
            scales = [
                settings.base_linear_velocity_scale_m_s,
                settings.base_linear_velocity_scale_m_s,
                settings.base_angular_velocity_scale_rad_s,
            ]
            base_score += (
                settings.base_velocity_weight * base_gate * np.exp(-np.sum((base_velocity_error / scales) ** 2))
            )
        base_acceleration = (base_velocity - self.base_velocity) / dt
        base_jerk = (base_acceleration - self.base_acceleration) / dt
        action = np.asarray(action, dtype=float)
        scaled_jerk = np.r_[jerk / (4 * np.asarray(p.acceleration)), base_jerk / [1.0, 1.0, 4.0]]
        jerk_cost = np.sum(np.sqrt(1 + scaled_jerk**2) - 1)
        command_cost = np.sum((action[:-1] - self.action[:-1]) ** 2)
        effort = np.sum(action[:-1] ** 2)
        force = current_contact_force(e)
        force_cost = (max(0, force - settings.force_free_N) / settings.force_scale_N) ** 2
        bad = bool(
            e.monitor.wheel_lift or e._physics_invalid or force > p.unsafe_force or getattr(e, "retention_lost", False)
        )
        groups = np.array(
            [
                dt * (hand_score + base_score) + progress,
                settings.stability_scale
                * (
                    -dt * (settings.jerk_weight * jerk_cost + settings.effort_weight * effort)
                    - settings.command_delta_weight * command_cost
                ),
                -dt * settings.force_weight * force_cost - settings.guard_penalty * bad,
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
            "hand_pose_component": float(hand_pose_score if c.hand_mask else 0),
            "hand_velocity_component": float(hand_velocity_score if c.hand_mask else 0),
            "hand_pose_gate": float(pose_gate if c.hand_mask else 1),
            "hand_progress_reward": float(progress),
            "linear_velocity_error_m_s": float(np.linalg.norm(velocity_error[:3])),
            "angular_velocity_error_rad_s": float(np.linalg.norm(velocity_error[3:])),
            "base_velocity_error": base_velocity_error.tolist(),
            "jerk_cost": float(jerk_cost),
            "command_cost": float(command_cost),
            "force_excess_cost": float(force_cost),
            "guard_failed": bad,
        }
        return groups
