"""Shared state, goal, observation and actuation code for training and serving."""

from dataclasses import dataclass

import mujoco
import numpy as np

from emet.rl.model import PROXIES
from emet.rl.profile import MARS


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def invalid_physics(data):
    # MuJoCo can reset diverging state to finite values; inspect warnings as well.
    return bool(
        not np.isfinite(data.qpos).all()
        or not np.isfinite(data.qvel).all()
        or any(
            data.warning[k].number
            for k in (
                mujoco.mjtWarning.mjWARN_BADQPOS,
                mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC,
            )
        )
    )


@dataclass
class Goal:
    base: np.ndarray
    arm: np.ndarray
    mask: np.ndarray

    @classmethod
    def parse(cls, payload, control):
        if not isinstance(payload, dict) or set(payload) - {"base", "arm", "frame"}:
            raise ValueError("goal requires base and/or named arm targets")
        if payload.get("frame", "world") != "world":
            raise ValueError("reactive goals use the world frame")
        base, arm = control.base_pose(), control.arm_positions().copy()
        mask = np.zeros(6)
        if "base" in payload:
            base = np.asarray(payload["base"], dtype=float)
            if base.shape != (3,) or not np.isfinite(base).all():
                raise ValueError("base target must contain three finite coordinates")
            base = base.copy()
            base[2] = wrap(base[2])
            mask[0] = 1
        if "arm" in payload:
            targets = payload["arm"]
            if not isinstance(targets, dict) or set(targets) - set(control.profile.arm_joints):
                raise ValueError("unknown reaching joint (gripper and head are held separately)")
            for name, value in targets.items():
                i = control.profile.arm_joints.index(name)
                value = float(value)
                if not np.isfinite(value) or not control.limits[i, 0] <= value <= control.limits[i, 1]:
                    raise ValueError(f"target outside limits: {name}")
                arm[i], mask[i + 1] = value, 1
        if not mask.any():
            raise ValueError("empty reactive goal")
        return cls(base, arm, mask)

    def payload(self, profile=MARS):
        result = {"frame": "world"}
        if self.mask[0]:
            result["base"] = self.base.tolist()
        if self.mask[1:].any():
            result["arm"] = {
                n: float(v)
                for n, v, enabled in zip(profile.arm_joints, self.arm, self.mask[1:], strict=True)
                if enabled
            }
        return result


class MarsControl:
    """Owns targets, never the physics clock. Call apply() before each mj_step."""

    def __init__(self, model, data, profile=MARS):
        self.model, self.data, self.profile = model, data, profile
        self.arm_ids = np.array([model.joint(n).id for n in profile.arm_joints])
        self.arm_q = model.jnt_qposadr[self.arm_ids]
        self.arm_v = model.jnt_dofadr[self.arm_ids]
        self.arm_act = np.array([model.actuator(n).id for n in profile.arm_joints])
        self.base_ids = np.array([model.joint(n).id for n in profile.base_joints])
        self.base_q = model.jnt_qposadr[self.base_ids]
        self.base_v = model.jnt_dofadr[self.base_ids]
        self.base_act = np.array([model.actuator(n).id for n in profile.base_joints])
        self.base_body = model.body("base_link").id
        self.limits = model.jnt_range[self.arm_ids].copy()
        self.robot_geoms = [model.geom(f"rl_{n}").id for n in PROXIES]
        self.environment_geoms = [
            i
            for i in range(model.ngeom)
            if i not in self.robot_geoms and (model.geom_contype[i] or model.geom_conaffinity[i])
        ]
        excludes = {int(s) for s in model.exclude_signature}
        self.pairs = []
        for i, a in enumerate(self.robot_geoms):
            for b in self.environment_geoms + self.robot_geoms[i + 1 :]:
                ba, bb = sorted((int(model.geom_bodyid[a]), int(model.geom_bodyid[b])))
                if ba == bb or (ba << 16) + bb in excludes:
                    continue
                # MuJoCo filters parent-child contacts by default.
                if ba != 0 and (model.body_parentid[bb] == ba or model.body_parentid[ba] == bb):
                    continue
                self.pairs.append((a, b))
        self.ee_body = model.body("ee_link").id
        self._goal_data = mujoco.MjData(model)
        self.reset()

    def arm_positions(self):
        return self.data.qpos[self.arm_q]

    def base_pose(self):
        rotation = self.data.xmat[self.base_body].reshape(3, 3)
        return np.array([*self.data.xpos[self.base_body, :2], np.arctan2(rotation[1, 0], rotation[0, 0])])

    def reset(self):
        self._history = None
        self._history_time = None
        self._guard_time = float(self.data.time)
        self._blocked_time = np.zeros(7)
        self.obstruction = np.zeros(7, dtype=bool)
        self._blocked_direction = np.zeros(7)
        self.last_action = np.zeros(7)
        self.targets = self.arm_positions().copy()
        self.held = {
            n: float(self.data.qpos[self.model.jnt_qposadr[self.model.joint(n).id]]) for n in self.profile.held_joints
        }
        self.goal = Goal(self.base_pose(), self.targets.copy(), np.ones(6))
        self.settle_time = 0.0
        self.set_goal(self.goal)
        self.apply()

    def set_goal(self, goal):
        self.goal = Goal.parse(goal, self) if isinstance(goal, dict) else goal
        # Inactive joints hold the configuration sampled when accepting this goal.
        self.targets[~self.goal.mask[1:].astype(bool)] = self.goal.arm[~self.goal.mask[1:].astype(bool)]
        self.settle_time = 0.0
        # FK produces a reachable hand/body pair from the resolved joint goal.
        d = self._goal_data
        d.qpos[:] = self.data.qpos
        delta = self.goal.base[:2] - self.base_pose()[:2]
        axes = self.data.xaxis[self.base_ids]
        d.qpos[self.base_q[:2]] += np.linalg.solve(axes[:2, :2].T, delta)
        d.qpos[self.base_q[2]] += wrap(self.goal.base[2] - self.base_pose()[2]) / axes[2, 2]
        d.qpos[self.arm_q] = self.goal.arm
        mujoco.mj_forward(self.model, d)
        self.goal_pose = self.pose_features(d)

    def pose_features(self, data=None):
        d = self.data if data is None else data
        rotation = d.xmat[self.base_body].reshape(3, 3)
        # World poses, continuous rotation columns; the base is planar.
        return np.r_[
            d.xpos[self.ee_body],
            d.xmat[self.ee_body].reshape(3, 3)[:, :2].ravel(),
            d.xpos[self.base_body, :2],
            rotation[0, 0],
            rotation[1, 0],
        ]

    def stop(self):
        self.last_action[:] = 0
        if np.isfinite(self.arm_positions()).all():
            self.targets = self.arm_positions().copy()
        self.apply()

    def accept_action(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("policy action must contain seven finite values")
        action = np.clip(action, -1, 1)
        action[:2] *= self.goal.mask[0]
        action[2:] *= self.goal.mask[1:]
        # A latched obstruction prevents repeated policy commands from winding up.
        retreat = self.obstruction & (action * self._blocked_direction < 0)
        self.obstruction[retreat] = False
        self._blocked_time[retreat] = 0
        action[self.obstruction] = 0
        self.targets = np.clip(
            self.targets + action[2:] * self.profile.max_joint_speed * self.profile.control_dt,
            self.limits[:, 0],
            self.limits[:, 1],
        )
        self.targets = np.clip(
            self.targets,
            self.arm_positions() - self.profile.max_target_lead,
            self.arm_positions() + self.profile.max_target_lead,
        )
        self.targets = np.clip(self.targets, self.limits[:, 0], self.limits[:, 1])
        self.last_action = action.copy()

    def apply(self):
        # Run independently of inference at physics/feedback cadence. This detects
        # sustained tracking failure, not contact force. Retreat or reset clears the latch.
        now = float(self.data.time)
        dt = max(0.0, now - self._guard_time)
        self._guard_time = now
        residual = self.targets - self.arm_positions()
        stalled_arm = (np.abs(residual) > 0.06) & (np.abs(self.data.qvel[self.arm_v]) < 0.03)
        velocity = self.base_velocity()[[0, 2]]
        desired = self.last_action[:2] * [self.profile.max_v, self.profile.max_w]
        stalled_base = (np.abs(desired) > [0.08, 0.2]) & (velocity * np.sign(desired) < 0.5 * np.abs(desired))
        stalled = np.r_[stalled_base, stalled_arm]
        # Leaky evidence tolerates brief solver/encoder jitter without treating a
        # single delayed feedback sample as an obstruction.
        self._blocked_time = np.maximum(0.0, self._blocked_time + np.where(stalled, dt, -dt))
        detected = self._blocked_time >= self.profile.obstruction_seconds
        newly_blocked = detected & ~self.obstruction
        self._blocked_direction[newly_blocked] = np.sign(np.r_[desired, residual])[newly_blocked]
        self.obstruction |= detected
        self.last_action[self.obstruction] = 0
        blocked_arm = self.obstruction[2:]
        self.targets[blocked_arm] = self.arm_positions()[blocked_arm]
        self.targets = np.clip(
            self.targets,
            self.arm_positions() - self.profile.max_target_lead,
            self.arm_positions() + self.profile.max_target_lead,
        )
        self.data.ctrl[self.base_act] = 0
        if np.any(self.last_action[:2]):
            yaw = self.base_pose()[2]
            velocity = self.last_action[0] * self.profile.max_v * np.array([np.cos(yaw), np.sin(yaw)])
            # Planar axes may be rotated by scene composition; use their world axes.
            axes = self.data.xaxis[self.base_ids]
            planar = np.linalg.solve(axes[:2, :2].T, velocity)
            self.data.ctrl[self.base_act] = [*planar, self.last_action[1] * self.profile.max_w / axes[2, 2]]
        self.data.ctrl[self.arm_act] = self.targets
        for name, value in self.held.items():
            self.data.ctrl[self.model.actuator(name).id] = value

    def base_velocity(self):
        axes = self.data.xaxis[self.base_ids]
        world = axes[:2, :2].T @ self.data.qvel[self.base_v[:2]]
        yaw = self.base_pose()[2]
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array(
            [c * world[0] + s * world[1], -s * world[0] + c * world[1], axes[2, 2] * self.data.qvel[self.base_v[2]]]
        )

    def obstacle_features(self):
        features = np.zeros((len(self.robot_geoms), 5))
        features[:, 0] = self.profile.obstacle_range
        indices = {g: i for i, g in enumerate(self.robot_geoms)}
        nearest = np.empty(6)
        minimum = self.profile.obstacle_range
        for a, b in self.pairs:
            distance = mujoco.mj_geomDistance(self.model, self.data, a, b, self.profile.obstacle_range, nearest)
            minimum = min(minimum, distance)
            # Local obstacle input describes the scene. Fixed self-clearances and
            # the supporting floor would otherwise hide every nearby obstacle.
            # They remain part of collision detection above.
            if b in indices or self.model.geom_type[b] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if distance >= self.profile.obstacle_range:
                continue
            direction = nearest[3:] - nearest[:3]
            direction /= max(np.linalg.norm(direction), 1e-8)
            for geom, sign in ((a, 1), (b, -1)):
                if geom in indices and distance < features[indices[geom], 0]:
                    rotation = self.data.xmat[self.model.geom_bodyid[geom]].reshape(3, 3)
                    features[indices[geom]] = [distance, *(rotation.T @ (sign * direction)), 1]
        return features.astype(np.float32).ravel(), float(minimum)

    def errors(self):
        base = self.base_pose()
        delta = self.goal.base[:2] - base[:2]
        yaw = wrap(self.goal.base[2] - base[2])
        arm = (self.goal.arm - self.arm_positions()) * self.goal.mask[1:]
        return delta, yaw, arm

    def observe(self):
        delta, yaw_error, arm = self.errors()
        yaw = self.base_pose()[2]
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])
        obstacles, clearance = self.obstacle_features()
        obs = {
            "state": np.concatenate(
                [
                    self.arm_positions() / np.pi,
                    self.data.qvel[self.arm_v] / self.profile.max_joint_speed,
                    (self.targets - self.arm_positions()) / (self.profile.max_joint_speed * self.profile.control_dt),
                    self.base_velocity() / [self.profile.max_v, self.profile.max_v, self.profile.max_w],
                    self.last_action,
                ]
            ).astype(np.float32),
            "goal": np.concatenate(
                [
                    local * self.goal.mask[0],
                    np.array([np.sin(yaw_error), np.cos(yaw_error)]) * self.goal.mask[0],
                    arm,
                    self.goal.mask,
                ]
            ).astype(np.float32),
            "obstacles": obstacles,
        }
        obs["poses"] = np.r_[self.pose_features(), self.goal_pose].astype(np.float32)
        token = np.concatenate([obs[k] for k in ("state", "goal", "obstacles", "poses")])
        now = float(self.data.time)
        if self._history is None:
            # Repeat the initial measurement: no invented zero poses or stale episode data.
            self._history = np.repeat(token[None], self.profile.history_steps, axis=0)
            self._history_time = now
        elif now - self._history_time >= self.profile.control_dt - 1e-9:
            self._history = np.roll(self._history, -1, axis=0)
            self._history[-1] = token
            self._history_time = now
        elif abs(now - self._history_time) < 1e-9:
            self._history[-1] = token
        obs["history"] = self._history.copy()
        goal_columns = self.goal_pose[3:9].reshape(3, 2)
        goal_rotation = np.column_stack([goal_columns, np.cross(goal_columns[:, 0], goal_columns[:, 1])])
        current_rotation = self.data.xmat[self.ee_body].reshape(3, 3)
        ee_angle = np.arccos(np.clip((np.trace(current_rotation.T @ goal_rotation) - 1) / 2, -1, 1))
        metrics = {
            "ee_orientation_error": float(ee_angle),
            "obstruction": bool(self.obstruction.any()),
            "max_target_lead": float(np.max(np.abs(self.targets - self.arm_positions()))),
            "xy_error": float(np.linalg.norm(delta) * self.goal.mask[0]),
            "yaw_error": float(abs(yaw_error) * self.goal.mask[0]),
            "joint_error": float(np.max(np.abs(arm))),
            "mean_joint_error": float(np.mean(np.abs(arm))),
            "ee_position_error": float(np.linalg.norm(self.pose_features()[:3] - self.goal_pose[:3])),
            "clearance": clearance,
            "collision": clearance < -0.001,
        }
        metrics["obstacle_clearance"] = float(obstacles.reshape(-1, 5)[:, 0].min())
        return obs, metrics

    def settled(self, metrics, dt):
        p = self.profile
        small = (
            metrics["xy_error"] <= p.xy_tolerance
            and metrics["yaw_error"] <= p.yaw_tolerance
            and metrics["joint_error"] <= p.joint_tolerance
            and np.max(np.abs(self.data.qvel[self.arm_v])) < 0.08
            and np.linalg.norm(self.base_velocity()[:2]) < 0.03
            and abs(self.base_velocity()[2]) < 0.08
        )
        self.settle_time = self.settle_time + dt if small else 0.0
        return self.settle_time + 1e-9 >= p.settle_seconds


def baseline_action(control):
    delta, yaw_error, _ = control.errors()
    yaw = control.base_pose()[2]
    distance = np.linalg.norm(delta)
    bearing = wrap(np.arctan2(delta[1], delta[0]) - yaw)
    reverse = abs(bearing) > np.pi / 2
    if reverse:
        bearing = wrap(bearing + np.pi)
    v = (1 if not reverse else -1) * min(distance * 1.5, control.profile.max_v) * max(0, np.cos(bearing))
    w = 2.5 * (bearing if distance > control.profile.xy_tolerance * 0.6 else yaw_error)
    if distance <= control.profile.xy_tolerance * 0.6:
        v = 0
    arm = (control.goal.arm - control.targets) / (control.profile.max_joint_speed * control.profile.control_dt)
    return np.clip(np.r_[v / control.profile.max_v, w / control.profile.max_w, arm], -1, 1)
