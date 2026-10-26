"""Shared state, goal, observation and actuation code for training and serving."""

from dataclasses import dataclass

import mujoco
import numpy as np

from emet.rl.model import PROXIES
from emet.rl.profile import MARS


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def invalid_physics(data):
    # Snapshot each MuJoCo view once. Re-fetching these C-backed array
    # properties during long collision rollouts can segfault on some builds.
    qpos = np.asarray(data.qpos).copy()
    qvel = np.asarray(data.qvel).copy()
    qacc = np.asarray(data.qacc).copy()
    state_invalid = bool(
        not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
        or not np.isfinite(qacc).all()
        or np.max(np.abs(qpos), initial=0.0) > 1e3
        or np.max(np.abs(qvel), initial=0.0) > 1e4
        or np.max(np.abs(qacc), initial=0.0) > 1e5
    )
    if state_invalid:
        return True
    # MuJoCo may clear constraint arrays after arena exhaustion while leaving
    # contact efc_address values unusable. Stop before Python touches contacts.
    if data.ncon >= data.model.nconmax or data.nefc >= data.model.njmax:
        return True
    if data.maxuse_arena >= data.narena - 4096:
        return True
    return False


@dataclass
class Goal:
    base: np.ndarray
    arm: np.ndarray
    mask: np.ndarray

    @classmethod
    def parse(cls, payload, control):
        if not isinstance(payload, dict) or set(payload) - {"base", "arm", "ee_pose", "base_locked", "frame"}:
            raise ValueError("goal requires base, arm joints, or an end-effector pose")
        if payload.get("frame", "world") != "world":
            raise ValueError("reactive goals use the world frame")
        if "base_locked" in payload and not isinstance(payload["base_locked"], bool):
            raise ValueError("base_locked must be a boolean")
        if "base_locked" in payload and "ee_pose" not in payload:
            raise ValueError("base_locked applies only to end-effector pose targets")
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
        if "ee_pose" in payload:
            pose = np.asarray(payload["ee_pose"], dtype=float)
            if pose.shape != (9,) or not np.isfinite(pose).all():
                raise ValueError("ee_pose must contain xyz and the first two rotation-matrix columns")
            if "arm" in payload:
                raise ValueError("provide either ee_pose or arm joint targets, not both")
            locked = bool(payload.get("base_locked", "base" in payload))
            if "base" in payload and not locked:
                raise ValueError("a commanded base pose alongside ee_pose must be locked")
            base, arm = control.solve_ee_pose(pose, base=base, base_locked=locked)
            mask[0] = float(not locked or "base" in payload)
            mask[1:] = 1
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
        self.held_act = np.array([model.actuator(n).id for n in profile.held_joints], dtype=int)
        self.base_body = model.body("base_link").id
        self.limits = model.jnt_range[self.arm_ids].copy()
        self.robot_geoms = [model.geom(f"rl_{n}").id for n in PROXIES]
        robot_bodies = set()
        for body in range(1, model.nbody):
            ancestor = body
            while ancestor and ancestor != self.base_body:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == self.base_body:
                robot_bodies.add(body)
        self.environment_geoms = [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] not in robot_bodies
            and (model.geom_contype[i] or model.geom_conaffinity[i])
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
        self.held_values = np.array(
            [self.data.qpos[self.model.jnt_qposadr[self.model.joint(n).id]] for n in self.profile.held_joints],
            dtype=float,
        )
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

    def solve_ee_pose(self, pose, *, base, base_locked):
        """Damped least-squares IK over the arm and, when allowed, SE(2) base."""
        target_position = np.asarray(pose[:3], dtype=float)
        columns = np.asarray(pose[3:], dtype=float).reshape(3, 2)
        first = columns[:, 0] / max(np.linalg.norm(columns[:, 0]), 1e-8)
        second = columns[:, 1] - first * np.dot(first, columns[:, 1])
        second /= max(np.linalg.norm(second), 1e-8)
        target_rotation = np.column_stack((first, second, np.cross(first, second)))
        if abs(np.linalg.det(target_rotation) - 1) > 1e-3:
            raise ValueError("ee_pose orientation must define a right-handed rotation")
        d = self._goal_data
        d.qpos[:] = self.data.qpos
        base = np.asarray(base, dtype=float).copy()
        axes = self.data.xaxis[self.base_ids]
        delta = base[:2] - self.base_pose()[:2]
        d.qpos[self.base_q[:2]] += np.linalg.solve(axes[:2, :2].T, delta)
        d.qpos[self.base_q[2]] += wrap(base[2] - self.base_pose()[2]) / axes[2, 2]
        active = np.r_[[] if base_locked else self.base_v, self.arm_v].astype(int)
        for _ in range(100):
            mujoco.mj_forward(self.model, d)
            current_position = d.xpos[self.ee_body].copy()
            current_rotation = d.xmat[self.ee_body].reshape(3, 3).copy()
            position_error = target_position - current_position
            rotation_error = 0.5 * sum(
                (np.cross(current_rotation[:, i], target_rotation[:, i]) for i in range(3)),
                start=np.zeros(3),
            )
            if np.linalg.norm(position_error) < 0.004 and np.linalg.norm(rotation_error) < 0.03:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, d, jacp, jacr, self.ee_body)
            jac = np.vstack((jacp[:, active], jacr[:, active]))
            error = np.r_[position_error, rotation_error]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 2e-3, error)
            dq = np.clip(dq, -0.025, 0.025)
            cursor = 0
            if not base_locked:
                d.qpos[self.base_q] += dq[:3]
                cursor = 3
            d.qpos[self.arm_q] = np.clip(
                d.qpos[self.arm_q] + dq[cursor:], self.limits[:, 0], self.limits[:, 1]
            )
        mujoco.mj_forward(self.model, d)
        position_error = float(np.linalg.norm(target_position - d.xpos[self.ee_body]))
        current_rotation = d.xmat[self.ee_body].reshape(3, 3)
        angle_error = float(np.arccos(np.clip((np.trace(target_rotation.T @ current_rotation) - 1) / 2, -1, 1)))
        if position_error > 0.015 or angle_error > 0.12:
            raise ValueError("end-effector pose is not reachable under the base-motion constraint")
        resolved_base = np.array([*d.xpos[self.base_body, :2], np.arctan2(d.xmat[self.base_body, 3], d.xmat[self.base_body, 0])])
        return resolved_base, d.qpos[self.arm_q].copy()

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
        # Rate-limit velocity setpoints to avoid discontinuous base commands.
        scales = np.array([self.profile.max_v, self.profile.max_w])
        acceleration = np.array([
            self.profile.max_linear_acceleration,
            self.profile.max_angular_acceleration,
        ])
        previous_velocity = self.last_action[:2] * scales
        desired_velocity = action[:2] * scales
        limited_velocity = previous_velocity + np.clip(
            desired_velocity - previous_velocity,
            -acceleration * self.profile.control_dt,
            acceleration * self.profile.control_dt,
        )
        action[:2] = limited_velocity / scales
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
        self.data.ctrl[self.held_act] = self.held_values

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
        minimum = self.profile.obstacle_range
        geom_xpos = np.asarray(self.data.geom_xpos).copy()
        geom_xmat = np.asarray(self.data.geom_xmat).copy().reshape(self.model.ngeom, 3, 3)
        geom_aabb = np.asarray(self.model.geom_aabb)
        world_center = np.empty_like(geom_xpos)
        world_extent = np.empty_like(geom_xpos)
        for geom in range(self.model.ngeom):
            rotation = geom_xmat[geom]
            world_center[geom] = geom_xpos[geom] + rotation @ geom_aabb[geom, :3]
            world_extent[geom] = np.abs(rotation) @ geom_aabb[geom, 3:]
        for a, b in self.pairs:
            # Use conservative world-aligned bounds for policy features;
            # collision termination still uses MuJoCo's solved contacts.
            delta = world_center[b] - world_center[a]
            gaps = np.maximum(
                np.maximum(
                    world_center[a] - world_extent[a] - (world_center[b] + world_extent[b]),
                    world_center[b] - world_extent[b] - (world_center[a] + world_extent[a]),
                ),
                0.0,
            )
            overlaps = world_extent[a] + world_extent[b] - np.abs(delta)
            distance = (
                float(np.linalg.norm(gaps))
                if np.any(gaps > 0)
                else -float(np.min(overlaps))
            )
            # Local obstacle input describes the scene. Fixed self-clearances and
            # the supporting floor would otherwise hide every nearby obstacle.
            # They remain part of collision detection above.
            if b in indices or self.model.geom_type[b] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            minimum = min(minimum, distance)
            if distance >= self.profile.obstacle_range:
                continue
            direction = delta / max(float(np.linalg.norm(delta)), 1e-8)
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
            "hand_active": bool(self.goal.mask[1:].any()),
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
            and (not self.goal.mask[1:].any() or (
                metrics["ee_position_error"] <= p.ee_position_tolerance
                and metrics["ee_orientation_error"] <= p.ee_orientation_tolerance
            ))
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
    deadband = control.profile.xy_tolerance * 0.6
    if control.goal.mask[1:].any():
        # A coarse base deadband can leave the hand outside its tighter pose
        # tolerance even after the arm reaches its final joint reference.
        deadband = min(deadband, control.profile.ee_position_tolerance * 0.25)
    w = 2.5 * (bearing if distance > deadband else yaw_error)
    if distance <= deadband:
        v = 0
    arm = (control.goal.arm - control.targets) / (control.profile.max_joint_speed * control.profile.control_dt)
    return np.clip(np.r_[v / control.profile.max_v, w / control.profile.max_w, arm], -1, 1)
