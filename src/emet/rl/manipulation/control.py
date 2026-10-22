"""Measured-state trajectory servo with bounded grip corrections and explicit release."""

import mujoco
import numpy as np

from emet.rl.manipulation.profiles import require_finite


class ManipulationControl:
    def __init__(self, model, data, profile):
        self.model, self.data, self.profile = model, data, profile
        self.q = [np.array([model.joint(n).qposadr[0] for n in group]) for group in profile.joint_groups]
        self.v = [np.array([model.joint(n).dofadr[0] for n in group]) for group in profile.joint_groups]
        self.act = np.array([model.actuator(n).id for n in profile.actuators])
        self.grip_q = model.joint(profile.gripper_joint).qposadr[0]
        self.grip_act = model.actuator(profile.gripper_actuator).id
        self.limits = np.array(
            [np.sum([model.joint(n).range for n in group], axis=0) for group in profile.joint_groups]
        )
        self.grip_limits = model.joint(profile.gripper_joint).range.copy()
        self.ee = model.body(profile.ee_body).id
        self.base = model.body(profile.base_body).id
        base_joint = int(model.body_jntadr[self.base])
        self.base_dof = int(model.jnt_dofadr[base_joint]) if base_joint >= 0 else -1
        self.wheel_act = np.array(
            [model.actuator(name).id for name in ("left_wheel_vel", "right_wheel_vel")] if profile.wheel_bodies else [],
            dtype=int,
        )
        self.wheel_gear = model.actuator_gear[self.wheel_act, 0].copy() if len(self.wheel_act) else np.zeros(0)
        self.axle_offset = np.zeros(3)
        if profile.wheel_bodies:
            axle = np.mean([data.body(name).xpos for name in profile.wheel_bodies[:2]], axis=0)
            self.axle_offset = data.xmat[self.base].reshape(3, 3).T @ (axle - data.xpos[self.base])
        self.base_hold_act = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ("base_x", "base_y", "base_yaw")],
            dtype=int,
        )
        self.tracking_integral = np.zeros(profile.arm_size)
        self._trim_time = float(data.time)
        self.pose_only_targets = False
        self.preview_enabled = False
        self.preview_sampler = None
        self.action_mode = "residual"
        self.base_xy_tolerance = 0.02
        self.reset()

    def positions(self):
        return np.array([self.data.qpos[group].sum() for group in self.q])

    def velocities(self):
        return np.array([self.data.qvel[group].sum() for group in self.v])

    def initialize(self, q, grip):
        """Reset/fixture initialization only; runtime moves use actuator controls."""
        q = require_finite(q, (self.profile.arm_size,))
        for group, value in zip(self.q, q, strict=True):
            self.data.qpos[group] = value / len(group)
        self.data.qpos[self.grip_q] = grip
        # Set coupled finger coordinates consistently at initialization.
        for index in range(self.model.neq):
            if self.model.eq_type[index] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            a, b = self.model.eq_obj1id[index], self.model.eq_obj2id[index]
            if b >= 0 and self.model.jnt_qposadr[b] == self.grip_q:
                self.data.qpos[self.model.jnt_qposadr[a]] = np.polynomial.polynomial.polyval(
                    grip, self.model.eq_data[index, :5]
                )
            elif a >= 0 and b >= 0 and self.model.jnt_qposadr[a] == self.grip_q:
                coefficients = self.model.eq_data[index, :5]
                if coefficients[1] != 0 and not np.any(coefficients[2:]):
                    # Mars declares driver = -follower, the reverse of
                    # Stretch's follower = f(driver) equality ordering.
                    self.data.qpos[self.model.jnt_qposadr[b]] = (grip - coefficients[0]) / coefficients[1]
        mujoco.mj_forward(self.model, self.data)
        self.reset()

    def reset(self):
        self.target = self.positions()
        self.reference = self.target.copy()
        self.velocity = np.zeros(self.profile.arm_size)
        self.grip_reference = float(self.data.qpos[self.grip_q])
        self.grip_target = self.grip_reference
        self.task_trajectory = None
        self.preview_sampler = None
        self.phase = "reach"
        self.release_authorized = False
        self.stopped = False
        self.history = None
        self.last_action = np.zeros(self.profile.action_size)
        self.base_command = np.zeros(2)
        self.base_goal = self.base_pose()
        self.base_mask = np.zeros(3, dtype=bool)
        self._base_align = True
        self._base_direction = 1
        self._base_turning = True
        self._base_yaw_integral = 0.0
        self.hand_goal = self.ee_pose()
        self.hand_mask = True
        self._history_time = None
        self.tracking_integral[:] = 0
        self._trim_time = float(self.data.time)
        self.apply()

    def base_pose(self):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        return np.array([*self.data.xpos[self.base, :2], np.arctan2(rotation[1, 0], rotation[0, 0])])

    def ee_pose(self):
        return np.r_[self.data.xpos[self.ee], self.data.xmat[self.ee].reshape(3, 3)[:, :2].ravel()]

    def set_pose_targets(self, base, base_mask, hand_pose, hand_mask=True):
        base = require_finite(base, (3,))
        mask = np.asarray(base_mask, dtype=bool)
        changed = not np.allclose(base, self.base_goal, rtol=0, atol=1e-6) or not np.array_equal(mask, self.base_mask)
        self.base_goal = base
        self.base_mask = mask
        if changed:
            delta = self._base_tracking_delta()
            bearing = np.arctan2(delta[1], delta[0]) - self.base_pose()[2]
            bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
            self._base_direction = -1 if abs(bearing) > np.pi / 2 + 0.05 else 1
            self._base_align = np.linalg.norm(delta) < 0.006
            self._base_turning = True
            self._base_yaw_integral = 0.0
        if self.base_mask.shape != (3,):
            raise ValueError("base target mask must have three components")
        self.hand_goal = require_finite(hand_pose, (9,))
        self.hand_mask = bool(hand_mask)

    def set_task_trajectory(self, trajectory):
        """Install a TAMP pose schedule without IK, resetting state, or clearing faults."""
        if not self.preview_enabled or self.action_mode != "direct_velocity":
            raise ValueError("task trajectories require a direct velocity preview policy")
        if self.stopped:
            raise ValueError("cannot install a trajectory on a stopped controller")
        from emet.rl.manipulation.tracking import TaskPoseTrajectory

        if not isinstance(trajectory, TaskPoseTrajectory):
            raise ValueError("expected a validated task pose trajectory")
        if np.any(trajectory.gripper < self.grip_limits[0]) or np.any(trajectory.gripper > self.grip_limits[1]):
            raise ValueError("gripper trajectory exceeds physical limits")
        self.task_trajectory = trajectory
        self.task_trajectory_start = float(self.data.time)
        self.preview_sampler = lambda time: trajectory.sample(time - self.task_trajectory_start)
        base, hand, _ = trajectory.sample(0)
        self.set_pose_targets(base, trajectory.base_mask, hand, trajectory.hand_mask)

    def set_reference(self, reference, *, phase="reach", grip=None, release=False):
        reference = require_finite(reference, (self.profile.arm_size,))
        if np.any(reference < self.limits[:, 0]) or np.any(reference > self.limits[:, 1]):
            raise ValueError("trajectory outside joint limits")
        if phase not in {"reach", "pregrasp", "close", "carry", "place", "release", "disturbance"}:
            raise ValueError("unknown manipulation phase")
        if phase == "release" and not release:
            raise ValueError("release requires executor authorization")
        if grip is not None:
            if not np.isfinite(grip) or not self.grip_limits[0] <= grip <= self.grip_limits[1]:
                raise ValueError("invalid grip reference")
            self.grip_reference = float(grip)
        self.reference = reference.copy()
        self.phase, self.release_authorized = phase, bool(release)
        # Changing the reference does not clear history or a latched safety stop.

    def _base_tracking_delta(self):
        # Differential drive rotates about the wheel axle, which need not pass
        # through base_link. Tracking base_link XY while turning causes loops.
        yaw = self.base_goal[2] if self.base_mask[2] else self.base_pose()[2]
        c, s = np.cos(yaw), np.sin(yaw)
        goal_axle = self.base_goal[:2] + np.array([[c, -s], [s, c]]) @ self.axle_offset[:2]
        current_axle = self.data.xpos[self.base] + self.data.xmat[self.base].reshape(3, 3) @ self.axle_offset
        return (goal_axle - current_axle[:2]) * self.base_mask[:2]

    def base_feedback(self):
        """Nonholonomic pose servo; zero policy residual tracks a base target.

        Travel and final-heading phases use position hysteresis. Locking the
        travel direction at goal acceptance avoids forward/reverse flips for
        lateral targets. Slew limits are applied by accept(), also on retarget.
        """
        if not self.base_mask.any():
            return np.zeros(2)
        p = self.profile
        pose = self.base_pose()
        delta = self._base_tracking_delta()
        distance = float(np.linalg.norm(delta))
        was_aligning = self._base_align
        if self.base_xy_tolerance == 0.02:
            enter, leave = 0.006, (0.014 if self.hand_mask else 0.018)
        else:
            enter, leave = 0.8 * self.base_xy_tolerance, self.base_xy_tolerance
        self._base_align = distance < (leave if self._base_align else enter)
        if was_aligning and not self._base_align:
            # A caster can move the axle slightly during final alignment.
            # Re-enter travel using the nearest direction at the new heading;
            # retaining the original direction can demand a complete circle.
            bearing = np.arctan2(delta[1], delta[0]) - pose[2]
            bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
            self._base_direction = -1 if abs(bearing) > np.pi / 2 else 1
            self._base_turning = True
        if not self._base_align:
            bearing = np.arctan2(delta[1], delta[0]) - pose[2]
            if self._base_direction < 0:
                bearing += np.pi
            bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
            speed = min(2.0 * distance, p.base_speed, 0.7 * np.sqrt(2 * p.base_acceleration * max(0, distance - 0.003)))
            if self._base_turning:
                self._base_turning = abs(bearing) > 0.12
            elif abs(bearing) > 0.5:
                self._base_turning = True
            linear = 0.0 if self._base_turning else self._base_direction * speed * max(0.0, np.cos(bearing))
            angular = np.clip(3.0 * bearing, -p.base_yaw_speed, p.base_yaw_speed)
            return np.array([linear, angular])
        if not self.base_mask[2]:
            return np.zeros(2)
        error = self.base_goal[2] - pose[2]
        error = float(np.arctan2(np.sin(error), np.cos(error)))
        if abs(error) < 0.015:
            self._base_yaw_integral = 0.0
            return np.zeros(2)
        self._base_yaw_integral = float(np.clip(self._base_yaw_integral + 0.2 * error * p.dt, -0.12, 0.12))
        angular = np.clip(2.5 * error + self._base_yaw_integral, -p.base_yaw_speed, p.base_yaw_speed)
        stopping_speed = min(0.2, 0.7 * np.sqrt(2 * p.base_yaw_acceleration * abs(error)))
        return np.array([0.0, np.clip(angular, -stopping_speed, stopping_speed)])

    def _arm_teacher_velocity(self):
        """Arm prior without advancing the stateful base servo."""
        p = self.profile
        reference = self.reference.copy()
        feedforward = np.zeros(p.arm_size)
        if p.robot == "innate_mars" and self.hand_mask and self.base_mask.any():
            yaw = self.base_goal[2] - self.base_pose()[2]
            reference[0] = np.clip(reference[0] + np.arctan2(np.sin(yaw), np.cos(yaw)), *self.limits[0])
            spatial = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, spatial, 0)
            feedforward[0] = -spatial[2]
        arm = np.clip(4 * (reference - self.target), -np.asarray(p.speed), p.speed) + feedforward
        arm *= float(self.hand_mask)
        return arm

    def teacher_action(self):
        """Normalized complete velocity command, for imitation or comparison."""
        p = self.profile
        base = self.base_feedback() / [p.base_speed, p.base_yaw_speed] if p.wheel_bodies else []
        return np.clip(np.r_[base, self._arm_teacher_velocity() / p.speed, 0.0], -1, 1)

    def accept(self, action):
        p = self.profile
        action = np.clip(require_finite(action, (p.action_size,)), -1, 1)
        if self.stopped:
            return
        if p.wheel_bodies:
            scales = np.array([p.base_speed, p.base_yaw_speed])
            prior = self.base_feedback() if self.action_mode == "residual" else np.zeros(2)
            requested = np.clip(prior + action[:2] * scales, -scales, scales)
            requested *= bool(
                self.base_mask.any() or ((self.preview_enabled or self.pose_only_targets) and self.hand_mask)
            )
            acceleration = np.array([p.base_acceleration, p.base_yaw_acceleration]) * p.dt
            self.base_command += np.clip(requested - self.base_command, -acceleration, acceleration)
            arm_action = action[2:-1]
        else:
            arm_action = action[:-1]
        # Direct policies own the complete velocity command. The same physical
        # slew, position, grip, and stop limits apply to both action semantics.
        prior = self._arm_teacher_velocity() / p.speed if self.action_mode == "residual" else 0
        desired = np.clip((prior + arm_action * float(self.hand_mask)) * p.speed, -np.asarray(p.speed), p.speed)
        dv = np.asarray(p.acceleration) * p.dt
        self.velocity += np.clip(desired - self.velocity, -dv, dv)
        self.target += self.velocity * p.dt
        lead = np.minimum(np.asarray(p.speed) * 0.5, 0.08)
        self.target = np.clip(self.target, self.positions() - lead, self.positions() + lead)
        self.target = np.clip(self.target, self.limits[:, 0], self.limits[:, 1])
        grip_span = float(np.ptp(self.grip_limits))
        correction = action[-1] * p.grip_fraction * grip_span
        # Corrections are only permitted during closed-grip phases.
        if self.phase not in {"carry", "place", "disturbance"}:
            correction = 0
        goal = self.grip_reference + correction
        if self.release_authorized:
            goal = p.gripper_open
        rate = (
            p.gripper_speed
            * (p.grip_rate_fraction if self.phase in {"carry", "place", "release", "disturbance"} else 1)
            * p.dt
        )
        self.grip_target += float(np.clip(goal - self.grip_target, -rate, rate))
        self.grip_target = float(np.clip(self.grip_target, *self.grip_limits))
        self.last_action = action.copy()
        if p.wheel_bodies:
            self.last_action[:2] = self.base_command / [p.base_speed, p.base_yaw_speed]
        self.apply()

    def stop(self):
        self.stopped = True
        measured = self.positions()
        # Corrupt feedback must not turn a stop into a NaN actuator command.
        self.target = np.where(np.isfinite(measured), measured, self.target)
        self.velocity[:] = 0
        self.base_command[:] = 0
        self.last_action[:] = 0
        measured_grip = float(self.data.qpos[self.grip_q])
        if self.release_authorized:
            # Freeze an authorized release at its measured position on fault.
            self.grip_reference = measured_grip
            self.grip_target = measured_grip
        else:
            # Discard residual grip relaxation and return to the phase target.
            self.grip_target = self.grip_reference
        self.release_authorized = False
        self.apply()

    def apply(self):
        now = float(self.data.time)
        elapsed = max(0.0, now - self._trim_time)
        if elapsed >= self.model.opt.timestep - 1e-9 and not self.stopped:
            tracking_error = self.target - self.positions()
            self.tracking_integral = np.clip(
                self.tracking_integral + tracking_error * elapsed,
                -self.profile.tracking_trim_limit / self.profile.tracking_trim_gain,
                self.profile.tracking_trim_limit / self.profile.tracking_trim_gain,
            )
            self._trim_time = now
        trim = self.profile.tracking_trim_gain * self.tracking_integral
        self.data.ctrl[self.act] = np.clip(self.target + trim, self.limits[:, 0], self.limits[:, 1])
        self.data.ctrl[self.grip_act] = self.grip_target
        if len(self.wheel_act):
            drive = self.base_command.copy()
            if self.profile.robot == "innate_mars" and not self.stopped:
                spatial = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, spatial, 0)
                forward = self.data.xmat[self.base].reshape(3, 3)[:, 0]
                measured = np.array([spatial[3:] @ forward, spatial[2]])
                limits = np.array([self.profile.base_speed, self.profile.base_yaw_speed])
                drive = np.clip(drive + 2.0 * (drive - measured), -limits, limits)
            left = (drive[0] - drive[1] * self.profile.wheel_separation / 2) / self.profile.wheel_radius
            right = (drive[0] + drive[1] * self.profile.wheel_separation / 2) / self.profile.wheel_radius
            # Velocity actuator targets are transmission-space velocities:
            # length_dot = gear * joint_velocity (not joint_velocity / gear).
            commands = np.array([left, right]) * self.wheel_gear
            limited = self.model.actuator_ctrllimited[self.wheel_act].astype(bool)
            ranges = self.model.actuator_ctrlrange[self.wheel_act]
            commands[limited] = np.clip(commands[limited], ranges[limited, 0], ranges[limited, 1])
            self.data.ctrl[self.wheel_act] = commands
        valid = self.base_hold_act[self.base_hold_act >= 0]
        self.data.ctrl[valid] = 0

    def update_task_reference(self):
        if self.task_trajectory is not None:
            base, hand, _ = self.task_trajectory.sample(float(self.data.time) - self.task_trajectory_start)
            self.set_pose_targets(base, self.task_trajectory.base_mask, hand, self.task_trajectory.hand_mask)

    def observe(self):
        p = self.profile
        self.update_task_reference()
        from emet.rl.pose_targets import PoseTargets

        # No object pose, solver contact, or wheel-force ground truth in actor input.
        phase = np.zeros(7)
        phase[("reach", "pregrasp", "close", "carry", "place", "release", "disturbance").index(self.phase)] = 1
        target_tokens, target_mask = PoseTargets(
            self.base_goal,
            self.base_mask,
            self.hand_goal[None],
            np.array([self.hand_mask], dtype=bool),
        ).actor_tokens(max_hands=1)
        pose_only = self.preview_enabled or self.pose_only_targets
        if pose_only:
            target_tokens[~target_mask] = 0
            target_tokens[0, :2] *= self.base_mask[:2]
            target_tokens[0, 2:4] *= self.base_mask[2]
        token = np.r_[
            self.positions(),
            self.velocities(),
            self.base_pose(),
            self.last_action[:2] if p.wheel_bodies else np.zeros(2),
            self.target - self.positions(),
            np.zeros(p.arm_size)
            if self.preview_enabled or self.pose_only_targets
            else self.reference - self.positions(),
            self.data.qpos[self.grip_q],
            self.grip_target - self.data.qpos[self.grip_q],
            self.grip_reference,
            self.last_action,
            phase,
            self.base_pose(),
            self.ee_pose(),
            np.where(self.base_mask, self.base_goal, 0) if pose_only else self.base_goal,
            self.base_mask.astype(float),
            self.hand_goal * float(self.hand_mask) if pose_only else self.hand_goal,
            float(self.hand_mask),
            self.data.qvel[self.base_dof : self.base_dof + 6] if self.base_dof >= 0 else np.zeros(6),
            target_tokens.ravel(),
            target_mask.astype(float),
        ].astype(np.float32)
        now = float(self.data.time)
        if self.history is None:
            self.history = np.repeat(token[None], p.history_steps, axis=0)
            self._history_time = now
        elif now - self._history_time >= p.dt - 1e-8:
            self.history = np.roll(self.history, -1, axis=0)
            self._history_time = now
        self.history[-1] = token
        observation = {"history": self.history.copy()}
        if self.preview_enabled:
            from emet.rl.manipulation.tracking import encode_preview

            observation["preview"] = encode_preview(self, self.preview_sampler)
        return observation
