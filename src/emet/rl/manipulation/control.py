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
        mujoco.mj_forward(self.model, self.data)
        self.reset()

    def reset(self):
        self.target = self.positions()
        self.reference = self.target.copy()
        self.velocity = np.zeros(self.profile.arm_size)
        self.grip_reference = float(self.data.qpos[self.grip_q])
        self.grip_target = self.grip_reference
        self.phase = "reach"
        self.release_authorized = False
        self.stopped = False
        self.history = None
        self.last_action = np.zeros(self.profile.arm_size + 1)
        self._history_time = None
        self.apply()

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

    def accept(self, action):
        p = self.profile
        action = np.clip(require_finite(action, (p.arm_size + 1,)), -1, 1)
        if self.stopped:
            return
        # Feedback is a prior; PPO can correct direction and speed within limits.
        feedback = np.clip(4 * (self.reference - self.target), -np.asarray(p.speed), p.speed)
        desired = np.clip(feedback + action[:-1] * p.speed, -np.asarray(p.speed), p.speed)
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
        rate = p.gripper_speed * (p.grip_rate_fraction if self.phase in {"carry", "place", "disturbance"} else 1) * p.dt
        self.grip_target += float(np.clip(goal - self.grip_target, -rate, rate))
        self.grip_target = float(np.clip(self.grip_target, *self.grip_limits))
        self.last_action = action.copy()
        self.apply()

    def stop(self):
        self.stopped = True
        measured = self.positions()
        # Corrupt feedback must not turn a stop into a NaN actuator command.
        self.target = np.where(np.isfinite(measured), measured, self.target)
        self.velocity[:] = 0
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
        self.data.ctrl[self.act] = self.target
        self.data.ctrl[self.grip_act] = self.grip_target
        for name in ("left_wheel_vel", "right_wheel_vel", "base_x", "base_y", "base_yaw"):
            index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if index >= 0:
                self.data.ctrl[index] = 0

    def observe(self):
        p = self.profile
        # No object pose, solver contact, or wheel-force ground truth in actor input.
        phase = np.zeros(7)
        phase[("reach", "pregrasp", "close", "carry", "place", "release", "disturbance").index(self.phase)] = 1
        token = np.r_[
            self.positions(),
            self.velocities(),
            self.target - self.positions(),
            self.reference - self.positions(),
            self.data.qpos[self.grip_q],
            self.grip_target - self.data.qpos[self.grip_q],
            self.grip_reference,
            self.last_action,
            phase,
        ].astype(np.float32)
        now = float(self.data.time)
        if self.history is None:
            self.history = np.repeat(token[None], p.history_steps, axis=0)
            self._history_time = now
        elif now - self._history_time >= p.dt - 1e-8:
            self.history = np.roll(self.history, -1, axis=0)
            self._history_time = now
        self.history[-1] = token
        return {"history": self.history.copy()}
