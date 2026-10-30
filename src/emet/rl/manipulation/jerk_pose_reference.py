"""Continuous FK targets for an admitted arm path; no IK in the actor loop.

The internal v2 record retains the actual polynomial reference rather than
mislabeling a linearized pose export as the executed curve. It is intentionally
not accepted by the public v1 command decoder. Actor observations remain the
ordinary masked SE(2)/SE(3) preview. Dynamic/contact feasibility needs rollouts.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.jerk_reference import compile_joint_path
from emet.rl.manipulation.tracking import TaskPoseTrajectory


class FKPoseReference(TaskPoseTrajectory):
    def __init__(
        self,
        fk,
        base,
        path,
        gripper,
        certificate,
        *,
        goal_pose=None,
        phase_acceleration=0,
        phase_jerk=0,
        sample_dt=0.05,
    ):
        if not np.isfinite(sample_dt) or sample_dt <= 0:
            raise ValueError("positive finite reference sample period required")
        self.fk, self.path, self.certificate = fk, path, certificate
        self.fixed_base = np.asarray(base, dtype=float).copy()
        self.grip = float(gripper)
        self.prefix_end = path.end
        self.correction = None
        self.endpoint = fk(base, path.sample(path.end)[0]).copy()
        # Task-space correction keeps the original requested endpoint. Its
        # norm bounds are explicit experimental task limits; they do not claim
        # a joint witness or a real-robot Cartesian rating for this short tail.
        if goal_pose is not None:
            goal = np.asarray(goal_pose, dtype=float)
            # Validate even an allegedly tiny correction before Rotation use.
            TaskPoseTrajectory([0, 1], [base, base], [self.endpoint, goal], [gripper] * 2, base_mask=[True] * 3)
            a = self.endpoint[3:].reshape(3, 2)
            b = goal[3:].reshape(3, 2)
            self.start_rotation = np.column_stack([a, np.cross(a[:, 0], a[:, 1])])
            goal_rotation = np.column_stack([b, np.cross(b[:, 0], b[:, 1])])
            self.rotation_delta = Rotation.from_matrix(goal_rotation @ self.start_rotation.T).as_rotvec()
            self.position_delta = goal[:3] - self.endpoint[:3]
            distance, angle = np.linalg.norm(self.position_delta), np.linalg.norm(self.rotation_delta)
            if max(distance, angle) > 1e-10:
                v = min(0.05 / max(distance, 1e-12), 0.3 / max(angle, 1e-12))
                amax = min(0.2 / max(distance, 1e-12), 1.0 / max(angle, 1e-12))
                j = min(1.0 / max(distance, 1e-12), 4.0 / max(angle, 1e-12))
                self.correction, self.correction_certificate = compile_joint_path(
                    [[0.0], [1.0]],
                    [v],
                    [amax],
                    [j],
                    phase_acceleration=phase_acceleration,
                    phase_jerk=phase_jerk,
                )
                self.goal_pose = goal.copy()
        end = self.prefix_end + (self.correction.end if self.correction else 0)
        times = sorted([*np.arange(0, end, sample_dt), end, self.prefix_end])
        times = np.array([t for i, t in enumerate(times) if i == 0 or t - times[i - 1] > 1e-9])
        hands = np.array([self._pose(t) for t in times])
        super().__init__(
            times, np.tile(base, (len(times), 1)), hands, np.full(len(times), gripper), base_mask=[True] * 3
        )

    def _tail(self, time):
        return self.correction.sample(time - self.prefix_end)

    def _pose(self, time):
        if self.correction is None or time <= self.prefix_end:
            return self.fk(self.fixed_base, self.path.sample(time)[0]).copy()
        blend = float(self._tail(time)[0][0])
        rotation = Rotation.from_rotvec(blend * self.rotation_delta).as_matrix() @ self.start_rotation
        return np.r_[self.endpoint[:3] + blend * self.position_delta, rotation[:, :2].ravel()]

    def sample(self, time):
        if not np.isfinite(time):
            raise ValueError("finite reference time required")
        return self.fixed_base.copy(), self._pose(float(time)), self.grip

    def sample_motion(self, time, *, rate=1.0, phase_acceleration=0.0, phase_jerk=0.0):
        """World hand twist/acceleration, including derivatives of retiming.

        Linear/angular units are m/s and rad/s (and corresponding accelerations).
        Prefix joint derivatives are exact polynomials; FK twist and acceleration
        use the Jacobian and its time derivative on private scratch state.
        """
        if not np.isfinite([time, rate, phase_acceleration, phase_jerk]).all() or not 0 <= rate <= 1:
            raise ValueError("finite time/phase derivatives and rate in [0,1] required")
        base, hand, grip = self.sample(time)
        outside = time < 0 or time > self.end
        if self.correction is not None and time > self.prefix_end:
            _, v, a, _ = self._tail(time)
            speed = float(v[0]) * rate
            acceleration = float(a[0]) * rate**2 + float(v[0]) * phase_acceleration
            twist = np.r_[self.position_delta, self.rotation_delta] * speed
            spatial_acceleration = np.r_[self.position_delta, self.rotation_delta] * acceleration
            joint = None
        else:
            q, v, a, j = self.path.sample(time)
            velocity = v * rate
            acceleration = a * rate**2 + v * phase_acceleration
            jerk = j * rate**3 + 3 * a * rate * phase_acceleration + v * phase_jerk
            d, m, c = self.fk.data, self.fk.model, self.fk.control
            # sample() has already positioned the private FK data.
            d.qvel[:] = 0
            qdot, qddot = np.zeros(m.nv), np.zeros(m.nv)
            for group, vv, aa in zip(c.v, velocity, acceleration, strict=True):
                qdot[group] = vv / len(group)
                qddot[group] = aa / len(group)
            d.qvel[:] = qdot
            mujoco.mj_comPos(m, d)
            mujoco.mj_comVel(m, d)
            jp, jr, dp, dr = (np.zeros((3, m.nv)) for _ in range(4))
            mujoco.mj_jacBody(m, d, jp, jr, c.ee)
            mujoco.mj_jacDot(m, d, dp, dr, d.xpos[c.ee].copy(), c.ee)
            jac, derivative = np.vstack([jp, jr]), np.vstack([dp, dr])
            twist = jac @ qdot
            spatial_acceleration = jac @ qddot + derivative @ qdot
            joint = {
                "position": q.tolist(),
                "velocity": velocity.tolist(),
                "acceleration": acceleration.tolist(),
                "jerk": jerk.tolist(),
            }
        if outside:
            twist[:], spatial_acceleration[:] = 0, 0
        return {
            "base": base.tolist(),
            "hand": hand.tolist(),
            "gripper": grip,
            "hand_twist_world": twist.tolist(),
            "hand_acceleration_world": spatial_acceleration.tolist(),
            "joint_witness": joint,
            "joint_bounds_certified": joint is not None,
        }

    def record(self):
        record = super().record()
        record.update(
            format="task_pose_fk_reference_v2",
            interpolation="continuous_private_fk_and_optional_quintic_se3_tail",
            arm_trajectory=self.path.record(),
            arm_derivative_certificate=self.certificate,
            fk_anchor_qpos=self.fk.qpos.tolist(),
            fixed_base=self.fixed_base.tolist(),
            terminal_correction=(
                {
                    "goal_pose": self.goal_pose.tolist(),
                    "certificate": self.correction_certificate,
                    "linear_norm_limits": [0.05, 0.2, 1.0],
                    "angular_norm_limits": [0.3, 1.0, 4.0],
                    "joint_witness_certified": False,
                    "dynamic_feasibility_verified": False,
                }
                if self.correction
                else None
            ),
            sampled_arrays="visualization_only_not_execution_interpolation",
            dynamic_feasibility_verified=False,
        )
        return record
