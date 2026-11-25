"""Measured-state Mars waypoint execution using the shared WBC pose sampler."""

import threading
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.command_runtime import task_pose_teacher_action
from emet.rl.manipulation.pose_stream_reference import SingleHandPoseStream
from emet.rl.manipulation.whole_body_observation import actor_observation, pose_rotation
from emet.robots.innate_mars.wbc_interface import MarsWBCInterface


class MarsPoseExecutor:
    """Own a single robot-side stream from a verified transport and shadow FK model."""

    def __init__(self, *, transport, control, published_limits, policy=None, translation_limit_m=0):
        self.control = control
        self.stream = SingleHandPoseStream()
        self._wbc_options = {
            "transport": transport,
            "policy": policy if policy is not None else lambda _: task_pose_teacher_action(control),
            "profile": control.profile,
            "published_limits": published_limits,
            "enabled": True,
            "translation_limit_m": translation_limit_m,
        }
        self.wbc = MarsWBCInterface(**self._wbc_options)
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._worker = None
        self._result = None
        self._finishing = False
        self.control.action_mode = "direct_velocity"
        self.control.preview_enabled = True
        self.control.retention_context_enabled = True

    def available(self):
        if self.wbc.driver.capabilities()["whole_body_velocity"]:
            return True
        return self._wbc_options["transport"].can_rearm()

    def _measure(self):
        c = self.control
        feedback = self.wbc.driver.feedback()
        if self.wbc.session._age(feedback) > 0.2 or feedback.get("arm_healthy") is not True:
            raise RuntimeError("fresh healthy Mars arm and odometry required")
        q = np.asarray(feedback["joint_position"], float)
        dq = np.asarray(feedback["joint_velocity"], float)
        pose = np.asarray(feedback["base_pose"], float)
        twist = np.asarray(feedback["base_twist"], float)
        grip = np.asarray(feedback["gripper_state"], float)
        for value, size in ((q, 5), (dq, 5), (pose, 3), (twist, 6), (grip, 2)):
            if value.shape != (size,) or not np.isfinite(value).all():
                raise RuntimeError("invalid measured Mars state")
        joint = c.model.body_jntadr[c.base]
        offset = c.model.jnt_qposadr[joint]
        c.data.qpos[offset : offset + 2] = pose[:2]
        c.data.qpos[offset + 3 : offset + 7] = [np.cos(pose[2] / 2), 0, 0, np.sin(pose[2] / 2)]
        c.data.qvel[c.base_dof : c.base_dof + 3] = Rotation.from_euler("z", pose[2]).apply(twist[:3])
        c.data.qvel[c.base_dof + 3 : c.base_dof + 6] = twist[3:]
        for ids, value in zip(c.q, q, strict=True):
            c.data.qpos[ids] = value / len(ids)
        for ids, value in zip(c.v, dq, strict=True):
            c.data.qvel[ids] = value / len(ids)
        c.data.qpos[c.grip_q], c.data.qvel[c.grip_v] = grip
        target = self.wbc.driver.last_arm_target
        c.target = q.copy() if target is None else target.copy()
        c.grip_reference = c.grip_target = float(feedback["gripper_target_rad"])
        c.data.time = time.monotonic()
        mujoco.mj_forward(c.model, c.data)
        return feedback

    def start(self, goal):
        with self._lock:
            if self._worker is not None:
                if self._worker.is_alive():
                    raise RuntimeError("previous Mars stream is still stopping")
                self._wbc_options["transport"].rearm()
                self.wbc = MarsWBCInterface(**self._wbc_options)
                self._done.clear()
                self._result = None
                self._finishing = False
                self.stream = SingleHandPoseStream()
                self.control.clear_task_trajectory()
                self.control.history = None
                self.control.last_action[:] = 0
            self._measure()
            self.control.set_task_trajectory(self.stream.trajectory(self.control, goal, new=True))
            self.wbc.start()
            self._worker = threading.Thread(target=self._run, name="mars-wbc", daemon=True)
            self._worker.start()
        return {"motion_mode": "task_pose"}

    def update(self, goal):
        with self._lock:
            if self._worker is None or self._done.is_set() or self._result is not None:
                raise RuntimeError("no active Mars pose stream")
            if self.stream.is_finish(goal):
                self._finishing = True
            elif self._finishing:
                raise RuntimeError("Mars pose stream is finishing")
            else:
                self._measure()
                self.control.set_task_trajectory(self.stream.trajectory(self.control, goal, new=False))
            return {
                "motion_mode": "task_pose",
                "pose_stream_lease_s": max(3.0, self.control.task_trajectory.end + 2),
            }

    def result(self):
        with self._lock:
            return self._result

    def stop(self):
        self._done.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            if self._worker.is_alive():
                return False
        return self.wbc.stop()

    def _run(self):
        c = self.control
        period = c.profile.dt
        try:
            while not self._done.wait(period):
                with self._lock:
                    feedback = self._measure()
                    observation = actor_observation(c)
                    path = c.task_trajectory
                    hand = c.ee_pose()
                    position_error = np.linalg.norm(hand[:3] - path.hand[-1, :3])
                    angle_error = Rotation.from_matrix(pose_rotation(path.hand[-1]) @ pose_rotation(hand).T).magnitude()
                    base_error = c.base_pose() - path.base[-1]
                    base_error[2] = np.arctan2(np.sin(base_error[2]), np.cos(base_error[2]))
                    reached_base = not path.base_mask.any() or (
                        np.linalg.norm(base_error[:2]) < 0.02 and abs(base_error[2]) < 0.1
                    )
                    reached = (
                        c.trajectory_elapsed() >= path.end
                        and reached_base
                        and position_error < 0.02
                        and angle_error < 0.14
                        and np.max(np.abs(feedback["joint_velocity"])) < 0.03
                        and np.linalg.norm(feedback["base_twist"]) < 0.03
                    )
                    if reached and self._finishing:
                        self._result = ("succeeded", {"hand_error_m": float(position_error)})
                        break
                    if reached:
                        self.wbc.session.command(
                            np.zeros(7), base_target=path.base[-1] if path.base_mask.any() else c.base_pose()
                        )
                        continue
                    self.wbc.step(observation, base_target=path.base[-1] if path.base_mask.any() else c.base_pose())
                    limits = np.r_[c.profile.base_speed, c.profile.base_yaw_speed, c.profile.speed]
                    c.last_action = np.r_[self.wbc.session.last_velocity / limits, 0]
        except Exception as exc:
            with self._lock:
                self._result = ("failed", {"reason": str(exc)})
        finally:
            self._done.set()
            self.wbc.stop()
