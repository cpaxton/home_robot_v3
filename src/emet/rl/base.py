"""Base-only local pose control with bounded acceleration and short history."""

from dataclasses import replace

import mujoco
import numpy as np
from gymnasium import spaces

from emet.rl.control import Goal, MarsControl, baseline_action
from emet.rl.env import EnvConfig, MarsReachEnv
from emet.rl.model import PROXIES
from emet.rl.profile import MARS

BASE = replace(MARS, history_steps=10, xy_tolerance=0.02, yaw_tolerance=0.035, settle_seconds=0.30)
# Experimental simulator command limits: m/s², rad/s².
ACCELERATION = np.array([0.4, 1.2])
BASE_CONTRACT = {"version": 2, "acceleration": ACCELERATION.tolist(), "history_width": 12 + 2 * len(PROXIES)}


class MarsBaseControl(MarsControl):
    def __init__(self, model, data):
        super().__init__(model, data, profile=BASE)

    def reset(self):
        self._base_history = None
        self._base_history_time = None
        super().reset()

    def set_goal(self, goal):
        if isinstance(goal, dict):
            if "arm" in goal or "base" not in goal:
                raise ValueError("base-only controller requires a base goal and cannot command the arm")
            goal = Goal.parse(goal, self)
        goal = Goal(goal.base.copy(), self.arm_positions().copy(), np.array([1, 0, 0, 0, 0, 0]))
        # Hold the original arm targets even when the base is retargeted.
        arm_hold = self.targets.copy()
        super().set_goal(goal)
        self.targets = arm_hold

    def accept_action(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("base policy action must contain two finite values")
        scales = np.array([self.profile.max_v, self.profile.max_w])
        previous = self.last_action[:2] * scales
        desired = np.clip(action, -1, 1) * scales
        velocity = previous + np.clip(
            desired - previous, -ACCELERATION * BASE.control_dt, ACCELERATION * BASE.control_dt
        )
        super().accept_action(np.r_[velocity / scales, np.zeros(5)])

    def baseline_action(self):
        return baseline_action(self)[:2]

    def observe(self):
        full, metrics = super().observe()
        obstacle = full["obstacles"].reshape(-1, 5)
        local = full["goal"][:2]
        distance = np.linalg.norm(local)
        token = np.r_[
            full["goal"][:4],
            self.base_velocity() / [BASE.max_v, BASE.max_v, BASE.max_w],
            self.last_action[:2],
            obstacle[:, 0] / BASE.obstacle_range,
            obstacle[:, 4],
            distance,
            local / max(distance, 0.01),
        ].astype(np.float32)
        now = float(self.data.time)
        if self._base_history is None:
            self._base_history = np.repeat(token[None], BASE.history_steps, axis=0)
            self._base_history_time = now
        elif now - self._base_history_time >= BASE.control_dt - 1e-9:
            self._base_history = np.roll(self._base_history, -1, axis=0)
            self._base_history[-1] = token
            self._base_history_time = now
        elif abs(now - self._base_history_time) < 1e-9:
            self._base_history[-1] = token
        return {"history": self._base_history.copy()}, metrics


class MarsBaseEnv(MarsReachEnv):
    def __init__(self, config=None, render_mode=None):
        super().__init__(config or EnvConfig(task="base"), render_mode)
        self.control = MarsBaseControl(self.model, self.data)
        self._nominal_gain = self.model.actuator_gainprm.copy()
        self._nominal_bias = self.model.actuator_biasprm.copy()
        self._nominal_damping = self.model.dof_damping.copy()
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "history": spaces.Box(
                    -np.inf, np.inf, shape=(BASE.history_steps, BASE_CONTRACT["history_width"]), dtype=np.float32
                )
            }
        )

    def expert_action(self):
        return self.control.baseline_action()

    def _sample_configuration(self):
        c = self.control
        radius = 0.15 if self.config.stage == 0 else 0.5
        for _ in range(300):
            self.data.qpos[c.base_q] = [*self.np_random.uniform(-radius, radius, 2), self.np_random.uniform(-0.5, 0.5)]
            self.data.qpos[c.arm_q] = [0, -0.4, 0, 0, 0]
            mujoco.mj_forward(self.model, self.data)
            if c.obstacle_features()[1] > 0.0005:
                return c.base_pose(), c.arm_positions().copy()
        raise RuntimeError("could not sample a collision-free base pose")

    def _sample_goal(self):
        qpos = self.data.qpos.copy()
        try:
            base, _ = self._sample_configuration()
        finally:
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
        return Goal(base, self.control.arm_positions().copy(), np.array([1, 0, 0, 0, 0, 0]))

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Later curriculum stages vary drive response and actuation delay. Keep
        # stage 0 deterministic in dynamics to isolate smooth local reaching.
        self.model.actuator_gainprm[:] = self._nominal_gain
        self.model.actuator_biasprm[:] = self._nominal_bias
        self.model.dof_damping[:] = self._nominal_damping
        self._delay_steps = 0
        if self.config.stage >= 1 or self.config.base_scenario != "standard":
            gain = self.np_random.uniform(0.8, 1.2)
            self.model.actuator_gainprm[self.control.base_act] *= gain
            self.model.actuator_biasprm[self.control.base_act] *= gain
            self.model.dof_damping[self.control.base_v] *= self.np_random.uniform(0.7, 1.3)
            self._delay_steps = int(self.np_random.integers(0, 3))
        self._pending_actions = [np.zeros(2) for _ in range(self._delay_steps)]
        self._velocity = self.control.base_velocity()[[0, 2]]
        self._acceleration = np.zeros(2)
        self._command = self.control.last_action[:2] * [BASE.max_v, BASE.max_w]
        self._command_acceleration = np.zeros(2)
        self._segment_start = self.control.base_pose()[:2].copy()
        self._scenario = self.config.base_scenario
        if self._scenario == "mixed":
            self._scenario = self.np_random.choice(["standard", "retarget", "push", "contact_retreat"])
        self._scenario_ready = self._scenario == "standard"
        self._event_count = 0
        self._last_event_time = None
        self._approach_distance = 0.0
        self._event_step = int(self.np_random.integers(30, 60))
        self._push = self.np_random.uniform(-1, 1, 3) * [3.0, 3.0, 0.12]
        self._start_pose = self.control.base_pose().copy()
        self.data.xfrc_applied[:] = 0
        if self.config.base_scenario != "standard":
            # Obstacles are present only in the contact exercise. Do not let a
            # random reset placement accidentally contaminate other scenarios.
            self.data.mocap_pos[:] = [5, 5, 0.2]
            if self._scenario == "contact_retreat":
                yaw = self._start_pose[2]
                forward = np.array([np.cos(yaw), np.sin(yaw)])
                self.data.mocap_pos[0] = [*(self._start_pose[:2] + 0.65 * forward), 0.2]
                self.set_goal({"base": [*(self._start_pose[:2] + 0.5 * forward), float(yaw)]})
                self._event_step = 80
            mujoco.mj_forward(self.model, self.data)
            obs, info = self.control.observe()
            self._previous_cost = self._cost(info)
        return obs, info

    def _scenario_events(self):
        if self._scenario == "push":
            active = self._event_step <= self.steps < self._event_step + 6
            self.data.xfrc_applied[self.control.base_body, [0, 1, 5]] = self._push if active else 0
            if self.steps == self._event_step:
                self._event_count += 1
                self._last_event_time = float(self.data.time)
            self._scenario_ready = self.steps >= self._event_step + 6
        elif self.steps == self._event_step and self._scenario in {"retarget", "contact_retreat"}:
            if self._scenario == "retarget":
                self.set_goal(self._sample_goal())
            else:
                # Keep the obstacle in place. A new goal behind the robot must
                # unload contact and clear only the opposing-direction latch.
                self._approach_distance = float(np.linalg.norm(self.control.base_pose()[:2] - self._start_pose[:2]))
                self.set_goal({"base": self._start_pose.tolist()})
            self._event_count += 1
            self._last_event_time = float(self.data.time)
            self._scenario_ready = True

    def set_goal(self, payload):
        super().set_goal(payload)
        self._segment_start = self.control.base_pose()[:2].copy()

    @staticmethod
    def _cost(metrics):
        return 3 * metrics["xy_error"] + 0.5 * metrics["yaw_error"]

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("reset required before step")
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("base policy action must contain two finite values")
        self._scenario_events()
        self._pending_actions.append(action.copy())
        action = self._pending_actions.pop(0)
        before = self.data.time
        obs, reward, terminated, truncated, info = super().step(action)
        dt = max(self.data.time - before, self.model.opt.timestep)
        velocity = self.control.base_velocity()[[0, 2]]
        acceleration = (velocity - self._velocity) / dt
        jerk = (acceleration - self._acceleration) / dt
        command = self.control.last_action[:2] * [BASE.max_v, BASE.max_w]
        command_acceleration = (command - self._command) / dt
        command_jerk = (command_acceleration - self._command_acceleration) / dt
        self._velocity, self._acceleration = velocity.copy(), acceleration
        self._command, self._command_acceleration = command, command_acceleration
        direction = self.control.goal.base[:2] - self._segment_start
        direction /= max(np.linalg.norm(direction), 1e-9)
        overshoot = max(0.0, float((self.control.base_pose()[:2] - self.control.goal.base[:2]) @ direction))
        reward -= 0.01 * float(np.sum((acceleration / ACCELERATION) ** 2))
        reward -= 0.002 * float(np.sum((command_jerk / (ACCELERATION / BASE.control_dt)) ** 2))
        reward -= 0.2 * overshoot
        if info["is_success"]:
            reward -= self.config.terminal_accuracy_weight * (info["xy_error"] + 0.2 * info["yaw_error"])
        info.update(
            scenario=str(self._scenario),
            event_count=self._event_count,
            approach_distance=self._approach_distance,
            event_elapsed=None if self._last_event_time is None else float(self.data.time) - self._last_event_time,
            measured_velocity=velocity.tolist(),
            measured_acceleration=acceleration.tolist(),
            measured_jerk=jerk.tolist(),
            command_acceleration=command_acceleration.tolist(),
            command_jerk=command_jerk.tolist(),
            overshoot=overshoot,
            settling_time=info["time"] if info["is_success"] else None,
        )
        return obs, float(reward), terminated, truncated, info
