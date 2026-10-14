"""Realtime adapter for the existing MuJoCo server; physics retains one writer."""

import time

import mujoco
import numpy as np

from emet.rl.control import Goal, MarsControl, baseline_action, invalid_physics
from emet.rl.profile import MARS


class ReactiveController:
    def __init__(self, model, data, *, policy=None, base_only=False):
        self.base_only = base_only or getattr(policy, "task", "reach") == "base"
        if self.base_only:
            from emet.rl.base import MarsBaseControl

            self.control = MarsBaseControl(model, data)
        else:
            self.control = MarsControl(model, data)
        if not np.isclose(model.opt.timestep, MARS.physics_dt):
            raise ValueError("reactive model timestep does not match control profile")
        self.policy = policy
        self.active = False
        self.stopping = False
        self.outcome = None
        self.last_step_wall = time.monotonic()
        self.next_tick = 0.0
        self.pending_goal = None

    def validate(self, payload):
        if self.base_only and (not isinstance(payload, dict) or "arm" in payload or "base" not in payload):
            raise ValueError("base-only policy requires a base goal")
        return Goal.parse(payload, self.control)

    def start(self, payload):
        goal = self.validate(payload)
        self.control.reset()
        self.control.set_goal(goal)
        self.pending_goal = None
        self.active, self.stopping = True, False
        self.outcome = None
        self.last_step_wall = time.monotonic()
        self.next_tick = self.control.data.time
        return {"resolved_goal": goal.payload(), "frame": "world", "motion_mode": "reactive"}

    def update(self, payload):
        if not self.active or self.stopping or self.outcome is not None:
            raise ValueError("reactive controller is not running")
        self.pending_goal = self.validate(payload)
        return {"resolved_goal": self.pending_goal.payload(), "frame": "world", "motion_mode": "reactive"}

    def stop(self):
        if not self.stopping:
            self.control.stop()
        self.stopping = True
        self.pending_goal = None

    def stopped(self):
        c = self.control
        return bool(np.max(np.abs(c.data.qvel[np.r_[c.base_v, c.arm_v]])) < 0.03)

    def step(self):
        c = self.control
        self.last_step_wall = time.monotonic()
        if not self.stopping and c.data.time + 1e-9 >= self.next_tick:
            begin = time.monotonic()
            try:
                if self.pending_goal is not None:
                    c.set_goal(self.pending_goal)
                    self.pending_goal = None
                if invalid_physics(c.data):
                    raise RuntimeError("invalid physics state")
                obs, metrics = c.observe()
                if metrics["collision"]:
                    raise RuntimeError("forbidden collision")
                if c.settled(metrics, MARS.control_dt):
                    self.outcome = ("succeeded", metrics)
                    self.stop()
                else:
                    action = (
                        (c.baseline_action() if self.base_only else baseline_action(c))
                        if self.policy is None
                        else self.policy(obs)
                    )
                    # The servo executes under the physics lock. Allow one
                    # missed 20 Hz interval for inference and host scheduling;
                    # the command's own deadline and stale-physics watchdog
                    # still bound end-to-end motion.
                    if time.monotonic() - begin > 2 * MARS.control_dt:
                        raise RuntimeError("policy control deadline exceeded")
                    c.accept_action(action)
                self.next_tick = c.data.time + MARS.control_dt
            except Exception as exc:
                self.outcome = ("failed", {"failure": str(exc)})
                self.stop()
        c.apply()
        mujoco.mj_step(c.model, c.data)
        mujoco.mj_forward(c.model, c.data)
        if invalid_physics(c.data):
            self.outcome = ("failed", {"failure": "invalid physics state"})
            self.stop()
        if not self.stopping:
            robot = c.robot_geoms
            if any(
                contact.dist < -0.001 and (contact.geom1 in robot or contact.geom2 in robot)
                for contact in c.data.contact
            ):
                self.outcome = ("failed", {"failure": "forbidden collision"})
                self.stop()


class ReactiveSimMixin:
    """Adapter hooks. Called under CommandRuntime's lock, then the physics lock."""

    def supports_reactive_control(self):
        return getattr(self, "_reactive_controller_kind", "legacy") != "legacy"

    def _init_reactive_controller(self):
        if not self.supports_reactive_control():
            return
        policy = None
        if self._reactive_controller_kind == "rl":
            from emet.rl.policy import PPOPolicy

            policy = PPOPolicy(self._reactive_policy_path)
        self._reactive_controller = ReactiveController(
            self._mjmodel, self._mjdata, policy=policy, base_only=self._reactive_controller_kind == "base"
        )

    def validate_reactive_goal(self, goal):
        with self._mj_lock:
            controller = getattr(self, "_reactive_controller", None)
            if controller is None:
                raise ValueError("reactive controller is not initialized")
            controller.validate(goal)

    def start_reactive_command(self, goal):
        with self._mj_lock:
            self._nav_yaw_slew = None
            self._nav_goal_world = None
            self._at_goal = False
            return self._reactive_controller.start(goal)

    def update_reactive_command(self, goal):
        with self._mj_lock:
            return self._reactive_controller.update(goal)

    def reactive_command_result(self, context):
        with self._mj_lock:
            controller = self._reactive_controller
            if time.monotonic() - controller.last_step_wall > 0.25:
                controller.stop()
                return "failed", {"failure": "stale physics state"}
            return controller.outcome

    def cancel_reactive_command(self):
        controller = getattr(self, "_reactive_controller", None)
        if controller is None:
            return True  # Startup failed before there was an actuator writer.
        with self._mj_lock:
            controller.stop()
        deadline = time.monotonic() + 0.75
        while time.monotonic() < deadline:
            with self._mj_lock:
                if controller.stopped():
                    controller.active = False
                    self._snapshot_stationary_planar_base_qpos()
                    self._preserve_joint_ctrl_hold_from_ctrl()
                    if self._joint_ctrl_hold_client_pin is not None:
                        self._joint_ctrl_hold_client_pin[:] = True
                    self._at_goal = bool(controller.outcome and controller.outcome[0] == "succeeded")
                    return True
            time.sleep(0.005)
        return False
