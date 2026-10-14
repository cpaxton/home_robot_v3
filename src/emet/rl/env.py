"""Synchronous, seeded Mars reaching environment. No transport or realtime sleeps."""

from dataclasses import asdict, dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from emet.rl.control import Goal, MarsControl, invalid_physics
from emet.rl.model import PROXIES, build_model
from emet.rl.profile import MARS


@dataclass(frozen=True)
class EnvConfig:
    task: str = "reach"
    scene: str = "procedural"
    stage: int = 0
    horizon_seconds: float = 10.0
    obstacle_count: int = 3
    contact_training: bool = False
    unsafe_contact_force: float = 30.0  # Experimental simulator threshold, N.
    base_scenario: str = "standard"
    terminal_accuracy_weight: float = 0.0
    arm_only: bool = False

    def __post_init__(self):
        if self.arm_only and (self.task != "reach" or self.stage != 0):
            raise ValueError("arm_only currently requires reach task at stage 0")
        if not np.isfinite(self.terminal_accuracy_weight) or self.terminal_accuracy_weight < 0:
            raise ValueError("terminal accuracy weight must be finite and nonnegative")
        if self.base_scenario not in {"standard", "mixed", "retarget", "push", "contact_retreat"}:
            raise ValueError("invalid base scenario")
        if self.base_scenario != "standard" and (self.task != "base" or self.stage != 0 or self.scene != "procedural"):
            raise ValueError("base scenarios require base task, procedural scene and stage 0")
        if self.task not in {"base", "reach"}:
            raise ValueError("task must be base or reach")
        if self.scene not in {"procedural", "table"} or self.stage not in range(5):
            raise ValueError("scene must be procedural/table and stage must be 0..4")
        if not np.isfinite(self.horizon_seconds) or self.horizon_seconds <= 0 or self.obstacle_count < 0:
            raise ValueError("invalid horizon or obstacle count")
        if not np.isfinite(self.unsafe_contact_force) or self.unsafe_contact_force <= 0:
            raise ValueError("unsafe contact force must be finite and positive")


class MarsReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config=None, render_mode=None):
        self.config = config if isinstance(config, EnvConfig) else EnvConfig(**(config or {}))
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        self.render_mode = render_mode
        obstacles = self.config.obstacle_count if self.config.stage >= 3 else 0
        if self.config.base_scenario in {"mixed", "contact_retreat"}:
            obstacles = max(obstacles, 1)
        self.model = build_model(self.config.scene, obstacles)
        # Policy uses geometry features, not cameras/rangefinders. Avoid 360 unused
        # sensor raycasts on every training physics step.
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.control = MarsControl(self.model, self.data)
        self.action_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "history": spaces.Box(
                    -np.inf, np.inf, shape=(MARS.history_steps, 66 + len(PROXIES) * 5), dtype=np.float32
                ),
                "poses": spaces.Box(-np.inf, np.inf, shape=(26,), dtype=np.float32),
                "state": spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32),
                "goal": spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32),
                "obstacles": spaces.Box(-np.inf, np.inf, shape=(len(PROXIES) * 5,), dtype=np.float32),
            }
        )
        self.substeps = round(MARS.control_dt / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, MARS.control_dt):
            raise ValueError("control period must be an integer number of physics steps")
        self._renderer = None
        self._needs_reset = True

    def _sample_configuration(self):
        c = self.control
        # Sample close enough to complete under the conservative velocity limits.
        radius = 0.15 if self.config.stage == 0 else 0.5
        for _ in range(300):
            self.data.qpos[c.base_q] = [*self.np_random.uniform(-radius, radius, 2), self.np_random.uniform(-0.5, 0.5)]
            if self.config.arm_only:
                self.data.qpos[c.base_q] = 0
                self.data.qpos[c.arm_q] = self.np_random.uniform(c.limits[:, 0] + 0.08, c.limits[:, 1] - 0.08)
            else:
                self.data.qpos[c.arm_q] = self.np_random.uniform(-0.25, 0.25, 5)
                self.data.qpos[c.arm_q[1]] -= 0.4
            mujoco.mj_forward(self.model, self.data)
            if c.obstacle_features()[1] > 0.0005:
                return c.base_pose(), c.arm_positions().copy()
        raise RuntimeError("could not sample a collision-free Mars configuration")

    def _sample_goal(self):
        qpos = self.data.qpos.copy()
        try:
            for _ in range(60):
                base, arm = self._sample_configuration()
                if self.config.arm_only and self.np_random.random() < 0.5:
                    arm = np.clip(
                        qpos[self.control.arm_q] + self.np_random.uniform(-0.25, 0.25, 5),
                        self.control.limits[:, 0] + 0.08,
                        self.control.limits[:, 1] - 0.08,
                    )
                if not self.config.arm_only:
                    break
                # Initial curriculum: reject paths with sampled geometric contact.
                # Discrete checks are not a continuous swept-volume certificate.
                start = qpos[self.control.arm_q]
                count = max(2, int(np.ceil(np.max(np.abs(arm - start)) / 0.04)) + 1)
                clear = True
                for fraction in np.linspace(0, 1, count):
                    self.data.qpos[self.control.arm_q] = start + fraction * (arm - start)
                    mujoco.mj_forward(self.model, self.data)
                    if self.control.obstacle_features()[1] <= 0.0005:
                        clear = False
                        break
                if clear:
                    break
            else:
                raise RuntimeError("could not sample a collision-free arm interpolation")
        finally:
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
        mask = np.ones(6)
        if self.config.arm_only:
            mask[0] = 0
            return Goal(self.control.base_pose(), arm, mask)
        # Train masked modes too, holding the omitted dimensions at the start.
        kind = self.np_random.choice(3, p=[0.15, 0.15, 0.70])
        if kind == 0:
            mask[0], base = 0, self.control.base_pose()
        elif kind == 1:
            mask[1:], arm = 0, self.control.arm_positions().copy()
        return Goal(base, arm, mask)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        self._needs_reset = False
        for i in range(self.model.nmocap):
            self.data.mocap_pos[i] = [*self.np_random.uniform(-0.8, 0.8, 2), 0.20]
        if self.config.base_scenario != "standard":
            # The base scenario positions its own obstacle after sampling the
            # start. A random box can otherwise cover the entire small start
            # region and make rejection sampling fail on a later reset.
            self.data.mocap_pos[:] = [5, 5, 0.2]
        self._obstacle_origin = self.data.mocap_pos.copy()
        self.sampling_rejected_starts = 0
        for _ in range(30):
            self._sample_configuration()
            self.control.reset()
            try:
                goal = self._sample_goal()
            except RuntimeError:
                if not self.config.arm_only:
                    raise
                self.sampling_rejected_starts += 1
                continue
            self.control.set_goal(goal)
            break
        else:
            raise RuntimeError("could not sample connected arm start/goal pair")
        if options and "goal" in options:
            self.control.set_goal(options["goal"])
        mujoco.mj_forward(self.model, self.data)
        obs, info = self.control.observe()
        self._previous_cost = self._cost(info)
        return obs, {
            **info,
            "is_success": False,
            "config": asdict(self.config),
            "sampling_rejected_starts": self.sampling_rejected_starts,
        }

    def set_goal(self, payload):
        """Explicit goal change; reward is rebased so switches cannot create progress."""
        self.control.set_goal(payload)
        self._previous_cost = self._cost(self.control.observe()[1])

    @staticmethod
    def _cost(metrics):
        return (
            2 * metrics["xy_error"]
            + 0.3 * metrics["yaw_error"]
            + metrics["joint_error"]
            + 2 * metrics["mean_joint_error"]
        )

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("reset required before step")
        previous_action = self.control.last_action.copy()
        if self.config.arm_only:
            action = np.asarray(action).copy()
            action[:2] = 0
        self.control.accept_action(action)
        goal_changed = self.config.stage >= 2 and self.steps == 60
        if goal_changed:
            self.set_goal(self._sample_goal())
        if self.config.stage >= 2 and self.steps == 100:
            self.data.qvel[self.control.base_v[:2]] += self.np_random.uniform(-0.15, 0.15, 2)
        collision = False
        peak_contact_force = 0.0
        contact_impulse = 0.0
        wrench = np.zeros(6)
        invalid = False
        for _ in range(self.substeps):
            if self.config.stage >= 4:
                self.data.mocap_pos[:, 1] = self._obstacle_origin[:, 1] + 0.12 * np.sin(2 * self.data.time)
            self.control.apply()
            mujoco.mj_step(self.model, self.data)
            # Read the forces from the solved step, before recomputing forward dynamics.
            force_sum = 0.0
            for index, contact in enumerate(self.data.contact):
                if contact.geom1 not in self.control.robot_geoms and contact.geom2 not in self.control.robot_geoms:
                    continue
                # The mobile base's support contact is not an obstacle impact.
                geoms = (contact.geom1, contact.geom2)
                if self.model.geom("rl_base_link").id in geoms and any(
                    self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in geoms
                ):
                    continue
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                force_sum += float(np.linalg.norm(wrench[:3]))
            peak_contact_force = max(peak_contact_force, force_sum)
            contact_impulse += force_sum * self.model.opt.timestep
            mujoco.mj_forward(self.model, self.data)
            invalid |= invalid_physics(self.data)
            # Latch contacts across substeps, even if the solver separates them.
            robot = self.control.robot_geoms
            collision |= any(
                contact.dist < -0.001 and (contact.geom1 in robot or contact.geom2 in robot)
                for contact in self.data.contact
            )
            if invalid or peak_contact_force > self.config.unsafe_contact_force:
                break
        mujoco.mj_forward(self.model, self.data)
        obs, metrics = self.control.observe()
        collision |= metrics["collision"]
        unsafe_contact = peak_contact_force > self.config.unsafe_contact_force
        contact_failure = unsafe_contact or (collision and not self.config.contact_training)
        if contact_failure or invalid:
            self.control.stop()
            obs, metrics = self.control.observe()
        settled = self.control.settled(metrics, MARS.control_dt)
        # Recovery scenarios must actually encounter their scheduled events.
        success = (
            settled
            and not contact_failure
            and not invalid
            and (self.config.stage < 2 or self.steps >= 100)
            and getattr(self, "_scenario_ready", True)
        )
        cost = self._cost(metrics)
        smoothness = float(np.sum((self.control.last_action - previous_action) ** 2))
        joint_margin = np.minimum(
            self.control.arm_positions() - self.control.limits[:, 0],
            self.control.limits[:, 1] - self.control.arm_positions(),
        )
        reward = (
            10 * (self._previous_cost - cost)
            - 0.05 * cost
            - 0.002 * smoothness
            - 0.001 * float(np.sum(self.control.last_action**2))
            - 0.01 * float(np.sum(np.maximum(0, 0.1 - joint_margin)))
            + 10 * success
            - (0.0 if self.config.contact_training else 10 * collision)
            - 0.02 * peak_contact_force
            - 0.2 * contact_impulse
            - 10 * unsafe_contact
            - 0.1 * metrics["obstruction"]
            - 10 * invalid
        )
        self._previous_cost = cost
        self.steps += 1
        terminated = bool(success or contact_failure or invalid)
        horizon = self.config.horizon_seconds
        if self.config.base_scenario == "mixed" and getattr(self, "_scenario", "standard") == "standard":
            horizon = min(horizon, 10.0)
        truncated = bool(self.steps * MARS.control_dt >= horizon and not terminated)
        self._needs_reset = terminated or truncated
        return (
            obs,
            float(reward),
            terminated,
            truncated,
            {
                **metrics,
                "collision": bool(collision),
                "peak_contact_force": peak_contact_force,
                "contact_impulse": contact_impulse,
                "unsafe_contact": bool(unsafe_contact),
                "is_success": bool(success),
                "invalid_physics": bool(invalid),
                "goal_changed": goal_changed,
                "action_change": smoothness,
                "joint_limit_violation": bool(np.any(joint_margin < -0.001)),
                "time": float(self.data.time),
            },
        )

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
