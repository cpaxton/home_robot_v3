"""Physical grasp/retention curriculum. Reset assistance ends before scored motion."""

from dataclasses import asdict, dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from emet.rl.control import invalid_physics
from emet.rl.manipulation.control import ManipulationControl
from emet.rl.manipulation.models import build_manipulation_model
from emet.rl.manipulation.profiles import PROFILES
from emet.rl.manipulation.safety import ContactMonitor, relative_pose, retention_error

SCENARIOS = ("reach", "pregrasp", "carry", "place", "disturbance")
MIXTURE = (0.25, 0.20, 0.30, 0.15, 0.10)
DISTURBANCE_FORCE_N = (1.0, 4.0)
DISTURBANCE_TORQUE_NM = (0.03, 0.20)
DISTURBANCE_WARMUP_EPISODES = 20


def sample_disturbance(rng, difficulty=1.0):
    """Draw a short world-frame wrench for a compliant end-effector push."""
    difficulty = float(np.clip(difficulty, 0, 1))
    force_direction = rng.normal(size=3)
    force_direction /= np.linalg.norm(force_direction)
    torque_direction = rng.normal(size=3)
    torque_direction /= np.linalg.norm(torque_direction)
    force_range = (0.5 + 0.5 * difficulty, 1.0 + 3.0 * difficulty)
    torque_range = (0.01 + 0.02 * difficulty, 0.04 + 0.16 * difficulty)
    force = force_direction * rng.uniform(*force_range)
    torque = torque_direction * rng.uniform(*torque_range)
    return np.r_[force, torque]


def running_cost(*, dt, force, jerk, action, rotation, translation):
    """Integrate physical costs over time; do not charge them once per sample.

    Sustaining an ordinary grasp must not cost more than dropping it early.
    Terminal safety/retention failures still receive a separate -20 penalty.
    """
    return dt * (
        0.4
        + 0.02 * force
        + 0.00001 * float(np.sum(np.asarray(jerk) ** 2))
        + 0.01 * float(np.sum(np.asarray(action) ** 2))
        + 2 * rotation
        + 5 * translation
    )


@dataclass(frozen=True)
class ManipulationConfig:
    robot: str = "stretch"
    scenario: str = "mixed"
    horizon: float = 8.0
    initialization_attempts: int = 20
    diagnostics: bool = False
    disturbance_curriculum: bool = False

    def __post_init__(self):
        if self.robot not in PROFILES or self.scenario not in (*SCENARIOS, "mixed"):
            raise ValueError("unknown robot or manipulation scenario")
        if not np.isfinite(self.horizon) or self.horizon <= 0 or not 1 <= self.initialization_attempts <= 20:
            raise ValueError("invalid horizon or initialization attempts")


class ManipulationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        self.config = config if isinstance(config, ManipulationConfig) else ManipulationConfig(**(config or {}))
        self.profile = PROFILES[self.config.robot]
        self.model, self.fingerprint = build_manipulation_model(self.config.robot, diagnostics=self.config.diagnostics)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.control = ManipulationControl(self.model, self.data, self.profile)
        self.monitor = ContactMonitor(self.model, self.profile)
        self.substeps = round(self.profile.dt / self.model.opt.timestep)
        self.action_space = spaces.Box(-1, 1, (self.profile.arm_size + 1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {"history": spaces.Box(-np.inf, np.inf, self.control.observe()["history"].shape, dtype=np.float32)}
        )
        self.mass = self.model.body_mass.copy()
        self.inertia = self.model.body_inertia.copy()
        self.friction = self.model.geom_friction.copy()
        self.disturbance_episodes = 0
        self._done = True

    def _tick(self, action, *, score=True):
        self.control.accept(action)
        last = None
        for _ in range(self.substeps):
            self.control.apply()
            mujoco.mj_step(self.model, self.data)
            last = self.monitor.update(self.data, self.payload)
            if score and (
                last["force"] > self.profile.unsafe_force or self.monitor.wheel_lift or invalid_physics(self.data)
            ):
                self.control.stop()
                break
        mujoco.mj_forward(self.model, self.data)
        return last

    def _initialize(self):
        m, d, p = self.model, self.data, self.profile
        mujoco.mj_resetData(m, d)
        m.body_mass[:] = self.mass
        m.body_inertia[:] = self.inertia
        m.geom_friction[:] = self.friction
        self.payload_name = str(self.np_random.choice(("object1", "object2")))
        self.payload = m.body(self.payload_name).id
        mass_factor = self.np_random.uniform(0.8, 1.2)
        m.body_mass[self.payload] *= mass_factor
        m.body_inertia[self.payload] *= mass_factor
        friction_factor = self.np_random.uniform(0.8, 1.2)
        payload_geom = m.body_geomadr[self.payload]
        # Randomize the manipulated object's grip friction only. Scaling every
        # geom also changes table and wheel/floor traction, confounding carry.
        m.geom_friction[payload_geom, 0] *= friction_factor
        mujoco.mj_setConst(m, d)
        if p.robot == "stretch":
            q = np.array(
                [
                    self.np_random.uniform(0.45, 0.65),
                    self.np_random.uniform(0.10, 0.22),
                    self.np_random.uniform(-0.10, 0.10),
                    -0.2618,
                    self.np_random.uniform(-0.04, 0.04),
                ]
            )
        else:
            q = np.array([0, -0.4, 0.2, 0.2, 0]) + self.np_random.uniform(-0.05, 0.05, 5)
        self.control.initialize(q, p.gripper_open)
        points = [
            d.body(name).xpos + d.body(name).xmat.reshape(3, 3) @ np.array([0, 0, 0.01]) for name in p.finger_bodies
        ]
        center = np.mean(points, axis=0)
        # A real support surface holds the object during physical closure.
        table = m.body("table").id
        table_geom = m.body_geomadr[table]
        payload_geom = m.body_geomadr[self.payload]
        size = m.geom_size[payload_geom]
        half_height = size[1] if m.geom_type[payload_geom] == mujoco.mjtGeom.mjGEOM_CYLINDER else size[2]
        m.body_pos[table] = [center[0], center[1], center[2] - half_height - m.geom_size[table_geom, 2]]
        m.geom_size[table_geom, :2] = 0.15
        for name in ("object1", "object2"):
            address = m.jnt_qposadr[m.body(name).jntadr[0]]
            d.qpos[address : address + 3] = center if name == self.payload_name else [4, 4, 0.5]
        mujoco.mj_forward(m, d)
        grasp_reference = self.control.positions().copy()
        if self.scenario == "pregrasp":
            self.control.initialize(grasp_reference + [0.025, -0.025, 0, 0, 0], p.gripper_open)
        self.monitor.reset()
        # Allow the initialized chassis to establish ground contact; these steps
        # are reset initialization, not a claimed no-lift rollout.
        for _ in range(20):
            self._tick(np.zeros(p.arm_size + 1), score=False)
        self.monitor.reset()
        self.initial_object_height = float(d.xpos[self.payload, 2])
        self.randomization = {"mass_factor": mass_factor, "friction_factor": friction_factor}
        if self.scenario == "disturbance":
            horizon_steps = int(self.config.horizon / p.dt)
            latest_start = max(10, horizon_steps - 12)
            self.disturbance_start = int(self.np_random.integers(10, latest_start + 1))
            self.disturbance_steps = int(self.np_random.integers(6, 13))
            self.disturbance_wrench = sample_disturbance(self.np_random, self.disturbance_difficulty)
        else:
            self.disturbance_start = None
            self.disturbance_steps = 0
            self.disturbance_wrench = np.zeros(6)
        if self.scenario == "reach":
            return True
        if self.scenario == "pregrasp":
            self.control.set_reference(grasp_reference, phase="pregrasp", grip=p.gripper_open)
            return True
        self.control.set_reference(self.control.positions(), phase="close", grip=p.gripper_closed)
        bilateral_steps = 0
        for _ in range(100):
            evidence = self._tick(np.zeros(p.arm_size + 1))
            bilateral_steps = bilateral_steps + 1 if evidence["bilateral_contact"] else 0
            if self.control.stopped:
                return False
        if bilateral_steps < 20:
            return False
        self.control.set_reference(self.control.positions() + np.array([0.08, 0, 0, 0, 0]), phase="carry")
        stable_reference = None
        stable_steps = 0
        for index in range(80):
            evidence = self._tick(np.zeros(p.arm_size + 1))
            if self.control.stopped:
                return False
            if index == 59:
                stable_reference = relative_pose(d, self.control.ee, self.payload)
            if index >= 60:
                translation, rotation = retention_error(
                    stable_reference, relative_pose(d, self.control.ee, self.payload)
                )
                if not evidence["bilateral_contact"] or translation > 0.02 or rotation > 0.1:
                    return False
                stable_steps += 1
        lifted = d.xpos[self.payload, 2] - self.initial_object_height > 0.04
        return bool(lifted and stable_steps == 20)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("frozen fixtures and arbitrary physics injection are not training reset options")
        self.scenario = (
            str(self.np_random.choice(SCENARIOS, p=MIXTURE))
            if self.config.scenario == "mixed"
            else self.config.scenario
        )
        if self.scenario == "disturbance":
            self.disturbance_difficulty = (
                min(1.0, self.disturbance_episodes / DISTURBANCE_WARMUP_EPISODES)
                if self.config.disturbance_curriculum
                else 1.0
            )
            self.disturbance_episodes += 1
        else:
            self.disturbance_difficulty = 1.0
        self.initialization_rejections = 0
        self.initialization_failed = True
        for _ in range(self.config.initialization_attempts):
            if self._initialize():
                self.initialization_failed = False
                break
            self.initialization_rejections += 1
        self.steps = 0
        self.start_time = float(self.data.time)
        self._done = False
        self.retention_reference = (
            relative_pose(self.data, self.control.ee, self.payload)
            if self.scenario in {"carry", "place", "disturbance"} and not self.initialization_failed
            else None
        )
        self.monitor.reset()
        self.max_translation = self.max_rotation = 0.0
        self.retention_lost = False
        self.supported_time = self.settle_time = self.bilateral_time = 0.0
        self.last_velocity = self.control.velocities()
        self.last_acceleration = np.zeros(self.profile.arm_size)
        self.peak_jerk = self.peak_acceleration = 0.0
        self._released = False
        self._disturbed = False
        self._lift_commanded = False
        self._closure_commanded = False
        self._previous_cost = 0.0
        self.command_delay = int(self.np_random.integers(0, 3))
        self.actions = [np.zeros(self.profile.arm_size + 1) for _ in range(self.command_delay)]
        self._make_reference()
        self._previous_cost = float(np.linalg.norm(self.control.reference - self.control.positions()))
        obs = self.control.observe()
        from emet.rl.manipulation.risk import trajectory_features

        self.risk_features = trajectory_features(
            obs["history"], self.control.reference[None], self.config.horizon
        ).tolist()
        return obs, self._info()

    def _make_reference(self):
        if self.scenario == "pregrasp":
            return
        q = self.control.positions()
        if self.scenario == "place":
            q[0] -= 0.06
        else:
            scale = np.asarray(self.profile.speed) * 0.5
            q += self.np_random.uniform(-1, 1, self.profile.arm_size) * scale
            if self.scenario != "reach":
                q[0] = max(q[0], self.control.positions()[0])
        q = np.clip(q, self.control.limits[:, 0] + 0.02, self.control.limits[:, 1] - 0.02)
        phase = "reach" if self.scenario == "reach" else self.scenario
        self.control.set_reference(q, phase=phase)

    def _info(self):
        return {
            **self.monitor.metrics(),
            "scenario": self.scenario,
            "robot": self.profile.robot,
            "initialization_failed": self.initialization_failed,
            "initialization_rejections": self.initialization_rejections,
            "retention_lost": self.retention_lost,
            "max_payload_translation": self.max_translation,
            "max_payload_rotation": self.max_rotation,
            "peak_measured_jerk": self.peak_jerk,
            "peak_measured_acceleration": self.peak_acceleration,
            "joint_error": float(np.max(np.abs(self.control.reference - self.control.positions()))),
            "time": float(self.data.time - self.start_time),
            "is_success": False,
            "unsafe_contact": self.monitor.peak_force > self.profile.unsafe_force,
            "randomization": getattr(self, "randomization", {}),
            "disturbance_force_n": float(np.linalg.norm(getattr(self, "disturbance_wrench", np.zeros(6))[:3])),
            "disturbance_torque_nm": float(np.linalg.norm(getattr(self, "disturbance_wrench", np.zeros(6))[3:])),
            "disturbance_difficulty": getattr(self, "disturbance_difficulty", 1.0),
            "source": "simulator_ground_truth_evaluation",
            "diagnostics_only": self.config.diagnostics,
        }

    def step(self, action):
        if self._done:
            raise RuntimeError("reset required")
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("invalid manipulation action")
        if self.initialization_failed:
            self._done = True
            self.control.stop()
            return self.control.observe(), -20.0, True, False, {**self._info(), "code": "initialization_failed"}
        self.actions.append(action.copy())
        action = self.actions.pop(0)
        if self.scenario == "disturbance":
            end = self.disturbance_start + self.disturbance_steps
            if self.disturbance_start <= self.steps < end:
                # Perturb the hand, so the held object is disturbed through
                # the real simulated grasp contacts rather than by teleporting
                # a force directly onto the payload.
                elapsed = self.steps - self.disturbance_start
                remaining = end - self.steps - 1
                ramp = 3
                ramp_up = np.clip(elapsed / (ramp - 1), 0, 1)
                ramp_down = np.clip(remaining / (ramp - 1), 0, 1)
                ramp_up = ramp_up**2 * (3 - 2 * ramp_up)
                ramp_down = ramp_down**2 * (3 - 2 * ramp_down)
                envelope = min(ramp_up, ramp_down)
                self.data.xfrc_applied[self.control.ee] = self.disturbance_wrench * envelope
                self._disturbed |= envelope > 0
            else:
                self.data.xfrc_applied[self.control.ee] = 0
        evidence = self._tick(action)
        dt = self.profile.dt
        velocity = self.control.velocities()
        acceleration = (velocity - self.last_velocity) / dt
        jerk = (acceleration - self.last_acceleration) / dt
        self.peak_acceleration = max(self.peak_acceleration, float(np.max(abs(acceleration))))
        self.peak_jerk = max(self.peak_jerk, float(np.max(abs(jerk))))
        self.last_velocity, self.last_acceleration = velocity, acceleration
        translation, rotation = retention_error(
            self.retention_reference, relative_pose(self.data, self.control.ee, self.payload)
        )
        if translation is not None and not self._released:
            self.max_translation, self.max_rotation = (
                max(self.max_translation, translation),
                max(self.max_rotation, rotation),
            )
            self.retention_lost |= translation > 0.02 or rotation > 0.1
        if self.scenario == "pregrasp":
            if (
                not self._closure_commanded
                and np.max(abs(self.control.reference - self.control.positions())) < 0.02
                and np.max(abs(velocity)) < 0.03
            ):
                self.control.set_reference(self.control.reference, phase="close", grip=self.profile.gripper_closed)
                self._closure_commanded = True
            self.bilateral_time = self.bilateral_time + dt if evidence["bilateral_contact"] else 0
            if self.bilateral_time >= 1 and not self._lift_commanded:
                self.retention_reference = relative_pose(self.data, self.control.ee, self.payload)
                self.control.set_reference(self.control.positions() + [0.08, 0, 0, 0, 0], phase="carry")
                self._lift_commanded = True
                self._previous_cost = float(np.linalg.norm(self.control.reference - self.control.positions()))
        error = float(np.max(np.abs(self.control.reference - self.control.positions())))
        settled = error < 0.025 and np.max(np.abs(velocity)) < 0.03
        if self.scenario == "place" and settled and not self._released:
            self.control.set_reference(self.control.reference, phase="release", release=True)
            self._released = True
        if self.scenario == "place":
            table_body = self.model.body("table").id
            supported = any(
                set(self.model.geom_bodyid[[c.geom1, c.geom2]]) == {self.payload, table_body} for c in self.data.contact
            )
            object_velocity = self.data.qvel[
                self.model.body(self.payload).dofadr[0] : self.model.body(self.payload).dofadr[0] + 6
            ]
            geom = self.model.body_geomadr[self.payload]
            size = self.model.geom_size[geom].copy()
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                size = np.array([size[0], size[0], size[1]])
            extent = np.abs(self.data.geom_xmat[geom].reshape(3, 3)) @ size
            inside = np.all(
                np.abs(self.data.xpos[self.payload, :2] - self.model.body_pos[table_body, :2]) + extent[:2] <= 0.155
            )
            open_grip = abs(self.data.qpos[self.control.grip_q] - self.profile.gripper_open) < 0.003
            settled = (
                self._released
                and supported
                and inside
                and open_grip
                and settled
                and np.max(abs(object_velocity)) < 0.02
            )
        if self.scenario == "pregrasp":
            settled &= self._lift_commanded and self.data.xpos[self.payload, 2] - self.initial_object_height > 0.04
        if self.scenario == "disturbance":
            settled &= self._disturbed and self.steps >= self.disturbance_start + self.disturbance_steps
        self.settle_time = self.settle_time + dt if settled else 0
        unsafe = self.monitor.peak_force > self.profile.unsafe_force
        invalid = invalid_physics(self.data)
        failure = unsafe or self.monitor.wheel_lift or self.retention_lost or invalid
        success = self.settle_time >= (2 if self.scenario == "place" else 1) and not failure
        if failure or success:
            self.control.stop()
        cost = float(np.linalg.norm(self.control.reference - self.control.positions()))
        reward = (
            10 * (self._previous_cost - cost)
            - running_cost(
                dt=dt,
                force=evidence["force"],
                jerk=jerk,
                action=action,
                rotation=self.max_rotation,
                translation=self.max_translation,
            )
            + 10 * success
            - 20 * failure
        )
        self._previous_cost = cost
        self.steps += 1
        truncated = self.steps * dt >= self.config.horizon and not (failure or success)
        self._done = bool(failure or success or truncated)
        code = (
            "wheel_lift"
            if self.monitor.wheel_lift
            else "unsafe_contact"
            if unsafe
            else "retention_lost"
            if self.retention_lost
            else "invalid_physics"
            if invalid
            else "ok"
            if success
            else "tracking_timeout"
            if truncated
            else "running"
        )
        info = {
            **self._info(),
            "is_success": bool(success),
            "code": code,
            "invalid_physics": bool(invalid),
            "config": asdict(self.config),
            "risk_features": self.risk_features if self._done else None,
        }
        return self.control.observe(), float(reward), bool(failure or success), bool(truncated), info
