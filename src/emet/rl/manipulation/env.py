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


def running_cost(*, dt, force, jerk, action, rotation, translation, tracking=0.0, motion=0.0):
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
        + 4 * tracking
        + 0.2 * motion
    )


class ResetMotionUnreachable(ValueError):
    """A procedural reset cannot produce its requested kinematic lift."""


@dataclass(frozen=True)
class ManipulationConfig:
    robot: str = "stretch"
    scenario: str = "mixed"
    horizon: float = 8.0
    initialization_attempts: int = 20
    diagnostics: bool = False
    disturbance_curriculum: bool = False
    randomization_scale: float = 1.0
    payload_mass_kg: float | None = None
    payload_asset_set: str = "training"
    payload_object: str | None = None
    initial_heading_range: float = 0.0
    target_scale: float = 1.0
    retarget_probability: float = 0.0
    reach_push_probability: float = 0.0
    certify_reach_paths: bool = False
    dynamic_references: bool = False
    action_mode: str = "residual"
    base_xy_tolerance: float = 0.02
    base_heading_tolerance: float = 0.04
    base_only_min_distance: float = 0.0
    base_only_max_distance: float = 0.0
    reach_target_mode: str = "mixed"
    differential_base_reward: bool = False
    grasp_lift_height: float = 0.06
    support_half_width: float = 0.012
    grasp_depth: float = 0.03
    hand_position_tolerance: float = 0.015
    hand_orientation_tolerance: float = 0.10
    trajectory_preview: bool = False
    tracking_reward: bool = False
    tracking_precision: int = 0
    endpoint_motion_weight: float = 0.0
    goal_gated_settling: bool = False

    def __post_init__(self):
        if self.payload_asset_set not in {"training", "native_table"}:
            raise ValueError("invalid payload asset set")
        if self.payload_object not in {None, "object1", "object2"}:
            raise ValueError("invalid payload object")
        if self.payload_mass_kg is not None and (not np.isfinite(self.payload_mass_kg) or self.payload_mass_kg <= 0):
            raise ValueError("payload mass must be finite and positive")
        if not np.isfinite(self.base_only_min_distance) or not 0 <= self.base_only_min_distance <= 0.5:
            raise ValueError("base-only minimum distance must be in [0, .5]")
        if not np.isfinite(self.base_only_max_distance) or not 0 <= self.base_only_max_distance <= 0.75:
            raise ValueError("base-only maximum distance must be in [0, .75]")
        if self.base_only_max_distance and (
            not self.base_only_min_distance or self.base_only_max_distance < self.base_only_min_distance
        ):
            raise ValueError("base-only maximum requires a positive minimum no larger than maximum")
        if self.reach_target_mode not in {"mixed", "base", "hand", "combined"}:
            raise ValueError("invalid reach target mode")
        if self.base_only_min_distance and self.base_only_min_distance <= self.base_xy_tolerance:
            raise ValueError("base-only minimum distance must exceed the accepted XY radius")
        if not np.isfinite(self.base_heading_tolerance) or not 0 < self.base_heading_tolerance <= np.pi:
            raise ValueError("base heading tolerance must be in (0, pi]")
        if not np.isfinite(self.endpoint_motion_weight) or not 0 <= self.endpoint_motion_weight <= 2:
            raise ValueError("endpoint motion weight must be in [0, 2]")
        if type(self.tracking_precision) is not int or not 0 <= self.tracking_precision < 5:
            raise ValueError("tracking precision must be a stage from 0 to 4")
        if not (
            np.isfinite(self.hand_position_tolerance)
            and 0 < self.hand_position_tolerance <= 0.015
            and np.isfinite(self.hand_orientation_tolerance)
            and 0 < self.hand_orientation_tolerance <= 0.10
        ):
            raise ValueError("hand tolerances may only tighten the 15 mm / 0.1 rad contract")
        if not np.isfinite(self.grasp_depth) or not 0 <= self.grasp_depth <= 0.05:
            raise ValueError("grasp depth must be between zero and 50 mm")
        if not np.isfinite(self.support_half_width) or not 0.012 <= self.support_half_width <= 0.15:
            raise ValueError("support half width must be between 12 and 150 mm")
        if not np.isfinite(self.grasp_lift_height) or not 0.045 <= self.grasp_lift_height <= 0.10:
            raise ValueError("grasp lift height must be between 45 and 100 mm")
        if self.action_mode not in {"residual", "direct_velocity"}:
            raise ValueError("unknown action mode")
        if not np.isfinite(self.base_xy_tolerance) or not 0.005 <= self.base_xy_tolerance <= 0.10:
            raise ValueError("base tolerance must be between 5 and 100 mm")
        if self.robot not in PROFILES or self.scenario not in (*SCENARIOS, "mixed"):
            raise ValueError("unknown robot or manipulation scenario")
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in (self.retarget_probability, self.reach_push_probability)):
            raise ValueError("event probabilities must be in [0, 1]")
        if not np.isfinite(self.target_scale) or not 0 < self.target_scale <= 1:
            raise ValueError("target scale must be in (0, 1]")
        if not np.isfinite(self.initial_heading_range) or not 0 <= self.initial_heading_range <= np.pi:
            raise ValueError("initial heading range must be in [0, pi]")
        if not np.isfinite(self.randomization_scale) or not 0 <= self.randomization_scale <= 1:
            raise ValueError("randomization scale must be in [0, 1]")
        if not np.isfinite(self.horizon) or self.horizon <= 0 or not 1 <= self.initialization_attempts <= 20:
            raise ValueError("invalid horizon or initialization attempts")


class ManipulationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        self.config = config if isinstance(config, ManipulationConfig) else ManipulationConfig(**(config or {}))
        self.profile = PROFILES[self.config.robot]
        self.model, self.fingerprint = build_manipulation_model(
            self.config.robot, diagnostics=self.config.diagnostics, payload_asset_set=self.config.payload_asset_set
        )
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.control = ManipulationControl(self.model, self.data, self.profile)
        self._reference_fk = None
        self.tracking_precision = self.config.tracking_precision
        self._configure_preview()
        self.monitor = ContactMonitor(self.model, self.profile)
        self.substeps = round(self.profile.dt / self.model.opt.timestep)
        self.action_space = spaces.Box(-1, 1, (self.profile.action_size,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
                for key, value in self.control.observe().items()
            }
        )
        self.mass = self.model.body_mass.copy()
        self.inertia = self.model.body_inertia.copy()
        self.friction = self.model.geom_friction.copy()
        self.randomization_scale = self.config.randomization_scale
        self.target_scale = self.config.target_scale
        self.disturbance_episodes = 0
        self._done = True

    def _configure_preview(self):
        self.control.pose_only_targets = self.config.tracking_reward or self.config.trajectory_preview
        self.control.preview_enabled = self.config.trajectory_preview
        if self.config.trajectory_preview and self.control.task_trajectory is not None:
            self.control.preview_sampler = lambda time: self.control.task_trajectory.sample(
                time - self.control.task_trajectory_start
            )
        else:
            self.control.preview_sampler = self._sample_task_reference if self.config.trajectory_preview else None
        if hasattr(self, "observation_space"):
            observation_spaces = {"history": self.observation_space["history"]}
            if self.config.trajectory_preview:
                from emet.rl.manipulation.tracking import PREVIEW_OFFSETS, PREVIEW_WIDTH

                observation_spaces["preview"] = spaces.Box(
                    -np.inf, np.inf, (len(PREVIEW_OFFSETS), PREVIEW_WIDTH), dtype=np.float32
                )
            self.observation_space = spaces.Dict(observation_spaces)

    def set_tracking_precision(self, level):
        if type(level) is not int or not 0 <= level < 5:
            raise ValueError("tracking precision must be a stage from 0 to 4")
        self.tracking_precision = level

    def _sample_task_reference(self, time):
        path = getattr(self, "arm_trajectory", None)
        if path is None or self._reference_fk is None:
            return self.control.base_goal.copy(), self.control.hand_goal.copy()
        elapsed = max(0.0, time - self.start_time)
        # A reset freezes test events for pairing, but future unannounced goals
        # must not leak to the actor through the precomputed second segment.
        if path.replan_time is not None and not self.retargets:
            arm = path.segments[0].sample(max(path.segments[0].start, elapsed))[0]
        else:
            arm = path.sample(elapsed)[0]
        base = self.control.base_goal.copy()
        return base, self._reference_fk(base, arm)

    def set_task_trajectory(self, trajectory):
        """Replace procedural goals with an executor-supplied SE(2)/SE(3) path."""
        self.control.set_task_trajectory(trajectory)
        self.arm_trajectory = None
        self.retarget_step = None
        self.retarget_goal = None
        self.settle_time = 0
        self.hold_base_anchor = None
        self.first_settle_time = None
        self._previous_cost = self._tracking_cost()

    def _reference_complete(self):
        return self._reference_end_elapsed() >= 0

    def _reference_end_elapsed(self):
        c = self.control
        if c.task_trajectory is not None:
            return float(self.data.time - c.task_trajectory_start - c.task_trajectory.end)
        if self.arm_trajectory is not None:
            return float(self.data.time - self.start_time - self.arm_trajectory.end)
        return float(self.data.time - max(self.start_time, c.goal_changed_time))

    def teacher_action(self):
        return (
            self.control.teacher_action()
            if self.config.action_mode == "direct_velocity"
            else np.zeros(self.action_space.shape)
        )

    def set_randomization_scale(self, scale):
        if not np.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError("randomization scale must be in [0, 1]")
        self.randomization_scale = float(scale)

    def set_target_scale(self, scale):
        if not np.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("target scale must be in (0, 1]")
        self.target_scale = float(scale)

    def _tick(self, action, *, score=True):
        self.control.accept(action)
        safe_qpos = self.data.qpos.copy()
        safe_qvel = self.data.qvel.copy()
        safe_time = float(self.data.time)
        last = None
        for _ in range(self.substeps):
            self.control.apply()
            mujoco.mj_step(self.model, self.data)
            # Do not inspect contact or geometry buffers after a divergent
            # integration step. Some MuJoCo builds can segfault while exposing
            # those views even when the generalized state already contains NaNs.
            if invalid_physics(self.data):
                self.data = mujoco.MjData(self.model)
                self.data.qpos[:] = safe_qpos
                self.data.qvel[:] = safe_qvel
                self.data.time = safe_time
                mujoco.mj_forward(self.model, self.data)
                self.control = type(self.control)(self.model, self.data, self.profile)
                self._configure_preview()
                self.control.stop()
                self.monitor = ContactMonitor(self.model, self.profile)
                self._physics_invalid = True
                return {"force": 0.0, "bilateral_contact": False, "invalid_physics": True}
            last = self.monitor.update(self.data, self.payload)
            if last["invalid_contact"]:
                self.control.stop()
                self._physics_invalid = True
                return {"force": 0.0, "bilateral_contact": False, "invalid_physics": True}
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
        sampled_payload = str(self.np_random.choice(("object1", "object2")))
        self.payload_name = self.config.payload_object or sampled_payload
        self.payload = m.body(self.payload_name).id
        mass_factor = self.np_random.uniform(1 - 0.2 * self.randomization_scale, 1 + 0.2 * self.randomization_scale)
        density_factor = (
            self.config.payload_mass_kg / self.mass[self.payload] if self.config.payload_mass_kg is not None else 1.0
        )
        m.body_mass[self.payload] *= density_factor * mass_factor
        m.body_inertia[self.payload] *= density_factor * mass_factor
        friction_factor = self.np_random.uniform(1 - 0.2 * self.randomization_scale, 1 + 0.2 * self.randomization_scale)
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
            if self.scenario != "reach":
                q[0] = q[4] = 0
                q[3] = -q[1] - q[2]
        self.control.initialize(q, p.gripper_open)
        if self.scenario == "reach" and self.config.initial_heading_range:
            base_joint = int(m.body_jntadr[self.control.base])
            if base_joint < 0 or m.jnt_type[base_joint] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError("heading randomization requires a free chassis")
            address = int(m.jnt_qposadr[base_joint]) + 3
            yaw = self.np_random.uniform(-self.config.initial_heading_range, self.config.initial_heading_range)
            quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            result = np.empty(4)
            mujoco.mju_mulQuat(result, quat, d.qpos[address : address + 4])
            d.qpos[address : address + 4] = result
            mujoco.mj_forward(m, d)
            self.control.reset()
        from emet.rl.manipulation.payload_assets import provenance

        self.randomization = {
            **provenance(self.config.payload_asset_set),
            "mass_factor": mass_factor,
            "payload_mass_kg": float(m.body_mass[self.payload]),
            "payload_inertia_kg_m2": m.body_inertia[self.payload].tolist(),
            "payload_com_local_m": m.body_ipos[self.payload].tolist(),
            "friction_factor": friction_factor,
            "scale": self.randomization_scale,
        }
        if self.config.payload_asset_set == "native_table" and self.scenario != "reach" and p.robot == "innate_mars":
            from emet.rl.manipulation.payload_assets import nominal_pinch_admission

            admission = nominal_pinch_admission(m, d, p, self.payload, self.config.grasp_depth)
            self.randomization["nominal_pinch_admission"] = admission
            if not admission["accepted"]:
                self.initialization_failure_reason = admission["reason"]
                return False
        points = [
            d.body(name).xpos
            + d.body(name).xmat.reshape(3, 3)
            @ np.array([self.config.grasp_depth, 0, 0] if p.robot == "innate_mars" else [0, 0, 0.01])
            for name in p.finger_bodies
        ]
        center = np.mean(points, axis=0)
        # A real support surface holds the object during physical closure.
        table = m.body("table").id
        table_geom = m.body_geomadr[table]
        payload_geom = m.body_geomadr[self.payload]
        size = m.geom_size[payload_geom]
        half_height = size[1] if m.geom_type[payload_geom] == mujoco.mjtGeom.mjGEOM_CYLINDER else size[2]
        m.body_pos[table] = [center[0], center[1], center[2] - half_height - m.geom_size[table_geom, 2]]
        m.geom_size[table_geom, :2] = self.config.support_half_width if p.robot == "innate_mars" else 0.15
        for name in ("object1", "object2"):
            address = m.jnt_qposadr[m.body(name).jntadr[0]]
            d.qpos[address : address + 3] = center if name == self.payload_name else [4, 4, 0.5]
        if self.scenario == "reach":
            # Free-space reaching includes base turns. A table under the hand
            # becomes an unintended obstacle during those turns; loaded/grasp
            # scenarios retain the physical support fixture.
            m.body_pos[table] = [4, 4, 0.1]
            for index, name in enumerate(("object1", "object2")):
                address = m.jnt_qposadr[m.body(name).jntadr[0]]
                d.qpos[address : address + 3] = [3 + index * 0.1, 3, 0.05]
        mujoco.mj_forward(m, d)
        grasp_reference = self.control.positions().copy()
        if self.scenario == "pregrasp":
            approach = (
                self._vertical_reference(0.025)
                if p.robot == "innate_mars"
                else grasp_reference + [0.025, -0.025, 0, 0, 0]
            )
            self.control.initialize(approach, p.gripper_open)
        self.monitor.reset()
        # Allow the initialized chassis to establish ground contact; these steps
        # are reset initialization, not a claimed no-lift rollout.
        for _ in range(20):
            self._tick(np.zeros(p.action_size), score=False)
            if self._physics_invalid:
                return False
        self.monitor.reset()
        self.initial_object_height = float(d.xpos[self.payload, 2])
        self._scheduled_push = self.scenario == "disturbance" or (
            self.scenario == "reach"
            and self.config.reach_push_probability > 0
            and self.np_random.random() < self.config.reach_push_probability
        )
        if self._scheduled_push:
            horizon_steps = int(self.config.horizon / p.dt)
            latest_start = (
                max(10, min(20, horizon_steps - 12)) if self.scenario == "reach" else max(10, horizon_steps - 12)
            )
            self.disturbance_start = int(self.np_random.integers(10, latest_start + 1))
            self.disturbance_steps = int(self.np_random.integers(6, 13))
            self.disturbance_wrench = (
                sample_disturbance(self.np_random, self.disturbance_difficulty) * self.randomization_scale
            )
            if self.scenario == "reach":
                force = self.disturbance_wrench[:3].copy()
                norm = np.linalg.norm(force)
                self.disturbance_wrench[:] = 0
                if norm:
                    self.disturbance_wrench[:3] = force / norm * 0.5 * self.randomization_scale
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
            evidence = self._tick(np.zeros(p.action_size))
            bilateral_steps = bilateral_steps + 1 if evidence["bilateral_contact"] else 0
            if self.control.stopped:
                return False
            if p.robot == "innate_mars" and bilateral_steps >= 20:
                # Lift after sustained bilateral contact, keeping bounded
                # closure torque to maintain grasp pressure.
                break
        if bilateral_steps < 20:
            return False
        lift_height = float(d.xpos[self.control.ee, 2]) + 0.06
        lift_rotation = d.xmat[self.control.ee].reshape(3, 3).copy()
        self.control.set_reference(self._vertical_reference(0.003 if p.robot == "innate_mars" else 0.06), phase="carry")
        stable_reference = None
        stable_steps = 0
        for index in range(160 if p.robot == "innate_mars" else 80):
            if p.robot == "innate_mars":
                remaining = max(0.0, lift_height - float(d.xpos[self.control.ee, 2]))
                self.control.set_reference(
                    self._vertical_reference(min(0.003, remaining), rotation=lift_rotation), phase="carry"
                )
            evidence = self._tick(np.zeros(p.action_size))
            if self.control.stopped:
                return False
            if index == (139 if p.robot == "innate_mars" else 59):
                stable_reference = relative_pose(d, self.control.ee, self.payload)
            if index >= (140 if p.robot == "innate_mars" else 60):
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
        self._reference_fk = None
        self.arm_trajectory = None
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
        self.initialization_failure_reason = None
        self.initialization_rejections = 0
        self.initialization_failed = True
        for _ in range(self.config.initialization_attempts):
            self._physics_invalid = False
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.control.action_mode = "residual"
            self.control.base_xy_tolerance = self.config.base_xy_tolerance
            self.control.reset()
            self.monitor.reset()
            try:
                initialized = self._initialize()
            except ResetMotionUnreachable:
                initialized = False
            if initialized:
                self.initialization_failed = False
                break
            self.initialization_rejections += 1
        self.control.action_mode = self.config.action_mode
        self._configure_preview()
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
        self.peak_hand_position_error = self.peak_hand_orientation_error = 0.0
        self.hand_squared_error = self.hand_tracking_time = 0.0
        self.retention_lost = False
        self.supported_time = self.settle_time = self.bilateral_time = 0.0
        self.first_settle_time = None
        self.hold_base_anchor = None
        self.peak_hold_drift = 0.0
        self.last_actor_action = np.zeros(self.profile.action_size)
        self.last_velocity = self.control.velocities()
        self.last_base_pose = self.control.base_pose()
        self.last_base_velocity = np.zeros(2)
        self.last_base_acceleration = np.zeros(2)
        self.last_acceleration = np.zeros(self.profile.arm_size)
        self.peak_jerk = self.peak_acceleration = 0.0
        self.peak_base_jerk = self.peak_base_acceleration = 0.0
        self.peak_joint_jerk = np.zeros(self.profile.arm_size)
        self.peak_base_axis_jerk = np.zeros(2)
        self._released = False
        self._disturbed = False
        self._lift_commanded = False
        self._motion_infeasible = False
        self._closure_commanded = False
        self._previous_cost = 0.0
        self.command_delay = (
            int(self.np_random.integers(0, 3))
            if self.randomization_scale == 1
            or (self.randomization_scale > 0 and self.np_random.random() < self.randomization_scale)
            else 0
        )
        self.actions = [np.zeros(self.profile.action_size) for _ in range(self.command_delay)]
        try:
            self._make_reference()
        except ResetMotionUnreachable:
            self.initialization_failed = True
            self.control.set_reference(self.control.positions(), phase="reach")
            self.control.stop()
        self._set_pose_targets()
        if self.control.pose_only_targets:
            from emet.rl.manipulation.tracking import TaskFrameFK

            self._reference_fk = TaskFrameFK(self.model, self.data, self.control)
        self.retargets = 0
        self.retarget_step = None
        self.retarget_goal = None
        if (
            self.scenario == "reach"
            and self.config.retarget_probability > 0
            and self.np_random.random() < self.config.retarget_probability
        ):
            self.retarget_step = int(self.np_random.integers(8, 20))
            # Freeze absolute targets at reset so teacher/policy cases remain
            # identical even when their states differ at the retarget time.
            arm = np.clip(
                self.control.positions()
                + self.np_random.uniform(-1, 1, self.profile.arm_size)
                * np.asarray(self.profile.speed)
                * 0.5
                * self.target_scale,
                self.control.limits[:, 0] + 0.02,
                self.control.limits[:, 1] - 0.02,
            )
            base = self.control.base_pose()
            if self.control.base_mask.any() or (self.control.pose_only_targets and self.control.hand_mask):
                base += self.np_random.uniform([-0.12, -0.12, -0.25], [0.12, 0.12, 0.25]) * self.target_scale
                if not self.control.hand_mask:
                    base = self._base_only_goal(base)
            if not self.control.hand_mask:
                arm = self.control.positions()
            self.retarget_goal = (base, arm, self._fk_hand_pose(base, arm))
        self.arm_trajectory = None
        self.trajectory_rejections = []
        self.trajectory_admission = {"accepted": None, "reason": "not_requested"}
        if self.scenario == "reach" and (self.config.certify_reach_paths or self.config.dynamic_references):
            self._prepare_reach_trajectory()
        self._previous_cost = self._tracking_cost()
        obs = self.control.observe()
        from emet.rl.manipulation.risk import trajectory_features

        self.risk_features = trajectory_features(
            obs["history"][:, :36], self.control.reference[None], self.config.horizon
        ).tolist()
        return obs, self._info()

    def _prepare_reach_trajectory(self):
        from emet.rl.manipulation.dynamic_trajectory import ArmTrajectory, admit, build, segment

        q0 = self.control.positions()
        for attempt in range(20):
            if attempt:
                arm = np.clip(
                    q0
                    + self.np_random.uniform(-1, 1, self.profile.arm_size)
                    * np.asarray(self.profile.speed)
                    * 0.5
                    * self.target_scale,
                    self.control.limits[:, 0] + 0.02,
                    self.control.limits[:, 1] - 0.02,
                )
                if not self.control.hand_mask:
                    arm = q0.copy()
                self.control.set_reference(arm, phase="reach")
                if self.retarget_goal is not None:
                    base, _, _ = self.retarget_goal
                    other = np.clip(
                        q0
                        + self.np_random.uniform(-1, 1, self.profile.arm_size)
                        * np.asarray(self.profile.speed)
                        * 0.5
                        * self.target_scale,
                        self.control.limits[:, 0] + 0.02,
                        self.control.limits[:, 1] - 0.02,
                    )
                    if not self.control.hand_mask:
                        other = q0.copy()
                    self.retarget_goal = (base, other, self._fk_hand_pose(base, other))
            goal = self.control.reference.copy()
            replan_time = None if self.retarget_step is None else self.retarget_step * self.profile.dt
            if self.config.dynamic_references:
                path = build(
                    q0,
                    goal,
                    self.profile,
                    self.np_random,
                    replan_time=replan_time,
                    replan_goal=None if self.retarget_goal is None else self.retarget_goal[1],
                )
            else:
                path = ArmTrajectory([segment(q0, goal, self.profile.speed, self.profile.acceleration)])
            admission = admit(self, path)
            # A static retarget is also checked from the reset configuration;
            # execution from the future measured state still requires rollout validation.
            if admission["accepted"] and not self.config.dynamic_references and self.retarget_goal is not None:
                admission = admit(
                    self,
                    ArmTrajectory([segment(q0, self.retarget_goal[1], self.profile.speed, self.profile.acceleration)]),
                )
            if admission["accepted"] and self.config.dynamic_references and path.end + 1 > self.config.horizon:
                admission = {"accepted": False, "reason": "reference_exceeds_horizon"}
            if admission["accepted"]:
                self.trajectory_admission = admission
                self.control.set_pose_targets(
                    self.control.base_goal,
                    self.control.base_mask.copy(),
                    self._fk_hand_pose(self.control.base_goal, goal),
                    self.control.hand_mask,
                )
                if self.config.dynamic_references:
                    self.arm_trajectory = path
                    self._advance_dynamic_reference()
                return
            self.trajectory_rejections.append(
                {
                    **admission,
                    "goal": goal.tolist(),
                    "trajectory": path.record(),
                    "base": self.control.base_goal.tolist(),
                    "base_mask": self.control.base_mask.tolist(),
                    "hand_active": bool(self.control.hand_mask),
                    "q_start": q0.tolist(),
                }
            )
        self.trajectory_admission = {"accepted": False, "reason": "sampling_exhausted"}
        self.initialization_failed = True
        self.control.stop()

    def _advance_dynamic_reference(self):
        if self.arm_trajectory is None:
            return
        elapsed = float(self.data.time - self.start_time)
        if self.arm_trajectory.replan_time is not None and not self.retargets:
            arm = self.arm_trajectory.segments[0].sample(max(self.arm_trajectory.segments[0].start, elapsed))[0]
        else:
            arm = self.arm_trajectory.sample(elapsed)[0]
        self.control.set_reference(arm, phase="reach")
        self.control.set_pose_targets(
            self.control.base_goal,
            self.control.base_mask.copy(),
            self._fk_hand_pose(self.control.base_goal, arm),
            self.control.hand_mask,
        )
        self._previous_cost = self._tracking_cost()

    def _vertical_reference(self, height, *, rotation=None, xy=None):
        """Robot-specific vertical translation, using a private FK/Jacobian solve."""
        if self.profile.robot == "stretch":
            q = self.control.positions()
            q[0] += height
            return np.clip(q, self.control.limits[:, 0], self.control.limits[:, 1])
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, scratch)
        goal = scratch.xpos[self.control.ee].copy() + [0, 0, height]
        if xy is not None:
            goal[:2] = xy
        rotation = scratch.xmat[self.control.ee].reshape(3, 3).copy() if rotation is None else rotation
        jac = np.zeros((3, self.model.nv))
        jac_rotation = np.zeros_like(jac)
        columns = [int(v[0]) for v in self.control.v]
        addresses = [int(q[0]) for q in self.control.q]
        for _ in range(200):
            error = goal - scratch.xpos[self.control.ee]
            current_rotation = scratch.xmat[self.control.ee].reshape(3, 3)
            angular = 0.5 * sum(np.cross(current_rotation[:, k], rotation[:, k]) for k in range(3))
            if np.linalg.norm(error) < 0.001 and np.linalg.norm(angular) < 0.005:
                return scratch.qpos[addresses].copy()
            mujoco.mj_jacBody(self.model, scratch, jac, jac_rotation, self.control.ee)
            j = np.vstack([jac[:, columns], 0.2 * jac_rotation[:, columns]])
            error = np.r_[error, 0.2 * angular]
            step = j.T @ np.linalg.solve(j @ j.T + 1e-5 * np.eye(6), error)
            scratch.qpos[addresses] = np.clip(
                scratch.qpos[addresses] + np.clip(step, -0.03, 0.03),
                self.control.limits[:, 0],
                self.control.limits[:, 1],
            )
            mujoco.mj_forward(self.model, scratch)
        raise ResetMotionUnreachable("vertical reset motion is unreachable")

    def _make_reference(self):
        if self.scenario == "pregrasp":
            return
        q = self.control.positions()
        if self.scenario == "place":
            q = self._vertical_reference(-0.05)
        else:
            scale = np.asarray(self.profile.speed) * 0.5
            if self.profile.robot == "innate_mars" and self.scenario != "reach":
                # Same nearby joint-space ranges as generated TAMP component
                # trajectories, rather than Stretch-scale angular jumps.
                scale = np.array([0.03, 0.02, 0.12, 0.08, 0.08])
            q += self.np_random.uniform(-1, 1, self.profile.arm_size) * scale * self.target_scale
            if self.scenario != "reach":
                if self.profile.robot == "stretch":
                    q[0] = max(q[0], self.control.positions()[0])
        q = np.clip(q, self.control.limits[:, 0] + 0.02, self.control.limits[:, 1] - 0.02)
        phase = "reach" if self.scenario == "reach" else self.scenario
        self.control.set_reference(q, phase=phase)

    def _base_only_goal(self, base):
        base = base.copy()
        if self.config.base_only_min_distance:
            angle = self.np_random.uniform(-np.pi, np.pi)
            minimum = self.config.base_only_min_distance
            maximum = (
                minimum + self.target_scale * (self.config.base_only_max_distance - minimum)
                if self.config.base_only_max_distance
                else 1.5 * minimum
            )
            radius = self.np_random.uniform(minimum, maximum)
            base[:2] = self.control.base_pose()[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
        return base

    def _set_pose_targets(self):
        control, profile = self.control, self.profile
        base = control.base_pose()
        if profile.wheel_bodies:
            if self.scenario == "reach":
                self.target_mode = (
                    int(self.np_random.choice((1, 2, 3), p=(0.2, 0.4, 0.4)))
                    if self.config.reach_target_mode == "mixed"
                    else {"base": 1, "hand": 2, "combined": 3}[self.config.reach_target_mode]
                )
            else:
                self.target_mode = int(self.np_random.choice((2, 3), p=(0.5, 0.5)))
            base_mask = np.full(3, self.target_mode in (1, 3), dtype=bool)
            if self.target_mode in (1, 3) or (
                self.scenario == "reach" and self.control.pose_only_targets and self.target_mode == 2
            ):
                # Sample a reachable whole-body configuration even when only
                # the hand is constrained. The latent base pose is not an
                # actor command or an IK hint.
                base = base + self.np_random.uniform([-0.12, -0.12, -0.25], [0.12, 0.12, 0.25]) * self.target_scale
                if self.target_mode == 1 and self.config.base_only_min_distance:
                    base = self._base_only_goal(base)
            if self.target_mode == 1:
                control.set_reference(control.positions(), phase="reach")
        else:
            self.target_mode = 2
            base_mask = np.zeros(3, dtype=bool)
        hand_active = self.target_mode in (2, 3)
        hand_pose = self._fk_hand_pose(base, control.reference)
        control.set_pose_targets(base, base_mask, hand_pose, hand_active)

    def _fk_hand_pose(self, base_target, arm_configuration):
        """Evaluate the hand pose at the commanded base and arm configuration."""
        control = self.control
        if self.control.pose_only_targets and self._reference_fk is not None:
            return self._reference_fk(base_target, arm_configuration)
        scratch = mujoco.MjData(self.model)
        mujoco.mj_copyData(scratch, self.model, self.data)
        current_base = control.base_pose()
        delta = np.asarray(base_target) - current_base
        if control.profile.wheel_bodies:
            joint_id = int(self.model.body_jntadr[control.base])
            qadr = int(self.model.jnt_qposadr[joint_id])
            scratch.qpos[qadr : qadr + 2] += delta[:2]
            yaw_delta = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
            half = yaw_delta / 2
            w, x, y, z = scratch.qpos[qadr + 3 : qadr + 7].copy()
            scratch.qpos[qadr + 3 : qadr + 7] = [
                np.cos(half) * w - np.sin(half) * z,
                np.cos(half) * x - np.sin(half) * y,
                np.cos(half) * y + np.sin(half) * x,
                np.cos(half) * z + np.sin(half) * w,
            ]
        else:
            yaw_delta = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
            for name, offset in (("base_x", delta[0]), ("base_y", delta[1]), ("base_yaw", yaw_delta)):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    scratch.qpos[self.model.jnt_qposadr[jid]] += offset
        for group, value in zip(control.q, arm_configuration, strict=True):
            scratch.qpos[group] = value / len(group)
        mujoco.mj_forward(self.model, scratch)
        rotation = scratch.xmat[control.ee].reshape(3, 3)
        return np.r_[scratch.xpos[control.ee], rotation[:, :2].ravel()]

    def _tracking_cost(self):
        control = self.control
        cost = 0.0  # Score the active world-frame hand contract, not a redundant IK solution.
        if control.base_mask.any():
            delta = control.base_goal - control.base_pose()
            yaw = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
            cost += float(
                max(0, np.linalg.norm(delta[:2] * control.base_mask[:2]) - self.config.base_xy_tolerance)
                + 0.2 * abs(yaw) * control.base_mask[2]
            )
        if control.hand_mask:
            position_error, angle_error = self._hand_pose_error()
            cost += position_error + 0.1 * angle_error
        return cost

    def payload_support_load(self):
        """Solved upward support load; contact candidates alone are insufficient."""
        from emet.rl.contact import solved_contact_force

        table = self.model.body("table").id
        count = min(int(self.data.ncon), len(self.data.contact))
        bodies = self.model.geom_bodyid.copy()
        total = 0.0
        for index in range(count):
            c = self.data.contact[index]
            if set(bodies[[c.geom1, c.geom2]]) == {self.payload, table} and abs(c.frame[2]) > 0.8:
                total += solved_contact_force(self.model, self.data, index)[0]
        return total

    def _hand_pose_error(self):
        control = self.control
        current = control.ee_pose()
        current_columns = current[3:].reshape(3, 2)
        target_columns = control.hand_goal[3:].reshape(3, 2)
        current_rotation = np.column_stack([current_columns, np.cross(current_columns[:, 0], current_columns[:, 1])])
        target_rotation = np.column_stack([target_columns, np.cross(target_columns[:, 0], target_columns[:, 1])])
        angle = float(np.arccos(np.clip((np.trace(target_rotation.T @ current_rotation) - 1) / 2, -1, 1)))
        return float(np.linalg.norm(control.hand_goal[:3] - current[:3])), angle

    def _refresh_hand_target(self):
        control = self.control
        hand_pose = self._fk_hand_pose(control.base_goal, control.reference)
        control.set_pose_targets(control.base_goal, control.base_mask, hand_pose, control.hand_mask)

    def _info(self):
        hand_position_error, hand_orientation_error = self._hand_pose_error()
        heading_delta = self.control.base_goal[2] - self.control.base_pose()[2]
        return {
            **self.monitor.metrics(),
            "scenario": self.scenario,
            "hand_target_active": bool(self.control.hand_mask),
            "base_target_active": bool(self.control.base_mask.any()),
            "retargets": self.retargets,
            "trajectory_admission": self.trajectory_admission,
            "rejected_trajectories": len(self.trajectory_rejections),
            "robot": self.profile.robot,
            "initialization_failed": self.initialization_failed,
            "initialization_failure_reason": self.initialization_failure_reason,
            "initialization_rejections": self.initialization_rejections,
            "retention_lost": self.retention_lost,
            "max_payload_translation": self.max_translation,
            "max_payload_rotation": self.max_rotation,
            "peak_measured_jerk": max(self.peak_jerk, self.peak_base_jerk),
            "peak_measured_acceleration": max(self.peak_acceleration, self.peak_base_acceleration),
            "peak_base_jerk": self.peak_base_jerk,
            "peak_joint_jerk": self.peak_joint_jerk.tolist(),
            "peak_base_linear_jerk_m_s3": float(self.peak_base_axis_jerk[0]),
            "peak_base_yaw_jerk_rad_s3": float(self.peak_base_axis_jerk[1]),
            "peak_base_acceleration": self.peak_base_acceleration,
            "joint_error": float(np.max(np.abs(self.control.reference - self.control.positions()))),
            "ee_position_error": hand_position_error,
            "peak_hand_position_error": self.peak_hand_position_error,
            "peak_hand_orientation_error": self.peak_hand_orientation_error,
            "rms_hand_position_error": float(np.sqrt(self.hand_squared_error / max(self.hand_tracking_time, 1e-9))),
            "ee_orientation_error": hand_orientation_error,
            "base_linear_velocity_m_s": float(self.last_base_velocity[0]),
            "base_yaw_velocity_rad_s": float(self.last_base_velocity[1]),
            "base_heading_error_rad": float(abs(np.arctan2(np.sin(heading_delta), np.cos(heading_delta))))
            if self.control.base_mask[2]
            else 0.0,
            "max_joint_velocity": float(np.max(np.abs(self.control.velocities()))),
            "settled_seconds": self.settle_time,
            "goal_age_seconds": max(0.0, float(self.data.time - self.control.goal_changed_time)),
            "time_to_first_settle": self.first_settle_time,
            "peak_hold_base_drift_m": self.peak_hold_drift,
            "tracking_precision": self.tracking_precision,
            "base_xy_tolerance": self.config.base_xy_tolerance,
            "base_heading_tolerance": self.config.base_heading_tolerance,
            "base_contract_violation": max(
                0.0,
                float(
                    np.linalg.norm(
                        (self.control.base_goal[:2] - self.control.base_pose()[:2]) * self.control.base_mask[:2]
                    )
                )
                - self.config.base_xy_tolerance,
            ),
            "base_goal_error": float(
                np.linalg.norm(self.control.base_goal[:2] - self.control.base_pose()[:2])
                if self.control.base_mask[:2].any()
                else 0.0
            ),
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
        if self._scheduled_push:
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
        if self._physics_invalid:
            self._done = True
            return (
                self.control.observe(),
                -20.0,
                True,
                False,
                {**self._info(), "code": "invalid_physics", "invalid_physics": True},
            )
        if self.config.tracking_reward and self.arm_trajectory is not None:
            self._advance_dynamic_reference()
        self.control.update_task_reference()
        dt = self.profile.dt
        velocity = self.control.velocities()
        acceleration = (velocity - self.last_velocity) / dt
        jerk = (acceleration - self.last_acceleration) / dt
        self.peak_acceleration = max(self.peak_acceleration, float(np.max(abs(acceleration))))
        self.peak_jerk = max(self.peak_jerk, float(np.max(abs(jerk))))
        self.peak_joint_jerk = np.maximum(self.peak_joint_jerk, abs(jerk))
        self.last_velocity, self.last_acceleration = velocity, acceleration
        base_pose = self.control.base_pose()
        base_delta = base_pose - self.last_base_pose
        base_delta[2] = np.arctan2(np.sin(base_delta[2]), np.cos(base_delta[2]))
        base_velocity = np.array([np.linalg.norm(base_delta[:2]) / dt, base_delta[2] / dt])
        base_acceleration = (base_velocity - self.last_base_velocity) / dt
        base_jerk = (base_acceleration - self.last_base_acceleration) / dt
        self.peak_base_acceleration = max(self.peak_base_acceleration, float(np.max(abs(base_acceleration))))
        self.peak_base_jerk = max(self.peak_base_jerk, float(np.max(abs(base_jerk))))
        self.peak_base_axis_jerk = np.maximum(self.peak_base_axis_jerk, abs(base_jerk))
        self.last_base_pose, self.last_base_velocity, self.last_base_acceleration = (
            base_pose,
            base_velocity,
            base_acceleration,
        )
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
            if self.control.pose_only_targets:
                position_error, angle_error = self._hand_pose_error()
                base_error = self.control.base_goal - self.control.base_pose()
                base_yaw_error = np.arctan2(np.sin(base_error[2]), np.cos(base_error[2]))
                approach_reached = (
                    position_error < self.config.hand_position_tolerance
                    and angle_error < self.config.hand_orientation_tolerance
                    and np.linalg.norm(base_error[:2] * self.control.base_mask[:2]) < self.config.base_xy_tolerance
                    and (not self.control.base_mask[2] or abs(base_yaw_error) < self.config.base_heading_tolerance)
                    and abs(self.last_base_velocity[0]) < 0.005
                    and abs(self.last_base_velocity[1]) < 0.02
                )
            else:
                approach_reached = np.max(abs(self.control.reference - self.control.positions())) < 0.02
            if not self._closure_commanded and approach_reached and np.max(abs(velocity)) < 0.03:
                self.control.set_reference(self.control.reference, phase="close", grip=self.profile.gripper_closed)
                self._closure_commanded = True
            self.bilateral_time = self.bilateral_time + dt if evidence["bilateral_contact"] else 0
            if self.bilateral_time >= 1 and not self._lift_commanded:
                self.retention_reference = relative_pose(self.data, self.control.ee, self.payload)
                try:
                    self._lift_height = float(self.data.xpos[self.control.ee, 2]) + self.config.grasp_lift_height
                    self._lift_rotation = self.data.xmat[self.control.ee].reshape(3, 3).copy()
                    self._lift_xy = self.data.xpos[self.control.ee, :2].copy()
                    self.control.set_reference(
                        self._vertical_reference(0.003, rotation=self._lift_rotation, xy=self._lift_xy), phase="carry"
                    )
                except ResetMotionUnreachable:
                    self._motion_infeasible = True
                    self.control.stop()
                self._refresh_hand_target()
                self._lift_commanded = True
                self._previous_cost = self._tracking_cost()
        if self.scenario == "pregrasp" and self._lift_commanded and not self.control.stopped:
            remaining = max(0.0, self._lift_height - float(self.data.xpos[self.control.ee, 2]))
            if remaining > 0.001:
                try:
                    self.control.set_reference(
                        self._vertical_reference(min(0.003, remaining), rotation=self._lift_rotation, xy=self._lift_xy),
                        phase="carry",
                    )
                    self._refresh_hand_target()
                except ResetMotionUnreachable:
                    self._motion_infeasible = True
                    self.control.stop()
        base_pose = self.control.base_pose()
        base_delta = self.control.base_goal - base_pose
        base_delta[2] = np.arctan2(np.sin(base_delta[2]), np.cos(base_delta[2]))
        base_within_pose = not self.control.base_mask.any() or (
            np.linalg.norm(base_delta[:2] * self.control.base_mask[:2]) < self.config.base_xy_tolerance
            and (not self.control.base_mask[2] or abs(base_delta[2]) < self.config.base_heading_tolerance)
        )
        base_motion_allowed = self.control.base_mask.any() or (
            self.config.trajectory_preview and self.control.hand_mask
        )
        base_settled = base_within_pose and (
            not base_motion_allowed
            or (abs(self.last_base_velocity[0]) < 0.005 and abs(self.last_base_velocity[1]) < 0.02)
        )
        hand_position_error, hand_orientation_error = self._hand_pose_error()
        if self.control.hand_mask:
            self.peak_hand_position_error = max(self.peak_hand_position_error, hand_position_error)
            self.peak_hand_orientation_error = max(self.peak_hand_orientation_error, hand_orientation_error)
            self.hand_squared_error += hand_position_error**2 * dt
            self.hand_tracking_time += dt
        hand_settled = not self.control.hand_mask or (
            hand_position_error < self.config.hand_position_tolerance
            and hand_orientation_error < self.config.hand_orientation_tolerance
        )
        settled = np.max(np.abs(velocity)) < 0.03 and base_settled and hand_settled
        if self.scenario == "place":
            table_body = self.model.body("table").id
            supported = self.payload_support_load() >= 0.5 * self.model.body_mass[self.payload] * 9.81
            if settled and supported and self.control.phase == "place" and not self._released:
                self.control.set_reference(self.control.reference, phase="release", release=True)
                self._released = True
            object_velocity = self.data.qvel[
                self.model.body(self.payload).dofadr[0] : self.model.body(self.payload).dofadr[0] + 6
            ]
            geom = self.model.body_geomadr[self.payload]
            size = self.model.geom_size[geom].copy()
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                size = np.array([size[0], size[0], size[1]])
            extent = np.abs(self.data.geom_xmat[geom].reshape(3, 3)) @ size
            inside = np.all(
                np.abs(self.data.xpos[self.payload, :2] - self.model.body_pos[table_body, :2]) + extent[:2]
                <= self.model.geom_size[self.model.body_geomadr[table_body], :2]
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
        if self._scheduled_push:
            settled &= (
                self._disturbed or not np.any(self.disturbance_wrench)
            ) and self.steps >= self.disturbance_start + self.disturbance_steps
        settled &= self._reference_complete()
        self.settle_time = self.settle_time + dt if settled else 0
        unsafe = self.monitor.peak_force > self.profile.unsafe_force
        invalid = invalid_physics(self.data)
        failure = unsafe or self.monitor.wheel_lift or self.retention_lost or invalid or self._motion_infeasible
        success = self.settle_time >= (2 if self.scenario == "place" else 1) and not failure
        continuing = self.config.tracking_reward and self.scenario == "reach"
        success_event = success and self.first_settle_time is None
        if success_event:
            self.first_settle_time = float(self.data.time - self.start_time)
            self.hold_base_anchor = self.control.base_pose()[:2].copy()
        if self.hold_base_anchor is not None:
            self.peak_hold_drift = max(
                self.peak_hold_drift, float(np.linalg.norm(self.control.base_pose()[:2] - self.hold_base_anchor))
            )
        terminal_success = success and not continuing
        if failure or terminal_success:
            self.control.stop()
        cost = self._tracking_cost()
        at_reference_end = self._reference_complete()
        physical_cost = running_cost(
            dt=dt,
            force=evidence["force"],
            jerk=np.r_[jerk, base_jerk],
            action=action,
            rotation=self.max_rotation,
            translation=self.max_translation,
            tracking=0 if self.config.tracking_reward else cost,
            motion=(
                float(
                    np.sum((velocity / self.profile.speed) ** 2)
                    + (base_velocity[0] / max(self.profile.base_speed, 1e-6)) ** 2
                    + (base_velocity[1] / max(self.profile.base_yaw_speed, 1e-6)) ** 2
                )
                if hand_settled and base_within_pose and at_reference_end
                else 0
            ),
        )
        if self.config.tracking_reward:
            from emet.rl.manipulation.tracking import (
                PRECISION_STAGES,
                differential_base_score,
                endpoint_motion_cost,
                stationary_tracking_score,
                tracking_score,
            )

            pose_score = tracking_score(hand_position_error, hand_orientation_error, self.tracking_precision)
            base_sigma = max(self.config.base_xy_tolerance, PRECISION_STAGES[self.tracking_precision][0])
            base_error = max(
                0.0, float(np.linalg.norm(base_delta[:2] * self.control.base_mask[:2])) - self.config.base_xy_tolerance
            )
            base_score = np.exp(-base_error / base_sigma - abs(base_delta[2]) * self.control.base_mask[2] / 0.2)
            if self.config.differential_base_reward and self.control.base_mask.any():
                delta = self.control._base_tracking_delta()
                bearing = np.arctan2(delta[1], delta[0]) - self.control.base_pose()[2]
                base_score = differential_base_score(
                    base_error, base_delta[2] * self.control.base_mask[2], bearing, base_sigma
                )
            density = 4 * pose_score * self.control.hand_mask + base_score * self.control.base_mask.any()
            in_goal = base_within_pose and hand_settled
            if at_reference_end and (not self.config.goal_gated_settling or in_goal):
                density += base_score * stationary_tracking_score(
                    hand_position_error,
                    hand_orientation_error,
                    velocity,
                    base_velocity,
                    hand_active=self.control.hand_mask,
                )
            action_rate = 0.02 * float(np.sum((action - self.last_actor_action) ** 2))
            endpoint_cost = self.config.endpoint_motion_weight * endpoint_motion_cost(
                self._reference_end_elapsed(), velocity, base_velocity, hand_active=self.control.hand_mask
            )
            if self.config.goal_gated_settling and not in_goal:
                endpoint_cost = 0.0
            reward = dt * (density - action_rate - endpoint_cost) - physical_cost + 10 * success_event - 20 * failure
        else:
            reward = 10 * (self._previous_cost - cost) - physical_cost + 10 * success - 20 * failure
        self.last_actor_action = action.copy()
        self._previous_cost = cost
        self.steps += 1
        truncated = self.steps * dt >= self.config.horizon and not (failure or terminal_success)
        self._done = bool(failure or terminal_success or truncated)
        if truncated:
            self.control.stop()
        code = (
            "wheel_lift"
            if self.monitor.wheel_lift
            else "unsafe_contact"
            if unsafe
            else "retention_lost"
            if self.retention_lost
            else "invalid_physics"
            if invalid
            else "motion_infeasible"
            if self._motion_infeasible
            else "ok"
            if success
            else "tracking_timeout"
            if truncated
            else "running"
        )
        if not self._done and self.retarget_step == self.steps:
            base, arm, hand = self.retarget_goal
            self.control.set_reference(arm, phase="reach")
            self.control.set_pose_targets(base, self.control.base_mask.copy(), hand, self.control.hand_mask)
            self.settle_time = 0
            self.hold_base_anchor = None
            self.retargets += 1
            self._previous_cost = self._tracking_cost()
        if not self._done and self.arm_trajectory is not None:
            self._advance_dynamic_reference()
        info = {
            **self._info(),
            "is_success": bool(success),
            "code": code,
            "invalid_physics": bool(invalid),
            "config": asdict(self.config),
            "risk_features": self.risk_features if self._done else None,
        }
        return self.control.observe(), float(reward), bool(failure or terminal_success), bool(truncated), info
