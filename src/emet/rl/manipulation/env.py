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
from emet.rl.manipulation.safety import ContactMonitor, relative_pose, retention_error, whole_body_stopped

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


def running_cost(*, dt, force, jerk, action, rotation, translation, tracking=0.0, motion=0.0, motion_scale=1.0):
    """Integrate physical costs over time; do not charge them once per sample.

    Sustaining an ordinary grasp must not cost more than dropping it early.
    Terminal safety/retention failures still receive a separate -20 penalty.
    """
    return dt * (
        0.4
        + 0.02 * force
        + motion_scale
        * (0.00001 * float(np.sum(np.asarray(jerk) ** 2)) + 0.01 * float(np.sum(np.asarray(action) ** 2)))
        + 2 * rotation
        + 5 * translation
        + 4 * tracking
        + motion_scale * 0.2 * motion
    )


class ResetMotionUnreachable(ValueError):
    """A procedural reset cannot produce its requested kinematic lift."""


def pose_progress_potential(position_error, orientation_error):
    """Dense error signal outside the fine coupled tracking reward's support."""
    return -float(np.log1p(position_error / 0.02) + np.log1p(orientation_error / 0.2))


def sample_carry_events(config, rng, scenario):
    """Choose training categories once per reset, not once per grasp retry."""
    mode = config.carry_target_mode
    if scenario == "carry" and mode == "mixed":
        mode = str(rng.choice(("translation", "orientation", "combined")))
    push = (
        scenario == "carry"
        and config.carry_push_scale > 0
        and config.carry_push_probability > 0
        and (config.carry_push_probability == 1 or rng.random() < config.carry_push_probability)
    )
    return mode, push


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
    two_supports: bool = False
    contact_cone: str = "pyramidal"
    contact_impratio: float = 1.0
    balanced_grasp_setup: bool = False
    tamp_carry_trajectories: bool = False
    carry_target_mode: str = "combined"
    carry_push_scale: float = 0.0
    carry_push_probability: float = 1.0
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
    grasp_pitch_rad: float = 0.0
    hand_position_tolerance: float = 0.015
    hand_orientation_tolerance: float = 0.10
    trajectory_preview: bool = False
    tracking_reward: bool = False
    tracking_precision: int = 0
    endpoint_motion_weight: float = 0.0
    goal_gated_settling: bool = False
    hand_gated_base_reward: bool = False
    motion_penalty_scale: float = 1.0
    smoothness_curriculum: bool = False
    jerk_limited_lift_reference: bool = False
    grasp_lift_phase_weights: tuple[float, ...] | None = None
    scenario_weights: tuple[float, ...] | None = None
    retention_context: bool = False
    empty_grasp_probability: float = 0.0
    carry_hold_probability: float = 0.0
    grasp_lift_probability: float = 0.0
    grasp_lift_payload_object: str | None = None
    payload_mass_range_kg: tuple[float, float] | None = None
    payload_com_randomization_m: float = 0.0
    reach_hold_probability: float = 0.0
    hold_pregrasp_fixture: bool = False
    fixed_base_probability: float = 0.0
    fixed_base_xy_tolerance: float = 0.005
    categorical_features_unscaled: bool = False
    body_motion_weight: float = 0.0
    feedback_policy: bool = False
    restoring_yaw_feedback: bool = False
    feedback_orientation_scale: float = 1.0
    wheel_feedback_gain: float = 2.0
    physical_wheel_slew: bool = False
    wheel_jerk_limits: tuple[float, float] | None = None
    precision_tracking_reward: bool = False
    terminal_hold_seconds: float = 1.0
    normalized_jerk_weight: float = 0.0
    command_smoothness_weight: float = 0.0
    base_turn_progress_guard: bool = False
    whole_body_settling: bool = False
    reach_local_probability: float = 0.0
    reach_local_translation_probability: float = 0.0
    fixed_base_contract: bool = False
    pose_progress_weight: float = 0.0

    def __post_init__(self):
        from emet.rl.manipulation.retention_training import validate_mixture

        if self.scenario_weights is not None:
            object.__setattr__(self, "scenario_weights", validate_mixture(self.scenario_weights))
        for value in (
            self.reach_hold_probability,
            self.fixed_base_probability,
            self.reach_local_probability,
            self.reach_local_translation_probability,
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("hold/fixed-base probabilities must be in [0, 1]")
        if self.reach_local_probability + self.reach_hold_probability > 1:
            raise ValueError("local and hold reach shares must sum to at most one")
        if self.reach_local_probability and not self.hold_pregrasp_fixture:
            raise ValueError("nearby pose curriculum requires pregrasp fixtures")
        if self.reach_local_translation_probability and not self.reach_local_probability:
            raise ValueError("translation share requires a local reach curriculum")
        if not np.isfinite(self.wheel_feedback_gain) or not 0 <= self.wheel_feedback_gain <= 2:
            raise ValueError("wheel feedback gain must be in [0, 2]")
        if type(self.restoring_yaw_feedback) is not bool or (
            self.restoring_yaw_feedback
            and (
                not self.feedback_policy
                or self.robot != "innate_mars"
                or not self.categorical_features_unscaled
                or not self.trajectory_preview
            )
        ):
            raise ValueError("restoring yaw feedback requires Mars feedback with native masks and preview")
        if type(self.physical_wheel_slew) is not bool:
            raise ValueError("physical wheel slew must be boolean")
        if self.wheel_jerk_limits is not None:
            limits = np.asarray(self.wheel_jerk_limits, dtype=float)
            if (
                not self.physical_wheel_slew
                or limits.shape != (2,)
                or not np.isfinite(limits).all()
                or np.any(limits <= 0)
            ):
                raise ValueError("positive linear/yaw jerk limits require physical wheel slew")
            object.__setattr__(self, "wheel_jerk_limits", tuple(float(v) for v in limits))
        if type(self.precision_tracking_reward) is not bool or (
            self.precision_tracking_reward
            and (not self.tracking_reward or not self.trajectory_preview or self.differential_base_reward)
        ):
            raise ValueError("precision tracking reward requires tracking and trajectory preview")
        if (
            isinstance(self.terminal_hold_seconds, bool)
            or not np.isfinite(self.terminal_hold_seconds)
            or not 0.5 <= self.terminal_hold_seconds <= 5
        ):
            raise ValueError("terminal hold must be between 0.5 and 5 seconds")
        if not np.isfinite(self.normalized_jerk_weight) or not 0 <= self.normalized_jerk_weight <= 2:
            raise ValueError("normalized jerk weight must be in [0,2]")
        if self.normalized_jerk_weight and not self.precision_tracking_reward:
            raise ValueError("normalized jerk cost requires precision tracking reward")
        if type(self.base_turn_progress_guard) is not bool:
            raise ValueError("base turn progress guard must be boolean")
        if not np.isfinite(self.command_smoothness_weight) or not 0 <= self.command_smoothness_weight <= 2:
            raise ValueError("command smoothness weight must be in [0,2]")
        if self.command_smoothness_weight and not self.precision_tracking_reward:
            raise ValueError("command smoothness requires precision tracking reward")
        if not np.isfinite(self.feedback_orientation_scale) or not 0.05 <= self.feedback_orientation_scale <= 1:
            raise ValueError("feedback orientation scale must be in [0.05, 1] rad")
        if self.feedback_orientation_scale != 1 and not self.feedback_policy:
            raise ValueError("orientation feedback scaling requires the feedback policy")
        if type(self.whole_body_settling) is not bool:
            raise ValueError("whole-body settling must be boolean")
        if type(self.fixed_base_contract) is not bool or (self.fixed_base_contract and self.robot != "innate_mars"):
            raise ValueError("fixed-base execution contract currently requires Mars")
        if not np.isfinite(self.pose_progress_weight) or not 0 <= self.pose_progress_weight <= 2:
            raise ValueError("pose progress weight must be in [0, 2]")
        if self.pose_progress_weight and not self.tracking_reward:
            raise ValueError("pose progress shaping requires tracking rewards")
        for value in (self.hand_gated_base_reward, self.smoothness_curriculum, self.jerk_limited_lift_reference):
            if type(value) is not bool or (value and not self.tracking_reward):
                raise ValueError("reward/reference curriculum flags require boolean values and tracking rewards")
        if not np.isfinite(self.motion_penalty_scale) or not 0 <= self.motion_penalty_scale <= 1:
            raise ValueError("motion penalty scale must be in [0,1]")
        if self.jerk_limited_lift_reference and (
            self.robot != "innate_mars" or not self.trajectory_preview or self.action_mode != "direct_velocity"
        ):
            raise ValueError("jerk-limited lift references require Mars direct velocities and preview")
        if self.grasp_lift_phase_weights is not None:
            values = np.asarray(self.grasp_lift_phase_weights, dtype=float)
            if (
                values.shape != (4,)
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or not np.isclose(values.sum(), 1)
            ):
                raise ValueError("lift phase weights require four nonnegative probabilities summing to one")
            if not self.jerk_limited_lift_reference:
                raise ValueError("lift phase resets require jerk-limited references")
            object.__setattr__(self, "grasp_lift_phase_weights", tuple(float(v) for v in values))
        if self.fixed_base_probability and self.robot != "innate_mars":
            raise ValueError("fixed-base manipulation sampling currently requires Mars")
        if not np.isfinite(self.body_motion_weight) or not 0 <= self.body_motion_weight <= 2:
            raise ValueError("body motion weight must be in [0, 2]")
        if self.body_motion_weight and not self.tracking_reward:
            raise ValueError("body motion penalty requires pose tracking reward")
        if type(self.feedback_policy) is not bool or (
            self.feedback_policy
            and (
                not self.categorical_features_unscaled
                or not self.trajectory_preview
                or self.action_mode != "direct_velocity"
            )
        ):
            raise ValueError("feedback policy requires native command features, preview and direct velocities")
        if type(self.hold_pregrasp_fixture) is not bool or type(self.categorical_features_unscaled) is not bool:
            raise ValueError("hold fixture and categorical normalization options must be boolean")
        if self.hold_pregrasp_fixture and (self.robot != "innate_mars" or not self.balanced_grasp_setup):
            raise ValueError("pregrasp hold fixtures require balanced Mars setup")
        if not np.isfinite(self.fixed_base_xy_tolerance) or not 0.005 <= self.fixed_base_xy_tolerance <= 0.01:
            raise ValueError("fixed-base tolerance must be between 5 and 10 mm")
        if type(self.retention_context) is not bool:
            raise ValueError("retention context must be boolean")
        if not np.isfinite(self.empty_grasp_probability) or not 0 <= self.empty_grasp_probability <= 1:
            raise ValueError("empty grasp probability must be in [0, 1]")
        if self.empty_grasp_probability and not self.retention_context:
            raise ValueError("empty grasp counterexamples require retention context")
        if not np.isfinite(self.carry_hold_probability) or not 0 <= self.carry_hold_probability <= 1:
            raise ValueError("carry hold probability must be in [0, 1]")
        if not np.isfinite(self.grasp_lift_probability) or not 0 <= self.grasp_lift_probability <= 1:
            raise ValueError("grasp lift probability must be in [0, 1]")
        if self.grasp_lift_payload_object not in (None, "object1", "object2"):
            raise ValueError("unsupported grasp lift payload slot")
        if self.payload_object is not None and self.grasp_lift_payload_object not in (None, self.payload_object):
            raise ValueError("grasp lift payload conflicts with global payload selection")
        if self.grasp_lift_probability and (
            self.robot != "innate_mars"
            or not self.balanced_grasp_setup
            or not self.trajectory_preview
            or not self.tracking_reward
            or self.action_mode != "direct_velocity"
            or not self.retention_context
            or not self.fixed_base_contract
            or self.tamp_carry_trajectories
        ):
            raise ValueError(
                "grasp lift curriculum requires balanced Mars pose/preview/direct-velocity retention and fixed-base checks; separate from qualified TAMP carry"
            )
        if self.payload_mass_range_kg is not None:
            bounds = np.asarray(self.payload_mass_range_kg, dtype=float)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] <= bounds[1] <= 0.2:
                raise ValueError("payload mass range must be ordered, positive and at most 200 g")
            if self.payload_mass_kg is not None:
                raise ValueError("choose a fixed payload mass or a mass range")
            object.__setattr__(self, "payload_mass_range_kg", tuple(float(v) for v in bounds))
        if not np.isfinite(self.payload_com_randomization_m) or not 0 <= self.payload_com_randomization_m <= 0.005:
            raise ValueError("payload COM randomization must be in [0, 5 mm]")
        if type(self.two_supports) is not bool or (self.two_supports and self.robot != "innate_mars"):
            raise ValueError("two-support fixtures require Mars")
        if self.carry_target_mode not in {"translation", "orientation", "combined", "mixed"}:
            raise ValueError("invalid carry target mode")
        if not np.isfinite(self.carry_push_probability) or not 0 <= self.carry_push_probability <= 1:
            raise ValueError("carry push probability must be in [0, 1]")
        if not np.isfinite(self.carry_push_scale) or not 0 <= self.carry_push_scale <= 1:
            raise ValueError("carry push scale must be in [0, 1]")
        if self.tamp_carry_trajectories and (
            self.robot != "innate_mars"
            or not self.trajectory_preview
            or not self.tracking_reward
            or self.action_mode != "direct_velocity"
        ):
            raise ValueError("TAMP carry references require Mars direct-velocity pose tracking with preview")
        if self.balanced_grasp_setup and self.robot != "innate_mars":
            raise ValueError("balanced grasp setup currently supports Mars only")
        if not np.isfinite(self.contact_impratio) or not 1 <= self.contact_impratio <= 10:
            raise ValueError("contact impedance ratio must be in [1, 10]")
        if self.contact_impratio != 1 and self.contact_cone != "elliptic":
            raise ValueError("nondefault impedance ratio requires elliptic contacts")
        if self.contact_cone not in {"pyramidal", "elliptic"}:
            raise ValueError("invalid contact cone")
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
        if not np.isfinite(self.grasp_pitch_rad) or abs(self.grasp_pitch_rad) > np.pi / 3:
            raise ValueError("grasp pitch must be finite and within +/- pi/3")
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
        self.reach_hold_trial = self.fixed_base_trial = False
        self.reach_local_trial = False
        self.reach_local_mode = None
        self.reach_local_translation = None
        self.grasp_lift_trial = False
        self.grasp_lift_guard = None
        self.grasp_lift_admission = None
        self.grasp_lift_reset_attempts = []
        self.fixed_base_distance_m = self.fixed_base_rotation_rad = 0.0
        self.fixed_base_contract_violation = False
        self.model, self.fingerprint = build_manipulation_model(
            self.config.robot,
            diagnostics=self.config.diagnostics,
            payload_asset_set=self.config.payload_asset_set,
            two_supports=self.config.two_supports,
            contact_cone=self.config.contact_cone,
            contact_impratio=self.config.contact_impratio,
        )
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.control = ManipulationControl(self.model, self.data, self.profile)
        self.control.set_physical_wheel_slew(self.config.physical_wheel_slew, jerk_limits=self.config.wheel_jerk_limits)
        self.control.retention_context_enabled = self.config.retention_context
        self._reference_fk = None
        self.tracking_precision = self.config.tracking_precision
        self.motion_penalty_scale = self.config.motion_penalty_scale
        self.command_smoothness_fraction = 1.0
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
        self.body_ipos = self.model.body_ipos.copy()
        self.friction = self.model.geom_friction.copy()
        self.randomization_scale = self.config.randomization_scale
        self.target_scale = self.config.target_scale
        self.disturbance_episodes = 0
        self._done = True

    def _configure_preview(self):
        self.control.turn_progress_guard_enabled = self.config.base_turn_progress_guard
        self.control.base_velocity_feedback_gain = self.config.wheel_feedback_gain
        # The old teacher's 0.015-rad deadband exceeds this task's 0.01-rad
        # heading success condition. Correct the new task's comparator without
        # changing historical reset/teacher behavior for other distributions.
        self.control.base_yaw_deadband = 0.5 * self.active_base_heading_tolerance if self.grasp_lift_trial else 0.015
        self.control.pose_only_targets = self.config.tracking_reward or self.config.trajectory_preview
        self.control.preview_enabled = self.config.trajectory_preview
        if self.config.trajectory_preview and self.control.task_trajectory is not None:
            self.control.preview_sampler = lambda time: self.control.task_trajectory.sample(
                self.control.trajectory_elapsed() + (time - float(self.data.time)) * self.control.task_progress_rate
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

    def set_motion_penalty_scale(self, scale):
        if not np.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError("motion penalty scale must be in [0,1]")
        self.motion_penalty_scale = float(scale)

    def set_command_smoothness_fraction(self, fraction):
        """Training-only curriculum coefficient, independent of physical limits."""
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("command smoothness fraction must be in [0,1]")
        self.command_smoothness_fraction = float(fraction)

    def _sample_task_reference(self, time):
        path = getattr(self, "arm_trajectory", None)
        if path is None or self._reference_fk is None:
            return self.control.base_goal.copy(), self.control.hand_goal.copy()
        elapsed = max(0.0, time - self.start_time + self.reference_time_offset)
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
            return c.trajectory_end_elapsed()
        if self.arm_trajectory is not None:
            return float(self.data.time - self.start_time + self.reference_time_offset - self.arm_trajectory.end)
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
                last["force"] > self.profile.unsafe_force
                or self.monitor.wheel_lift
                or invalid_physics(self.data)
                or self.control.motion_fault
            ):
                self.control.stop()
                break
        mujoco.mj_forward(self.model, self.data)
        return last

    def _initialize(self):
        grasp_fixture = self.scenario != "reach" or (
            (self.reach_hold_trial or self.reach_local_trial) and self.config.hold_pregrasp_fixture
        )
        m, d, p = self.model, self.data, self.profile
        mujoco.mj_resetData(m, d)
        m.body_mass[:] = self.mass
        m.body_inertia[:] = self.inertia
        m.body_ipos[:] = self.body_ipos
        m.geom_friction[:] = self.friction
        sampled_payload = str(self.np_random.choice(("object1", "object2")))
        self.payload_name = (
            self.config.grasp_lift_payload_object
            if self.grasp_lift_trial and self.config.grasp_lift_payload_object
            else self.config.payload_object or sampled_payload
        )
        self.payload = m.body(self.payload_name).id
        mass_factor = self.np_random.uniform(1 - 0.2 * self.randomization_scale, 1 + 0.2 * self.randomization_scale)
        density_factor = (
            self.config.payload_mass_kg / self.mass[self.payload] if self.config.payload_mass_kg is not None else 1.0
        )
        m.body_mass[self.payload] *= density_factor * mass_factor
        m.body_inertia[self.payload] *= density_factor * mass_factor
        if self.config.payload_mass_range_kg is not None:
            low, high = self.config.payload_mass_range_kg
            middle = (low + high) / 2
            sampled_mass = self.np_random.uniform(
                middle + (low - middle) * self.randomization_scale,
                middle + (high - middle) * self.randomization_scale,
            )
            m.body_mass[self.payload] = sampled_mass
            m.body_inertia[self.payload] = self.inertia[self.payload] * sampled_mass / self.mass[self.payload]
            mass_factor = sampled_mass / self.mass[self.payload]
        if self.config.payload_com_randomization_m:
            bound = self.config.payload_com_randomization_m * self.randomization_scale
            m.body_ipos[self.payload] += self.np_random.uniform(-bound, bound, 3)
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
            if self.grasp_lift_trial:
                # Continuous nearby grasp postures include the straighter elbow
                # used by public execution, as well as ordinary flexed carry.
                # These are procedural training samples, not frozen test states.
                q[2] = self.np_random.uniform(-0.08, 0.25)
            if grasp_fixture:
                q[0] = q[4] = 0
                q[3] = -q[1] - q[2] + self.config.grasp_pitch_rad
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
            "contact_cone": self.config.contact_cone,
            "contact_impratio": self.config.contact_impratio,
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
        if self.scenario == "reach" and not grasp_fixture:
            # Free-space reaching includes base turns. A table under the hand
            # becomes an unintended obstacle during those turns; loaded/grasp
            # scenarios retain the physical support fixture.
            m.body_pos[table] = [4, 4, 0.1]
            for index, name in enumerate(("object1", "object2")):
                address = m.jnt_qposadr[m.body(name).jntadr[0]]
                d.qpos[address : address + 3] = [3 + index * 0.1, 3, 0.05]
        mujoco.mj_forward(m, d)
        if self.config.balanced_grasp_setup and grasp_fixture:
            from emet.rl.manipulation.grasp_setup import plan_setup

            plan = plan_setup(self)
            if plan is None:
                self.initialization_failure_reason = "no_stable_grasp_setup"
                return False
            # Procedural fixture initialization only, before any scored motion.
            d.qpos[:] = plan.pop("qpos")
            mujoco.mj_forward(m, d)
            self.control.reset()
            self.randomization["grasp_setup"] = plan
        grasp_reference = self.control.positions().copy()
        if self.scenario == "pregrasp" or (
            (self.reach_hold_trial or self.reach_local_trial) and self.config.hold_pregrasp_fixture
        ):
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
        self._scheduled_push = (
            self._carry_push_selected
            or self.scenario == "disturbance"
            or (
                self.scenario == "reach"
                and self.config.reach_push_probability > 0
                and self.np_random.random() < self.config.reach_push_probability
            )
        )
        if self._scheduled_push:
            horizon_steps = int(self.config.horizon / p.dt)
            latest_start = (
                max(10, min(20, horizon_steps - 12)) if self.scenario == "reach" else max(10, horizon_steps - 12)
            )
            if self.scenario == "carry":
                # The scored carry contract requires settled time AFTER the
                # push. A late window otherwise makes success impossible even
                # with perfect tracking. Retain two seconds for recovery.
                latest_start = horizon_steps - 12 - int(np.ceil(2.0 / p.dt))
                if latest_start < 10:
                    self.initialization_failure_reason = "carry_disturbance_exceeds_horizon"
                    return False
            self.disturbance_start = int(self.np_random.integers(10, latest_start + 1))
            self.disturbance_steps = int(self.np_random.integers(6, 13))
            self.disturbance_wrench = sample_disturbance(self.np_random, self.disturbance_difficulty) * (
                self.config.carry_push_scale if self.scenario == "carry" else self.randomization_scale
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
        readiness = None
        if self.grasp_lift_trial:
            from emet.rl.manipulation.grasp_lift import SupportedGraspReadiness

            readiness = SupportedGraspReadiness(p.dt)
        ready = False
        for _ in range(100):
            evidence = self._tick(np.zeros(p.action_size))
            bilateral_steps = bilateral_steps + 1 if evidence["bilateral_contact"] else 0
            if readiness is not None:
                ready = readiness.update(
                    relative_pose(d, self.control.ee, self.payload),
                    bilateral=evidence["bilateral_contact"],
                    grip_velocity=d.qvel[self.control.grip_v],
                    supported=self.payload_support_load() >= 0.1 * m.body_mass[self.payload] * 9.81,
                )
            if self.control.stopped:
                if readiness is not None:
                    self.grasp_lift_reset_attempts.append(
                        {"payload": self.payload_name, "reason": "closure_physical_stop", **readiness.last}
                    )
                return False
            if p.robot == "innate_mars" and bilateral_steps >= 20 and (readiness is None or ready):
                # Lift after sustained bilateral contact, keeping bounded
                # closure torque to maintain grasp pressure.
                break
        if readiness is not None:
            self.grasp_lift_reset_attempts.append(
                {"payload": self.payload_name, "bilateral_steps": bilateral_steps, **readiness.last}
            )
        if bilateral_steps < 20 or (readiness is not None and not ready):
            if readiness is not None:
                self.initialization_failure_reason = "supported_grasp_not_quiet"
            return False
        if self.grasp_lift_trial:
            # The learned rollout starts at physical, table-supported closure.
            # Do not perform the teacher lift that initializes ordinary carry.
            self.control.set_reference(self.control.positions(), phase="carry", grip=p.gripper_closed)
            return True
        lift_height = float(d.xpos[self.control.ee, 2]) + self.config.grasp_lift_height
        lift_rotation = d.xmat[self.control.ee].reshape(3, 3).copy()
        self.control.set_reference(
            self._vertical_reference(0.003 if p.robot == "innate_mars" else self.config.grasp_lift_height),
            phase="carry",
        )
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

    def resume_stage(self, scenario, *, reset_clock=False):
        """Resume a verified successful stage without resetting physical state.

        The executor must verify the preceding stage before calling this method;
        terminal failures cannot be resumed through this API.
        """
        if scenario not in ("carry", "place") or self.initialization_failed:
            raise ValueError("invalid continuous stage transition")
        if self._done and not self._last_stage_success:
            raise RuntimeError("cannot resume a failed contract")
        self.control.rearm()
        self.scenario = scenario
        self._last_stage_success = False
        self._done = False
        self.settle_time = 0
        if reset_clock:
            self.steps = 0

    def reset(self, *, seed=None, options=None):
        # A bounded target perturbation is reset-only planner input. It cannot
        # inject physics, clear a running fault, or bypass geometric admission.
        goal_delta = None
        if options:
            if (
                set(options) != {"lift_goal_delta"}
                or self.config.robot != "innate_mars"
                or self.config.scenario != "carry"
                or self.config.grasp_lift_probability != 1
            ):
                raise ValueError("frozen fixtures and arbitrary physics injection are not training reset options")
            goal_delta = np.asarray(options["lift_goal_delta"], dtype=float).copy()
            if goal_delta.shape != (5,) or not np.isfinite(goal_delta).all() or np.any(abs(goal_delta) > 0.03):
                raise ValueError("lift goal delta requires five finite joint offsets within 0.03 rad")
        super().reset(seed=seed)
        if self.grasp_lift_guard is not None:
            self.grasp_lift_guard.close()
            self.grasp_lift_guard = None
        self.grasp_lift_admission = None
        self.grasp_lift_reset_attempts = []
        self._reference_fk = None
        self.arm_trajectory = None
        self.tamp_reference_case = None
        self.scenario = (
            str(self.np_random.choice(SCENARIOS, p=self.config.scenario_weights or MIXTURE))
            if self.config.scenario == "mixed"
            else self.config.scenario
        )
        self.reach_hold_trial = bool(
            self.scenario == "reach"
            and self.config.reach_hold_probability > 0
            and self.np_random.random() < self.config.reach_hold_probability
        )
        self.reach_local_trial = bool(
            self.scenario == "reach"
            and not self.reach_hold_trial
            and self.config.reach_local_probability > 0
            and self.np_random.random() < self.config.reach_local_probability / (1 - self.config.reach_hold_probability)
        )
        self.reach_local_mode = None
        self.reach_local_translation = None
        self.fixed_base_trial = bool(
            self.scenario != "reach"
            and self.config.fixed_base_probability > 0
            and self.np_random.random() < self.config.fixed_base_probability
        )
        self.carry_target_mode, self._carry_push_selected = sample_carry_events(
            self.config, self.np_random, self.scenario
        )
        from emet.rl.manipulation.retention_training import sample_carry_subtype

        carry_subtype = sample_carry_subtype(
            self.scenario,
            self.np_random,
            lift_probability=self.config.grasp_lift_probability,
            hold_probability=self.config.carry_hold_probability,
        )
        self.grasp_lift_trial = carry_subtype == "grasp_lift"
        if self.grasp_lift_trial:
            self.fixed_base_trial = True
        self.carry_hold_trial = carry_subtype == "hold"
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
            self.control.base_xy_tolerance = self.active_base_xy_tolerance
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
        self.reference_time_offset = 0.0
        self.grasp_lift_start_mode = "start"
        self._last_stage_success = False
        self._done = False
        self.retention_reference = (
            relative_pose(self.data, self.control.ee, self.payload)
            if self.scenario in {"carry", "place", "disturbance"} and not self.initialization_failed
            else None
        )
        self.monitor.reset()
        self.max_translation = self.max_rotation = 0.0
        self.current_payload_translation = self.current_payload_rotation = 0.0
        self.peak_hand_position_error = self.peak_hand_orientation_error = 0.0
        self.hand_squared_error = self.hand_tracking_time = 0.0
        self.retention_lost = False
        self.supported_time = self.settle_time = self.bilateral_time = 0.0
        self.first_settle_time = None
        self.hold_base_anchor = None
        self.peak_hold_drift = 0.0
        self.last_actor_action = np.zeros(self.profile.action_size)
        self.last_command_acceleration = np.zeros(self.profile.action_size - 1)
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
        self._fixed_base_origin_position = self.data.xpos[self.control.base].copy()
        self._fixed_base_origin_rotation = self.data.xmat[self.control.base].reshape(3, 3).copy()
        self.fixed_base_distance_m = self.fixed_base_rotation_rad = 0.0
        self.fixed_base_contract_violation = False
        if self.control.pose_only_targets:
            from emet.rl.manipulation.tracking import TaskFrameFK

            self._reference_fk = TaskFrameFK(self.model, self.data, self.control)
        self.retargets = 0
        self.retarget_step = None
        self.retarget_goal = None
        if (
            self.scenario == "reach"
            and not self.reach_hold_trial
            and not self.reach_local_trial
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
        if (
            self.scenario == "reach"
            and not self.reach_hold_trial
            and (self.config.certify_reach_paths or self.config.dynamic_references)
        ):
            self._prepare_reach_trajectory()
        if (
            self.scenario == "carry"
            and self.config.tamp_carry_trajectories
            and not self.initialization_failed
            and not self.carry_hold_trial
        ):
            from emet.rl.manipulation.carry_training import prepare

            prepare(self)
        if self.grasp_lift_trial and not self.initialization_failed:
            from emet.rl.manipulation.grasp_lift import prepare, prepare_phase_reset

            prepare(self, goal_delta=goal_delta)
            if not self.initialization_failed and self.config.grasp_lift_phase_weights is not None:
                prepare_phase_reset(self)
        self._previous_cost = self._tracking_cost()
        # Closed-empty examples retain the ordinary reachable tracking task.
        # The object was already placed far away by the reach reset. No scored
        # state is teleported and no fabricated success/retention is introduced.
        self.empty_grasp_trial = bool(
            self.scenario == "reach"
            and not ((self.reach_hold_trial or self.reach_local_trial) and self.config.hold_pregrasp_fixture)
            and self.config.empty_grasp_probability > 0
            and self.np_random.random() < self.config.empty_grasp_probability
        )
        if self.empty_grasp_trial and not self.initialization_failed:
            self.control.grip_reference = self.profile.gripper_closed
            self.control.retain_object = True
        from emet.rl.manipulation.retention_training import HeldObjectLabel

        self.held_object_label = HeldObjectLabel()
        obs = self.control.observe()
        from emet.rl.manipulation.observation_layout import proprioceptive_layout
        from emet.rl.manipulation.risk import trajectory_features

        self.risk_features = trajectory_features(
            obs["history"][:, : proprioceptive_layout(self.profile)[1]],
            self.control.reference[None],
            self.config.horizon,
            profile=self.profile,
        ).tolist()
        return obs, self._info()

    def _prepare_reach_trajectory(self):
        from emet.rl.manipulation.dynamic_trajectory import ArmTrajectory, admit, build, segment

        q0 = self.control.positions()
        for attempt in range(20):
            if attempt and self.reach_local_trial:
                try:
                    self._make_reference()
                except ResetMotionUnreachable:
                    self.trajectory_rejections.append({"accepted": False, "reason": "local_sampling_exhausted"})
                    continue
            elif attempt:
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
        elapsed = float(self.data.time - self.start_time + self.reference_time_offset)
        if self.arm_trajectory.replan_time is not None and not self.retargets:
            arm = self.arm_trajectory.segments[0].sample(max(self.arm_trajectory.segments[0].start, elapsed))[0]
        else:
            arm = self.arm_trajectory.sample(elapsed)[0]
        self.control.set_reference(arm, phase="pregrasp" if self.reach_local_trial else self.scenario)
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
        if self.reach_hold_trial:
            phase = "pregrasp" if self.config.hold_pregrasp_fixture else "reach"
            self.control.set_reference(self.control.positions(), phase=phase)
            return
        if self.reach_local_trial:
            if (
                self.config.reach_local_translation_probability
                and self.np_random.random() < self.config.reach_local_translation_probability
            ):
                from emet.rl.manipulation.translation_curriculum import sample_translation_reference

                self.reach_local_mode = "translation"
                self.reach_local_translation = None
                try:
                    q, record = sample_translation_reference(self.control, self.np_random)
                except ValueError as error:
                    raise ResetMotionUnreachable(str(error)) from error
                self.reach_local_mode = "translation"
                self.reach_local_translation = record
                self.control.set_reference(q, phase="pregrasp")
                return
            self.reach_local_translation = None
            # FK witnesses near the actual public-tool pregrasp distribution.
            # Half isolate reachable wrist rotations; the remainder include
            # shoulder/elbow translation. They are certified below before use.
            q = self.control.positions()
            if self.np_random.random() < 0.5:
                self.reach_local_mode = "orientation"
                q[-1] += self.np_random.choice([-1, 1]) * self.np_random.uniform(0.08, 0.20)
            else:
                self.reach_local_mode = "combined"
                q += self.np_random.uniform(-1, 1, self.profile.arm_size) * [0.04, 0.08, 0.10, 0.08, 0.15]
            q = np.clip(q, self.control.limits[:, 0] + 0.02, self.control.limits[:, 1] - 0.02)
            self.control.set_reference(q, phase="pregrasp")
            return
        if self.scenario == "pregrasp":
            return
        if self.carry_hold_trial:
            self.control.set_reference(self.control.positions(), phase="carry")
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
        if self.carry_hold_trial or self.reach_hold_trial or self.reach_local_trial or self.fixed_base_trial:
            self.target_mode = 3 if profile.wheel_bodies else 2
            hand = (
                control.ee_pose()
                if self.carry_hold_trial or self.reach_hold_trial
                else self._fk_hand_pose(base, control.reference)
            )
            control.set_pose_targets(base, np.full(3, bool(profile.wheel_bodies)), hand, True)
            return
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

    @property
    def active_base_xy_tolerance(self):
        return (
            self.config.fixed_base_xy_tolerance
            if self.reach_hold_trial or self.reach_local_trial or self.fixed_base_trial
            else self.config.base_xy_tolerance
        )

    @property
    def fixed_base_contract_active(self):
        return self.config.fixed_base_contract and (
            self.reach_hold_trial or self.reach_local_trial or self.fixed_base_trial
        )

    @property
    def active_base_heading_tolerance(self):
        return (
            min(0.01, self.config.base_heading_tolerance)
            if self.fixed_base_contract_active
            else self.config.base_heading_tolerance
        )

    def _tracking_cost(self):
        control = self.control
        cost = 0.0  # Score the active world-frame hand contract, not a redundant IK solution.
        if control.base_mask.any():
            delta = control.base_goal - control.base_pose()
            yaw = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
            cost += float(
                max(0, np.linalg.norm(delta[:2] * control.base_mask[:2]) - self.active_base_xy_tolerance)
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

    def _hand_pose_error(self, pose=None):
        control = self.control
        current = control.ee_pose() if pose is None else pose
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
        base_velocity = (
            self.data.qvel[self.control.base_dof : self.control.base_dof + 6].copy()
            if self.control.base_dof >= 0
            else np.zeros(6)
        )
        return {
            **self.monitor.metrics(),
            "empty_grasp_trial": getattr(self, "empty_grasp_trial", False),
            "carry_hold_trial": getattr(self, "carry_hold_trial", False),
            "held_object_supervision": self.held_object_label.record()
            if hasattr(self, "held_object_label")
            else {"target": 0.0, "valid": False},
            "scenario": self.scenario,
            "grasp_lift_start_mode": self.grasp_lift_start_mode,
            "reference_time_offset_s": self.reference_time_offset,
            "carry_target_mode": self.carry_target_mode,
            "carry_push_selected": self._carry_push_selected,
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
            "current_payload_translation": self.current_payload_translation,
            "current_payload_rotation": self.current_payload_rotation,
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
            "base_linear_norm_m_s": float(np.linalg.norm(base_velocity[:3])),
            "base_angular_norm_rad_s": float(np.linalg.norm(base_velocity[3:])),
            "base_heading_error_rad": float(abs(np.arctan2(np.sin(heading_delta), np.cos(heading_delta))))
            if self.control.base_mask[2]
            else 0.0,
            "max_joint_velocity": float(np.max(np.abs(self.control.velocities()))),
            "settled_seconds": self.settle_time,
            "goal_age_seconds": max(0.0, float(self.data.time - self.control.goal_changed_time)),
            "time_to_first_settle": self.first_settle_time,
            "peak_hold_base_drift_m": self.peak_hold_drift,
            "tracking_precision": self.tracking_precision,
            "base_xy_tolerance": self.active_base_xy_tolerance,
            "base_heading_tolerance": self.active_base_heading_tolerance,
            "reach_hold_trial": self.reach_hold_trial,
            "reach_local_trial": self.reach_local_trial,
            "grasp_lift_trial": self.grasp_lift_trial,
            "grasp_lift_admission": self.grasp_lift_admission,
            "grasp_lift_contacts": self.grasp_lift_guard.contacts if self.grasp_lift_guard else [],
            "reach_local_mode": self.reach_local_mode,
            "reach_local_translation": self.reach_local_translation,
            "fixed_base_contract_active": self.fixed_base_contract_active,
            "fixed_base_contract_violation": self.fixed_base_contract_violation,
            "fixed_base_distance_m": self.fixed_base_distance_m,
            "fixed_base_rotation_rad": self.fixed_base_rotation_rad,
            "fixed_base_trial": self.fixed_base_trial,
            "base_contract_violation": max(
                0.0,
                float(
                    np.linalg.norm(
                        (self.control.base_goal[:2] - self.control.base_pose()[:2]) * self.control.base_mask[:2]
                    )
                )
                - self.active_base_xy_tolerance,
            ),
            "base_goal_error": float(
                np.linalg.norm(self.control.base_goal[:2] - self.control.base_pose()[:2])
                if self.control.base_mask[:2].any()
                else 0.0
            ),
            "time": float(self.data.time - self.start_time),
            "is_success": False,
            "unsafe_contact": self.monitor.peak_force > self.profile.unsafe_force,
            "motion_fault": self.control.motion_fault,
            "turn_guard_evidence": self.control.turn_progress_guard.evidence,
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
        before_hand = self.control.ee_pose().copy() if self.config.pose_progress_weight else None
        before_base = self.control.base_pose().copy() if self.config.precision_tracking_reward else None
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
        self.current_payload_translation = self.current_payload_rotation = 0.0
        if translation is not None and not self._released:
            self.current_payload_translation, self.current_payload_rotation = translation, rotation
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
                    and np.linalg.norm(base_error[:2] * self.control.base_mask[:2]) < self.active_base_xy_tolerance
                    and (not self.control.base_mask[2] or abs(base_yaw_error) < self.active_base_heading_tolerance)
                    and abs(self.last_base_velocity[0]) < 0.005
                    and abs(self.last_base_velocity[1]) < 0.02
                )
            else:
                approach_reached = np.max(abs(self.control.reference - self.control.positions())) < 0.02
            if self.config.whole_body_settling:
                approach_reached &= whole_body_stopped(self.control)
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
            np.linalg.norm(base_delta[:2] * self.control.base_mask[:2]) < self.active_base_xy_tolerance
            and (not self.control.base_mask[2] or abs(base_delta[2]) < self.active_base_heading_tolerance)
        )
        base_motion_allowed = self.control.base_mask.any() or (
            self.config.trajectory_preview and self.control.hand_mask
        )
        base_settled = base_within_pose and (
            not base_motion_allowed
            or (abs(self.last_base_velocity[0]) < 0.005 and abs(self.last_base_velocity[1]) < 0.02)
        )
        if self.config.whole_body_settling:
            base_settled = base_settled and whole_body_stopped(self.control)
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
            open_grip = self.control.gripper_is_open()
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
        if self.grasp_lift_trial:
            settled &= self.data.xpos[self.payload, 2] - self.initial_object_height > 0.04
        if self._scheduled_push:
            settled &= (
                self._disturbed or not np.any(self.disturbance_wrench)
            ) and self.steps >= self.disturbance_start + self.disturbance_steps
        settled &= self._reference_complete()
        self.settle_time = self.settle_time + dt if settled else 0
        unsafe = self.monitor.peak_force > self.profile.unsafe_force
        invalid = invalid_physics(self.data)
        if self.fixed_base_contract_active:
            self.fixed_base_distance_m = float(
                np.linalg.norm(self.data.xpos[self.control.base] - self._fixed_base_origin_position)
            )
            rotation = self.data.xmat[self.control.base].reshape(3, 3).copy()
            self.fixed_base_rotation_rad = float(
                np.arccos(np.clip((np.trace(self._fixed_base_origin_rotation.T @ rotation) - 1) / 2, -1, 1))
            )
            self.fixed_base_contract_violation |= (
                not np.isfinite([self.fixed_base_distance_m, self.fixed_base_rotation_rad]).all()
                or self.fixed_base_distance_m > 0.01
                or self.fixed_base_rotation_rad > 0.02
            )
        grasp_lift_collision = self.grasp_lift_guard is not None and self.grasp_lift_guard.collides(self.data)
        failure = (
            unsafe
            or bool(self.control.motion_fault)
            or self.monitor.wheel_lift
            or self.retention_lost
            or invalid
            or self._motion_infeasible
            or self.fixed_base_contract_violation
            or grasp_lift_collision
        )
        hold_seconds = max(
            self.config.terminal_hold_seconds, 2 if self.scenario == "place" or self.reach_hold_trial else 1
        )
        success = self.settle_time >= hold_seconds and not failure
        continuing = self.config.tracking_reward and (
            self.scenario == "reach"
            or (
                self.scenario == "carry"
                and (self.config.tamp_carry_trajectories or self.carry_hold_trial or self.grasp_lift_trial)
            )
        )
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
            motion_scale=self.motion_penalty_scale,
            force=evidence["force"],
            jerk=np.r_[jerk, base_jerk],
            action=action,
            rotation=self.current_payload_rotation,
            translation=self.current_payload_translation,
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
        body_cost = 0.0
        if self.config.tracking_reward:
            from emet.rl.manipulation.tracking import (
                PRECISION_STAGES,
                base_progress_potential,
                body_motion_cost,
                command_motion_cost,
                differential_base_score,
                endpoint_motion_cost,
                normalized_jerk_cost,
                precision_base_score,
                stationary_tracking_score,
                tracking_score,
                whole_body_quiet_score,
            )

            pose_score = tracking_score(hand_position_error, hand_orientation_error, self.tracking_precision)
            base_sigma = max(self.active_base_xy_tolerance, PRECISION_STAGES[self.tracking_precision][0])
            base_error = max(
                0.0, float(np.linalg.norm(base_delta[:2] * self.control.base_mask[:2])) - self.active_base_xy_tolerance
            )
            heading_sigma = self.active_base_heading_tolerance if self.fixed_base_contract_active else 0.2
            base_score = np.exp(
                -base_error / base_sigma - abs(base_delta[2]) * self.control.base_mask[2] / heading_sigma
            )
            if self.config.precision_tracking_reward:
                base_score = precision_base_score(
                    base_error,
                    base_delta[2] * self.control.base_mask[2],
                    self.active_base_xy_tolerance,
                    self.active_base_heading_tolerance,
                )
            if self.config.differential_base_reward and self.control.base_mask.any():
                delta = self.control._base_tracking_delta()
                bearing = np.arctan2(delta[1], delta[0]) - self.control.base_pose()[2]
                base_score = differential_base_score(
                    base_error, base_delta[2] * self.control.base_mask[2], bearing, base_sigma
                )
            base_weight = 4 if self.fixed_base_contract_active else 1
            hand_density = 4 * pose_score * self.control.hand_mask
            base_density = base_weight * base_score * self.control.base_mask.any()
            # Use the same newly active goal as native hand scoring, including
            # reference advancement. Base-only commands keep their own reward.
            if self.config.hand_gated_base_reward and self.control.hand_mask:
                base_density *= pose_score
            density = hand_density + base_density
            in_goal = base_within_pose and hand_settled
            quiet_density = 0.0
            if self.config.precision_tracking_reward and at_reference_end:
                full_velocity = (
                    self.data.qvel[self.control.base_dof : self.control.base_dof + 6].copy()
                    if self.control.base_dof >= 0
                    else np.zeros(6)
                )
                quiet_density = (
                    2
                    * base_score
                    * (pose_score if self.control.hand_mask else 1)
                    * whole_body_quiet_score(velocity, full_velocity)
                )
            elif at_reference_end and (not self.config.goal_gated_settling or in_goal):
                quiet_density = base_score * stationary_tracking_score(
                    hand_position_error,
                    hand_orientation_error,
                    velocity,
                    base_velocity,
                    hand_active=self.control.hand_mask,
                )
            density += quiet_density
            action_rate = self.motion_penalty_scale * 0.02 * float(np.sum((action - self.last_actor_action) ** 2))
            endpoint_cost = (
                self.motion_penalty_scale
                * self.config.endpoint_motion_weight
                * endpoint_motion_cost(
                    self._reference_end_elapsed(), velocity, base_velocity, hand_active=self.control.hand_mask
                )
            )
            if self.config.goal_gated_settling and not in_goal:
                endpoint_cost = 0.0
            if at_reference_end and self.control.base_dof >= 0:
                body_cost = (
                    self.motion_penalty_scale
                    * self.config.body_motion_weight
                    * body_motion_cost(self.data.qvel[self.control.base_dof : self.control.base_dof + 6].copy())
                )
            reward = (
                dt * (density - action_rate - endpoint_cost - body_cost)
                - physical_cost
                + 10 * success_event
                - 20 * failure
            )
            normalized_jerk = 0.0
            if self.config.precision_tracking_reward:
                normalized_jerk = (
                    dt
                    * self.config.normalized_jerk_weight
                    * normalized_jerk_cost(
                        jerk,
                        base_jerk,
                        np.asarray(self.profile.acceleration) * 2,
                        np.array(
                            [
                                max(self.profile.base_acceleration * 2, 0.01),
                                max(self.profile.base_yaw_acceleration * 2, 0.01),
                            ]
                        ),
                    )
                )
                reward -= normalized_jerk
            command_cost = 0.0
            if self.config.command_smoothness_weight:
                wheels = bool(self.profile.wheel_bodies)
                active_base = self.control.base_mask.any() or self.control.hand_mask
                mask = np.r_[[active_base] * 2 if wheels else [], [self.control.hand_mask] * self.profile.arm_size]
                speeds = np.r_[
                    [self.profile.base_speed, self.profile.base_yaw_speed] if wheels else [], self.profile.speed
                ]
                acceleration_scales = np.r_[
                    [self.profile.base_acceleration, self.profile.base_yaw_acceleration] if wheels else [],
                    self.profile.acceleration,
                ]
                value, self.last_command_acceleration = command_motion_cost(
                    action[:-1],
                    self.last_actor_action[:-1],
                    self.last_command_acceleration,
                    dt,
                    speeds,
                    acceleration_scales,
                    2 * acceleration_scales,
                    mask,
                )
                command_cost = dt * self.config.command_smoothness_weight * self.command_smoothness_fraction * value
                reward -= command_cost
            progress_reward = 0.0
            if before_hand is not None and self.control.hand_mask:
                before_position, before_angle = self._hand_pose_error(before_hand)
                # Both endpoints use the same newly active goal. Goal changes
                # and reference-clock updates cannot manufacture progress.
                progress_reward = self.config.pose_progress_weight * (
                    pose_progress_potential(hand_position_error, hand_orientation_error)
                    - pose_progress_potential(before_position, before_angle)
                )
                reward += progress_reward
            base_progress = 0.0
            if before_base is not None and self.control.base_mask.any():
                previous = self.control.base_goal - before_base
                previous[2] = np.arctan2(np.sin(previous[2]), np.cos(previous[2]))

                def potential(delta):
                    outside = max(
                        0.0,
                        float(np.linalg.norm(delta[:2] * self.control.base_mask[:2])) - self.active_base_xy_tolerance,
                    )
                    return base_progress_potential(outside, delta[2] * self.control.base_mask[2])

                base_progress = self.config.pose_progress_weight * (potential(base_delta) - potential(previous))
                reward += base_progress
        else:
            reward = 10 * (self._previous_cost - cost) - physical_cost + 10 * success - 20 * failure
        self.last_actor_action = action.copy()
        self._previous_cost = cost
        self.steps += 1
        truncated = self.steps * dt >= self.config.horizon and not (failure or terminal_success)
        self._last_stage_success = bool(terminal_success)
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
            else self.control.motion_fault
            if self.control.motion_fault
            else "fixed_base_contract_violation"
            if self.fixed_base_contract_violation
            else "grasp_lift_collision"
            if grasp_lift_collision
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
        if self.empty_grasp_trial:
            self.control.retain_object = True
        self.held_object_label.update(
            bilateral=evidence["bilateral_contact"],
            clearly_empty=(
                np.linalg.norm(self.data.xpos[self.payload] - self.data.xpos[self.control.ee]) > 0.18
                or self.control.gripper_is_open()
            ),
            dt=dt,
            physics_valid=not invalid and not self._physics_invalid,
        )
        info = {
            **self._info(),
            "reward_terms": {
                "hand_tracking_reward": float(dt * hand_density) if self.config.tracking_reward else 0.0,
                "base_tracking_reward": float(dt * base_density) if self.config.tracking_reward else 0.0,
                "terminal_quiet_reward": float(dt * quiet_density) if self.config.tracking_reward else 0.0,
                "motion_penalty_scale": self.motion_penalty_scale,
                "body_motion_cost": float(dt * body_cost),
                "pose_progress_reward": float(progress_reward) if self.config.tracking_reward else 0.0,
                "base_progress_reward": float(base_progress) if self.config.tracking_reward else 0.0,
                "normalized_jerk_cost": float(normalized_jerk) if self.config.tracking_reward else 0.0,
                "command_motion_cost": float(command_cost) if self.config.tracking_reward else 0.0,
                "command_smoothness_fraction": self.command_smoothness_fraction,
                "physical_cost": float(physical_cost),
                "success_bonus": float(10 * success_event if self.config.tracking_reward else 10 * success),
                "failure_penalty": float(20 * failure),
                "tracking_and_progress": float(
                    reward
                    - (10 * success_event if self.config.tracking_reward else 10 * success)
                    + 20 * failure
                    + physical_cost
                ),
            },
            "is_success": bool(success),
            "code": code,
            "invalid_physics": bool(invalid),
            "config": asdict(self.config),
            "risk_features": self.risk_features if self._done else None,
        }
        if self._done and self.grasp_lift_guard is not None:
            self.grasp_lift_guard.close()
        return self.control.observe(), float(reward), bool(failure or terminal_success), bool(truncated), info

    def close(self):
        if self.grasp_lift_guard is not None:
            self.grasp_lift_guard.close()
        super().close()
