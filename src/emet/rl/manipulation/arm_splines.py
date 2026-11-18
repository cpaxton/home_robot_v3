"""Seeded random arm splines; FK witnesses never become actor joint commands.

Mars starts with a stationary, fully masked base target, not a welded base.
Every reset/path admission is recorded; no audit identity enters training.
"""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.interpolate import CubicSpline

from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.coupled_tracking import admit_reference, spline_bounds
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.jerk_reference import compile_joint_path, polynomial_peaks
from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory
from emet.rl.manipulation.whole_body_observation import (
    PRIVILEGED_WIDTH,
    actor_observation,
    privileged_state,
    reference_velocity,
)
from emet.rl.manipulation.whole_body_reward import RewardMeter

ANCHORS = {
    "ready": [0, -0.4, 0.2, 0.2, 0],
    "raised": [0, -0.6, -0.2, 0.8, 0],
    "left_reach": [-1.2, -0.6, 0.2, 0.4, 0],
    "folded_right": [1.35, -1.3, 0.6, 1.05, 0],
}
MIXTURE = {"hand": 0.10, "base": 0.10, "goal_changes": 0.10, "arm_splines": 0.20, "tamp": 0.50}
NAMESPACES = {"train": 460_000_000, "audit": 470_000_000}


@dataclass(frozen=True)
class SplineLevel:
    amplitude_rad: float
    speed_fraction: float
    posture_jitter_rad: float
    knot_count: int = 4
    initial_hold_s: float = 1.0
    maximum_duration_s: float = 28.0

    def __post_init__(self):
        values = [
            self.amplitude_rad,
            self.speed_fraction,
            self.posture_jitter_rad,
            self.initial_hold_s,
            self.maximum_duration_s,
        ]
        if any(isinstance(v, (bool, np.bool_)) for v in values):
            raise ValueError("numeric spline settings cannot be booleans")
        if not np.isfinite(values).all() or not 0 < self.amplitude_rad <= 0.7 or not 0 < self.speed_fraction <= 1:
            raise ValueError("finite bounded spline amplitude and speed required")
        if (
            not 0 <= self.posture_jitter_rad <= 0.2
            or not 0 <= self.initial_hold_s <= 2
            or not 3 <= self.maximum_duration_s <= 30
        ):
            raise ValueError("bounded posture jitter, hold and duration required")
        if type(self.knot_count) is not int or not 3 <= self.knot_count <= 6:
            raise ValueError("three to six spline knots required")


LEVELS = (
    SplineLevel(0.12, 0.25, 0.04),
    SplineLevel(0.30, 0.40, 0.08),
    SplineLevel(0.55, 0.60, 0.12),
)


def manifest(train_count=32, audit_count=16):
    if any(type(n) is not int or not 1 <= n <= 1000 for n in (train_count, audit_count)):
        raise ValueError("bounded positive population counts required")
    return {
        "format": "mars-arm-splines-v1",
        "robot": "innate_mars",
        "training_mix": MIXTURE.copy(),
        "reset_anchors_rad": {k: list(v) for k, v in ANCHORS.items()},
        "levels": [asdict(x) for x in LEVELS],
        "curriculum": {
            "initial_level": 0,
            "advance_minimum_success": 0.85,
            "advance_maximum_safety_failures": 0,
            "advance_scope": "development probes only; audit never selects a level",
        },
        "cases": [
            {
                "id": f"mars:arm-splines-v1:{split}:l{level}:{i}",
                "seed": start + level * 10000 + i,
                "split": split,
                "level": level,
                "posture": list(ANCHORS)[i % len(ANCHORS)],
            }
            for split, start, count in [
                ("train", NAMESPACES["train"], train_count),
                ("audit", NAMESPACES["audit"], audit_count),
            ]
            for level in range(len(LEVELS))
            for i in range(count)
        ],
    }


def load_manifest(path):
    value = json.loads(Path(path).read_text())
    if (
        value.get("format") != "mars-arm-splines-v1"
        or value.get("robot") != "innate_mars"
        or value.get("training_mix") != MIXTURE
        or value.get("reset_anchors_rad") != ANCHORS
    ):
        raise ValueError("versioned Mars arm-spline mixture required")
    levels = tuple(SplineLevel(**v) for v in value["levels"])
    if not levels or len(levels) > 3:
        raise ValueError("one to three declared curriculum levels required")
    ids, seeds, populations = set(), set(), set()
    for case in value["cases"]:
        validate_case(case, levels)
        split, level, seed = case["split"], case["level"], case["seed"]
        if case["id"] in ids or seed in seeds:
            raise ValueError("disjoint spline identities and seeds required")
        ids.add(case["id"])
        seeds.add(seed)
        populations.add((split, level))
    if populations != {(s, l) for s in NAMESPACES for l in range(len(levels))}:
        raise ValueError("train and audit coverage required for every level")
    return value


def validate_case(case, levels):
    if not isinstance(case, dict) or set(case) != {"id", "seed", "split", "level", "posture"}:
        raise ValueError("complete seeded spline case required")
    split, level, seed = case["split"], case["level"], case["seed"]
    if (
        not isinstance(split, str)
        or split not in NAMESPACES
        or type(level) is not int
        or not 0 <= level < len(levels)
        or type(seed) is not int
        or not NAMESPACES[split] + level * 10000 <= seed < NAMESPACES[split] + (level + 1) * 10000
        or not isinstance(case["posture"], str)
        or case["posture"] not in ANCHORS
        or not isinstance(case["id"], str)
        or not case["id"].startswith(f"mars:arm-splines-v1:{split}:l{level}:")
    ):
        raise ValueError("disjoint split/level identity and posture required")


def spline_joint_extrema(spline):
    """Exact coordinatewise extrema, including between-knot overshoot."""
    values = [spline(spline.x)]
    for interval, width in enumerate(np.diff(spline.x)):
        for joint in range(spline.c.shape[-1]):
            a, b, c, _ = spline.c[:, interval, joint]
            for root in np.roots([3 * a, 2 * b, c]):
                if abs(root.imag) < 1e-9 and 0 < root.real < width:
                    values.append(spline(spline.x[interval] + root.real)[None])
    values = np.concatenate(values)
    return values.min(axis=0), values.max(axis=0)


class ArmSplineReference(TaskPoseTrajectory):
    """Continuous cubic geometry with a globally retimed rest-to-rest phase.

    Interior knots retain their spline tangents rather than stopping every joint
    at every waypoint. Position/velocity/acceleration are continuous; analytical
    chain-rule bounds constrain velocity, acceleration and jerk throughout.
    """

    def __init__(self, control, points, level):
        if control.profile.robot != "innate_mars":
            raise ValueError("arm-spline posture sampling currently supports Mars")
        points = np.asarray(points, float)
        if points.shape != (level.knot_count, control.profile.arm_size) or not np.isfinite(points).all():
            raise ValueError("complete finite arm spline knots required")
        if not np.allclose(points[0], control.positions(), atol=1e-10, rtol=0):
            raise ValueError("spline must start at measured reset posture")
        self.fk = TaskFrameFK(control.model, control.data, control)
        self.start_base = control.base_pose().copy()
        self.fixed_hand = None
        self.initial_hold_s = level.initial_hold_s
        phases = np.linspace(0, 1, len(points))
        self.joints = CubicSpline(phases, points, axis=0, bc_type="natural")
        low, high = spline_joint_extrema(self.joints)
        if np.any(low < control.limits[:, 0] + 0.01) or np.any(high > control.limits[:, 1] - 0.01):
            raise ValueError("between-knot joint margin")
        q1, q2, q3 = spline_bounds(self.joints)
        p = control.profile
        speed = np.asarray(p.speed) * level.speed_fraction
        acceleration = np.asarray(p.acceleration) * 0.5
        jerk = 4 * acceleration
        scale = 1.0
        for _ in range(20):
            path, _ = compile_joint_path([[0.0], [1.0]], [1 / scale], [1 / scale**2], [1 / scale**3])
            v, a, j = (float(x[0]) for x in polynomial_peaks(path.segments[0]))
            bounds = np.array([q1 * v, q2 * v * v + q1 * a, q3 * v**3 + 3 * q2 * v * a + q1 * j])
            ratios = np.max(bounds / np.array([speed, acceleration, jerk]), axis=1)
            factor = max(ratios[0], np.sqrt(ratios[1]), np.cbrt(ratios[2]))
            if factor <= 1 + 1e-9:
                break
            scale *= factor * 1.01
        else:
            raise ValueError("spline timing did not converge")
        self.path = path
        duration = self.initial_hold_s + path.end
        if duration > level.maximum_duration_s:
            raise ValueError("spline exceeds declared duration")
        self.certificate = {
            "level": asdict(level),
            "speed": speed.tolist(),
            "acceleration": acceleration.tolist(),
            "jerk": jerk.tolist(),
            "derivative_bounds": bounds.tolist(),
            "joint_minimum": low.tolist(),
            "joint_maximum": high.tolist(),
            "joint_waypoints": points.tolist(),
            "base_pose_stationary": True,
            "base_physically_welded": False,
            "initial_hold_s": self.initial_hold_s,
            "dynamic_feasibility_verified": False,
        }
        times = np.r_[np.arange(0, duration, p.dt), duration]
        hands = [self.fk(*self.coordinates(t)) for t in times]
        base_origin = control.data.xpos[control.base].copy()
        if np.max(np.linalg.norm(np.asarray(hands)[:, :3] - base_origin, axis=1)) > 0.7:
            raise ValueError("hand path outside nearby workspace")
        super().__init__(
            times,
            np.tile(self.start_base, (len(times), 1)),
            hands,
            np.full(len(times), control.grip_reference),
            base_mask=[True] * 3,
        )

    def coordinates(self, time):
        phase = float(self.path.sample(float(np.clip(time - self.initial_hold_s, 0, self.path.end)))[0][0])
        return self.start_base.copy(), self.joints(phase)

    def sample(self, time):
        base, joints = self.coordinates(time)
        return base, self.fk(base, joints), float(self.gripper[0])

    def record(self):
        return {
            **super().record(),
            "format": "arm-spline-fk-v1",
            "certificate": self.certificate,
            "phase_path": self.path.record(),
            "joint_phase_knots": self.joints.x.tolist(),
            "joint_phase_coefficients": self.joints.c.tolist(),
            "fk_anchor_qpos": self.fk.qpos.tolist(),
        }


class ArmSplinePlant(ManipulationEnv):
    def __init__(self):
        self.reset_posture = np.array(ANCHORS["ready"], float)
        super().__init__(
            ManipulationConfig(
                robot="innate_mars",
                scenario="reach",
                horizon=35,
                randomization_scale=0,
                action_mode="direct_velocity",
                trajectory_preview=True,
                tracking_reward=True,
                tracking_precision=4,
                retention_context=True,
                whole_body_settling=True,
                terminal_hold_seconds=2,
                hand_position_tolerance=0.003,
                hand_orientation_tolerance=0.02,
                base_xy_tolerance=0.05,
                base_heading_tolerance=float(np.deg2rad(5)),
                wheel_feedback_current_velocity=True,
                wheel_feedback_gain=0.5,
                physical_wheel_slew=True,
                wheel_feedback_base_only_gains=(0.0, 0.5),
                retarget_probability=0,
                reach_push_probability=0,
                initialization_attempts=1,
            ),
            mesh_storage="binary",
        )

    def _initial_arm_posture(self, sampled):
        return self.reset_posture.copy()

    def _make_reference(self):
        self.control.set_reference(self.control.positions(), phase="reach")


class ArmSplineTrackingEnv(gym.Wrapper):
    def __init__(self, cases, *, levels=LEVELS, reward_config=None, maximum_attempts=64):
        if not cases or type(maximum_attempts) is not int or not 1 <= maximum_attempts <= 64:
            raise ValueError("seeded cases and bounded sampling attempts required")
        self.cases = [dict(c) for c in cases]
        self.levels, self.maximum_attempts = tuple(levels), maximum_attempts
        if not self.levels or any(not isinstance(level, SplineLevel) for level in self.levels):
            raise ValueError("declared spline levels required")
        for case in self.cases:
            validate_case(case, self.levels)
        if len({c["id"] for c in self.cases}) != len(self.cases) or len({c["seed"] for c in self.cases}) != len(
            self.cases
        ):
            raise ValueError("unique spline identities and seeds required")
        self.cursor = 0
        super().__init__(ArmSplinePlant())
        self.action_space = self.env.action_space
        _, width = history_layout(self.env.profile, retention_context=True)
        self.observation_space = spaces.Dict(
            {
                "history": spaces.Box(-np.inf, np.inf, (20, width), np.float32),
                "preview": spaces.Box(-np.inf, np.inf, (7, 18), np.float32),
                "motion": spaces.Box(-np.inf, np.inf, (9,), np.float32),
                "critic": spaces.Box(-np.inf, np.inf, (PRIVILEGED_WIDTH,), np.float32),
            }
        )
        self.reward_config = reward_config
        self.ready = False

    def observe(self, raw):
        e = self.unwrapped
        return actor_observation(e.control, raw) | {
            "critic": privileged_state(e, e.steps * e.profile.dt, e.config.horizon)
        }

    def reset(self, *, seed=None, options=None):
        if options is not None and set(options) != {"case_id"}:
            raise ValueError("only a declared case identity may be selected")
        case = (
            next((c for c in self.cases if c["id"] == options["case_id"]), None)
            if options
            else self.cases[self.cursor % len(self.cases)]
        )
        if case is None:
            raise ValueError("unknown arm-spline case")
        self.cursor += 1
        self.case = case
        self.ready = False
        self.rejections = []
        e = self.unwrapped
        level = self.levels[case["level"]]
        rng = np.random.default_rng(case["seed"])
        anchor = np.array(ANCHORS[case["posture"]], float)
        for attempt in range(self.maximum_attempts):
            # Sampling bounds choose a synthetic reset. No measured robot state
            # is clamped and no runtime reference can write generalized state.
            low = np.maximum(anchor - level.posture_jitter_rad, e.control.limits[:, 0] + 0.03)
            high = np.minimum(anchor + level.posture_jitter_rad, e.control.limits[:, 1] - 0.03)
            e.reset_posture = rng.uniform(low, high)
            raw, info = e.reset(seed=case["seed"])
            if info.get("initialization_failed"):
                self.rejections.append({"attempt": attempt, "reason": "initialization_failed"})
                continue
            initial = e.control.positions().copy()
            low = np.maximum(initial - level.amplitude_rad, e.control.limits[:, 0] + 0.03)
            high = np.minimum(initial + level.amplitude_rad, e.control.limits[:, 1] - 0.03)
            points = np.vstack([initial, rng.uniform(low, high, size=(level.knot_count - 1, e.profile.arm_size))])
            if rng.random() < 0.5:
                points[-1] = initial
            try:
                reference = ArmSplineReference(e.control, points, level)
                admission = admit_reference(e, reference)
                if not admission["accepted"]:
                    raise ValueError(str(admission["reason"]))
            except ValueError as exc:
                self.rejections.append({"attempt": attempt, "reason": str(exc)})
                continue
            e.reach_hold_trial = False
            e.fixed_base_trial = False
            e.control.pose_only_targets = True
            # Two seconds of required quiet hold plus one second of settling
            # allowance, rather than mostly collecting a 35-second static hold.
            e.config = replace(e.config, horizon=float(np.ceil((reference.end + 3.0) / e.profile.dt) * e.profile.dt))
            e.set_task_trajectory(reference)
            e.control.history = None
            self.reference = reference
            self.admission = admission
            self.meter = RewardMeter(e, base_heading_region=True, config=self.reward_config)
            self.peak_path_error = np.zeros(2)
            self.limit_push_ticks = 0
            self.ready = True
            return self.observe(e.control.observe()), {
                **info,
                "case_id": case["id"],
                "spline_admission": admission,
                "spline_rejections": self.rejections.copy(),
                "reference_sha256": digest(reference.record()),
                "spline_certificate": reference.certificate,
                "reward_contract": self.meter.record(),
            }
        e.control.stop()
        raise ValueError("arm_spline_sampling_exhausted:" + json.dumps(self.rejections))

    def step(self, action):
        if not self.ready or self.unwrapped.control.stopped:
            raise RuntimeError("reset an active arm-spline episode")
        e = self.unwrapped
        joints = e.control.positions()
        requested = np.asarray(action)[2:-1]
        near_upper = joints >= e.control.limits[:, 1] - 0.02
        near_lower = joints <= e.control.limits[:, 0] + 0.02
        self.limit_push_ticks += int(np.any((near_upper & (requested > 0)) | (near_lower & (requested < 0))))
        motion = reference_velocity(e.control)
        self.meter.before_step()
        raw, _, done, truncated, info = e.step(action)
        rewards = self.meter.tick(action, motion)
        phase = e.control.trajectory_elapsed()
        if self.reference.initial_hold_s <= phase <= self.reference.end + 1e-8:
            self.peak_path_error = np.maximum(
                self.peak_path_error, [info["ee_position_error"], info["ee_orientation_error"]]
            )
        success = bool(
            info.get("is_success") and phase >= self.reference.end and np.all(self.peak_path_error <= [0.003, 0.02])
        )
        if done or truncated:
            rewards[0] += self.meter.terminal_bonus(success)
            self.ready = False
        return (
            self.observe(raw),
            float(rewards.sum()),
            bool(done or truncated),
            False,
            {
                **info,
                "native_is_success": bool(info.get("is_success")),
                "is_success": success,
                "code": "spline_path_tracking_failed"
                if (done or truncated) and info.get("is_success") and not success
                else info.get("code"),
                "case_id": self.case["id"],
                "arm_spline_case": self.case.copy(),
                "spline_rejections": self.rejections.copy(),
                "native_terminated": bool(done),
                "native_truncated": bool(truncated),
                "episode_contract": "finite_tracking_deadline_v1",
                "reward_groups": rewards,
                "reward_terms": self.meter.last_terms.copy(),
                "elapsed_control_ticks": 1,
                "primitive_success": success,
                "full_pick_place_success": False,
                "spline_peak_path_error": self.peak_path_error.tolist(),
                "joint_limit_push_ticks": self.limit_push_ticks,
            },
        )


def training_mixture(environments, splines):
    """Preserve base/goal/TAMP coverage; add 20% overall random-arm ticks."""
    from emet.rl.manipulation.whole_body_env import ControlStepMixture

    if set(environments) != {"hand", "base", "goal_changes", "tamp"}:
        raise ValueError("all existing control families must be retained")
    if any(case["split"] != "train" for case in splines.cases):
        raise ValueError("audit spline cases must never enter training")
    members = {**environments, "arm_splines": splines}
    return ControlStepMixture({name: members[name] for name in MIXTURE}, list(MIXTURE.values()))


def make_training_environment(manifest_path, *, level=0, reward_config=None):
    """Use the existing general/TAMP sources with a dedicated random-arm share.

    The level is explicit. Its teacher preflight and development evaluation
    belong to the training launcher; held-out cases never enter this factory.
    """
    from dataclasses import replace

    from emet.rl.manipulation.coupled_tracking import COARSE_SUITE, CoupledTrackingEnv
    from emet.rl.manipulation.goal_change_tracking import GoalChangeTrackingEnv, load_curriculum
    from emet.rl.manipulation.tamp_primitive_env import TampPrimitiveEnv
    from emet.rl.manipulation.whole_body_training import load_development

    root = Path(__file__).resolve().parents[4]
    spec = load_manifest(manifest_path)
    if type(level) is not int or not 0 <= level < len(spec["levels"]):
        raise ValueError("declared curriculum level required")
    cases = json.loads((root / "paper/data/whole_body_control/long_finetune_v1/cases.json").read_text())
    general = [c for c in cases if c["split"] == "train"]
    goals = load_curriculum(root / "configs/benchmarks/mars_goal_changes_v1.json")
    goal_cases = [c for c in goals["robots"]["innate_mars"] if c["split"] == "train"]
    development, _ = load_development()
    environments = {}
    try:
        for name, base_only in (("hand", False), ("base", True)):
            env = CoupledTrackingEnv(
                "innate_mars",
                [c for c in general if (c["mode"] == "base_only") == base_only],
                suite_path=COARSE_SUITE,
                reward_config=reward_config,
            )
            environments[name] = env
            env.unwrapped.config = replace(env.unwrapped.config, wheel_feedback_base_only_gains=(0.0, 0.5))
        environments["goal_changes"] = GoalChangeTrackingEnv(
            "innate_mars",
            goal_cases,
            changes={c["id"]: goals["goal_changes"][c["id"]] for c in goal_cases},
            suite_path=COARSE_SUITE,
            reward_config=reward_config,
        )
        goal_env = environments["goal_changes"].unwrapped
        goal_env.config = replace(goal_env.config, wheel_feedback_base_only_gains=(0.0, 0.5))
        environments["tamp"] = TampPrimitiveEnv(
            development["train_cases"],
            primitive="grasp_pose",
            planning_scope="unloaded_prefix",
            wheel_feedback_current_velocity=True,
            wheel_feedback_base_only_gains=(0.0, 0.5),
            reward_config=reward_config,
        )
        splines = ArmSplineTrackingEnv(
            [c for c in spec["cases"] if c["split"] == "train" and c["level"] == level],
            levels=tuple(SplineLevel(**v) for v in spec["levels"]),
            reward_config=reward_config,
        )
        environments["arm_splines"] = splines
        return training_mixture({k: v for k, v in environments.items() if k != "arm_splines"}, splines)
    except Exception:
        for env in environments.values():
            env.close()
        raise
