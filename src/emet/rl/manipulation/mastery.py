"""Small, deterministic Mars tracking problems; no task decisions in the actor.

Progression: base -> base+arm -> +coupled -> +loaded, retaining earlier cases.
All commands remain ordinary masked task poses with measured history/preview.
This is a capability/overfit benchmark, not public TAMP or hardware qualification.
"""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.dynamic_trajectory import ArmTrajectory, Segment
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.jerk_reference import compile_joint_path, polynomial_peaks
from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory

ROOT = Path(__file__).resolve().parents[4]
SUITE = ROOT / "configs/benchmarks/mars_tracking_mastery_v1.json"
STAGES = ("base", "arm", "coupled", "loaded")


def load_suite(path=SUITE):
    suite = json.loads(Path(path).read_text())
    if (
        suite.get("format") not in {"mars-tracking-mastery-v1", "mars-tracking-mastery-v2"}
        or suite.get("robot") != "innate_mars"
    ):
        raise ValueError("unsupported mastery suite")
    cfg = ManipulationConfig(**suite["config"])
    if suite["format"] == "mars-tracking-mastery-v2":
        frozen = json.loads(SUITE.read_text())
        if any(suite[key] != frozen[key] for key in ("cases", "reference", "gate", "stages", "robot")):
            raise ValueError("v2 must preserve frozen v1 cases, references and acceptance gates")
        if not cfg.precision_tracking_reward or cfg.terminal_hold_seconds != 2 or cfg.normalized_jerk_weight <= 0:
            raise ValueError("v2 requires precision tracking, normalized jerk and aligned quiet duration")
    if "training_population" in suite:
        if suite["format"] != "mars-tracking-mastery-v2" or suite["training_population"] not in (
            {"format": "base-primitive-posture-cross-product-v1"},
            {"format": "base-posture-yaw-scales-v1", "yaw_magnitudes": [0.02, 0.04, 0.06]},
        ):
            raise ValueError("unsupported training population")
    if (
        cfg.action_mode != "direct_velocity"
        or not cfg.feedback_policy
        or not cfg.physical_wheel_slew
        or cfg.randomization_scale != 0
        or cfg.hand_position_tolerance != 0.003
        or cfg.hand_orientation_tolerance != 0.02
        or cfg.base_xy_tolerance != 0.005
        or cfg.base_heading_tolerance != 0.01
        or not cfg.whole_body_settling
        or suite["gate"]["terminal_hold_s"] != 2
        or cfg.terminal_hold_seconds != suite["gate"]["terminal_hold_s"]
        or suite["stages"] != list(STAGES)
    ):
        raise ValueError("mastery requires the frozen tracking, settling and servo contract")
    if any(not np.isfinite(v) or v <= 0 for v in [*suite["reference"].values(), *suite["gate"].values()]):
        raise ValueError("reference and gate scales must be positive and finite")
    ids, seeds, counts = set(), {"train": set(), "validation": set()}, Counter()
    for case in suite["cases"]:
        if (
            case["stage"] not in STAGES
            or case["split"] not in seeds
            or case["id"] in ids
            or type(case["reset_seed"]) is not int
            or not 0 <= case["reset_seed"] < 2**31
        ):
            raise ValueError("invalid or duplicate mastery case")
        values = np.asarray([case["travel_m"], case["turn_rad"], *case["arm_delta"]], dtype=float)
        if (
            values.shape != (7,)
            or not np.isfinite(values).all()
            or np.any(abs(values) > [0.05, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2])
        ):
            raise ValueError("mastery case exceeds nearby motion bounds")
        if case["stage"] in {"base", "loaded"} and np.any(values[2:]):
            raise ValueError("base/loaded cases do not take an arm delta")
        if case["stage"] in {"arm", "loaded"} and np.any(values[:2]):
            raise ValueError("fixed-base cases cannot command base travel")
        ids.add(case["id"])
        seeds[case["split"]].add(case["reset_seed"])
        counts[case["split"], case["stage"]] += 1
    if seeds["train"] & seeds["validation"] or any(
        counts[split, stage] != n for split in seeds for stage, n in zip(STAGES, (5, 4, 2, 2), strict=True)
    ):
        raise ValueError("expected disjoint 13-case train/validation partitions")
    if not np.isfinite(suite["reference"]["loaded_time_scale"]) or suite["reference"]["loaded_time_scale"] < 1:
        raise ValueError("loaded references may only slow down")
    scales = suite.get("arm_training_scales")
    if scales is not None and (
        suite["format"] != "mars-tracking-mastery-v2"
        or not isinstance(scales, list)
        or not scales
        or any(type(x) not in (int, float) or not np.isfinite(x) or not 0 < x <= 1 for x in scales)
        or scales != sorted(set(scales))
        or scales[-1] != 1
    ):
        raise ValueError("arm training scales require ordered unique positive amplitudes through 1")
    weights = suite.get("training_stage_weights")
    if weights is not None and (
        not isinstance(weights, dict)
        or not weights
        or any(
            k not in STAGES or type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for k, v in weights.items()
        )
        or not np.isclose(sum(weights.values()), 1, rtol=0, atol=1e-8)
    ):
        raise ValueError("training stage weights must be positive probabilities over known stages")
    return suite


def select_cases(suite, stages, split, *, population="canonical"):
    """Keep canonical evaluation fixed; explicitly expand training/admission only.

    Cross each existing training base primitive with each existing training reset
    posture. The five original pairs remain under their original IDs; the twenty
    added pairs get distinct IDs. Validation cases and non-base stages never
    enter this expansion. No case metadata is exposed to the policy.
    """
    if (
        not stages
        or len(set(stages)) != len(stages)
        or any(s not in STAGES for s in stages)
        or split not in {"train", "validation", "all"}
        or population not in {"canonical", "training", "admission"}
    ):
        raise ValueError("invalid mastery stage or split")
    if population == "training" and split != "train":
        raise ValueError("training population requires the training split")
    cases = list(suite["cases"])
    if population != "canonical" and "training_population" in suite:
        base = [c for c in cases if c["stage"] == "base" and c["split"] == "train"]
        for primitive in base:
            for posture in base:
                if primitive["id"] != posture["id"]:
                    cases.append(
                        {
                            **primitive,
                            "id": primitive["id"] + "@posture:" + posture["id"].rsplit(":", 1)[-1],
                            "reset_seed": posture["reset_seed"],
                        }
                    )
                for magnitude in suite["training_population"].get("yaw_magnitudes", []):
                    if primitive["turn_rad"] and magnitude != abs(primitive["turn_rad"]):
                        cases.append(
                            {
                                **primitive,
                                "id": primitive["id"]
                                + "@posture:"
                                + posture["id"].rsplit(":", 1)[-1]
                                + f"@yaw:{magnitude}",
                                "reset_seed": posture["reset_seed"],
                                "turn_rad": float(np.copysign(magnitude, primitive["turn_rad"])),
                            }
                        )
    if population != "canonical":
        for case in suite["cases"]:
            if case["stage"] == "arm" and case["split"] == "train":
                for scale in suite.get("arm_training_scales", []):
                    if scale != 1:
                        cases.append(
                            {
                                **case,
                                "id": case["id"] + f"@amplitude:{scale:g}",
                                "arm_delta": (np.asarray(case["arm_delta"]) * scale).tolist(),
                            }
                        )
    return [c for c in cases if c["stage"] in stages and (split == "all" or c["split"] == split)]


class TrainingCaseSampler:
    """Stage probabilities are independent of family size; cycle every case within a stage.

    Without explicit weights preserve the original shuffled full-case cycles.
    This object only selects reset metadata; nothing is added to actor inputs.
    """

    def __init__(self, cases, weights=None, seed=0):
        self.cases, self.weights = cases, weights
        if not cases:
            raise ValueError("nonempty training cases required")
        self.groups = {stage: [i for i, c in enumerate(cases) if c["stage"] == stage] for stage in STAGES}
        self.groups = {stage: rows for stage, rows in self.groups.items() if rows}
        if weights is not None and not set(self.groups).issubset(weights):
            raise ValueError("training stage weights must cover every selected stage")
        self.probabilities = None if weights is None else np.asarray([weights[s] for s in self.groups], dtype=float)
        if self.probabilities is not None:
            if not np.isfinite(self.probabilities).all() or np.any(self.probabilities <= 0):
                raise ValueError("positive finite stage weights required")
            self.probabilities /= self.probabilities.sum()
        self.reset(seed)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.orders = {}

    def next(self):
        stage = "all" if self.weights is None else str(self.rng.choice(list(self.groups), p=self.probabilities))
        if not self.orders.get(stage):
            indices = range(len(self.cases)) if stage == "all" else self.groups[stage]
            self.orders[stage] = list(self.rng.permutation(indices))
        return self.cases[self.orders[stage].pop()]


def slow_path(path, factor):
    if not np.isfinite(factor) or factor < 1 or path.replan_time is not None:
        raise ValueError("positive uniform slowdown of a non-retargeted path required")
    return ArmTrajectory(
        [
            Segment(p.start * factor, p.duration * factor, p.coefficients.copy() / (factor ** np.arange(6))[:, None])
            for p in path.segments
        ]
    )


class MasteryReference(TaskPoseTrajectory):
    """Private continuous FK of an axle-valid base path and arm witness."""

    def __init__(self, env, path, *, loaded=False):
        self.path, self.loaded = path, loaded
        self.fk = TaskFrameFK(env.model, env.data, env.control)
        self.origin = env.control.base_pose().copy()
        self.offset = env.control.axle_offset[:2].copy()
        self.axle = self.origin[:2] + self.yaw_rotation(self.origin[2]) @ self.offset
        self.grip = env.control.grip_reference
        times = np.r_[np.arange(0, path.end, env.profile.dt), path.end]
        samples = [self.sample(float(t)) for t in times]
        super().__init__(
            times,
            [x[0] for x in samples],
            [x[1] for x in samples],
            [x[2] for x in samples],
            base_mask=[True] * 3,
            hand_mask=env.mastery_case["stage"] != "base",
        )

    @staticmethod
    def yaw_rotation(yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s], [s, c]])

    def coordinates(self, time):
        values = self.path.sample(float(time))[0]
        if self.loaded:
            return self.origin.copy(), values
        travel, turn = values[:2]
        base = self.origin.copy()
        base[2] += turn
        base[:2] = self.axle + self.yaw_rotation(base[2]) @ (np.array([travel, 0.0]) - self.offset)
        return base, values[2:]

    def sample(self, time):
        if not np.isfinite(time):
            raise ValueError("finite reference time required")
        base, arm = self.coordinates(time)
        return base, self.fk(base, arm).copy(), self.grip

    def record(self):
        record = super().record()
        record.update(
            format="mastery_continuous_fk_v1",
            sampled_arrays="visualization_only",
            generalized_path=self.path.record(),
            fk_anchor_qpos=self.fk.qpos.tolist(),
            axle_offset=self.offset.tolist(),
            loaded=self.loaded,
            joint_reference_actor_input=False,
            dynamic_feasibility="requires_teacher_and_policy_rollout",
        )
        return record


class MasteryEnv(ManipulationEnv):
    """Reset-only case selection and path generation; full actor action space."""

    def __init__(self, suite=None, *, stages=("base",), split="train", population="canonical"):
        self.suite = load_suite() if suite is None else suite
        self.cases = select_cases(self.suite, stages, split, population=population)
        self.sampler = TrainingCaseSampler(
            self.cases, self.suite.get("training_stage_weights") if population == "training" else None
        )
        self.mastery_case = self.mastery_reference = self.mastery_checker = None
        self.mastery_scratch = None
        self.mastery_admission = None
        self.case_visits = Counter()
        super().__init__(ManipulationConfig(**self.suite["config"]))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.sampler.reset(seed)
        if options:
            if set(options) != {"case_id"} or options["case_id"] not in {c["id"] for c in self.cases}:
                raise ValueError("only a selected mastery case ID is a reset option")
            case = next(c for c in self.cases if c["id"] == options["case_id"])
        else:
            case = self.sampler.next()
        self.mastery_case = case
        self.mastery_reference = self.mastery_checker = None
        self.mastery_scratch = None
        loaded = case["stage"] == "loaded"
        cfg = ManipulationConfig(**self.suite["config"])
        self.config = (
            replace(cfg, scenario="carry", reach_hold_probability=0, grasp_lift_probability=1) if loaded else cfg
        )
        self.case_visits[case["id"]] += 1
        obs, info = super().reset(seed=case["reset_seed"])
        self.path_failed = False
        self.mastery_admission = {"accepted": False, "reason": "initialization_failed", "case_id": case["id"]}
        if info["initialization_failed"]:
            return obs, {**info, "mastery": self.mastery_admission.copy()}
        initial = {k: getattr(self.data, k).copy().tolist() for k in ("qpos", "qvel", "ctrl")}
        initial["time"] = float(self.data.time)
        self.reach_hold_trial = False
        self.fixed_base_trial = case["stage"] in {"arm", "loaded"}
        if loaded:
            path = slow_path(self.arm_trajectory, self.suite["reference"]["loaded_time_scale"])
        else:
            p = self.profile
            q = self.control.positions().copy()
            start = np.r_[0.0, 0.0, q]
            turned = np.r_[0.0, case["turn_rad"], q]
            goal = np.r_[case["travel_m"], case["turn_rad"], q + case["arm_delta"]]
            path, _ = compile_joint_path(
                [start, turned, goal],
                np.r_[p.base_speed, p.base_yaw_speed, p.speed] * 0.35,
                np.r_[p.base_acceleration, p.base_yaw_acceleration, p.acceleration] * 0.35,
                np.r_[p.base_acceleration, p.base_yaw_acceleration, p.acceleration] * 2,
            )
            path = slow_path(path, max(1.0, self.suite["reference"]["minimum_duration_s"] / path.end))
        if path.end + self.suite["gate"]["terminal_hold_s"] > self.config.horizon:
            return self.reject("reference_exceeds_horizon", initial)
        reference = MasteryReference(self, path, loaded=loaded)
        if not loaded:
            from emet.eval.physical_tamp import make_scene_checker
            from emet.rl.manipulation.grasp_setup import support_margin

            self.mastery_checker = make_scene_checker(
                self.model, {"object_body": self.payload_name, "gripper_bodies": self.profile.finger_bodies}
            )
            self.mastery_scratch = mujoco.MjData(self.model)
            for time in np.r_[np.arange(0, path.end, 0.025), path.end]:
                base, arm = reference.coordinates(float(time))
                if np.any(arm < self.control.limits[:, 0] + 0.02) or np.any(arm > self.control.limits[:, 1] - 0.02):
                    return self.reject("joint_margin", initial)
                reference.fk(base, arm)
                if self.mastery_checker.configuration_collides(self.model, reference.fk.data):
                    return self.reject("planned_collision", initial)
                if support_margin(self.model, reference.fk.data, self.profile, None)["margin_m"] < 0.005:
                    return self.reject("planned_support_margin", initial)
        self.mastery_reference = reference
        self.set_task_trajectory(reference)
        self.control.base_yaw_deadband = 0.005
        self.control.history = None  # Reset-only; discard the procedural reset's old command tokens.
        self.update_witness()
        self.control.base_xy_tolerance = self.active_base_xy_tolerance
        self.mastery_admission = {
            "accepted": True,
            "case_id": case["id"],
            "initial_state": initial,
            "initial_state_sha256": digest(initial),
            "reference_sha256": digest(reference.record()),
            "reference": reference.record(),
            "nominal_derivative_peaks": [
                np.max([x[i] for x in [polynomial_peaks(p) for p in path.segments]], axis=0).tolist() for i in range(3)
            ],
            "dynamic_feasibility_verified": False,
        }
        assert initial == {
            **{k: getattr(self.data, k).copy().tolist() for k in ("qpos", "qvel", "ctrl")},
            "time": float(self.data.time),
        }
        return self.control.observe(), {
            **self._info(),
            "initialization_failed": False,
            "mastery": self.mastery_admission.copy(),
        }

    def reject(self, reason, initial):
        self.initialization_failed = True
        self.initialization_failure_reason = reason
        self.control.stop()
        self.mastery_admission = {
            "accepted": False,
            "reason": reason,
            "case_id": self.mastery_case["id"],
            "initial_state": initial,
        }
        return self.control.observe(), {**self._info(), "mastery": self.mastery_admission.copy()}

    def update_witness(self):
        if self.mastery_reference is not None and not self.control.stopped:
            _, q = self.mastery_reference.coordinates(self.control.trajectory_elapsed())
            self.control.set_reference(q, phase="carry" if self.mastery_case["stage"] == "loaded" else "reach")

    def step(self, action):
        self.update_witness()
        obs, reward, done, truncated, info = super().step(action)
        if not self.initialization_failed and not info.get("invalid_physics"):
            ref = self.suite["reference"]
            position = info["ee_position_error"] if self.control.hand_mask else 0.0
            angle = info["ee_orientation_error"] if self.control.hand_mask else 0.0
            reason = None
            if (
                position > ref["hand_position_tube_m"]
                or angle > ref["hand_orientation_tube_rad"]
                or info["base_goal_error"] > ref["base_position_tube_m"]
                or abs(info["base_heading_error_rad"]) > ref["base_heading_tube_rad"]
            ):
                reason = "mastery_tracking_envelope"
            if self.mastery_checker is not None:
                scratch = self.mastery_scratch
                mujoco.mj_copyData(scratch, self.model, self.data)
                if self.mastery_checker.configuration_collides(self.model, scratch):
                    reason = "mastery_measured_collision"
            if reason and info["code"] in {"running", "ok", "tracking_timeout"}:
                self.path_failed = True
                self.control.stop()
                self._done = True
                reward -= 20
                done = True
                truncated = False
                info = {
                    **info,
                    "code": reason,
                    "is_success": False,
                    "reward_terms": {**info.get("reward_terms", {}), "mastery_envelope_penalty": 20.0},
                }
        self.update_witness()
        spatial = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.control.base, spatial, 1)
        info["mastery"] = {
            "case_id": self.mastery_case["id"],
            "stage": self.mastery_case["stage"],
            "split": self.mastery_case["split"],
            "visit_count": self.case_visits[self.mastery_case["id"]],
            "path_failed": self.path_failed,
            "base_velocity_command": self.control.base_command.copy().tolist(),
            # Existing base_linear_velocity_m_s is displacement speed (unsigned).
            # A body-local measured twist preserves direction for reverse audits.
            "base_velocity_measured": [float(spatial[3]), float(spatial[2])],
        }
        return self.control.observe(), reward, done, truncated, info


def save_report(destination, report):
    destination = Path(destination)
    raw = (json.dumps(report, allow_nan=False) + "\n").encode()
    destination.write_bytes(gzip.compress(raw, mtime=0))
    return hashlib.sha256(raw).hexdigest()


def read_report(path):
    from emet.rl.artifacts import resolve_artifact

    return json.loads(gzip.decompress(resolve_artifact(Path(path).absolute()).read_bytes()))


@contextmanager
def preserve_training_rng():
    """Bundle loading during evaluation must not reseed PPO exploration."""
    import random

    import torch

    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def summarize(report):
    result = {}
    for stage in STAGES:
        for split in ("train", "validation"):
            rows = [c for c in report["cases"] if c["stage"] == stage and c["split"] == split]
            if rows:
                result[f"{stage}/{split}"] = {
                    "cases": len(rows),
                    "admitted": sum(c["admitted"] for c in rows),
                    "mastered": sum(c["mastered"] for c in rows),
                    "tracking_and_settling_passed": sum(c["tracking_and_settling_pass"] for c in rows),
                    "smoothness_passed": sum(c["smoothness_pass"] for c in rows),
                    "failure_reasons": dict(Counter(reason for c in rows for reason in c["failure_reasons"])),
                    "codes": dict(Counter(c["code"] for c in rows)),
                    "all_pass": all(c["mastered"] for c in rows),
                }
    return {
        "complete": report["complete"],
        "controller": report["controller"],
        "population": report.get("population", "canonical"),
        "groups": result,
        "all_pass": bool(result) and report["complete"] and all(v["all_pass"] for v in result.values()),
        "safety_promoted": False,
    }


def evaluate_suite(suite, *, stages=STAGES, policy=None, teacher_report=None, population="canonical"):
    from emet.rl.manipulation.command_runtime import task_pose_teacher_action
    from emet.rl.manipulation.hand_audit import hand_tracking_diagnostics

    report = {
        "format": "mars-mastery-report-v1",
        "scope": __doc__,
        "suite_sha256": digest(suite),
        "population": population,
        "source_hashes": source_hashes(),
        "controller": "task_pose_teacher" if policy is None else policy.manifest["weights_sha256"],
        "cases": [],
        "complete": False,
        "safety_promoted": False,
        "hardware_used": False,
    }
    expected = select_cases(suite, stages, "all", population=population)
    teacher_rows = {} if teacher_report is None else {c["id"]: c for c in teacher_report["cases"]}
    env = MasteryEnv(suite, stages=stages, split="all", population=population)
    try:
        if policy is not None and (
            policy.robot != env.profile.robot or policy.manifest["fingerprint"] != env.fingerprint
        ):
            raise ValueError("mastery policy/model mismatch")
        for case in expected:
            obs, info = env.reset(options={"case_id": case["id"]})
            row = {
                **case,
                "admitted": not info["initialization_failed"],
                "admission": env.mastery_admission,
                "trace": [],
            }
            report["cases"].append(row)
            if row["admitted"]:
                for _ in range(round(env.config.horizon / env.profile.dt) + 1):
                    action = task_pose_teacher_action(env.control) if policy is None else policy(obs)
                    obs, reward, done, truncated, info = env.step(action)
                    row["trace"].append(
                        {
                            "time": info["time"],
                            "hand": env.control.ee_pose().tolist(),
                            "base": env.control.base_pose().tolist(),
                            "reference_base": env.control.base_goal.copy().tolist(),
                            "reference_hand": env.control.hand_goal.copy().tolist(),
                            "action": np.asarray(action).tolist(),
                            "reward": reward,
                            "reward_terms": info.get("reward_terms", {}),
                            "base_velocity_command": info["mastery"]["base_velocity_command"],
                            "base_velocity_measured": info["mastery"]["base_velocity_measured"],
                            "code": info["code"],
                            "hand_position_error_m": info["ee_position_error"],
                            "hand_orientation_error_rad": info["ee_orientation_error"],
                            "base_error_m": info["base_goal_error"],
                            "base_heading_error_rad": info["base_heading_error_rad"],
                            "settled_seconds": info["settled_seconds"],
                            "hand_control": hand_tracking_diagnostics(env.control) if env.control.hand_mask else None,
                        }
                    )
                    if done or truncated:
                        break
            metrics = {
                "peak_force_N": float(info["peak_contact_force"]),
                "base_linear_jerk": float(env.peak_base_axis_jerk[0]),
                "base_yaw_jerk": float(env.peak_base_axis_jerk[1]),
                "joint_jerk": float(np.max(env.peak_joint_jerk)),
                "hand_position_error_m": info["ee_position_error"],
                "hand_orientation_error_rad": info["ee_orientation_error"],
                "base_error_m": info["base_goal_error"],
                "base_heading_error_rad": info["base_heading_error_rad"],
                "hold_drift_m": env.peak_hold_drift,
                "settled_seconds": env.settle_time,
            }
            smooth = bool(row["admitted"])
            failures = []
            if not row["admitted"]:
                failures.append("initialization")
            if not info.get("is_success", False):
                failures.append(info.get("code", "native_tracking_or_safety"))
            if env.settle_time + 1e-8 < suite["gate"]["terminal_hold_s"]:
                failures.append("terminal_hold")
            if env.peak_hold_drift > suite["gate"]["hold_drift_m"]:
                failures.append("hold_drift")
            if teacher_report is not None and row["admitted"]:
                reference = teacher_rows[case["id"]]
                if (
                    not reference["admitted"]
                    or row["admission"]["initial_state_sha256"] != reference["admission"]["initial_state_sha256"]
                    or row["admission"]["reference_sha256"] != reference["admission"]["reference_sha256"]
                ):
                    raise ValueError("teacher/policy initial state or reference differs")
                floors = {
                    "peak_force_N": suite["gate"]["force_floor_N"],
                    "base_linear_jerk": suite["gate"]["base_linear_jerk_floor_m_s3"],
                    "base_yaw_jerk": suite["gate"]["base_yaw_jerk_floor_rad_s3"],
                    "joint_jerk": suite["gate"]["joint_jerk_floor_rad_s3"],
                }
                smooth_failures = [
                    k
                    for k in floors
                    if metrics[k] > max(floors[k], suite["gate"]["smoothness_ratio"] * reference["metrics"][k])
                ]
                smooth = not smooth_failures
                failures += ["smoothness/" + k for k in smooth_failures]
            passed = (
                row["admitted"]
                and info.get("is_success", False)
                and env.settle_time + 1e-8 >= suite["gate"]["terminal_hold_s"]
                and env.peak_hold_drift <= suite["gate"]["hold_drift_m"]
                and not env.path_failed
                and smooth
            )
            row.update(
                code=info.get("code", env.initialization_failure_reason or "initialization_failed"),
                mastered=bool(passed),
                tracking_and_settling_pass=bool(
                    row["admitted"]
                    and info.get("is_success", False)
                    and env.settle_time + 1e-8 >= suite["gate"]["terminal_hold_s"]
                ),
                smoothness_pass=bool(smooth),
                failure_reasons=failures,
                metrics=metrics,
                final_info=info,
            )
            print("MASTERY", case["id"], row["code"], "pass" if row["mastered"] else "fail", flush=True)
    finally:
        env.close()
    if report["source_hashes"] != source_hashes():
        raise RuntimeError("sources changed during mastery evaluation")
    report["complete"] = True
    return report


def validate_preflight(report, suite, stages):
    expected = {c["id"] for c in select_cases(suite, stages, "all", population="admission")}
    rows = [r for r in report["cases"] if r["id"] in expected]
    if (
        not report.get("complete")
        or report.get("controller") != "task_pose_teacher"
        or report.get("suite_sha256") != digest(suite)
        or report.get("source_hashes") != source_hashes()
        or {r["id"] for r in rows} != expected
        or len(rows) != len(expected)
        or not all(r["mastered"] for r in rows)
    ):
        raise ValueError("current-source teacher must master every selected training and validation case")


def plot_report(report, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5), layout="constrained")
    base = fig.add_subplot(131)
    hand = fig.add_subplot(132, projection="3d")
    errors = fig.add_subplot(133)
    for row in report["cases"]:
        if not row["admitted"]:
            continue
        reference = row["admission"]["reference"]
        b, h = np.asarray(reference["base"]), np.asarray(reference["hand"])
        style = "-" if row["split"] == "train" else "--"
        base.plot(b[:, 0], b[:, 1], style, alpha=0.7)
        base.quiver(
            b[-1, 0],
            b[-1, 1],
            0.003 * np.cos(b[-1, 2]),
            0.003 * np.sin(b[-1, 2]),
            angles="xy",
            scale_units="xy",
            scale=1,
        )
        hand.plot(*h[:, :3].T, style, alpha=0.7)
        if len(report["cases"]) == 1:
            columns = h[-1, 3:].reshape(3, 2)
            axes = np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])
            for i, color in enumerate(("red", "green", "blue")):
                hand.quiver(*h[-1, :3], *(0.005 * axes[:, i]), color=color)
        if row["trace"] and row["stage"] != "base":
            errors.plot(
                [x["time"] for x in row["trace"]],
                [1000 * x["hand_position_error_m"] for x in row["trace"]],
                style,
                alpha=0.6,
            )
    base.set(xlabel="base world x (m)", ylabel="base world y (m)", title="Targets: solid train, dashed validation")
    base.set_aspect("equal", adjustable="datalim")
    hand.set(xlabel="hand x (m)", ylabel="hand y (m)", zlabel="hand z (m)", title="FK-witnessed pose paths")
    errors.axhline(3, color="black", linestyle=":")
    errors.set(xlabel="time (s)", ylabel="hand error (mm)", title="Measured hand tracking")
    fig.savefig(destination)
    plt.close(fig)


class RolloutDiagnostics:
    """Finite scalar summaries, split by case; no policy or simulator mutation."""

    def __init__(self):
        self.cases = {}

    def add(self, case_id, action, clipped_action, reward, info, done):
        action, clipped_action = np.asarray(action), np.asarray(clipped_action)
        if case_id not in self.cases:
            self.cases[case_id] = {
                "steps": 0,
                "reward_sum": 0.0,
                "reward_terms_sum": Counter(),
                "terminal_codes": Counter(),
                "action_sum": np.zeros_like(action, dtype=float),
                "action_square_sum": np.zeros_like(action, dtype=float),
                "action_clipped_count": np.zeros_like(action, dtype=int),
                "command_sum": np.zeros(2),
                "measured_velocity_sum": np.zeros(2),
                "base_error_sum_m": 0.0,
                "heading_error_sum_rad": 0.0,
            }
        row = self.cases[case_id]
        row["steps"] += 1
        row["reward_sum"] += float(reward)
        row["reward_terms_sum"].update(
            info.get("reward_terms", {"initialization_penalty": 20.0} if info["initialization_failed"] else {})
        )
        if done:
            row["terminal_codes"][info["code"]] += 1
        row["action_sum"] += action
        row["action_square_sum"] += action**2
        row["action_clipped_count"] += action != clipped_action
        row["command_sum"] += info["mastery"]["base_velocity_command"]
        row["measured_velocity_sum"] += info["mastery"]["base_velocity_measured"]
        row["base_error_sum_m"] += info["base_goal_error"]
        row["heading_error_sum_rad"] += info["base_heading_error_rad"]

    def record(self):
        result = {}
        for cid, row in self.cases.items():
            n = row["steps"]
            mean = row["action_sum"] / n
            result[cid] = {
                "steps": n,
                "reward_sum": row["reward_sum"],
                "reward_mean": row["reward_sum"] / n,
                "reward_terms_sum": dict(row["reward_terms_sum"]),
                "terminal_codes": dict(row["terminal_codes"]),
                "action_mean": mean.tolist(),
                "action_std": np.sqrt(np.maximum(0, row["action_square_sum"] / n - mean**2)).tolist(),
                "action_clipped_fraction": (row["action_clipped_count"] / n).tolist(),
                "base_velocity_command_mean": (row["command_sum"] / n).tolist(),
                "base_velocity_measured_mean": (row["measured_velocity_sum"] / n).tolist(),
                "base_error_mean_m": row["base_error_sum_m"] / n,
                "heading_error_mean_rad": row["heading_error_sum_rad"] / n,
            }
        return result


def make_training_vec_env(suite, stages, *, num_envs=1, backend="subproc"):
    """Independent native-physics worlds; no simulator/model shared across workers."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if type(num_envs) is not int or not 1 <= num_envs <= 64 or backend not in {"dummy", "subproc"}:
        raise ValueError("1-64 environments and dummy/subproc backend required")

    def factory():
        import torch

        torch.set_num_threads(1)
        return MasteryEnv(suite, stages=stages, split="train", population="training")

    factories = [factory for _ in range(num_envs)]
    return (
        DummyVecEnv(factories)
        if num_envs == 1 or backend == "dummy"
        else SubprocVecEnv(factories, start_method="spawn")
    )


def train(args, suite):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from emet.rl.manipulation.feedback_policy import FeedbackActorCriticPolicy, GoalFeedbackFeatures
    from emet.rl.manipulation.normalization import CommandVecNormalize
    from emet.rl.manipulation.policy import ManipulationPolicy, export
    from emet.rl.regression_gate import verify_regressions

    verify_regressions(args.regression_receipt)
    teacher = read_report(args.preflight)
    validate_preflight(teacher, suite, args.stages)
    out = Path(args.output)
    out.mkdir(exist_ok=False, parents=True)
    cfg = ManipulationConfig(**suite["config"])
    num_envs = getattr(args, "num_envs", 1)
    backend = getattr(args, "vec_backend", "subproc")
    rollout_steps = getattr(args, "rollout_steps", 256)
    if type(num_envs) is not int or not 1 <= num_envs <= 64:
        raise ValueError("1-64 environments required")
    rollout_size = num_envs * rollout_steps
    if type(rollout_steps) is not int or rollout_steps < 2 or args.steps % rollout_size:
        raise ValueError("steps must be a multiple of num_envs * rollout_steps, with rollout_steps >= 2")
    sources = source_hashes()
    normalizer = CommandVecNormalize(
        make_training_vec_env(suite, args.stages, num_envs=num_envs, backend=backend),
        config=cfg,
        norm_reward=False,
        clip_obs=20,
    )
    run = None
    try:
        model = PPO(
            FeedbackActorCriticPolicy,
            normalizer,
            device="cpu",
            seed=args.seed,
            n_steps=rollout_steps,
            batch_size=min(256, rollout_size),
            n_epochs=5,
            learning_rate=args.learning_rate,
            gamma=args.discount,
            target_kl=0.01,
            clip_range=0.1,
            policy_kwargs={
                "restoring_yaw_feedback": cfg.restoring_yaw_feedback,
                "turn_xy_tolerance": cfg.base_xy_tolerance,
                "features_extractor_class": GoalFeedbackFeatures,
                "features_extractor_kwargs": {
                    "robot": cfg.robot,
                    "retention_context": True,
                    "orientation_scale": cfg.feedback_orientation_scale,
                },
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            },
        )
        initialization = {"kind": "scratch", "optimizer_restored": False}
        if args.initialize_from:
            from emet.rl.manipulation.initialization import transfer_actor

            with preserve_training_rng():
                initialization = transfer_actor(
                    model, normalizer, args.initialize_from, cfg, normalizer.get_attr("fingerprint")[0]
                )
        with torch.no_grad():
            model.policy.log_std[:2].fill_(np.log(args.base_std))
            model.policy.log_std[2:].fill_(np.log(args.arm_std))
        record = {
            "suite": suite,
            "suite_sha256": digest(suite),
            "stages": args.stages,
            "num_envs": num_envs,
            "vec_backend": "dummy" if num_envs == 1 else backend,
            "rollout_steps_per_env": rollout_steps,
            "aggregate_rollout_steps": rollout_size,
            "source_hashes": sources,
            "env": asdict(cfg),
            "seed": args.seed,
            "steps": args.steps,
            "initialization": initialization,
            "base_std": args.base_std,
            "arm_std": args.arm_std,
            "learning_rate": args.learning_rate,
            "discount": args.discount,
            "preflight_sha256": hashlib.sha256(Path(args.preflight).read_bytes()).hexdigest(),
            "actor_task_id_input": False,
            "training_cases": select_cases(suite, args.stages, "train", population="training"),
            "canonical_evaluation_cases": select_cases(suite, args.stages, "all"),
            "teacher_action_loss": False,
            "audit_initial_admission": getattr(args, "audit_initial_admission", False),
            "command_smoothness_warmup_steps": getattr(args, "command_smoothness_warmup", 0),
            "safety_promoted": False,
        }
        (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        if args.wandb_mode == "offline":
            import wandb

            run = wandb.init(
                project="emet-mars-control",
                mode="offline",
                dir=str(out),
                name="mastery-" + "-".join(args.stages),
                config=record,
            )

        class Audit(BaseCallback):
            def __init__(self):
                super().__init__()
                self.last = 0
                self.counts = Counter()
                self.starts = Counter()
                self.episode_first = [True] * num_envs
                self.rollout = RolloutDiagnostics()
                self.last_update = 0

            def append(self, filename, metrics):
                with (out / filename).open("a") as f:
                    f.write(json.dumps(metrics, allow_nan=False) + "\n")

            def record_update(self):
                # SB3 calls rollout_end before train(), so optimizer diagnostics
                # belong at the next rollout_start and at training_end (last update).
                if self.model._n_updates == self.last_update:
                    return
                values = {
                    k: float(v) if np.isfinite(v) else None
                    for k, v in self.model.logger.name_to_value.items()
                    if k.startswith("train/") and np.isscalar(v)
                }
                metrics = {
                    "timesteps": self.num_timesteps,
                    "optimizer_epochs": self.model._n_updates,
                    **values,
                    "policy/action_std": self.model.policy.log_std.detach().exp().cpu().tolist(),
                }
                self.append("updates.jsonl", metrics)
                if run:
                    run.log(
                        {
                            **{k: v for k, v in metrics.items() if np.isscalar(v)},
                            **{f"policy/action_std/{i}": value for i, value in enumerate(metrics["policy/action_std"])},
                        }
                    )
                self.last_update = self.model._n_updates

            def _on_step(self):
                for rank, info in enumerate(self.locals["infos"]):
                    cid = info["mastery"]["case_id"]
                    self.counts[cid] += 1
                    if self.episode_first[rank]:
                        self.starts[cid] += 1
                    self.episode_first[rank] = bool(self.locals["dones"][rank])
                    self.rollout.add(
                        cid,
                        self.locals["actions"][rank],
                        self.locals["clipped_actions"][rank],
                        self.locals["rewards"][rank],
                        info,
                        self.locals["dones"][rank],
                    )
                return True

            def _on_rollout_end(self):
                metrics = {
                    "timesteps": self.num_timesteps,
                    "sampling/steps": dict(self.counts),
                    "sampling/episode_starts": dict(self.starts),
                    "rollout/cases": self.rollout.record(),
                }
                self.append("metrics.jsonl", metrics)
                if run:
                    run.log(
                        {
                            "timesteps": self.num_timesteps,
                            **{f"sampling/{k}/steps": v for k, v in self.counts.items()},
                            **{
                                f"rollout/{cid}/{key}": value
                                for cid, row in metrics["rollout/cases"].items()
                                for key, value in row.items()
                                if np.isscalar(value)
                            },
                        }
                    )
                self.rollout = RolloutDiagnostics()

            def checkpoint(self, label):
                if sources != source_hashes():
                    raise RuntimeError("training sources changed")
                destination = out / label
                export(
                    self.model,
                    normalizer,
                    destination,
                    cfg,
                    normalizer.get_attr("fingerprint")[0],
                    steps=self.num_timesteps,
                )
                with preserve_training_rng():
                    report = evaluate_suite(
                        suite, stages=args.stages, policy=ManipulationPolicy(destination), teacher_report=teacher
                    )
                save_report(destination / "evaluation.json.gz", report)
                (destination / "summary.json").write_text(json.dumps(summarize(report), indent=2) + "\n")
                if self.num_timesteps == 0 and getattr(args, "audit_initial_admission", False):
                    with preserve_training_rng():
                        expanded = evaluate_suite(
                            suite,
                            stages=args.stages,
                            policy=ManipulationPolicy(destination),
                            teacher_report=teacher,
                            population="admission",
                        )
                    save_report(destination / "admission-evaluation.json.gz", expanded)
                    (destination / "admission-summary.json").write_text(
                        json.dumps(summarize(expanded), indent=2) + "\n"
                    )
                if run:
                    run.log(
                        {
                            "timesteps": self.num_timesteps,
                            **{
                                f"evaluation/{group}/{key}": value
                                for group, row in summarize(report)["groups"].items()
                                for key, value in row.items()
                                if np.isscalar(value)
                            },
                        }
                    )
                self.last = self.num_timesteps

            def _on_rollout_start(self):
                self.record_update()
                warmup = getattr(args, "command_smoothness_warmup", 0)
                fraction = min(1.0, self.num_timesteps / warmup) if warmup else 1.0
                normalizer.env_method("set_command_smoothness_fraction", fraction)
                if self.num_timesteps and self.num_timesteps - self.last >= args.eval_every:
                    self.checkpoint(f"step-{self.num_timesteps}")

            def _on_training_end(self):
                self.record_update()

        audit = Audit()
        audit.init_callback(model)
        audit.checkpoint("step-0")
        model.learn(args.steps, callback=audit)
        audit.num_timesteps = model.num_timesteps
        audit.checkpoint("last")
        (out / "complete.json").write_text(
            json.dumps(
                {
                    "steps": model.num_timesteps,
                    "mastery": json.loads((out / "last/summary.json").read_text()),
                    "safety_promoted": False,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        normalizer.close()
        if run:
            run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "evaluate", "train", "plot"])
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--stages", nargs="+", choices=STAGES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--population",
        choices=["canonical", "admission"],
        default="canonical",
        help="policy evaluation population; preflight always covers the full admission population",
    )
    parser.add_argument("--case-id", help="select one case for visualization only")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--regression-receipt", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=90)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vec-backend", choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.9975)
    parser.add_argument("--base-std", type=float, default=0.03)
    parser.add_argument("--arm-std", type=float, default=0.10)
    parser.add_argument(
        "--command-smoothness-warmup",
        type=int,
        default=0,
        help="training-only linear penalty ramp in environment steps",
    )
    parser.add_argument("--wandb-mode", choices=["disabled", "offline"], default="disabled")
    parser.add_argument("--audit-initial-admission", action="store_true")
    args = parser.parse_args()
    if args.stages is None:
        args.stages = ["base"] if args.command == "train" else list(STAGES)
    import torch

    torch.set_num_threads(1)
    if args.case_id and args.command != "plot":
        parser.error("case sub-selection is visualization only; mastery gates use both complete splits")
    if args.command == "plot":
        if not args.preflight:
            parser.error("plot requires --preflight REPORT.json.gz")
        report = read_report(args.preflight)
        if args.case_id:
            report["cases"] = [c for c in report["cases"] if c["id"] == args.case_id]
            if not report["cases"]:
                parser.error("unknown case ID")
        args.output.mkdir(exist_ok=False, parents=True)
        plot_report(report, args.output / "targets.svg")
        return
    suite = load_suite(args.suite)
    if args.command == "train":
        if (
            not args.preflight
            or not args.regression_receipt
            or min(args.steps, args.eval_every) <= 0
            or not 1 <= args.num_envs <= 64
            or args.rollout_steps < 2
            or args.command_smoothness_warmup < 0
            or not np.isfinite([args.learning_rate, args.discount, args.base_std, args.arm_std]).all()
            or not 0 < args.discount < 1
            or min(args.learning_rate, args.base_std, args.arm_std) <= 0
        ):
            parser.error("training requires preflight, fresh regressions and positive finite settings")
        train(args, suite)
    else:
        from emet.rl.manipulation.policy import ManipulationPolicy

        if args.command == "evaluate" and (not args.policy or not args.preflight):
            parser.error("policy evaluation requires --policy and --preflight")
        teacher = read_report(args.preflight) if args.preflight else None
        if teacher:
            validate_preflight(teacher, suite, args.stages)
        report = evaluate_suite(
            suite,
            stages=args.stages,
            policy=ManipulationPolicy(args.policy) if args.command == "evaluate" else None,
            teacher_report=teacher,
            population="admission" if args.command == "preflight" else args.population,
        )
        args.output.mkdir(exist_ok=False, parents=True)
        save_report(args.output / "report.json.gz", report)
        (args.output / "summary.json").write_text(json.dumps(summarize(report), indent=2) + "\n")
        plot_report(report, args.output / "targets.svg")
        print(json.dumps(summarize(report), indent=2))


if __name__ == "__main__":
    main()
