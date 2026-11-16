"""Reachable coupled references, reset distribution and diagnostics for both robots.

FK/IK witnesses are private, reset-only geometry. The actor receives ordinary
masked hand/base previews and measured history. Certificates bound reference
kinematics; sampled admission and teacher rollouts do not certify hardware.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.jerk_reference import compile_joint_path, polynomial_peaks
from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory

ROOT = Path(__file__).resolve().parents[4]
SUITE = ROOT / "configs/benchmarks/coupled_tracking_v1.json"
COARSE_SUITE = ROOT / "configs/benchmarks/coupled_tracking_v2.json"
MODES = {"hand_free_base", "coupled", "base_only", "fixed_world_hand"}


def suite_contract(suite):
    """Versioned general acceptance; frozen manipulation contracts are separate."""
    contract = {
        "base_xy_tolerance": 0.02,
        "base_heading_tolerance": 0.04,
        "hand_position_tolerance": 0.003,
        "hand_orientation_tolerance": 0.02,
    }
    if suite["format"] == "coupled-tracking-v2":
        contract.update(base_xy_tolerance=0.05, base_heading_tolerance=float(np.deg2rad(5)))
        if suite.get("acceptance") != contract:
            raise ValueError("v2 acceptance must declare coarse base and unchanged precise hand tolerances")
    return contract


def load_suite(path=SUITE):
    suite = json.loads(Path(path).read_text())
    if suite.get("format") not in {"coupled-tracking-v1", "coupled-tracking-v2"} or set(suite["robots"]) != {
        "stretch",
        "innate_mars",
    }:
        raise ValueError("expected the two-robot coupled tracking distribution")
    contract = suite_contract(suite)
    coarse = suite["format"] == "coupled-tracking-v2"
    travel_bound, turn_bound = (0.2, 0.5) if coarse else (0.05, 0.1)
    ids, seeds = set(), set()
    for robot, cases in suite["robots"].items():
        for case in cases:
            if (
                set(case)
                - {
                    "id",
                    "seed",
                    "split",
                    "mode",
                    "travel_m",
                    "turn_rad",
                    "arm_delta",
                    "base_mask",
                    "retarget",
                    "self_motion_scale",
                    *({"purpose"} if coarse else set()),
                }
                or not isinstance(case["id"], str)
                or not case["id"].startswith(robot + ":")
                or case["id"] in ids
                or type(case["seed"]) is not int
                or not 0 <= case["seed"] < 2**31
                or case["seed"] in seeds
                or case["split"] not in {"train", "audit"}
                or case["mode"] not in MODES
            ):
                raise ValueError("invalid or repeated coupled case identity")
            motion = np.asarray([case["travel_m"], case["turn_rad"], *case["arm_delta"]], dtype=float)
            mask = case["base_mask"]
            if (
                motion.shape != (7,)
                or not np.isfinite(motion).all()
                or np.any(abs(motion) > [travel_bound, turn_bound, 0.1, 0.1, 0.1, 0.1, 0.1])
                or len(mask) != 3
                or any(type(v) is not bool for v in mask)
                or (case["mode"] == "hand_free_base" and any(mask))
                or (case["mode"] in {"base_only", "fixed_world_hand"} and not all(mask))
                or (case["mode"] == "base_only" and np.any(motion[2:]))
                or (case["mode"] == "fixed_world_hand" and np.any(motion))
                or (case["mode"] == "coupled" and not any(mask))
            ):
                raise ValueError("invalid coupled geometry or masks")
            if coarse:
                purpose = "hand_hold_nullspace_diagnostic" if case["mode"] == "fixed_world_hand" else "tracking"
                if case.get("purpose") != purpose:
                    raise ValueError("v2 must distinguish optional nullspace motion from required base progress")
                if (
                    case["mode"] != "fixed_world_hand"
                    and any(mask[:2])
                    and abs(motion[0]) <= 2 * contract["base_xy_tolerance"]
                ):
                    raise ValueError("v2 required base travel must exceed twice its acceptance radius")
            if case["mode"] == "fixed_world_hand" and (
                type(case.get("self_motion_scale")) not in (int, float)
                or not np.isfinite(case["self_motion_scale"])
                or not 0 < abs(case["self_motion_scale"]) <= 1
            ):
                raise ValueError("fixed hand requires a declared self-motion amplitude")
            if "retarget" in case:
                event = case["retarget"]
                if (
                    set(event) != {"at_fraction", "travel_m"}
                    or not 0 < event["at_fraction"] < 1
                    or not np.isfinite(event["travel_m"])
                    or abs(event["travel_m"]) > travel_bound
                ):
                    raise ValueError("invalid announced goal change")
            ids.add(case["id"])
            seeds.add(case["seed"])
        if {c["split"] for c in cases} != {"train", "audit"}:
            raise ValueError("both train and independent audit cases required")
    return suite


def rotation(pose):
    columns = np.asarray(pose)[3:].reshape(3, 2)
    return np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])


def spline_bounds(spline):
    """Exact coordinatewise derivative maxima of a piecewise cubic in phase."""
    a, b, c, _ = spline.c
    widths = np.diff(spline.x)[:, None]
    vertex = np.divide(-b, 3 * a, out=np.zeros_like(a), where=abs(a) > 1e-15)
    vertex = np.clip(vertex, 0, widths)
    q1 = np.max(
        np.stack([abs(c), abs(3 * a * widths**2 + 2 * b * widths + c), abs(3 * a * vertex**2 + 2 * b * vertex + c)]),
        axis=(0, 1),
    )
    q2 = np.max(np.stack([abs(2 * b), abs(6 * a * widths + 2 * b)]), axis=(0, 1))
    q3 = np.max(abs(6 * a), axis=0)
    return q1, q2, q3


def fixed_hand_witness(control, fk, amplitude):
    """Reset-only self-motion sampling on the full differential-drive Jacobian.

    Generate a compatible base path along a hand-task nullspace, rather than
    demanding arbitrary base motion that a five-coordinate arm cannot cancel.
    This private geometry sampler never supplies actions or joint goals to the actor.
    """
    c, m, d = control, control.model, fk.data
    base, q = control.base_pose().copy(), control.positions().copy()
    target = control.ee_pose().copy()
    scales = np.r_[max(abs(amplitude), 0.005), 0.08, np.asarray(c.profile.speed) * 0.1]
    phases = np.linspace(0, 1, 81)
    bases, joints, previous = [base.copy()], [q.copy()], None
    sign = 1 if amplitude >= 0 else -1
    metric = np.r_[np.ones(3), np.full(3, 0.15)]
    for _ in phases[1:]:
        hand = fk(base, q)
        mujoco.mj_comPos(m, d)  # mj_kinematics alone does not prepare Jacobian caches.
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, c.ee)
        jac = np.vstack([jp, jr])
        arm = np.column_stack([jac[:, group].sum(axis=1) / len(group) for group in c.v])
        yaw = base[2]
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        pivot = d.xpos[c.base].copy() + d.xmat[c.base].copy().reshape(3, 3) @ c.axle_offset
        base_jac = np.column_stack(
            [np.r_[heading, np.zeros(3)], np.r_[np.cross([0.0, 0.0, 1.0], hand[:3] - pivot), [0.0, 0.0, 1.0]]]
        )
        matrix = np.column_stack([base_jac, arm]) * scales * metric[:, None]
        _, _, vh = np.linalg.svd(matrix, full_matrices=True)
        direction = vh[-1]
        if previous is None:
            largest = np.argmax(abs(direction[:2]))
            direction *= sign if direction[largest] >= 0 else -sign
        elif direction @ previous < 0:
            direction *= -1
        previous = direction.copy()
        error = np.r_[target[:3] - hand[:3], Rotation.from_matrix(rotation(target) @ rotation(hand).T).as_rotvec()]
        delta = scales * (direction / (len(phases) - 1) + np.linalg.pinv(matrix, rcond=1e-7) @ (error * metric))
        forward, turn = delta[:2]
        r0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        yaw1 = yaw + turn
        r1 = np.array([[np.cos(yaw1), -np.sin(yaw1)], [np.sin(yaw1), np.cos(yaw1)]])
        axle = base[:2] + r0 @ c.axle_offset[:2]
        if abs(turn) < 1e-10:
            axle += forward * heading[:2]
        else:
            axle += forward / turn * np.array([np.sin(yaw1) - np.sin(yaw), np.cos(yaw) - np.cos(yaw1)])
        base = np.r_[axle - r1 @ c.axle_offset[:2], yaw1]
        q += delta[2:]
        bases.append(base.copy())
        joints.append(q.copy())
    return phases, np.array(bases), np.array(joints)


def reference_support_margin(model, data, profile):
    if profile.robot == "innate_mars":
        from emet.rl.manipulation.grasp_setup import support_margin

        return support_margin(model, data, profile, None)["margin_m"]
    if profile.robot != "stretch":
        raise ValueError("support geometry adapter required")
    base = model.body(profile.base_body).id
    # The pinned Stretch model represents its caster as one collision sphere
    # directly on base_link, rather than a separately articulated wheel body.
    caster = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == base
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    if len(caster) != 1:
        raise ValueError("unique modeled Stretch caster required")
    polygon = np.array(
        [*[data.body(n).xpos[:2].copy() for n in profile.wheel_bodies], data.geom_xpos[caster[0], :2].copy()]
    )
    midpoint = polygon.mean(axis=0)
    polygon = polygon[np.argsort(np.arctan2(polygon[:, 1] - midpoint[1], polygon[:, 0] - midpoint[0]))]
    ids = {base}
    for body in range(base + 1, model.nbody):
        if int(model.body_parentid[body]) in ids:
            ids.add(body)
    ids = np.array(sorted(ids))
    weights = model.body_mass[ids].copy()
    centers = data.xpos[ids].copy() + np.einsum(
        "nij,nj->ni", data.xmat[ids].copy().reshape(-1, 3, 3), model.body_ipos[ids].copy()
    )
    com = np.sum(centers * weights[:, None], axis=0) / weights.sum()
    margins = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge, delta = b - a, com[:2] - a
        length = np.linalg.norm(edge)
        if length < 1e-6:
            raise ValueError("degenerate support polygon")
        margins.append(float((edge[0] * delta[1] - edge[1] * delta[0]) / length))
    return min(margins)


class CoupledReference(TaskPoseTrajectory):
    """Continuous FK hand target on a differential-drive axle arc and arm path."""

    def __init__(self, control, case):
        self.fk = TaskFrameFK(control.model, control.data, control)
        self.case = dict(case)
        self.start_base = control.base_pose().copy()
        self.axle_offset = control.axle_offset[:2].copy()
        self.travel, self.turn = float(case["travel_m"]), float(case["turn_rad"])
        self.fixed_hand = control.ee_pose().copy() if case["mode"] == "fixed_world_hand" else None
        q0 = control.positions().copy()
        delta = np.asarray(case["arm_delta"], dtype=float)
        if delta.shape != q0.shape:
            raise ValueError("arm dimension does not match robot profile")
        self.base_spline = None
        if self.fixed_hand is None:
            phases, joints = np.array([0.0, 1.0]), np.array([q0, q0 + delta])
        else:
            phases, bases, joints = fixed_hand_witness(control, self.fk, 0.02 * case["self_motion_scale"])
            self.base_spline = CubicSpline(phases, bases, axis=0, bc_type="natural")
        self.joints = CubicSpline(phases, joints, axis=0, bc_type="natural")
        q1, q2, q3 = spline_bounds(self.joints)
        if self.base_spline is None:
            b1, b2, b3 = np.array([abs(self.travel), abs(self.turn)]), np.zeros(2), np.zeros(2)
        else:
            d1, d2, d3 = spline_bounds(self.base_spline)
            radius = float(np.linalg.norm(self.axle_offset))
            b1 = np.array([np.linalg.norm(d1[:2]) + radius * d1[2], d1[2]])
            b2 = np.array([np.linalg.norm(d2[:2]) + radius * (d2[2] + d1[2] ** 2), d2[2]])
            b3 = np.array([np.linalg.norm(d3[:2]) + radius * (d3[2] + 3 * d1[2] * d2[2] + d1[2] ** 3), d3[2]])
        q1, q2, q3 = np.r_[b1, q1], np.r_[b2, q2], np.r_[b3, q3]
        p = control.profile
        # Explicit experimental reference jerk limits, not hardware ratings.
        speed = np.r_[p.base_speed, p.base_yaw_speed, p.speed]
        accel = np.r_[p.base_acceleration, p.base_yaw_acceleration, p.acceleration]
        jerk = 4 * accel
        scale = 1.0
        for _ in range(20):
            path, _ = compile_joint_path([[0.0], [1.0]], [1 / scale], [1 / scale**2], [1 / scale**3])
            v, a, j = (float(x[0]) for x in polynomial_peaks(path.segments[0]))
            bounds = np.array([q1 * v, q2 * v * v + q1 * a, q3 * v**3 + 3 * q2 * v * a + q1 * j])
            ratios = np.max(bounds / np.array([speed, accel, jerk]), axis=1)
            factor = max(ratios[0], np.sqrt(ratios[1]), np.cbrt(ratios[2]))
            if factor <= 1 + 1e-9:
                break
            scale *= factor * 1.01
        else:
            raise ValueError("cannot time-scale the coupled reference")
        self.path = path
        self.certificate = {
            "speed": speed.tolist(),
            "acceleration": accel.tolist(),
            "jerk": jerk.tolist(),
            "derivative_bounds": bounds.tolist(),
            "geometry": "nonholonomic_axle_arc_with_continuous_arm_spline"
            if self.fixed_hand is None
            else "differential_drive_hand_nullspace_cubic_witness",
            "jerk_limit_source": "experimental_4_times_acceleration",
            "dynamic_feasibility_verified": False,
        }
        times = np.r_[np.arange(0, path.end, p.dt), path.end]
        base, hand = zip(*(self._sample(t) for t in times), strict=True)
        super().__init__(
            times,
            base,
            hand,
            np.full(len(times), control.grip_reference),
            base_mask=case["base_mask"],
            hand_mask=case["mode"] != "base_only",
        )

    def base_at(self, phase):
        if self.base_spline is not None:
            return self.base_spline(phase)
        yaw0 = self.start_base[2]
        yaw = yaw0 + self.turn * phase
        r0 = np.array([[np.cos(yaw0), -np.sin(yaw0)], [np.sin(yaw0), np.cos(yaw0)]])
        r = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        axle = self.start_base[:2] + r0 @ self.axle_offset
        if abs(self.turn) < 1e-10:
            axle += phase * self.travel * np.array([np.cos(yaw0), np.sin(yaw0)])
        else:
            axle += self.travel / self.turn * np.array([np.sin(yaw) - np.sin(yaw0), np.cos(yaw0) - np.cos(yaw)])
        return np.r_[axle - r @ self.axle_offset, np.arctan2(np.sin(yaw), np.cos(yaw))]

    def coordinates(self, time):
        phase = float(np.clip(self.path.sample(time)[0][0], 0, 1))
        return self.base_at(phase), self.joints(phase)

    def _sample(self, time):
        base, joints = self.coordinates(time)
        hand = self.fk(base, joints)
        return base, hand if self.fixed_hand is None else self.fixed_hand.copy()

    def sample(self, time):
        base, hand = self._sample(float(np.clip(time, 0, self.path.end)))
        return base, hand, float(self.gripper[0])

    def record(self):
        return {
            **super().record(),
            "format": "coupled_fk_reference_v1",
            "interpolation": "continuous_private_fk_cubic_arm_quintic_phase",
            "certificate": self.certificate,
            "joint_phase_knots": self.joints.x.tolist(),
            "joint_phase_coefficients": self.joints.c.tolist(),
            "start_base": self.start_base.tolist(),
            "axle_offset": self.axle_offset.tolist(),
            "travel_m": self.travel,
            "turn_rad": self.turn,
            "fixed_world_hand": self.fixed_hand is not None,
            "phase_path": self.path.record(),
            "fk_anchor_qpos": self.fk.qpos.tolist(),
            "base_phase_knots": None if self.base_spline is None else self.base_spline.x.tolist(),
            "base_phase_coefficients": None if self.base_spline is None else self.base_spline.c.tolist(),
        }


def admit_reference(env, reference):
    """Sample full-path geometry in scratch data; retain every rejected case."""
    from emet.eval.physical_tamp import make_scene_checker

    checker = make_scene_checker(
        env.model, {"object_body": env.payload_name, "gripper_bodies": env.profile.finger_bodies}
    )
    minimum, max_position, max_rotation = float("inf"), 0.0, 0.0
    for time in np.r_[np.arange(0, reference.end, 0.025), reference.end]:
        base, joints = reference.coordinates(time)
        if np.any(joints < env.control.limits[:, 0] + 0.002) or np.any(joints > env.control.limits[:, 1] - 0.002):
            return {"accepted": False, "reason": "joint_margin"}
        hand = reference.fk(base, joints)
        if checker.configuration_collides(env.model, reference.fk.data):
            return {"accepted": False, "reason": "planned_collision"}
        margin = reference_support_margin(env.model, reference.fk.data, env.profile)
        minimum = min(minimum, margin)
        if margin < 0.005:
            return {"accepted": False, "reason": "planned_support_margin", "minimum_support_margin_m": minimum}
        if reference.fixed_hand is not None:
            max_position = max(max_position, float(np.linalg.norm(hand[:3] - reference.fixed_hand[:3])))
            angle = Rotation.from_matrix(rotation(hand) @ rotation(reference.fixed_hand).T).magnitude()
            max_rotation = max(max_rotation, float(angle))
    return {
        "accepted": max_position <= 0.0001 and max_rotation <= 0.001,
        "reason": None if max_position <= 0.0001 and max_rotation <= 0.001 else "fixed_hand_witness_error",
        "minimum_support_margin_m": minimum,
        "fixed_hand_witness_position_m": max_position,
        "fixed_hand_witness_orientation_rad": max_rotation,
        "dynamic_feasibility_verified": False,
        "geometry_sample_period_s": 0.025,
    }


class CoupledTrackingEnv(gym.Wrapper):
    """Reusable general training share; task/case metadata stays outside the actor."""

    def __init__(self, robot, cases=None, *, suite_path=SUITE):
        from gymnasium import spaces

        from emet.rl.manipulation.observation_layout import history_layout
        from emet.rl.manipulation.whole_body_observation import PRIVILEGED_WIDTH

        if robot not in {"innate_mars", "stretch"}:
            raise ValueError("unsupported coupled robot")
        suite = load_suite(suite_path)
        self.suite_format = suite["format"]
        self.cases = cases if cases is not None else suite["robots"][robot]
        cfg = ManipulationConfig(
            robot=robot,
            scenario="reach",
            horizon=25,
            randomization_scale=0,
            action_mode="direct_velocity",
            trajectory_preview=True,
            retention_context=True,
            reach_hold_probability=1,
            tracking_reward=True,
            tracking_precision=4,
            whole_body_settling=True,
            terminal_hold_seconds=2,
            wheel_feedback_current_velocity=robot == "innate_mars",
            wheel_feedback_gain=0.5,
            physical_wheel_slew=True,
            **suite_contract(suite),
            retarget_probability=0,
            reach_push_probability=0,
        )
        super().__init__(ManipulationEnv(cfg))
        _, width = history_layout(self.unwrapped.profile, retention_context=True)
        self.observation_space = spaces.Dict(
            {
                "history": spaces.Box(-np.inf, np.inf, (20, width), np.float32),
                "preview": spaces.Box(-np.inf, np.inf, (7, 18), np.float32),
                "motion": spaces.Box(-np.inf, np.inf, (9,), np.float32),
                "critic": spaces.Box(-np.inf, np.inf, (PRIVILEGED_WIDTH,), np.float32),
            }
        )
        self.cursor = 0

    def observe(self, raw):
        from emet.rl.manipulation.whole_body_observation import actor_observation, privileged_state

        e = self.unwrapped
        return actor_observation(e.control, raw) | {
            "critic": privileged_state(e, e.steps * e.profile.dt, e.config.horizon)
        }

    def reset(self, *, seed=None, options=None):
        from emet.rl.manipulation.whole_body_reward import RewardMeter

        if options is not None and set(options) != {"case_id"}:
            raise ValueError("only a declared coupled case can be selected")
        case = (
            next((c for c in self.cases if c["id"] == options["case_id"]), None)
            if options
            else self.cases[self.cursor % len(self.cases)]
        )
        if case is None:
            raise ValueError("unknown coupled case")
        self.case = case
        self.cursor += 1
        e = self.unwrapped
        raw, info = e.reset(seed=case["seed"])
        if info.get("initialization_failed"):
            raise ValueError("coupled_initialization_failed")
        reference = CoupledReference(e.control, case)
        self.admission = admit_reference(e, reference)
        if not self.admission["accepted"]:
            raise ValueError("coupled_reference_rejected:" + str(self.admission["reason"]))
        e.reach_hold_trial = False
        e.fixed_base_trial = False
        e.control.pose_only_targets = True
        e.set_task_trajectory(reference)
        e.control.history = None
        self.retarget_at = reference.end * case["retarget"]["at_fraction"] if "retarget" in case else None
        self.retargeted = False
        self.meter = RewardMeter(e, base_heading_region=self.suite_format == "coupled-tracking-v2")
        raw = e.control.observe()
        return self.observe(raw), {
            **info,
            "case_id": case["id"],
            "coupled_case_id": case["id"],
            "coupled_admission": self.admission,
            "reference_sha256": digest(reference.record()),
        }

    def step(self, action):
        from emet.rl.manipulation.whole_body_observation import reference_velocity

        e = self.unwrapped
        if self.retarget_at is not None and not self.retargeted and e.control.trajectory_elapsed() >= self.retarget_at:
            updated = {**self.case, "travel_m": self.case["retarget"]["travel_m"]}
            updated.pop("retarget")
            reference = CoupledReference(e.control, updated)
            admission = admit_reference(e, reference)
            if not admission["accepted"]:
                e.control.stop()
                raise ValueError("coupled_retarget_rejected:" + str(admission["reason"]))
            e.set_task_trajectory(reference)
            self.retargeted = True
        motion = reference_velocity(e.control)
        raw, _, done, truncated, info = e.step(action)
        rewards = self.meter.tick(action, motion)
        success = bool(info.get("is_success", False))
        if done or truncated:
            rewards[0] += 2.0 if success else -2.0
        return (
            self.observe(raw),
            float(rewards.sum()),
            done,
            truncated,
            {
                **info,
                "reward_groups": rewards,
                "elapsed_control_ticks": 1,
                "primitive_success": success,
                "full_pick_place_success": False,
                "case_id": self.case["id"],
                "coupled_case_id": self.case["id"],
                "retarget_delivered": self.retargeted,
                "coupled_admission": self.admission,
                "reward_terms": self.meter.last_terms.copy(),
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", choices=("innate_mars", "stretch"), required=True)
    parser.add_argument(
        "--teacher", action="store_true", help="run the bounded physical teacher after geometric admission"
    )
    args = parser.parse_args()
    from emet.rl.manipulation.coupled_evaluation import export_distribution

    export_distribution(args.robot, args.output, suite_path=args.suite, teacher=args.teacher)


if __name__ == "__main__":
    main()
