"""Fresh, intentional arm motion for the ordinary whole-body task families."""

import copy
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FORMAT = "broad-joint-offsets-v1"


def _delta(seed, index):
    rng = np.random.default_rng(seed)
    sign = 1 if index % 2 == 0 else -1
    return [
        sign * float(rng.uniform(0.28, 0.38)),
        float(rng.uniform(-0.08, 0.08)),
        float(rng.uniform(-0.08, 0.08)),
        float(rng.uniform(-0.06, 0.06)),
        sign * float(rng.uniform(0.20, 0.30)),
    ]


def general_cases():
    """Keep base travel/masks; replace sub-tolerance nominal arm offsets.

    The source pool remains frozen. New hand identities use independent seeds;
    base-only identities are unchanged. FK and full-path admission still decide
    whether each concrete reference is usable; failed cases are never pruned.
    """
    original = json.loads((ROOT / "paper/data/whole_body_control/long_finetune_v1/cases.json").read_text())
    result = []
    indices = {"train": 0, "audit": 0}
    for item in original:
        case = copy.deepcopy(item)
        if case["mode"] != "base_only":
            split = case["split"]
            index = indices[split]
            indices[split] += 1
            seed = (530_000_000 if split == "train" else 540_000_000) + index
            case.update(id=f"innate_mars:broad-hand-v1:{split}:{index}", seed=seed, arm_delta=_delta(seed, index))
        result.append(case)
    return result


def goal_cases():
    from emet.rl.manipulation.goal_change_tracking import load_curriculum

    spec = load_curriculum(ROOT / "configs/benchmarks/mars_goal_changes_v1.json")
    result = []
    changes = {}
    indices = {"train": 0, "audit": 0}
    for original in spec["robots"]["innate_mars"]:
        case = copy.deepcopy(original)
        split = case["split"]
        index = indices[split]
        indices[split] += 1
        seed = (550_000_000 if split == "train" else 560_000_000) + index
        case.update(id=f"innate_mars:broad-handoff-v1:{split}:{index}", seed=seed, arm_delta=_delta(seed, index))
        change = copy.deepcopy(spec["goal_changes"][original["id"]])
        # A large second hand command returns toward the original arm posture;
        # it never anchors on the executing policy's error or resets physics.
        change["arm_delta"] = (-np.asarray(case["arm_delta"])).tolist()
        changes[case["id"]] = change
        result.append(case)
    return result, changes


def preflight_pools():
    from emet.rl.manipulation.whole_body_training import load_development

    general = general_cases()
    goals, _ = goal_cases()
    development, _ = load_development()
    return {
        "hand": [c for c in general if c["mode"] != "base_only"],
        "base": [c for c in general if c["mode"] == "base_only"],
        "goal_changes": goals,
        "tamp": [dict(c, split="train") for c in development["train_cases"]]
        + [dict(c, split="audit") for c in development["audit_cases"]],
    }


def make_reference_env(family, cases, requirements, *, settling_budget_s=6.0):
    from dataclasses import replace

    from emet.rl.manipulation.coupled_tracking import COARSE_SUITE, CoupledTrackingEnv
    from emet.rl.manipulation.goal_change_tracking import GoalChangeTrackingEnv

    kwargs = {
        "suite_path": COARSE_SUITE,
        "timing": "bounded_v2",
        "finish_after_hold": True,
        "spread_requirements": requirements if family != "base" else None,
        "settling_budget_s": settling_budget_s,
    }
    if family == "goal_changes":
        _, changes = goal_cases()
        env = GoalChangeTrackingEnv(
            "innate_mars", cases, changes={c["id"]: changes[c["id"]] for c in cases}, change_arm_limit=0.4, **kwargs
        )
    else:
        env = CoupledTrackingEnv("innate_mars", cases, **kwargs)
    env.unwrapped.config = replace(env.unwrapped.config, wheel_feedback_base_only_gains=(0.0, 0.5))
    return env


def teacher_preflight(pools, requirements, *, settling_budget_s=6.0):
    from emet.rl.manipulation.arm_spline_benchmark import stop
    from emet.rl.manipulation.command_runtime import TaskPoseTeacherConfig, task_pose_teacher_action
    from emet.rl.manipulation.whole_body_training import primitive_rollout

    result = {}
    for family, cases in pools.items():
        if family == "tamp":
            rows = primitive_rollout(
                cases,
                primitive="grasp_pose",
                planning_scope="unloaded_prefix",
                wheel_feedback_current_velocity=True,
                wheel_feedback_base_only_gains=(0.0, 0.5),
            )
            result[family] = [
                {
                    "case": case,
                    "success": row["success"],
                    "code": row["code"],
                    "admitted": not row.get("initialization_failed", True) and not row.get("execution_error"),
                    "stop_confirmed": row.get("stop_confirmed"),
                    "metrics": row.get("safety_flags", {}),
                }
                for case, row in zip(cases, rows, strict=True)
            ]
            continue
        env = make_reference_env(family, cases, requirements, settling_budget_s=settling_budget_s)
        result[family] = []
        try:
            for case in cases:
                row = {"case": case, "success": False, "admitted": False}
                try:
                    _, initial = env.reset(options={"case_id": case["id"]})
                    e = env.unwrapped
                    moving = 0
                    for _ in range(int(np.ceil(e.config.horizon / e.profile.dt)) + 1):
                        _, _, done, truncated, info = env.step(
                            task_pose_teacher_action(
                                e.control, config=TaskPoseTeacherConfig(coupled_base_tracking=True)
                            )
                        )
                        moving += info.get("reference_motion_ticks", 0)
                        if done or truncated:
                            break
                    row.update(
                        success=bool(info["primitive_success"]),
                        admitted=True,
                        code=info["code"],
                        duration_s=float(e.steps * e.profile.dt),
                        reference_motion_fraction=moving / e.steps,
                        trajectory_spread=initial["trajectory_spread"],
                        metrics={
                            k: info[k]
                            for k in (
                                "unsafe_contact",
                                "wheel_lift",
                                "invalid_physics",
                                "ee_position_error",
                                "ee_orientation_error",
                                "base_goal_error",
                                "base_heading_error_rad",
                                "max_joint_velocity",
                                "base_linear_velocity_m_s",
                                "base_yaw_velocity_rad_s",
                                "settled_seconds",
                                "peak_contact_force",
                                "peak_base_linear_jerk_m_s3",
                                "peak_base_yaw_jerk_rad_s3",
                            )
                        },
                    )
                except ValueError as exc:
                    row.update(code=str(exc), metrics={"invalid_physics": bool(env.unwrapped._physics_invalid)})
                row["stop_confirmed"] = stop(env)
                result[family].append(row)
                print("GENERAL", family, case["id"], row["success"], row["code"], flush=True)
        finally:
            env.close()
    return result
