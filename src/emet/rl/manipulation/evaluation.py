"""Stratified physical outcomes, preserving rejected initialization and failures."""

from collections import Counter
from dataclasses import replace

import numpy as np

from emet.rl.manipulation.env import SCENARIOS, ManipulationConfig, ManipulationEnv


def evaluate(policy=None, *, config=None, episodes=20, seed=200000, scenarios=None):
    config = config or (
        ManipulationConfig(**policy.manifest["environment"]) if policy is not None else ManipulationConfig()
    )
    if policy is not None:
        settings = policy.manifest.get("environment", {})
        config = replace(
            config,
            action_mode=policy.action_mode,
            trajectory_preview="preview" in policy.normalization,
            tracking_reward=settings.get("tracking_reward", config.tracking_reward),
            hand_position_tolerance=min(
                config.hand_position_tolerance, settings.get("hand_position_tolerance", config.hand_position_tolerance)
            ),
            hand_orientation_tolerance=min(
                config.hand_orientation_tolerance,
                settings.get("hand_orientation_tolerance", config.hand_orientation_tolerance),
            ),
        )
    scenarios = scenarios or (SCENARIOS if config.scenario == "mixed" else (config.scenario,))
    if episodes < 1:
        raise ValueError("episodes must be positive")
    rows = []
    for scenario_index, scenario in enumerate(scenarios):
        env = ManipulationEnv(replace(config, scenario=scenario, disturbance_curriculum=False))
        if policy is not None and (policy.robot != config.robot or policy.manifest["fingerprint"] != env.fingerprint):
            raise ValueError("evaluation policy/model mismatch")
        try:
            for index in range(episodes):
                episode_seed = seed + scenario_index * 10000 + index
                obs, info = env.reset(seed=episode_seed)
                initial_base_distance = float(np.linalg.norm(env.control.base_goal[:2] - env.control.base_pose()[:2]))
                initial = {
                    "base_pose": env.control.base_pose().tolist(),
                    "base_goal": env.control.base_goal.tolist(),
                    "base_mask": env.control.base_mask.tolist(),
                    "task_contract": {
                        "base_xy_tolerance": config.base_xy_tolerance,
                        "base_heading_tolerance": config.base_heading_tolerance,
                        "hand_position_tolerance": config.hand_position_tolerance,
                        "hand_orientation_tolerance": config.hand_orientation_tolerance,
                        "support_half_width": config.support_half_width,
                        "grasp_lift_height": config.grasp_lift_height,
                        "grasp_depth": config.grasp_depth,
                    },
                    "reference": env.control.reference.tolist(),
                    "positions": env.control.positions().tolist(),
                    "grip_reference": env.control.grip_reference,
                    "trajectory": None if env.arm_trajectory is None else env.arm_trajectory.record(),
                    "rejected_candidates": env.trajectory_rejections,
                    "actor_history": obs["history"].tolist(),
                    "actor_preview": obs["preview"].tolist() if "preview" in obs else None,
                    "scheduled_retarget": None
                    if env.retarget_goal is None
                    else {
                        "step": env.retarget_step,
                        "base": env.retarget_goal[0].tolist(),
                        "arm": env.retarget_goal[1].tolist(),
                        "hand": env.retarget_goal[2].tolist(),
                    },
                    "scheduled_push": {
                        "step": env.disturbance_start,
                        "steps": env.disturbance_steps,
                        "wrench": env.disturbance_wrench.tolist(),
                    },
                }
                for _ in range(round(config.horizon / env.profile.dt) + 1):
                    action = env.teacher_action() if policy is None else policy(obs)
                    obs, _, done, truncated, info = env.step(action)
                    if done or truncated:
                        break
                rows.append(
                    {
                        "seed": episode_seed,
                        "initial_base_distance_m": initial_base_distance,
                        **info,
                        "initial": initial,
                        "episode_id": f"{config.robot}:{scenario}:{episode_seed}",
                        "fingerprint": env.fingerprint,
                        "controller_id": "feedback-v1" if policy is None else policy.manifest["weights_sha256"],
                    }
                )
        finally:
            env.close()
    return {
        "robot": config.robot,
        "episodes": rows,
        "summary": summarize(rows),
        "scope": "simulation only; randomized curriculum, not frozen TAMP acceptance",
    }


def summarize(rows):
    result = {}
    for scenario in sorted({r["scenario"] for r in rows}):
        group = [r for r in rows if r["scenario"] == scenario]
        actual = [r for r in group if not r["initialization_failed"]]
        metrics = {
            "episodes": len(group),
            "admitted_reference_count": sum(r.get("trajectory_admission", {}).get("accepted") is True for r in group),
            "rejected_reference_count": sum(r.get("rejected_trajectories", 0) for r in group),
            "initialization_failure_rate": float(np.mean([r["initialization_failed"] for r in group])),
            "initialization_rejections": sum(r["initialization_rejections"] for r in group),
            "success_rate": float(np.mean([r["is_success"] for r in group])),
            "admission_rate": len(actual) / len(group),
            "conditional_execution_success_rate": (
                float(np.mean([r["is_success"] for r in actual])) if actual else None
            ),
            "candidate_rejections_by_reason": dict(
                Counter(
                    candidate["reason"]
                    for row in group
                    for candidate in row.get("initial", {}).get("rejected_candidates", [])
                )
            ),
        }
        settled_once = [r for r in actual if r.get("time_to_first_settle") is not None]
        metrics["ever_settled_rate"] = len(settled_once) / len(group)
        metrics["settled_then_lost_count"] = sum(not r["is_success"] for r in settled_once)
        # Endpoint diagnostics distinguish tracking failures from a robot that
        # reaches the pose but never stops. These do not alter success gates.
        diagnostic_rows = [r for r in actual if "config" in r and "base_linear_velocity_m_s" in r]
        if diagnostic_rows:
            hand_within, base_still = [], []
            hand_rows = [r for r in diagnostic_rows if r.get("hand_target_active", True)]
            base_rows = [r for r in diagnostic_rows if r.get("base_target_active", True)]
            metrics["hand_target_count"] = len(hand_rows)
            metrics["base_target_count"] = len(base_rows)
            for row in diagnostic_rows:
                cfg = row["config"]
                if row.get("hand_target_active", True):
                    hand_within.append(
                        row["ee_position_error"] < cfg["hand_position_tolerance"]
                        and row["ee_orientation_error"] < cfg["hand_orientation_tolerance"]
                    )
                base_still.append(
                    abs(row["base_linear_velocity_m_s"]) < 0.005 and abs(row["base_yaw_velocity_rad_s"]) < 0.02
                )
            metrics["endpoint_hand_within_rate"] = float(np.mean(hand_within)) if hand_within else None
            metrics["endpoint_base_still_rate"] = float(np.mean(base_still))
            metrics["endpoint_base_moving_count"] = sum(not s for s in base_still)
            metrics["endpoint_hand_within_but_base_moving_rate"] = (
                float(
                    np.mean(
                        [
                            h
                            and (
                                abs(r["base_linear_velocity_m_s"]) >= 0.005 or abs(r["base_yaw_velocity_rad_s"]) >= 0.02
                            )
                            for h, r in zip(hand_within, hand_rows, strict=True)
                        ]
                    )
                )
                if hand_rows
                else None
            )
            for name, rows_for_reason, predicate in (
                (
                    "hand_position_outside",
                    hand_rows,
                    lambda r: r["ee_position_error"] >= r["config"]["hand_position_tolerance"],
                ),
                (
                    "hand_orientation_outside",
                    hand_rows,
                    lambda r: r["ee_orientation_error"] >= r["config"]["hand_orientation_tolerance"],
                ),
                (
                    "base_xy_outside",
                    [r for r in base_rows if "base_contract_violation" in r],
                    lambda r: r["base_contract_violation"] > 0,
                ),
                (
                    "base_heading_outside",
                    [r for r in base_rows if "base_heading_error_rad" in r],
                    lambda r: r["base_heading_error_rad"] >= r["config"].get("base_heading_tolerance", 0.04),
                ),
                (
                    "arm_moving",
                    [r for r in hand_rows if "max_joint_velocity" in r],
                    lambda r: r["max_joint_velocity"] >= 0.03,
                ),
            ):
                count = sum(bool(predicate(r)) for r in rows_for_reason)
                metrics[f"endpoint_{name}_count"] = count if rows_for_reason else None
                metrics[f"endpoint_{name}_rate"] = count / len(rows_for_reason) if rows_for_reason else None
            # Diagnostic tolerance bands do not replace the declared contract.
            for mm, angle in ((5, 0.05), (10, 0.1), (20, 0.1)):
                metrics[f"endpoint_hand_within_{mm}mm_{angle:g}rad_rate"] = (
                    float(
                        np.mean(
                            [
                                r["ee_position_error"] < mm / 1000 and r["ee_orientation_error"] < angle
                                for r in hand_rows
                            ]
                        )
                    )
                    if hand_rows
                    else None
                )
        for reason in ("tracking_timeout", "wheel_lift", "unsafe_contact", "retention_lost"):
            metrics[f"terminal_{reason}_count"] = sum(r.get("code") == reason for r in group)
        for key in ("retention_lost", "unsafe_contact", "wheel_lift"):
            values = [r[key] for r in actual if r.get(key) is not None]
            metrics[key + "_rate"] = float(np.mean(values)) if values else None
        for key in (
            "initial_base_distance_m",
            "peak_base_linear_jerk_m_s3",
            "peak_base_yaw_jerk_rad_s3",
            "peak_contact_force",
            "contact_impulse",
            "peak_measured_jerk",
            "joint_error",
            "ee_position_error",
            "peak_hand_position_error",
            "peak_hand_orientation_error",
            "rms_hand_position_error",
            "peak_hold_base_drift_m",
            "ee_orientation_error",
            "base_goal_error",
            "base_contract_violation",
            "base_heading_error_rad",
            "max_joint_velocity",
            "goal_age_seconds",
            "max_payload_rotation",
            "max_payload_translation",
        ):
            measured = actual
            if key in {
                "joint_error",
                "ee_position_error",
                "ee_orientation_error",
                "peak_hand_position_error",
                "peak_hand_orientation_error",
                "rms_hand_position_error",
                "max_joint_velocity",
            }:
                measured = [r for r in actual if r.get("hand_target_active", True)]
            if key in {"base_goal_error", "base_contract_violation", "base_heading_error_rad"}:
                measured = [r for r in actual if r.get("base_target_active", True)]
            if key == "peak_hold_base_drift_m":
                measured = [r for r in actual if r.get("time_to_first_settle") is not None]
            values = [r[key] for r in measured if key in r]
            metrics[key + "_mean"] = float(np.mean(values)) if values else None
            metrics[key + "_p95"] = float(np.percentile(values, 95)) if values else None
            metrics[key + "_max"] = max(values) if values else None
        successful = [r.get("time_to_first_settle") or r["time"] for r in actual if r["is_success"]]
        metrics["successful_time"] = float(np.mean(successful)) if successful else None
        result[scenario] = metrics
    return result


def score(summary):
    groups = list(summary.values())
    # Worst category first; a strong reaching category cannot hide unsafe carry.
    return (
        -max((g["wheel_lift_rate"] if g["wheel_lift_rate"] is not None else 1) for g in groups),
        -max((g["unsafe_contact_rate"] if g["unsafe_contact_rate"] is not None else 1) for g in groups),
        -max(g["initialization_failure_rate"] for g in groups),
        min(g["success_rate"] for g in groups),
        float(np.mean([g["success_rate"] for g in groups])),
        -max(g["peak_contact_force_p95"] or 0 for g in groups),
        -max(g["peak_measured_jerk_p95"] or 0 for g in groups),
    )


def curriculum_ready(summary, episodes):
    """Require stratified, sufficiently sampled success without safety failures."""
    return (
        bool(summary)
        and episodes >= 20
        and all(
            g["success_rate"] >= 0.9
            and g["initialization_failure_rate"] == 0
            and all(g.get(k) == 0 for k in ("unsafe_contact_rate", "wheel_lift_rate", "retention_lost_rate"))
            for g in summary.values()
        )
    )


def tracking_precision_ready(summary, level, episodes):
    """Tighten reward precision using measured accuracy, not elapsed training."""
    from emet.rl.manipulation.tracking import PRECISION_STAGES

    if not summary or episodes < 20:
        return False
    position, orientation = PRECISION_STAGES[level]
    for group in summary.values():
        if group.get("initialization_failure_rate") != 0 or any(
            group.get(key) != 0 for key in ("unsafe_contact_rate", "wheel_lift_rate", "retention_lost_rate")
        ):
            return False
        for key, limit in (
            ("ee_position_error_p95", position),
            ("ee_orientation_error_p95", orientation),
            ("base_contract_violation_p95", position),
        ):
            value = group.get(key)
            if value is None or not np.isfinite(value) or value > limit:
                return False
    return True


def target_coverage_ready(summary, episodes):
    """Expand reach coverage at coarse pose accuracy, without relaxing acceptance.

    Requiring final 5 mm settled success before exposing larger motions can
    strand training on tiny targets. Dynamics randomization and checkpoint
    promotion deliberately retain their separate, stricter gates.
    """
    return set(summary) == {"reach"} and tracking_precision_ready(summary, 2, episodes)


def paired_comparison(teacher, learned):
    """Gate speed comparisons on identical cases and no measured regression.

    This is a checkpoint-selection gate; it is not hardware promotion evidence.
    """
    reasons = []

    def indexed(report):
        rows = report["episodes"]
        result = {r["episode_id"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate paired episode identity")
        return result

    baseline, candidate = indexed(teacher), indexed(learned)
    if set(baseline) != set(candidate) or not baseline or teacher.get("fingerprint") != learned.get("fingerprint"):
        # Older reports store fingerprints per episode, which are checked below.
        reasons.append("missing or mismatched cases")
    common = []
    for identity in baseline.keys() & candidate.keys():
        a, b = baseline[identity], candidate[identity]
        if a["fingerprint"] != b["fingerprint"] or a["initial"] != b["initial"]:
            reasons.append(f"{identity}: different initial state or nominal reference")
        for key in ("initialization_failed", "unsafe_contact", "wheel_lift", "retention_lost"):
            if b.get(key) is not False:
                reasons.append(f"{identity}: {key} not false")
        if a["is_success"] and not b["is_success"]:
            reasons.append(f"{identity}: lost teacher success")
        if a["is_success"] and b["is_success"]:
            common.append((a.get("time_to_first_settle") or a["time"], b.get("time_to_first_settle") or b["time"]))
    if set(teacher["summary"]) != set(learned["summary"]):
        reasons.append("different scenario categories")
    for scenario in teacher["summary"].keys() & learned["summary"].keys():
        a, b = teacher["summary"][scenario], learned["summary"][scenario]
        if b["success_rate"] < a["success_rate"]:
            reasons.append(f"{scenario}: lower success")
        # New reports compare base constraint violation, allowing different XY
        # endpoints inside the declared task region. Historical reports keep
        # their original strict base-error comparison.
        base_metric = (
            "base_contract_violation_p95"
            if all(g.get("base_contract_violation_p95") is not None for g in (a, b))
            else "base_goal_error_p95"
        )
        hand_active = not all(g.get("hand_target_count") == 0 for g in (a, b))
        base_active = not all(g.get("base_target_count") == 0 for g in (a, b))
        for key in (
            *(("joint_error_mean",) if hand_active and base_metric == "base_goal_error_p95" else ()),
            *(("ee_position_error_p95", "ee_orientation_error_p95") if hand_active else ()),
            *((base_metric,) if base_active else ()),
            "peak_contact_force_p95",
            "peak_contact_force_max",
            "contact_impulse_p95",
            "peak_measured_jerk_p95",
            "peak_measured_jerk_max",
        ):
            if (
                a.get(key) is None
                or b.get(key) is None
                or not np.isfinite([a[key], b[key]]).all()
                or b[key] > a[key] + 1e-8
            ):
                reasons.append(f"{scenario}: {key} worse or unsupported")
    if not common:
        reasons.append("no common successful cases")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "paired_success_count": len(common),
        "teacher_paired_time": float(np.mean([a for a, _ in common])) if common else None,
        "policy_paired_time": float(np.mean([b for _, b in common])) if common else None,
    }
