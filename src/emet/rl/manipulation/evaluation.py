"""Stratified physical outcomes, preserving rejected initialization and failures."""

from dataclasses import replace

import numpy as np

from emet.rl.manipulation.env import SCENARIOS, ManipulationConfig, ManipulationEnv


def evaluate(policy=None, *, config=None, episodes=20, seed=200000, scenarios=SCENARIOS):
    config = config or ManipulationConfig()
    if episodes < 1:
        raise ValueError("episodes must be positive")
    rows = []
    for scenario_index, scenario in enumerate(scenarios):
        env = ManipulationEnv(replace(config, scenario=scenario))
        if policy is not None and (policy.robot != config.robot or policy.manifest["fingerprint"] != env.fingerprint):
            raise ValueError("evaluation policy/model mismatch")
        try:
            for index in range(episodes):
                episode_seed = seed + scenario_index * 10000 + index
                obs, info = env.reset(seed=episode_seed)
                initial = {
                    "reference": env.control.reference.tolist(),
                    "positions": env.control.positions().tolist(),
                    "grip_reference": env.control.grip_reference,
                    "actor_history": obs["history"].tolist(),
                }
                for _ in range(round(config.horizon / env.profile.dt) + 1):
                    action = np.zeros(env.action_space.shape) if policy is None else policy(obs)
                    obs, _, done, truncated, info = env.step(action)
                    if done or truncated:
                        break
                rows.append(
                    {
                        "seed": episode_seed,
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
            "initialization_failure_rate": float(np.mean([r["initialization_failed"] for r in group])),
            "initialization_rejections": sum(r["initialization_rejections"] for r in group),
            "success_rate": float(np.mean([r["is_success"] for r in group])),
        }
        for key in ("retention_lost", "unsafe_contact", "wheel_lift"):
            values = [r[key] for r in actual if r.get(key) is not None]
            metrics[key + "_rate"] = float(np.mean(values)) if values else None
        for key in (
            "peak_contact_force",
            "contact_impulse",
            "peak_measured_jerk",
            "joint_error",
            "max_payload_rotation",
            "max_payload_translation",
        ):
            values = [r[key] for r in actual]
            metrics[key + "_mean"] = float(np.mean(values)) if values else None
            metrics[key + "_p95"] = float(np.percentile(values, 95)) if values else None
            metrics[key + "_max"] = max(values) if values else None
        successful = [r["time"] for r in actual if r["is_success"]]
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
