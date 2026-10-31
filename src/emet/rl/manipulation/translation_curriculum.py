"""Private FK sampling of reachable translations with unchanged hand attitude.

The angular Jacobian selects directions in normalized joint space. Every finite
candidate is checked by FK; the local linearization is not a reachability or
collision certificate. Live state is never changed. Ordinary path admission
must still check the returned joint witness before training or execution.
"""

import mujoco
import numpy as np


def sample_translation_reference(control, rng, *, attempts=64):
    """Return a 5--20 mm FK witness and its debugging record, or reject.

    Scaling by native joint speeds gives prismatic and revolute coordinates
    meaningful relative units. This supports both Mars and Stretch couplings.
    Sampling adds no observation fields. Pose policies retain their normal
    measured-history and pose-target/preview interface, without a Jacobian hint.
    """
    if type(attempts) is not int or not 1 <= attempts <= 256:
        raise ValueError("translation sampling attempts must be in [1, 256]")
    m = control.model
    scratch = mujoco.MjData(m)
    initial_qpos = np.asarray(control.data.qpos).copy()
    if not np.isfinite(initial_qpos).all():
        raise ValueError("translation sampling requires finite initial positions")
    scratch.qpos[:] = initial_qpos
    mujoco.mj_forward(m, scratch)
    start = np.asarray(scratch.xpos[control.ee]).copy()
    rotation = np.asarray(scratch.xmat[control.ee]).reshape(3, 3).copy()
    q0 = control.positions().copy()
    speed = np.asarray(control.profile.speed)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, scratch, jp, jr, control.ee)
    angular = np.column_stack([jr[:, group].mean(axis=1) for group in control.v]) * speed
    _, singular, right = np.linalg.svd(angular, full_matrices=True)
    rank = int(np.sum(singular > max(1e-8, singular.max(initial=0) * 1e-6)))
    null = right[rank:].T
    if not null.shape[1]:
        raise ValueError("no local orientation-preserving joint directions")
    for attempt in range(attempts):
        direction = null @ rng.normal(size=null.shape[1])
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        q = q0 + speed * direction / norm * rng.uniform(0.03, 0.25)
        if np.any(q < control.limits[:, 0] + 0.02) or np.any(q > control.limits[:, 1] - 0.02):
            continue
        scratch.qpos[:] = initial_qpos
        for group, value in zip(control.q, q, strict=True):
            scratch.qpos[group] = value / len(group)
        mujoco.mj_kinematics(m, scratch)
        goal = np.asarray(scratch.xpos[control.ee]).copy()
        goal_rotation = np.asarray(scratch.xmat[control.ee]).reshape(3, 3).copy()
        distance = float(np.linalg.norm(goal - start))
        angle = float(np.arccos(np.clip((np.trace(rotation.T @ goal_rotation) - 1) / 2, -1, 1)))
        if 0.005 <= distance <= 0.020 and angle <= 0.002:
            return q, {
                "format": "orientation_preserving_fk_translation_v1",
                "initial_hand": np.r_[start, rotation[:, :2].ravel()].tolist(),
                "goal_hand": np.r_[goal, goal_rotation[:, :2].ravel()].tolist(),
                "translation_world_m": (goal - start).tolist(),
                "distance_m": distance,
                "orientation_change_rad": angle,
                "joint_witness": q.tolist(),
                "sampling_attempts": attempt + 1,
                "collision_and_dynamic_admission": "required_separately",
            }
    raise ValueError("no finite reachable translation within the sampling bounds")


def main():
    """Export fresh Mars reset witnesses and nominal pose paths for inspection."""
    import argparse
    import json
    from dataclasses import asdict
    from pathlib import Path

    from emet.rl.manipulation.carry_benchmark import source_hashes
    from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
    from emet.rl.manipulation.tracking import TaskPoseTrajectory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2960000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 128:
        parser.error("samples must be between 1 and 128")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = ManipulationConfig(
        robot="innate_mars",
        scenario="reach",
        action_mode="direct_velocity",
        trajectory_preview=True,
        reach_local_probability=1,
        reach_local_translation_probability=1,
        hold_pregrasp_fixture=True,
        balanced_grasp_setup=True,
        support_half_width=0.04,
        contact_cone="elliptic",
        contact_impratio=3,
        randomization_scale=0,
        certify_reach_paths=True,
        dynamic_references=True,
        horizon=6,
        hand_position_tolerance=0.003,
        hand_orientation_tolerance=0.02,
        whole_body_settling=True,
        wheel_feedback_gain=0.5,
        fixed_base_contract=True,
    )
    env = ManipulationEnv(cfg)
    report = {
        "config": asdict(cfg),
        "fingerprint": env.fingerprint,
        "source_hashes": source_hashes(),
        "cases": [],
        "scope": "FK and sampled geometric/derivative admission; dynamic execution not evaluated",
        "safety_promoted": False,
    }
    try:
        for seed in range(args.seed, args.seed + args.samples):
            _, info = env.reset(seed=seed)
            row = {
                "seed": seed,
                "accepted": not info["initialization_failed"],
                "failure_reason": info["initialization_failure_reason"],
                "sample": env.reach_local_translation,
                "admission": env.trajectory_admission,
                "initial_qpos": env.data.qpos.copy().tolist(),
            }
            if row["accepted"] and env.arm_trajectory is not None:
                path = env.arm_trajectory
                times = [*np.arange(0, path.end, env.profile.dt), path.end]
                base = env.control.base_goal.copy()
                hand = [env._fk_hand_pose(base, path.sample(t)[0]) for t in times]
                row["pose_trajectory"] = TaskPoseTrajectory(
                    times,
                    np.tile(base, (len(times), 1)),
                    hand,
                    np.full(len(times), env.control.grip_reference),
                    base_mask=np.ones(3, dtype=bool),
                ).record()
                row["joint_trajectory"] = path.record()
            report["cases"].append(row)
    finally:
        env.close()
    (args.output / "targets.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    accepted = [r for r in report["cases"] if r["accepted"]]
    if accepted:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        delta = 1000 * np.asarray([r["sample"]["translation_world_m"] for r in accepted])
        fig = plt.figure(figsize=(7, 6))
        axis = fig.add_subplot(projection="3d")
        axis.scatter(*delta.T, c=delta[:, 2], cmap="coolwarm")
        axis.scatter(0, 0, 0, marker="x", c="black")
        limit = 1.1 * float(np.max(abs(delta)))
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
        axis.set_box_aspect((1, 1, 1))
        axis.set(xlabel="World x offset (mm)", ylabel="World y offset (mm)", zlabel="World z offset (mm)")
        axis.set_title(f"Fixed-orientation translations: {len(accepted)}/{args.samples} admitted\nFK/geometry only")
        fig.tight_layout()
        fig.savefig(args.output / "translation_targets.png", dpi=160)
        plt.close(fig)
    print(f"Admitted {len(accepted)}/{args.samples}; targets and nominal paths: {args.output}", flush=True)


if __name__ == "__main__":
    main()
