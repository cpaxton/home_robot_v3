"""Task-frame reference sampling and observable, timestamped trajectory preview."""

import mujoco
import numpy as np

PREVIEW_OFFSETS = (-0.05, 0.0, 0.05, 0.10, 0.20, 0.50, 1.0)
PREVIEW_WIDTH = 18
PRECISION_STAGES = ((0.15, 0.8), (0.06, 0.35), (0.025, 0.15), (0.01, 0.08), (0.005, 0.05))


def tracking_score(position_error, orientation_error, level):
    """One coupled pose score: precision in only one component is insufficient."""
    sigma_position, sigma_orientation = PRECISION_STAGES[level]
    return float(np.exp(-position_error / sigma_position - orientation_error / sigma_orientation))


def precision_base_score(outside_region, heading_error, position_tolerance, heading_tolerance):
    """Coarse capture plus fine contract-scale yaw credit at every control tick."""
    coarse = np.exp(-outside_region / max(0.05, position_tolerance) - abs(heading_error) / max(0.1, heading_tolerance))
    fine = np.exp(-outside_region / position_tolerance - abs(heading_error) / heading_tolerance)
    return float(0.25 * coarse + 0.75 * fine)


def base_progress_potential(outside_region, heading_error):
    return -float(np.log1p(outside_region / 0.02) + np.log1p(abs(heading_error) / 0.05))


def normalized_jerk_cost(joint_jerk, base_jerk, joint_scales, base_scales):
    """Smooth robust cost in explicit experimental rad/s^3 and m/s^3 scales."""
    values = np.r_[np.asarray(joint_jerk) / joint_scales, np.asarray(base_jerk) / base_scales]
    return float(np.sum(np.sqrt(1 + values**2) - 1))


def whole_body_quiet_score(joint_velocity, base_velocity):
    """Dense soft quiet credit; no waiting for the binary completion bonus."""
    motion = np.sum((np.asarray(joint_velocity) / 0.03) ** 2)
    motion += np.sum((np.asarray(base_velocity) / np.array([0.005] * 3 + [0.02] * 3)) ** 2)
    return float(np.exp(-motion))


def command_motion_cost(action, previous, previous_acceleration, dt, speeds, acceleration_scales, jerk_scales, mask):
    """Penalize requested motion before motor smoothing, in physical time/units.

    Inputs exclude the gripper's position-offset channel. Inactive arm/wheel
    channels cannot be charged for commands the executor would ignore. This is
    a training objective; it never modifies or delays an action.
    """
    acceleration = ((np.asarray(action) - previous) * speeds / dt) * mask
    jerk = (acceleration - np.asarray(previous_acceleration) * mask) / dt
    scaled = np.r_[acceleration / acceleration_scales, jerk / jerk_scales]
    cost = float(np.sum(np.sqrt(1 + scaled**2) - 1))
    return cost, acceleration


def differential_base_score(outside_region, final_heading_error, travel_bearing, position_scale):
    """Reward travel alignment far away and final heading near the goal region.

    Forward and reverse travel are equally valid; a differential base cannot
    translate sideways while maintaining its final heading.
    """
    travel_weight = float(np.clip(outside_region / 0.05, 0, 1))
    alignment = 0.2 + 0.8 * np.cos(travel_bearing) ** 2
    heading = np.exp(-abs(final_heading_error) / 0.2)
    return float(np.exp(-outside_region / position_scale) * (travel_weight * alignment + (1 - travel_weight) * heading))


def stationary_tracking_score(position_error, orientation_error, joint_velocity, base_velocity, *, hand_active):
    """Smooth incentive to hold an accurate final pose, without a dead zone.

    Call only once the timed reference has finished. Precision remains fixed
    at the final contract scale even during the coarse tracking curriculum.
    """
    pose = tracking_score(position_error, orientation_error, 4) if hand_active else 1.0
    arm_motion = np.sum((np.asarray(joint_velocity) / 0.03) ** 2) if hand_active else 0.0
    base_motion = np.sum((np.asarray(base_velocity) / np.array([0.005, 0.02])) ** 2)
    return float(pose * np.exp(-arm_motion - base_motion))


def endpoint_motion_cost(elapsed, joint_velocity, base_velocity, *, hand_active, ramp_seconds=2.0):
    """Bounded measured-motion cost, ramping after the scheduled reference ends."""
    ramp = float(np.clip(elapsed / ramp_seconds, 0, 1))
    base = float(np.sum((np.asarray(base_velocity) / [0.005, 0.02]) ** 2))
    arm = float(np.sum((np.asarray(joint_velocity) / 0.03) ** 2)) if hand_active else 0.0
    return ramp * (base / (1 + base) + arm / (1 + arm))


def body_motion_cost(velocity):
    """Bounded cost of all six measured base DOFs at the stop-check scales."""
    values = np.asarray(velocity, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("body motion cost requires six finite velocities")
    motion = float(np.sum((values / np.array([0.005] * 3 + [0.02] * 3)) ** 2))
    return motion / (1 + motion)


def preview_contract():
    return {
        "format": "task_pose_preview_v1",
        "joint_reference_hint": False,
        "inactive_target_values": "zeroed",
        "gripper_schedule": "preview_only_executor_authorizes_release",
        "mask_semantics": "inactive_base_target_allows_base_motion_for_active_hand",
        "offsets_seconds": list(PREVIEW_OFFSETS),
        "columns": [
            "time_offset",
            "base_error_in_base_frame:3",
            "base_mask:3",
            "hand_error_in_hand_frame_position_rotation6:9",
            "hand_mask",
            "grip_error",
        ],
    }


class TaskFrameFK:
    """Private FK anchored to the reset task frame, independent of live shakes.

    This generates nominal targets, never commands or edits live robot state.
    Reuse one scratch data object and run kinematics only for preview sampling.
    """

    def __init__(self, model, data, control):
        self.model, self.control = model, control
        self.data = mujoco.MjData(model)
        self.qpos = data.qpos.copy()
        self.base = control.base_pose().copy()
        self.base_address = int(model.jnt_qposadr[model.body_jntadr[control.base]])

    def __call__(self, base, joints):
        c, d, m = self.control, self.data, self.model
        d.qpos[:] = self.qpos
        a = self.base_address
        if c.profile.wheel_bodies:
            d.qpos[a : a + 2] += np.asarray(base)[:2] - self.base[:2]
            yaw = base[2] - self.base[2]
            quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            mujoco.mju_mulQuat(d.qpos[a + 3 : a + 7], quat, self.qpos[a + 3 : a + 7])
        else:
            for name, value in zip(("base_x", "base_y", "base_yaw"), np.asarray(base) - self.base, strict=True):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    d.qpos[m.jnt_qposadr[jid]] += value
        for group, value in zip(c.q, joints, strict=True):
            d.qpos[group] = value / len(group)
        mujoco.mj_kinematics(m, d)
        return np.r_[d.xpos[c.ee], d.xmat[c.ee].reshape(3, 3)[:, :2].ravel()]


def encode_preview(control, sampler=None):
    """Sample only announced targets; no contact, payload, or disturbance inputs.

    A runtime caller can set control.preview_sampler(time) -> (base, hand).
    Without a timed command, the current pose command is a stationary hold.
    Masks and the executor-authorized gripper reference apply to every row.
    """
    base_pose, hand_pose = control.base_pose(), control.ee_pose()
    yaw = base_pose[2]
    rotation2 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    columns = hand_pose[3:].reshape(3, 2)
    hand_rotation = np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])
    rows = []
    for offset in PREVIEW_OFFSETS:
        sample = sampler(float(control.data.time) + offset) if sampler else (control.base_goal, control.hand_goal)
        base, hand = sample[:2]
        delta = np.asarray(base) - base_pose
        delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
        delta *= control.base_mask
        delta[:2] = rotation2.T @ delta[:2]
        target_columns = np.asarray(hand)[3:].reshape(3, 2)
        target_rotation = np.column_stack([target_columns, np.cross(target_columns[:, 0], target_columns[:, 1])])
        hand_error = np.r_[
            hand_rotation.T @ (np.asarray(hand)[:3] - hand_pose[:3]), (hand_rotation.T @ target_rotation)[:, :2].ravel()
        ] * float(control.hand_mask)
        grip = (
            sample[2]
            if len(sample) == 3
            else control.profile.gripper_open
            if control.release_authorized
            else control.grip_reference
        )
        rows.append(
            np.r_[
                offset,
                delta,
                control.base_mask,
                hand_error,
                float(control.hand_mask),
                grip - control.data.qpos[control.grip_q],
            ]
        )
    return np.asarray(rows, dtype=np.float32)


class TaskPoseTrajectory:
    """Timed task-frame base/hand/gripper commands for Mars and Stretch.

    This first runtime adapter supports their single hand and coupled gripper.
    Base yaw follows the shortest angular path; hand orientation uses SLERP.
    Gripper targets are preview information, not authorization to release.
    """

    def __init__(self, times, base, hand, gripper, *, base_mask, hand_mask=True):
        from scipy.spatial.transform import Rotation, Slerp

        self.times = np.asarray(times, dtype=float).copy()
        self.base = np.asarray(base, dtype=float).copy()
        self.hand = np.asarray(hand, dtype=float).copy()
        self.gripper = np.asarray(gripper, dtype=float).copy()
        self.base_mask = np.asarray(base_mask, dtype=bool).copy()
        self.hand_mask = bool(hand_mask)
        if self.times.ndim != 1:
            raise ValueError("trajectory times must be a vector")
        n = len(self.times)
        if (
            self.times.shape != (n,)
            or n < 2
            or self.times[0] != 0
            or np.any(np.diff(self.times) <= 0)
            or self.base.shape != (n, 3)
            or self.hand.shape != (n, 9)
            or self.gripper.shape != (n,)
            or self.base_mask.shape != (3,)
            or not (self.base_mask.any() or self.hand_mask)
            or not all(np.isfinite(x).all() for x in (self.times, self.base, self.hand, self.gripper))
        ):
            raise ValueError("invalid task-space trajectory")
        columns = self.hand[:, 3:].reshape(n, 3, 2)
        rotations = np.concatenate([columns, np.cross(columns[:, :, 0], columns[:, :, 1])[:, :, None]], axis=2)
        if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4):
            raise ValueError("hand rotations must be orthonormal")
        self.base[:, 2] = np.unwrap(self.base[:, 2])
        self.rotation = Slerp(self.times, Rotation.from_matrix(rotations))
        self.end = float(self.times[-1])

    def sample(self, time):
        t = float(np.clip(time, 0, self.end))
        base = np.array([np.interp(t, self.times, self.base[:, k]) for k in range(3)])
        base[2] = np.arctan2(np.sin(base[2]), np.cos(base[2]))
        hand = np.r_[
            [np.interp(t, self.times, self.hand[:, k]) for k in range(3)], self.rotation(t).as_matrix()[:, :2].ravel()
        ]
        return base, hand, float(np.interp(t, self.times, self.gripper))

    def record(self):
        return {
            "format": "task_pose_trajectory_v1",
            "times": self.times.tolist(),
            "base": self.base.tolist(),
            "hand": self.hand.tolist(),
            "gripper": self.gripper.tolist(),
            "base_mask": self.base_mask.tolist(),
            "hand_mask": self.hand_mask,
        }


def plot_references(report, output):
    """Visualize the actual emitted TAMP hand/base reference distribution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 5))
    hand_axes = figure.add_subplot(121, projection="3d")
    base_axes = figure.add_subplot(122)
    count = 0
    for case in report["cases"]:
        record = case.get("task_pose_trajectory")
        if record is None:
            continue
        points = np.asarray(record["hand"])
        base = np.asarray(record["base"])
        label = f"{case['scenario']} seed {case['seed']}"
        line = hand_axes.plot(*points[:, :3].T, label=label)[0]
        hand_axes.scatter(*points[0, :3], marker="o", color=line.get_color())
        hand_axes.scatter(*points[-1, :3], marker="x", color=line.get_color())
        for index in (0, len(points) - 1):
            columns = points[index, 3:].reshape(3, 2)
            rotation = np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])
            for axis, color in enumerate(("r", "g", "b")):
                hand_axes.quiver(*points[index, :3], *rotation[:, axis], length=0.005, color=color)
        base_axes.plot(base[:, 0], base[:, 1], marker=".", label=label)
        count += 1
    if not count:
        raise ValueError("report has no emitted task pose trajectories")
    hand_axes.set(
        xlabel="world x (m)", ylabel="world y (m)", zlabel="world z (m)", title="Hand references: start o / goal x"
    )
    base_axes.set(xlabel="world x (m)", ylabel="world y (m)", title="Base references (stationary holds)")
    base_axes.set_aspect("equal", adjustable="datalim")
    base_axes.margins(0.5)
    hand_axes.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Plot emitted TAMP task pose trajectories")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_references(json.loads(args.report.read_text()), args.output)
