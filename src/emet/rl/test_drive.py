"""Local, simulation-only policy test drive with an interactive MuJoCo camera.

Run: MUJOCO_GL=egl python -m emet.rl.test_drive --port 8765
No robot client, transport, connection profile, or hardware command endpoint.
"""

import argparse
import io
import json
import os
import queue
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

MANIPULATION_MODES = {"stretch-arm", "mars-arm"}
MODES = ("stretch-arm", "mars-base", "mars-coordinated", "mars-arm")
SCENARIOS = {
    "stretch-arm": ("reach", "pregrasp", "carry", "place", "disturbance"),
    "mars-base": ("standard", "retarget", "push", "contact_retreat"),
    "mars-coordinated": ("reach",),
    "mars-arm": ("reach",),
}


def checkpoints(roots, mode):
    result = {}
    if mode in MANIPULATION_MODES:
        from emet.rl.manipulation.profiles import PROFILES

        robot = "innate_mars" if mode == "mars-arm" else "stretch"
    if mode in {"mars-base", "mars-coordinated"}:
        from emet.rl.base import BASE, BASE_CONTRACT
        from emet.rl.model import model_fingerprint

        base_profile = json.loads(json.dumps(BASE.metadata()))
        base_contract = json.loads(json.dumps(BASE_CONTRACT))
        fingerprint = model_fingerprint()
    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            continue
        for manifest in root.rglob("manifest.json"):
            try:
                data = json.loads(manifest.read_text())
                arm = data.get("format") == "emet-manipulation-v1" and data.get("robot") == (
                    "innate_mars" if mode == "mars-arm" else "stretch"
                )
                base = data.get("robot") == "innate_mars" and data.get("task") == "base"
                coordinated = data.get("robot") == "innate_mars" and data.get("task") == "coordinated"
                if not (
                    (mode in MANIPULATION_MODES and arm)
                    or (mode == "mars-base" and base)
                    or (mode == "mars-coordinated" and coordinated)
                ):
                    continue
                if mode in MANIPULATION_MODES:
                    action_mode = data.get("action_mode", "residual")
                    if action_mode not in {"residual", "direct_velocity"}:
                        continue
                    expected = json.loads(json.dumps(PROFILES[robot].contract(action_mode)))
                    if data.get("contract") != expected:
                        continue
                if mode == "mars-coordinated":
                    from emet.rl.profile import MARS

                    if (
                        data.get("profile") != json.loads(json.dumps(MARS.metadata()))
                        or data.get("model_fingerprint") != fingerprint
                        or data.get("target_contract") != "masked_base_se2_hand_se3_v1"
                    ):
                        continue
                if mode == "mars-base" and (
                    data.get("base_contract") != base_contract
                    or data.get("profile") != base_profile
                    or data.get("model_fingerprint") != fingerprint
                ):
                    continue
                path = manifest.parent.resolve()
                if not path.is_relative_to(root) or not (path / "policy.zip").exists():
                    continue
                result[str(path)] = {
                    "id": str(path),
                    "label": str(path.relative_to(root)),
                    "root": root.name,
                    "steps": data.get("steps"),
                    "reward": data.get("training_reward", "earlier/diagnostic"),
                }
            except (OSError, ValueError):
                continue  # An export may still be in progress; retry on refresh.
    return list(result.values())


def training_status(roots):
    rows = []
    for root in roots:
        for path in sorted(Path(root).glob("seed-*/run.json")):
            try:
                run = json.loads(path.read_text())
                metrics_file = path.parent / "metrics.jsonl"
                metrics = {}
                if metrics_file.exists():
                    with metrics_file.open("rb") as stream:
                        stream.seek(max(0, metrics_file.stat().st_size - 65536))
                        lines = stream.read().splitlines()
                    for line in reversed(lines):
                        try:
                            metrics = json.loads(line)
                            break
                        except ValueError:
                            continue
                tracking = path.parent / "wandb-run.json"
                steps = int(metrics.get("timesteps", 0))
                initial = int(run["initialization_steps"])
                rows.append(
                    {
                        "seed": run["seed"],
                        "steps": steps,
                        "ppo_steps": max(0, steps - initial),
                        "total": initial + int(run["steps"]),
                        "phase": "complete"
                        if (path.parent / "complete.json").exists()
                        else "PPO"
                        if steps > initial
                        else "initializing",
                        "url": json.loads(tracking.read_text()).get("url") if tracking.exists() else None,
                    }
                )
            except (OSError, ValueError, KeyError):
                continue
    return rows


class Simulation:
    """Owns only a local Gym/MuJoCo environment, driven by one worker thread."""

    def __init__(self, mode="stretch-arm", scenario=None, seed=600001):
        if mode not in MODES:
            raise ValueError("only local simulation modes are supported")
        scenario = scenario or SCENARIOS[mode][0]
        if scenario not in SCENARIOS[mode]:
            raise ValueError("invalid simulation scenario")
        if not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2^31)")
        self.mode, self.scenario, self.seed = mode, scenario, seed
        self.control_mode = 3
        if mode in MANIPULATION_MODES:
            from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv

            self.env = ManipulationEnv(
                ManipulationConfig(
                    robot="innate_mars" if mode == "mars-arm" else "stretch", scenario=scenario, horizon=30
                )
            )
        elif mode == "mars-base":
            from emet.rl.base import MarsBaseEnv
            from emet.rl.env import EnvConfig

            self.env = MarsBaseEnv(
                EnvConfig(task="base", contact_training=True, base_scenario=scenario, horizon_seconds=30)
            )
        else:
            from emet.rl.env import EnvConfig, MarsReachEnv

            self.env = MarsReachEnv(EnvConfig(task="coordinated", pose_targets=True, horizon_seconds=30))
        self.obs, self.info = self.env.reset(seed=seed)
        if mode in MANIPULATION_MODES:
            self.control_mode = self.env.target_mode
        self.done = False
        self.policy = None
        self.policy_name = "Feedback controller"
        self.policy_scope = "Feedback controller"
        self.push_ticks = 0
        self.interventions = []

    def use_policy(self, candidate, name):
        """Bind input/action contracts on selection and on simulation reset."""
        if self.mode in MANIPULATION_MODES:
            if candidate is None:
                settings = {"action_mode": "residual", "trajectory_preview": False, "tracking_reward": False}
            else:
                if (
                    candidate.robot != self.env.profile.robot
                    or candidate.manifest["fingerprint"] != self.env.fingerprint
                ):
                    raise ValueError("policy/model mismatch")
                source = candidate.manifest.get("environment", {})
                settings = {
                    "action_mode": candidate.action_mode,
                    "trajectory_preview": "preview" in candidate.normalization,
                    "tracking_reward": source.get("tracking_reward", False),
                    **{
                        key: source[key]
                        for key in (
                            "base_xy_tolerance",
                            "base_heading_tolerance",
                            "hand_position_tolerance",
                            "hand_orientation_tolerance",
                        )
                        if key in source
                    },
                }
            self.env.config = replace(self.env.config, **settings)
            self.env._configure_preview()
            self.env.control.history = None  # Contract changes restart inputs, never physics.
            self.env.control.action_mode = self.env.config.action_mode
            self.env.control.base_xy_tolerance = self.env.config.base_xy_tolerance
            self.obs = self.env.control.observe()
        self.policy, self.policy_name = candidate, name
        self.policy_scope = (
            "Feedback controller"
            if candidate is None
            else "Learned policy; interactive goals are not acceptance tests."
        )
        path = Path(name)
        if candidate is not None and path.is_absolute():
            for parent in (path, path.parent, path.parent.parent):
                run = parent / "run.json"
                if run.is_file():
                    try:
                        metadata = json.loads(run.read_text())
                        if metadata.get("training_options", {}).get("debug_reset_seed") is not None:
                            self.policy_scope = (
                                "Fixed-case diagnostic policy: distant goals and base-only control are not validated."
                            )
                    except (OSError, ValueError):
                        pass
                    break

    def load_policy(self, path, roots):
        if path == "feedback":
            self.use_policy(None, "Feedback controller")
            return
        allowed = {row["id"] for row in checkpoints(roots, self.mode)}
        if path not in allowed:
            raise ValueError("select a complete checkpoint from the configured local roots")
        if self.mode in MANIPULATION_MODES:
            from emet.rl.manipulation.policy import ManipulationPolicy

            candidate = ManipulationPolicy(path, robot=self.env.profile.robot, fingerprint=self.env.fingerprint)
        else:
            from emet.rl.policy import PPOPolicy

            candidate = PPOPolicy(path)
            expected = "base" if self.mode == "mars-base" else "coordinated"
            if candidate.task != expected:
                raise ValueError("policy task does not match this simulation mode")
        self.use_policy(candidate, path)

    def select_target_mode(self, selected, *, sampled=False):
        """Select the controlled target, retaining that choice across resets."""
        if self.mode not in {"mars-coordinated", *MANIPULATION_MODES} or selected not in {1, 2, 3}:
            raise ValueError("target mode must be 1, 2, or 3 for pose control")
        self.control_mode = selected
        if self.mode in MANIPULATION_MODES:
            c = self.env.control
            reference = c.reference.copy() if sampled and selected != 1 else c.positions()
            base = c.base_goal.copy() if sampled and selected != 2 else c.base_pose()
            hand = self.env._fk_hand_pose(base, reference) if sampled else c.ee_pose()
            c.set_reference(reference, phase=c.phase)
            base_active = selected != 3 or self.policy is None
            c.set_pose_targets(base, np.full(3, base_active), hand, selected != 1)
            self.env.target_mode = selected
            self.env.settle_time = 0
            self.env._previous_cost = self.env._tracking_cost()
            self.obs = c.observe()
        else:
            from emet.rl.control import Goal

            c = self.env.control
            base = c.goal.base.copy() if sampled and selected != 2 else c.base_pose()
            arm = c.goal.arm.copy() if sampled and selected != 1 else c.arm_positions().copy()
            self.env.set_goal(Goal(base, arm, np.r_[selected != 2, np.full(5, selected != 1)]))
            self.obs = self.env._with_pose_targets(c.observe()[0])

    def resume_completed_reach(self):
        """Rearm a successful/timed-out UI reach without resetting physics."""
        if not self.done:
            return True
        if self.mode not in MANIPULATION_MODES or self.env.scenario != "reach":
            return False
        if not (self.info.get("is_success") or self.info.get("code") == "tracking_timeout"):
            return False
        env = self.env
        if (
            env.initialization_failed
            or env.monitor.wheel_lift
            or env._physics_invalid
            or env.monitor.peak_force > env.profile.unsafe_force
            or env.retention_lost
        ):
            return False
        self.done = env._done = False
        env.control.stopped = False
        env.steps = 0
        env.start_time = float(env.data.time)
        env.settle_time = 0
        env._previous_cost = env._tracking_cost()
        self.obs = env.control.observe()
        return True

    def goal(self, values):
        if self.done:
            raise ValueError("reset the finished episode before changing its goal")
        if self.mode in MANIPULATION_MODES and self.control_mode == 1:
            target = np.asarray(values, dtype=float)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError("base target must contain x, y, yaw")
            control = self.env.control
            control.set_reference(control.positions(), phase=control.phase)
            control.set_pose_targets(target, [True, True, True], control.ee_pose(), False)
            self.env._previous_cost = self.env._tracking_cost()
            self.obs = control.observe()
            self.interventions.append({"time": float(self.env.data.time), "goal": target.tolist()})
            return
        if self.mode in MANIPULATION_MODES and self.control_mode in (2, 3):
            pose = np.asarray(values, dtype=float)
            if pose.shape != (9,) or not np.isfinite(pose).all():
                raise ValueError("end-effector target must contain xyz and two rotation columns")
            control = self.env.control
            if self.policy is None:
                base_goal, arm_goal = self._solve_manipulation_pose(pose, base_locked=self.control_mode == 2)
                control.set_reference(arm_goal, phase=control.phase)
            else:
                columns = pose[3:].reshape(3, 2)
                if not np.allclose(columns.T @ columns, np.eye(2), atol=1e-4):
                    raise ValueError("hand target requires orthonormal rotation columns")
                base_goal = control.base_goal.copy()
            control.set_pose_targets(
                base_goal,
                np.full(3, self.control_mode == 2 or self.policy is None),
                pose,
                True,
            )
            self.env._previous_cost = self.env._tracking_cost()
            self.obs = control.observe()
            self.interventions.append({"time": float(self.env.data.time), "goal": pose.tolist()})
            return
        if self.mode == "mars-coordinated":
            target = np.asarray(values, dtype=float)
            if self.control_mode == 1:
                if target.shape != (3,) or not np.isfinite(target).all():
                    raise ValueError("base target must contain x, y, yaw")
                self.env.set_goal({"base": target.tolist()})
            else:
                if target.shape != (9,) or not np.isfinite(target).all():
                    raise ValueError("end-effector target must contain xyz and two rotation columns")
                self.env.set_goal({"ee_pose": target.tolist(), "base_locked": self.control_mode == 2})
            self.obs = self.env._with_pose_targets(self.env.control.observe()[0])
            self.interventions.append({"time": float(self.env.data.time), "goal": target.tolist()})
            return
        target = np.asarray(values, dtype=float)
        if target.shape != ((5,) if self.mode in MANIPULATION_MODES else (3,)) or not np.isfinite(target).all():
            raise ValueError("invalid goal")
        if self.mode in MANIPULATION_MODES:
            self.env.control.set_reference(target, phase=self.env.control.phase)
            self.env._previous_cost = float(np.linalg.norm(target - self.env.control.positions()))
            self.env.settle_time = 0
            self.obs = self.env.control.observe()
        else:
            if np.any(abs(target[:2]) > 2) or abs(target[2]) > np.pi:
                raise ValueError("interactive base goals are limited to ±2 m and ±pi rad")
            self.env.set_goal({"base": target.tolist()})
            self.obs, _ = self.env.control.observe()
        self.interventions.append({"time": float(self.env.data.time), "goal": target.tolist()})

    def _solve_manipulation_pose(self, pose, *, base_locked):
        """Prefer a local drive-feasible adjustment; reject blocked IK branches."""
        failure = None
        for allow_lateral in (False,) if base_locked else (False, True):
            try:
                return self._solve_manipulation_pose_candidate(
                    pose, base_locked=base_locked, allow_lateral=allow_lateral
                )
            except ValueError as exc:
                failure = exc
        raise ValueError(f"end-effector pose is unreachable or blocked under the selected base constraint: {failure}")

    def _solve_manipulation_pose_candidate(self, pose, *, base_locked, allow_lateral):
        """Solve in private state, then validate sampled robot/scene clearance."""
        control, model = self.env.control, self.env.model
        position = np.asarray(pose[:3], dtype=float)
        columns = np.asarray(pose[3:], dtype=float).reshape(3, 2)
        first = columns[:, 0] / max(np.linalg.norm(columns[:, 0]), 1e-8)
        second = columns[:, 1] - first * np.dot(first, columns[:, 1])
        second /= max(np.linalg.norm(second), 1e-8)
        rotation_goal = np.column_stack([first, second, np.cross(first, second)])
        if np.linalg.det(rotation_goal) < 0.999:
            raise ValueError("invalid end-effector rotation")
        scratch = mujoco.MjData(model)
        mujoco.mj_copyData(scratch, model, self.env.data)
        arm_indices = np.concatenate(control.v)
        # Use the base's actual instantaneous controls (forward and yaw),
        # rather than a holonomic sideways joint the wheeled robot lacks.
        base_size = 0 if base_locked else (3 if allow_lateral else 2)
        size = len(arm_indices) + base_size
        basis = np.zeros((model.nv, size))
        weights = np.ones(len(arm_indices))
        if control.profile.robot == "innate_mars" and not base_locked:
            # Prefer moving the chassis over folding the shoulder/elbow into
            # its hull. This is a local posture preference, not an IK label.
            weights[1:4] = 0.08
        basis[arm_indices, np.arange(len(arm_indices)) + base_size] = weights
        for _ in range(300):
            mujoco.mj_forward(model, scratch)
            current = scratch.xpos[control.ee].copy()
            rotation = scratch.xmat[control.ee].reshape(3, 3).copy()
            error = np.r_[
                position - current,
                0.5 * sum((np.cross(rotation[:, i], rotation_goal[:, i]) for i in range(3)), start=np.zeros(3)),
            ]
            if np.linalg.norm(error[:3]) < 0.004 and np.linalg.norm(error[3:]) < 0.03:
                break
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, scratch, jacp, jacr, control.ee)
            if not base_locked:
                heading = scratch.xmat[control.base].reshape(3, 3)[:, 0]
                basis[control.base_dof : control.base_dof + 2, 0] = heading[:2]
                if allow_lateral:
                    basis[control.base_dof : control.base_dof + 2, 1] = [-heading[1], heading[0]]
                basis[control.base_dof + 5, base_size - 1] = 1
            jac = np.vstack([jacp, jacr]) @ basis
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-4, error)
            delta = np.clip(delta, -0.02, 0.02)
            full_delta = basis @ delta
            old_arm = np.array([scratch.qpos[group].sum() for group in control.q])
            mujoco.mj_integratePos(model, scratch.qpos, full_delta, 1.0)
            values = old_arm + np.array([full_delta[index].sum() for index in control.v])
            for group, value, limits in zip(control.q, values, control.limits, strict=True):
                scratch.qpos[group] = np.clip(value, *limits) / len(group)
        mujoco.mj_forward(model, scratch)
        final_position = np.linalg.norm(position - scratch.xpos[control.ee])
        final_rotation = scratch.xmat[control.ee].reshape(3, 3)
        angle = np.arccos(np.clip((np.trace(rotation_goal.T @ final_rotation) - 1) / 2, -1, 1))
        if final_position > 0.004 or angle > 0.035:
            raise ValueError("end-effector pose is unreachable under the selected base constraint")
        from emet.eval.physical_tamp import make_scene_checker

        checker = make_scene_checker(
            model, {"object_body": self.env.payload_name, "gripper_bodies": control.profile.finger_bodies}
        )
        end = scratch.qpos.copy()
        velocity = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, velocity, 1.0, self.env.data.qpos, end)
        samples = max(2, int(np.ceil(np.max(np.abs(velocity)) / 0.04)) + 1)
        for fraction in np.linspace(0, 1, samples):
            scratch.qpos[:] = self.env.data.qpos
            mujoco.mj_integratePos(model, scratch.qpos, velocity, float(fraction))
            if checker.configuration_collides(model, scratch):
                raise ValueError("sampled end-effector path is in collision")
        scratch.qpos[:] = end
        mujoco.mj_forward(model, scratch)
        base_rotation = scratch.xmat[control.base].reshape(3, 3)
        base_goal = np.r_[scratch.xpos[control.base, :2], np.arctan2(base_rotation[1, 0], base_rotation[0, 0])]
        arm_goal = np.array([scratch.qpos[group].sum() for group in control.q])
        if base_locked:
            base_goal = control.base_pose()
        return base_goal, arm_goal

    def nudge_goal(self, translation, rotation_z=0.0):
        """Move the active Cartesian goal a small amount in the rendered view plane."""
        delta = np.asarray(translation, dtype=float)
        if (
            delta.shape != (3,)
            or not np.isfinite(delta).all()
            or not np.isfinite(rotation_z)
            or np.linalg.norm(delta) > 0.08
            or abs(rotation_z) > 0.2
        ):
            raise ValueError("invalid target movement")
        if self.done:
            raise ValueError("reset the finished episode before changing its goal")
        if self.mode == "mars-base":
            target = self.env.control.goal.base.copy()
            target[:2] += delta[:2]
            target[2] = (target[2] + rotation_z + np.pi) % (2 * np.pi) - np.pi
            self.goal(target)
            return
        if self.mode in MANIPULATION_MODES and self.control_mode == 1:
            target = self.env.control.base_goal.copy()
            target[:2] += delta[:2]
            target[2] = (target[2] + rotation_z + np.pi) % (2 * np.pi) - np.pi
            self.goal(target)
            return
        if self.mode == "mars-coordinated" and self.control_mode == 1:
            target = self.env.control.goal.base.copy()
            target[:2] += delta[:2]
            target[2] = (target[2] + rotation_z + np.pi) % (2 * np.pi) - np.pi
            self.goal(target)
            return
        if self.mode == "mars-coordinated":
            pose = self.env.control.goal_pose[:9].copy()
            pose[:3] += delta
            rotation = pose[3:].reshape(3, 2)
            angle = float(rotation_z)
            rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            pose[3:] = (rz @ rotation).ravel()
            self.goal(pose)
            return
        if self.mode in MANIPULATION_MODES and self.control_mode in (2, 3):
            pose = self.env.control.hand_goal.copy()
            pose[:3] += delta
            rotation = pose[3:].reshape(3, 2)
            angle = float(rotation_z)
            rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            pose[3:] = (rz @ rotation).ravel()
            self.goal(pose)
            return
        control, model = self.env.control, self.env.model
        scratch = mujoco.MjData(model)
        mujoco.mj_copyData(scratch, model, self.env.data)
        for group, value in zip(control.q, control.reference, strict=True):
            scratch.qpos[group] = value / len(group)
        mujoco.mj_forward(model, scratch)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, scratch, jacp, jacr, control.ee)
        jac = np.column_stack(
            [np.r_[jacp[:, group].sum(axis=1), jacr[:, group].sum(axis=1)] / len(group) for group in control.v]
        )
        twist = np.r_[delta, [0.0, 0.0, rotation_z]]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-4, twist)
        dq = np.clip(dq, -0.025, 0.025)
        self.goal(np.clip(control.reference + dq, control.limits[:, 0], control.limits[:, 1]))

    def push(self):
        if self.done:
            raise ValueError("reset the finished episode first")
        self.push_ticks = 5
        self.interventions.append(
            {"time": float(self.env.data.time), "disturbance": "positive world X force for 0.25 s"}
        )

    def step(self):
        if self.done:
            return
        env = self.env
        body = (
            (env.control.ee if self.mode == "mars-arm" else env.payload)
            if self.mode in MANIPULATION_MODES
            else env.control.base_body
        )
        if self.mode in MANIPULATION_MODES:
            env.control.action_mode = self.policy.action_mode if self.policy else "residual"
        env.data.xfrc_applied[body, 0] = (1.0 if self.mode in MANIPULATION_MODES else 3.0) if self.push_ticks else 0
        self.push_ticks = max(0, self.push_ticks - 1)
        if self.mode == "mars-coordinated":
            from emet.rl.control import baseline_action

        action = (
            self.policy(self.obs)
            if self.policy
            else (
                np.zeros(env.action_space.shape)
                if self.mode in MANIPULATION_MODES
                else env.expert_action()
                if self.mode == "mars-base"
                else baseline_action(env.control)
            )
        )
        self.obs, reward, done, truncated, self.info = env.step(action)
        self.done = bool(done or truncated)
        self.info["reward"] = reward
        if self.done or not self.push_ticks:
            env.data.xfrc_applied[body, 0] = 0

    def state(self):
        env = self.env
        if self.mode in MANIPULATION_MODES and self.control_mode == 1:
            current, target = env.control.base_pose().tolist(), env.control.base_goal.tolist()
            limits = [[-2, 2], [-2, 2], [-np.pi, np.pi]]
            fields = ["Base X (m)", "Base Y (m)", "Base yaw (rad)"]
        elif self.mode in MANIPULATION_MODES:
            current, target = env.control.ee_pose().tolist(), env.control.hand_goal.tolist()
            limits = [[-2, 2], [-2, 2], [0, 2]] + [[-1, 1]] * 6
            fields = ["EE X (m)", "EE Y (m)", "EE Z (m)", "R00", "R01", "R10", "R11", "R20", "R21"]
        elif self.mode == "mars-base":
            current, target = env.control.base_pose().tolist(), env.control.goal.base.tolist()
            limits = [[-2, 2], [-2, 2], [-np.pi, np.pi]]
            fields = ["X (m)", "Y (m)", "Yaw (rad)"]
        elif self.mode == "mars-coordinated" and self.control_mode == 1:
            current, target = env.control.base_pose().tolist(), env.control.goal.base.tolist()
            limits = [[-2, 2], [-2, 2], [-np.pi, np.pi]]
            fields = ["Base X (m)", "Base Y (m)", "Base yaw (rad)"]
        elif self.mode == "mars-coordinated":
            current, target = env.control.pose_features()[:9].tolist(), env.control.goal_pose[:9].tolist()
            limits = [[-2, 2], [-2, 2], [0, 2]] + [[-1, 1]] * 6
            fields = ["EE X (m)", "EE Y (m)", "EE Z (m)", "R00", "R01", "R10", "R11", "R20", "R21"]
        keys = (
            "is_success",
            "code",
            "time",
            "joint_error",
            "ee_position_error",
            "ee_orientation_error",
            "base_goal_error",
            "base_heading_error_rad",
            "xy_error",
            "yaw_error",
            "peak_contact_force",
            "peak_measured_jerk",
            "retention_lost",
            "max_payload_translation",
            "max_payload_rotation",
            "wheel_lift",
            "min_wheel_load",
            "max_wheel_gap",
            "unsafe_contact",
            "reward",
            "initialization_failed",
        )
        metrics = {key: self.info[key] for key in keys if key in self.info}
        return {
            "mode": self.mode,
            "control_mode": self.control_mode,
            "scenario": self.scenario,
            "seed": self.seed,
            "done": self.done,
            "simulation_only": True,
            "policy": self.policy_name,
            "holding": bool(getattr(getattr(self.env, "control", None), "holding", False)),
            "policy_scope": self.policy_scope,
            "fields": fields,
            "current": current,
            "goal": target,
            "limits": limits,
            "metrics": metrics,
            "interventions": len(self.interventions),
            "wheel_support": "evaluated in simulator"
            if self.mode in MANIPULATION_MODES
            else "unsupported: planar Mars model",
            "evaluation_scope": "interactive experiment, not acceptance evidence",
        }

    def close(self):
        self.env.close()


class TestDrive:
    __test__ = False

    def __init__(self, roots, mode):
        self.roots = roots
        self.commands = queue.Queue(maxsize=32)
        self.lock = threading.Lock()
        self.snapshot = {"status": "initializing", "simulation_only": True, "mode": mode}
        self.frame = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(mode,), daemon=True)
        self.thread.start()

    def _run(self, mode):
        sim = renderer = None
        running, repeat, single_steps, error = True, False, 0, None
        last_episode = None
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation, camera.distance = 135, -25, (1.2 if mode == "mars-arm" else 2.8)
        option = mujoco.MjvOption()
        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        def replace_sim(new_mode, scenario, seed):
            nonlocal sim, renderer
            replacement = Simulation(new_mode, scenario, seed)
            if new_mode == sim.mode:
                replacement.use_policy(sim.policy, sim.policy_name)
                if new_mode in {"mars-coordinated", *MANIPULATION_MODES}:
                    replacement.select_target_mode(sim.control_mode, sampled=True)
            if new_mode != sim.mode:
                camera.distance = 1.2 if new_mode == "mars-arm" else 2.8
            sim.close()
            sim = replacement
            if renderer:
                renderer.close()
                renderer = None
            camera.lookat[:] = sim.env.data.body("base_link").xpos + [0, 0, 0.7 if sim.mode == "stretch-arm" else 0.25]

        def reset_if_finished():
            nonlocal last_episode
            if sim.done:
                last_episode = {"seed": sim.seed, "scenario": sim.scenario, "metrics": sim.state()["metrics"]}
                with self.lock:
                    self.snapshot["status"] = "resetting"
                if not sim.resume_completed_reach():
                    raise ValueError("reset after a safety fault or completed manipulation episode")

        try:
            sim = Simulation(mode)
            while not self.closed.is_set():
                started = time.monotonic()
                simulation_started = float(sim.env.data.time)
                try:
                    command = self.commands.get_nowait()
                except queue.Empty:
                    command = None
                if command:
                    try:
                        op = command["op"]
                        if op == "reset":
                            running, repeat, single_steps = True, False, 0
                            new_mode = command.get("mode", sim.mode)
                            with self.lock:
                                self.snapshot["status"] = "resetting"
                            replace_sim(new_mode, command.get("scenario"), command.get("seed", sim.seed))
                        elif op == "run":
                            repeat = bool(command.get("repeat", False))
                            if sim.done:
                                with self.lock:
                                    self.snapshot["status"] = "resetting"
                                replace_sim(sim.mode, sim.scenario, sim.seed)
                            running = True
                        elif op == "pause":
                            running, repeat, single_steps = False, False, 0
                        elif op == "step":
                            reset_if_finished()
                            single_steps, running, repeat = 1, False, False
                        elif op == "goal":
                            reset_if_finished()
                            sim.goal(command["values"])
                            running = True
                        elif op == "nudge":
                            reset_if_finished()
                            sim.nudge_goal(command["translation"], command.get("rotation_z", 0.0))
                            running = True
                        elif op == "push":
                            reset_if_finished()
                            sim.push()
                        elif op == "hold":
                            if sim.mode not in MANIPULATION_MODES or not sim.resume_completed_reach():
                                raise ValueError("hold requires a nonfaulted arm simulation")
                            sim.env.control.hold_position()
                            sim.obs = sim.env.control.observe()
                            running, repeat = True, False
                        elif op == "recenter":
                            if sim.mode not in MANIPULATION_MODES or not sim.resume_completed_reach():
                                raise ValueError("recenter requires a nonfaulted arm simulation")
                            sim.select_target_mode(sim.control_mode)
                            sim.env.control.hold_position()
                            sim.obs = sim.env.control.observe()
                            running, repeat = True, False
                        elif op == "policy":
                            sim.load_policy(command["id"], self.roots)
                        elif op == "target_mode":
                            reset_if_finished()
                            sim.select_target_mode(int(command["mode"]))
                        elif op == "camera":
                            values = np.asarray(command["values"], dtype=float)
                            if values.shape != (3,) or not np.isfinite(values).all():
                                raise ValueError("invalid camera")
                            camera.azimuth, camera.elevation, camera.distance = (
                                values[0] % 360,
                                np.clip(values[1], -85, -5),
                                np.clip(values[2], 0.4, 6),
                            )
                        else:
                            raise ValueError("unknown simulation command")
                        error = None
                    except (ValueError, KeyError, TypeError, OSError) as exc:
                        running = False
                        error = str(exc)
                if (running or single_steps) and not sim.done:
                    sim.step()
                    single_steps = max(0, single_steps - 1)
                if sim.done:
                    last_episode = {
                        "seed": sim.seed,
                        "scenario": sim.scenario,
                        "metrics": sim.state()["metrics"],
                    }
                    if repeat:
                        with self.lock:
                            self.snapshot["status"] = "resetting"
                        replace_sim(sim.mode, sim.scenario, (sim.seed + 1) % (2**31))
                        running = True
                    else:
                        running = False
                if renderer is None:
                    sim.env.model.vis.global_.offwidth = 960
                    sim.env.model.vis.global_.offheight = 640
                    renderer = mujoco.Renderer(sim.env.model, height=540, width=960)
                    sim.env.model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
                    camera.lookat[:] = sim.env.data.body("base_link").xpos + [
                        0,
                        0,
                        0.7 if sim.mode == "stretch-arm" else 0.25,
                    ]
                renderer.update_scene(sim.env.data, camera=camera, scene_option=option)
                state = sim.state()
                # Goal markers are visualization only and never collision geometry.
                if (
                    sim.mode == "mars-base"
                    or (sim.mode == "mars-coordinated" and sim.control_mode == 1)
                    or (sim.mode in MANIPULATION_MODES and sim.control_mode == 1)
                ):
                    point = [*state["goal"][:2], 0.04]
                    heading = [np.cos(state["goal"][2]), np.sin(state["goal"][2]), 0]
                    arrow_length = 0.22
                    marker_kind = "base"
                elif sim.mode == "mars-coordinated":
                    scratch = mujoco.MjData(sim.env.model)
                    mujoco.mj_copyData(scratch, sim.env.model, sim.env.data)
                    for qadr, value in zip(sim.env.control.arm_q, sim.env.control.goal.arm, strict=True):
                        scratch.qpos[qadr] = value
                    axes = sim.env.data.xaxis[sim.env.control.base_ids]
                    delta = sim.env.control.goal.base[:2] - sim.env.control.base_pose()[:2]
                    scratch.qpos[sim.env.control.base_q[:2]] += np.linalg.solve(axes[:2, :2].T, delta)
                    scratch.qpos[sim.env.control.base_q[2]] += (
                        sim.env.control.goal.base[2] - sim.env.control.base_pose()[2]
                    ) / axes[2, 2]
                    mujoco.mj_forward(sim.env.model, scratch)
                    point = scratch.xpos[sim.env.control.ee_body].copy()
                    heading = scratch.xmat[sim.env.control.ee_body].reshape(3, 3)[:, 0]
                    arrow_length = 0.10
                    marker_kind = "end_effector"
                elif sim.mode in MANIPULATION_MODES:
                    columns = sim.env.control.hand_goal[3:].reshape(3, 2)
                    point = sim.env.control.hand_goal[:3].copy()
                    heading = columns[:, 0]
                    arrow_length = 0.10
                    marker_kind = "end_effector"
                else:
                    scratch = mujoco.MjData(sim.env.model)
                    mujoco.mj_copyData(scratch, sim.env.model, sim.env.data)
                    for group, value in zip(sim.env.control.q, state["goal"], strict=True):
                        scratch.qpos[group] = value / len(group)
                    mujoco.mj_kinematics(sim.env.model, scratch)
                    point = scratch.xpos[sim.env.control.ee]
                    heading = scratch.xmat[sim.env.control.ee].reshape(3, 3)[:, 0]
                    arrow_length = 0.10
                    marker_kind = "end_effector"
                scene = renderer.scene
                if scene.ngeom < scene.maxgeom:
                    color = (
                        np.array([0.2, 0.65, 1.0, 0.9]) if marker_kind == "base" else np.array([0.2, 1.0, 0.55, 0.9])
                    )
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.018] * 3),
                        np.asarray(point),
                        np.eye(3).ravel(),
                        color,
                    )
                    scene.ngeom += 1
                if scene.ngeom < scene.maxgeom:
                    arrow = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(
                        arrow,
                        mujoco.mjtGeom.mjGEOM_ARROW,
                        np.array([0.01, 0.02, 0.0]),
                        np.asarray(point),
                        np.eye(3).ravel(),
                        color,
                    )
                    mujoco.mjv_connector(
                        arrow,
                        mujoco.mjtGeom.mjGEOM_ARROW,
                        0.012,
                        np.asarray(point),
                        np.asarray(point) + np.asarray(heading) * arrow_length,
                    )
                    scene.ngeom += 1
                cam = scene.camera[0]
                forward, up = np.asarray(cam.forward, dtype=float), np.asarray(cam.up, dtype=float)
                right = np.cross(forward, up)
                right /= max(np.linalg.norm(right), 1e-9)
                up /= max(np.linalg.norm(up), 1e-9)
                rel = np.asarray(point) - np.asarray(cam.pos)
                depth = float(rel @ forward)
                near = max(float(cam.frustum_near), 1e-6)
                half_height = max((float(cam.frustum_top) - float(cam.frustum_bottom)) * depth / near / 2, 1e-6)
                # MuJoCo leaves frustum_width at zero for perspective cameras;
                # derive it from the viewport aspect ratio in that case.
                half_width = max(
                    float(cam.frustum_width) * depth / near / 2, half_height * renderer.width / renderer.height
                )
                marker_pixel = (
                    [480 + float(rel @ right) / half_width * 480, 270 - float(rel @ up) / half_height * 270]
                    if depth > near
                    else None
                )
                buffer = io.BytesIO()
                Image.fromarray(renderer.render()).save(buffer, format="JPEG", quality=85)
                with self.lock:
                    self.frame = buffer.getvalue()
                    self.snapshot = {
                        **state,
                        "marker_kind": marker_kind,
                        "marker_pixel": marker_pixel,
                        "marker_right": right.tolist(),
                        "marker_up": up.tolist(),
                        "marker_m_per_pixel": 2 * half_height / 540,
                        "status": "finished" if sim.done else "running" if running else "paused",
                        "error": error,
                        "last_episode": last_episode,
                        "wall_time": time.time(),
                        "simulation_speed": max(0.0, float(sim.env.data.time) - simulation_started)
                        / max(0.05, time.monotonic() - started),
                        "base_speed_limit_m_s": sim.env.profile.base_speed if sim.mode in MANIPULATION_MODES else None,
                        "base_yaw_limit_rad_s": sim.env.profile.base_yaw_speed
                        if sim.mode in MANIPULATION_MODES
                        else None,
                    }
                self.closed.wait(max(0, 0.05 - (time.monotonic() - started)))
        except Exception as exc:
            with self.lock:
                self.snapshot.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            if renderer:
                renderer.close()
            if sim:
                sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", choices=MODES, default="stretch-arm")
    parser.add_argument("--policy-root", type=Path, action="append")
    args = parser.parse_args()
    roots = args.policy_root or [
        Path("logs/manipulation/mars-dynamic-v2"),
        Path("logs/manipulation/mars-corrected-v1"),
        Path("logs/manipulation/physical-v2"),
        Path("logs/mars-base-extended"),
    ]
    app = TestDrive(roots, args.mode)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, value, content_type="application/json", code=200):
            body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            hosts = {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}
            return self.headers.get("Host") in hosts and self.headers.get("Origin") in (
                None,
                *[f"http://{h}" for h in hosts],
            )

        def do_GET(self):
            if not self.allowed():
                self.reply({"error": "local access only"}, code=403)
            elif self.path == "/":
                self.reply(Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
            elif self.path.split("?")[0] == "/frame.jpg":
                with app.lock:
                    frame = app.frame
                self.reply(frame or b"", "image/jpeg", 200 if frame else 503)
            elif self.path == "/api/state":
                with app.lock:
                    snapshot = dict(app.snapshot)
                self.reply(snapshot)
            elif self.path == "/api/training":
                self.reply(training_status(roots))
            elif self.path == "/api/checkpoints":
                with app.lock:
                    mode = app.snapshot["mode"]
                self.reply(checkpoints(roots, mode))
            else:
                self.reply({"error": "not found"}, code=404)

        def do_POST(self):
            if not self.allowed() or self.path != "/api/command":
                self.reply({"error": "local simulation endpoint only"}, code=403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096 or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("small JSON command required")
                command = json.loads(self.rfile.read(size))
                if not isinstance(command, dict) or command.get("op") not in {
                    "reset",
                    "run",
                    "pause",
                    "step",
                    "goal",
                    "nudge",
                    "push",
                    "hold",
                    "recenter",
                    "policy",
                    "camera",
                    "target_mode",
                }:
                    raise ValueError("unsupported simulation command")
                app.commands.put_nowait(command)
                self.reply({"accepted": True}, code=202)
            except (ValueError, queue.Full) as exc:
                self.reply({"error": str(exc)}, code=400)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Simulation-only test drive: http://localhost:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        app.closed.set()
        app.thread.join(timeout=15)
        server.server_close()


if __name__ == "__main__":
    main()
