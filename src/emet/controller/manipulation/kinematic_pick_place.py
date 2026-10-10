# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Kinematic pick-and-place skill graph for registry robots (rby1) in MuJoCo.

Uses MuJoCo position IK + ZMQ joint streaming + sim kinematic attach. Optional collision
filters: voxel-map (2D nav obstacles) or AABB table solids. Not CuRobo / not contact physics.

Note: IK is currently **position-only**; grasp orientation from the oracle is used for
approach standoff but not enforced at the EE.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from emet.motion.aabb_arm_collision import AabbArmCollisionChecker
from emet.motion.arm_manip_profile import (
    ArmManipProfile,
    home_arm_q_array,
    robot_id_from_client,
)
from emet.motion.arm_rrt import plan_arm_joint_path, resolve_agent_manip_planner
from emet.motion.mujoco_arm_ik import pack_arm_into_actuator_dict, solve_position_ik_multiseed
from emet.motion.placement_geometry import PlacementCollisionChecker
from emet.motion.voxel_arm_collision import VoxelMapArmCollisionChecker
from emet.simulation.sim_manipulation import (
    resolve_sim_object_body,
    robot_zmq_attach_body,
    robot_zmq_detach_body,
    robot_zmq_set_body_pose,
)
from emet.utils.logger import Logger

logger = Logger(__name__)

PLACEMENT_STATE_FAILURES = frozenset({
    "placement_stale_observation", "placement_missing_joint_state", "placement_nonfinite_joint_state",
})


@dataclass
class KinematicPickPlaceResult:
    success: bool
    object_body: str | None
    ee_body: str
    grasp_err_m: float | None
    place_err_m: float | None
    message: str


def _targets_from_grasp_T(
    grasp_T_world: np.ndarray,
    *,
    pregrasp_standoff_m: float = 0.12,
    lift_m: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pregrasp_xyz, grasp_xyz, lift_xyz) from a world grasp pose."""
    T = np.asarray(grasp_T_world, dtype=np.float64).reshape(4, 4)
    grasp = T[:3, 3].copy()
    approach = -T[:3, 2]
    n = float(np.linalg.norm(approach))
    if n < 1e-9:
        approach = np.array([0.0, 0.0, 1.0])
    else:
        approach = approach / n
    pregrasp = grasp + approach * float(pregrasp_standoff_m)
    lift = grasp + np.array([0.0, 0.0, float(lift_m)])
    return pregrasp, grasp, lift


def write_offline_mjcf_base_xyt(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    xyt: np.ndarray,
    *,
    planar_joint_names: Sequence[str] | None = None,
    freejoint_name: str | None = None,
    z: float | None = None,
) -> bool:
    """Write world XYT into a standalone (unmerged) robot MJCF base.

    Prefers planar slide/hinge joints when all names resolve, otherwise a 7-DoF
    freejoint. Values are **raw world XYT** — correct for vendored robot MJCFs, not
    Robocasa-merged models (use :func:`emet.simulation.spawn_planar.write_planar_base_xyt`).
    """
    x, y, th = float(xyt[0]), float(xyt[1]), float(xyt[2])
    names = tuple(str(n) for n in planar_joint_names) if planar_joint_names else ()
    if len(names) != 3:
        probe = ("base_x", "base_y", "base_yaw")
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in probe]
        if all(jid >= 0 for jid in ids):
            names = probe
    if len(names) == 3:
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in names]
        if all(jid >= 0 for jid in ids):
            for jid, val in zip(ids, (x, y, th), strict=True):
                data.qpos[int(model.jnt_qposadr[jid])] = float(val)
            return True
    if freejoint_name:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(freejoint_name))
        if jid >= 0:
            qadr = int(model.jnt_qposadr[jid])
            z_use = float(data.qpos[qadr + 2]) if z is None else float(z)
            half = 0.5 * th
            data.qpos[qadr : qadr + 7] = [
                x,
                y,
                z_use,
                float(np.cos(half)),
                0.0,
                0.0,
                float(np.sin(half)),
            ]
            return True
    return False


class KinematicPickPlaceExecutor:
    """Navigate (optional) → IK pregrasp/grasp → attach → lift → place → detach."""

    def __init__(
        self,
        robot: Any,
        *,
        arm: str = "left",
        profile: ArmManipProfile | None = None,
        manip_collision: str = "none",
        manip_planner: str = "rrt_connect",
        voxel_map: Any | None = None,
        traj_dt: float = 0.04,
        traj_steps: int = 15,
        lift_m: float = 0.12,
        place_z_offset_m: float = 0.02,
        rrt_max_iter: int = 400,
        ik_tol_m: float = 0.035,
        ik_max_iters: int = 150,
        pregrasp_standoff_m: float = 0.12,
        place_xy_tol_m: float = 0.12,
        grasp_lift_verify_tol_m: float = 0.08,
        visualizer: Any | None = None,
        placement_geometry_provider: Any | None = None,
    ) -> None:
        self.robot = robot
        self.arm = str(arm).lower()
        if profile is None:
            profile = ArmManipProfile.for_robot(robot_id_from_client(robot), arm=self.arm)
        self.profile = profile
        self.manip_collision = str(manip_collision).lower()
        self.manip_planner = resolve_agent_manip_planner(config_mode=manip_planner)
        self.voxel_map = voxel_map
        self.traj_dt = float(traj_dt)
        self.traj_steps = int(traj_steps)
        self.lift_m = float(lift_m)
        self.place_z_offset_m = float(place_z_offset_m)
        self.rrt_max_iter = int(rrt_max_iter)
        self.ik_tol_m = float(ik_tol_m)
        self.ik_max_iters = int(ik_max_iters)
        self.pregrasp_standoff_m = float(pregrasp_standoff_m)
        self.place_xy_tol_m = float(place_xy_tol_m)
        self.grasp_lift_verify_tol_m = float(grasp_lift_verify_tol_m)
        self.visualizer = visualizer
        self.placement_geometry_provider = placement_geometry_provider
        self.last_placement_search = None
        self.ee_body = self.profile.ee_body
        self.joint_names = self.profile.joint_names
        self.link_bodies = list(self.profile.link_bodies)
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._collision: VoxelMapArmCollisionChecker | AabbArmCollisionChecker | PlacementCollisionChecker | None = None
        self._last_cmd_q: np.ndarray | None = None
        self.last_plan_waypoints: list[np.ndarray] = []
        self.last_ee_path_world: list[np.ndarray] = []
        self.last_targets: dict[str, np.ndarray] = {}

    def _scaled_dt(self, dt: float) -> float:
        from emet.core.zmq_protocol import motion_wait_timeout_scale, read_sim_to_real_ratio

        state = getattr(self.robot, "_state", None)
        ratio = read_sim_to_real_ratio(state) if isinstance(state, dict) else None
        return float(dt) * motion_wait_timeout_scale(ratio)

    def _sleep(self, seconds: float) -> None:
        time.sleep(self._scaled_dt(seconds))

    def _body_pos(self, body: str) -> np.ndarray | None:
        from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

        pl = read_sim_object_placements(self.robot.get_emet_session())
        if not pl or body not in pl:
            return None
        return np.asarray(pl[body]["pos"], dtype=np.float64).reshape(3)

    def _verify_grasp_lift(self, body: str, lift_xyz: np.ndarray, *, pre_pos: np.ndarray | None) -> bool:
        """Accept target proximity only; height gain and threshold are diagnostics."""
        after = self._body_pos(body)
        lift = np.asarray(lift_xyz, dtype=np.float64).reshape(3)
        error = None if after is None else float(np.linalg.norm(after - lift))
        dz = None if after is None or pre_pos is None else float(after[2] - pre_pos[2])
        minimum_lift = max(0.04, 0.4 * self.lift_m)
        accepted = bool(error is not None and error <= self.grasp_lift_verify_tol_m)
        # Height gain alone can accept an object metres from the gripper target.
        # A command step alone is not an
        # acknowledgement that this body pose has been observed after the lift.
        self.last_grasp_verification = {
            "body": body, "target_xyz": lift.tolist(),
            "observed_xyz": None if after is None else after.tolist(),
            "before_xyz": None if pre_pos is None else np.asarray(pre_pos).tolist(),
            "target_error_m": error, "lift_dz_m": dz,
            "target_tolerance_m": self.grasp_lift_verify_tol_m, "minimum_lift_m": minimum_lift,
            "command_step": getattr(self.robot, "_last_step", None),
            "session_step": getattr(self.robot, "_emet_session_cache_step", None),
            "accepted": accepted,
        }
        logger.info("KinematicPickPlace lift verification: " + json.dumps(self.last_grasp_verification))
        return accepted

    def _verify_place_xy(self, body: str, recep_xy: np.ndarray) -> tuple[bool, float]:
        after = self._body_pos(body)
        if after is None:
            return False, float("inf")
        err = float(np.linalg.norm(after[:2] - np.asarray(recep_xy, dtype=np.float64).reshape(2)))
        return err <= self.place_xy_tol_m, err

    def _fk_ee_path(self, waypoints: list[np.ndarray]) -> list[np.ndarray]:
        assert self._model is not None and self._data is not None
        from emet.motion.mujoco_arm_ik import joint_qpos_addrs

        qadr = joint_qpos_addrs(self._model, self.joint_names)
        ee_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body)
        if ee_id < 0:
            return []
        out: list[np.ndarray] = []
        for q in waypoints:
            for a, v in zip(qadr, np.asarray(q, dtype=np.float64).reshape(-1), strict=True):
                self._data.qpos[a] = float(v)
            mujoco.mj_forward(self._model, self._data)
            out.append(np.array(self._data.body(ee_id).xpos, dtype=np.float64).copy())
        return out

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        spec = getattr(self.robot, "_spec", None) or getattr(self.robot, "get_robot_spec", lambda: None)()
        mjcf = getattr(spec, "mjcf_path", None) if spec is not None else None
        if not mjcf:
            logger.error("KinematicPickPlace: robot spec missing mjcf_path")
            return False
        path = Path(str(mjcf))
        if not path.is_file():
            logger.error(f"KinematicPickPlace: MJCF not found: {path}")
            return False
        self._model = mujoco.MjModel.from_xml_path(str(path))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        if self.manip_collision == "voxel" and self.voxel_map is not None:
            self._collision = VoxelMapArmCollisionChecker.from_voxel_map(
                self.voxel_map,
                link_bodies=self.link_bodies,
                inflate_cells=1,
            )
        elif self.manip_collision == "aabb":
            self._collision = AabbArmCollisionChecker.for_default_table(link_bodies=self.link_bodies)
        return True

    def _actuator_names(self) -> list[str]:
        spec = getattr(self.robot, "_spec", None)
        if spec is not None and getattr(spec, "actuator_names", None):
            return list(spec.actuator_names)
        return list(self.profile.actuator_names)

    def _world_base_xyt(self) -> np.ndarray | None:
        """Base ``(x, y, θ)`` in MuJoCo world (not episode-relative GPS).

        ``robot.get_base_pose()`` is episode-relative for robosuite/Molmo; GT placements and
        the offline MJCF freejoint are world-frame. Prefer session ``navigation_origin_xyt``
        composition; fall back to ``base_xyz`` XY + episode yaw only when origin is missing.
        """
        from emet.utils.geometry import nav_xyt_to_world_xyt

        pose = np.asarray(self.robot.get_base_pose(timeout=2.0), dtype=np.float64).reshape(-1)
        if pose.size < 3:
            return None
        sess = None
        get_sess = getattr(self.robot, "get_emet_session", None)
        if callable(get_sess):
            raw = get_sess()
            if isinstance(raw, dict):
                sess = raw
        world = nav_xyt_to_world_xyt(pose[:3], sess)
        state = getattr(self.robot, "_state", None)
        if isinstance(state, dict) and state.get("base_xyz") is not None:
            try:
                xyz = np.asarray(state["base_xyz"], dtype=np.float64).reshape(-1)
                if xyz.size >= 2:
                    world = np.array([float(xyz[0]), float(xyz[1]), float(world[2])], dtype=np.float64)
            except Exception:
                pass
        return world

    def _planar_joint_names(self) -> tuple[str, ...] | None:
        spec = getattr(self.robot, "_spec", None)
        names = getattr(spec, "planar_base_joint_names", None) if spec is not None else None
        if names and len(names) == 3:
            return tuple(str(n) for n in names)
        return None

    def _sync_base_freejoint(self) -> None:
        assert self._model is not None and self._data is not None
        world = self._world_base_xyt()
        if world is None:
            return
        z = None
        state = getattr(self.robot, "_state", None)
        if isinstance(state, dict) and state.get("base_xyz") is not None:
            try:
                z = float(np.asarray(state["base_xyz"], dtype=np.float64).reshape(-1)[2])
            except Exception:
                pass
        write_offline_mjcf_base_xyt(
            self._model,
            self._data,
            world,
            planar_joint_names=self._planar_joint_names(),
            freejoint_name=getattr(self.profile, "base_freejoint_name", None),
            z=z,
        )

    def _actuator_to_joint_name(self, aname: str) -> str | None:
        m = re.match(r"(left|right)_arm(\d+)$", aname)
        if m:
            return f"{m.group(1)}_arm_joint{m.group(2)}"
        m = re.match(r"torso(\d+)$", aname)
        if m:
            return f"torso_joint{m.group(1)}"
        m = re.match(r"(left|right)_gripper(\d+)$", aname)
        if m:
            return f"{m.group(1)}_gripper_finger_joint{m.group(2)}"
        if aname in self.joint_names:
            return aname
        if aname.endswith("_act"):
            stem = aname[:-4]
            if stem in self.joint_names:
                return stem
        return None

    def begin_operation(self, operation_id: str) -> None:
        """Start a new task operator; previous measurements cannot attest to it."""
        self.operation_id = operation_id
        self.last_ee_verification = None
        self.last_grasp_verification = None

    def _sync_qpos_from_robot(self) -> bool:
        """Wait briefly for fresh measured joints; never substitute commanded q.

        Planning can temporarily outpace the state receiver. Waiting gives the
        receiver a chance to recover without relaxing the two-second age limit.
        Invalid/incomplete samples never partially overwrite the offline model.
        """
        assert self._model is not None and self._data is not None
        started = time.monotonic()
        deadline = started + 2.0
        self._last_motion_failure = None
        age = None
        while True:
            received = getattr(self.robot, "_state_received_monotonic", None)
            age = time.monotonic() - received if isinstance(received, (int, float)) else None
            stale = age is not None and (not np.isfinite(age) or age < 0 or age > 2.0)
            if stale:
                reason = "stale_observation"
            else:
                q, _, _ = self.robot.get_joint_state(timeout=max(.001, deadline - time.monotonic()))
                # Recheck after the potentially blocking read, before trusting q.
                received = getattr(self.robot, "_state_received_monotonic", None)
                age = time.monotonic() - received if isinstance(received, (int, float)) else None
                names = self._actuator_names()
                if age is not None and (not np.isfinite(age) or age < 0 or age > 2.0):
                    reason = "stale_observation"
                elif q is None or np.asarray(q).ndim != 1 or len(q) < len(names):
                    reason = "missing_joint_state"
                elif not np.all(np.isfinite(q)):
                    reason = "nonfinite_joint_state"
                else:
                    updates = {}
                    observed_joints = set()
                    for i, aname in enumerate(names):
                        jname = self._actuator_to_joint_name(aname)
                        if not jname:
                            continue
                        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                        if jid >= 0:
                            updates[int(self._model.jnt_qposadr[jid])] = float(q[i])
                            observed_joints.add(jname)
                    reason = None if all(name in observed_joints for name in self.joint_names) else "missing_joint_state"
                    if reason is None:
                        self._sync_base_freejoint()
                        for address, value in updates.items():
                            self._data.qpos[address] = value
                        mujoco.mj_forward(self._model, self._data)
            if reason is None or time.monotonic() >= deadline:
                self._last_motion_failure = reason
                self.last_state_sync = {
                    "code": reason or "ok", "wait_s": time.monotonic() - started,
                    "state_age_s": float(age) if age is not None and np.isfinite(age) else None,
                }
                if reason or self.last_state_sync["wait_s"] > .1:
                    logger.info("Measured state refresh: " + json.dumps(self.last_state_sync))
                return reason is None
            time.sleep(min(.01, max(0., deadline - time.monotonic())))

    def _wait_measured_ee(self, target: np.ndarray, *, timeout_s: float = 3.0) -> tuple[bool, float]:
        """Require measured joint FK to reach the IK target before attaching."""
        if not np.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("Measured arrival requires a finite nonnegative timeout")
        # Match trajectory/sleep timing to the advertised simulation rate. Three
        # wall seconds can be far less than three simulated seconds under load.
        wall_timeout = self._scaled_dt(timeout_s) if timeout_s else 0.0
        deadline = time.monotonic() + wall_timeout
        error = float('inf')
        observed = None
        while True:
            if self._sync_qpos_from_robot():
                observed = np.asarray(self._data.body(self.ee_body).xpos).copy()
                error = float(np.linalg.norm(observed - target))
            else:
                observed = None
                error = float('inf')
            if error <= self.ik_tol_m or time.monotonic() >= deadline:
                break
            time.sleep(.05)
        self.last_ee_verification = {
            'target_xyz': np.asarray(target).tolist(),
            'observed_xyz': None if observed is None else observed.tolist(),
            'error_m': error if np.isfinite(error) else None,
            'tolerance_m': self.ik_tol_m, 'accepted': bool(error <= self.ik_tol_m),
            'timeout_wall_s': wall_timeout,
        }
        self.last_ee_verification.update(self._joint_tracking_evidence())
        logger.info('KinematicPickPlace measured EE: ' + json.dumps(self.last_ee_verification))
        return error <= self.ik_tol_m, error

    def _joint_tracking_evidence(self) -> dict[str, Any]:
        """Separate the planner target, server-held commands, and observed joints."""
        from emet.motion.mujoco_arm_ik import joint_qpos_addrs

        if getattr(self, '_model', None) is None:
            return {}
        observed = np.asarray([self._data.qpos[a] for a in joint_qpos_addrs(self._model, self.joint_names)])
        command = getattr(self, '_last_cmd_q', None)
        state = getattr(self.robot, '_state', None)
        targets = state.get('actuator_targets') if isinstance(state, dict) else None
        by_joint = {}
        if targets is not None:
            for actuator, target in zip(self._actuator_names(), targets, strict=False):
                by_joint[self._actuator_to_joint_name(actuator)] = float(target)
        return {
            'joint_names': list(self.joint_names), 'observed_q': observed.tolist(),
            'planned_q': None if command is None else np.asarray(command).tolist(),
            'server_targets': [by_joint.get(name) for name in self.joint_names],
            'state_step': state.get('step') if isinstance(state, dict) else None,
        }

    def _stage_failure(self, stage: str) -> str:
        return f"{stage}_{getattr(self, '_last_motion_failure', None) or 'ik_failed'}"

    def _hold_actuator_dict(self) -> dict[str, float]:
        names = self._actuator_names()
        hold: dict[str, float] = {}
        q, _, _ = self.robot.get_joint_state(timeout=1.0)
        if q is not None and len(q) >= len(names):
            hold = {n: float(q[i]) for i, n in enumerate(names)}
        return hold

    def _set_gripper(self, *, open_: bool) -> None:
        names = self._actuator_names()
        hold = self._hold_actuator_dict()
        val = self.profile.gripper_open if open_ else self.profile.gripper_closed
        keys = [n for n in self.profile.actuator_names if "gripper" in n.lower()]
        if not keys:
            keys = ["right_gripper1", "right_gripper2"] if self.arm == "right" else ["left_gripper1", "left_gripper2"]
        for k in keys:
            if k in names:
                hold[k] = val
        if hold:
            self.robot.set_actuator_positions(hold)

    def _stream_arm_q(self, arm_q: np.ndarray) -> None:
        names = self._actuator_names()
        hold = pack_arm_into_actuator_dict(names, self.joint_names, arm_q, hold=self._hold_actuator_dict())
        self.robot.set_actuator_positions(hold)

    def _command_home_posture(self, settle_s: float = 1.5) -> None:
        """Drive torso/arms/grippers to the profile home (skip base steer/wheel actuators).

        Base freejoint / planar holds are owned by the sim server after nav teleport; pinning
        wheel velocity actuators to 0 from the client can fight that hold and destabilize limbs.
        """
        names = self._actuator_names()
        home_ctrl = self.profile.home_cmd
        skip_base = {"steer1", "wheel1", "steer2", "wheel2", "steer3", "wheel3"}
        cmd = {
            n: float(home_ctrl[i])
            for i, n in enumerate(self.profile.actuator_names)
            if i < len(home_ctrl) and n in names and n not in skip_base
        }
        if cmd:
            # Repeated holds: a single send after nav teleport often loses to contact transients.
            deadline = time.time() + max(float(settle_s), 0.5)
            while time.time() < deadline:
                self.robot.set_actuator_positions(cmd)
                self._sleep(0.05)
            self._sleep(0.25)
        self._last_cmd_q = home_arm_q_array(self.profile)

    def _plan_and_execute_ee(self, target_xyz_world: np.ndarray) -> tuple[bool, float]:
        self._last_motion_failure = None
        self.last_ee_verification = None
        assert self._model is not None and self._data is not None
        from emet.motion.mujoco_arm_ik import joint_qpos_addrs

        qadr = joint_qpos_addrs(self._model, self.joint_names)
        # A previous command is an IK seed, never evidence of the path start.
        # Contact may have prevented that posture from being reached.
        if not self._sync_qpos_from_robot():
            self._last_motion_failure = self._last_motion_failure or 'missing_joint_state'
            return False, float('inf')
        q0 = np.array([float(self._data.qpos[a]) for a in qadr], dtype=np.float64)
        seeds = []
        if self._last_cmd_q is not None and np.linalg.norm(self._last_cmd_q - q0) > 1e-3:
            seeds.append(self._last_cmd_q)
        result = solve_position_ik_multiseed(
            self._model,
            self._data,
            ee_body=self.ee_body,
            joint_names=self.joint_names,
            target_pos=target_xyz_world,
            seeds=seeds,
            try_midrange=True,
            # Reserve most of the measured-arrival budget for execution error.
            # Solving only to that full tolerance can fail after ordinary PD lag.
            tol_m=min(0.01, self.ik_tol_m * 0.25),
            max_iters=self.ik_max_iters,
        )
        if not result.success:
            self._last_motion_failure = 'ik_failed'
            return False, result.pos_error_m
        q1 = np.array([float(self._data.qpos[a]) for a in qadr], dtype=np.float64)
        plan = plan_arm_joint_path(
            self._model,
            self._data,
            joint_names=self.joint_names,
            q_start=q0,
            q_goal=q1,
            collision=self._collision,
            planner=self.manip_planner,
            max_iter=self.rrt_max_iter,
            linear_fallback=True,
            linear_steps=self.traj_steps,
        )
        if not plan.success:
            detail = getattr(plan, 'detail', None) or ''
            self._last_motion_failure = (detail.split(':')[0] if detail.startswith(('joint_bounds:', 'collision'))
                                         else 'planning_failed')
            logger.warning(
                f"KinematicPickPlace: path plan failed planner={plan.planner!r} "
                f"reason={plan.reason!r} detail={getattr(plan, 'detail', None)!r}"
            )
            return False, result.pos_error_m
        logger.info(f"KinematicPickPlace: path via {plan.planner} n_waypoints={len(plan.waypoints)}")
        self.last_plan_waypoints = [np.asarray(w, dtype=np.float64).copy() for w in plan.waypoints]
        self.last_ee_path_world = self._fk_ee_path(self.last_plan_waypoints)
        if self.visualizer is not None and hasattr(self.visualizer, "log_manip_ee_path"):
            try:
                self.visualizer.log_manip_ee_path(self.last_ee_path_world)
            except Exception as e:
                logger.debug(f"manip rerun log: {e}")
        for q in plan.waypoints:
            self._stream_arm_q(q)
            self._sleep(self.traj_dt)
        if plan.waypoints:
            self._stream_arm_q(plan.waypoints[-1])
            self._sleep(max(0.25, self.traj_dt * 3))
        self._last_cmd_q = q1.copy()
        reached, measured_error = self._wait_measured_ee(np.asarray(target_xyz_world))
        if not reached:
            self._last_motion_failure = self._last_motion_failure or 'tracking_failed'
        return reached, measured_error

    def _placements(self) -> dict[str, dict[str, Any]] | None:
        from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

        return read_sim_object_placements(self.robot.get_emet_session())

    def grasp_only(
        self,
        object_query: str,
        *,
        object_gt_body: str | None = None,
        grasp_T_world: np.ndarray | None = None,
    ) -> KinematicPickPlaceResult:
        if not self._ensure_model():
            return KinematicPickPlaceResult(False, None, self.ee_body, None, None, "mjcf_missing")
        body = object_gt_body or resolve_sim_object_body(self.robot, object_query)
        pl = self._placements()
        if not body or not pl or body not in pl:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, None, "object_not_in_gt")
        self._command_home_posture()
        obj_pos = np.asarray(pl[body]["pos"], dtype=np.float64).reshape(3)
        pre_pos = obj_pos.copy()
        try:
            self._set_gripper(open_=True)
        except Exception as e:
            logger.warning(f"KinematicPickPlace: open gripper before grasp failed: {e}")
        if grasp_T_world is not None:
            pregrasp, grasp, lift = _targets_from_grasp_T(
                grasp_T_world, pregrasp_standoff_m=self.pregrasp_standoff_m, lift_m=self.lift_m
            )
        else:
            pregrasp = obj_pos + np.array([0.0, 0.0, 0.15])
            grasp = obj_pos + np.array([0.0, 0.0, 0.02])
            lift = grasp + np.array([0.0, 0.0, self.lift_m])
        self.last_targets = {"pregrasp": pregrasp, "grasp": grasp, "lift": lift}
        ok, g_err = self._plan_and_execute_ee(pregrasp)
        if not ok:
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, self._stage_failure('pregrasp'))
        ok, g_err = self._plan_and_execute_ee(grasp)
        if not ok:
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, self._stage_failure('grasp'))
        self._sleep(0.4)
        try:
            self._set_gripper(open_=False)
        except Exception as e:
            logger.warning(f"KinematicPickPlace: close gripper at grasp failed: {e}")
        # Snap + attach in one ZMQ action so physics cannot drop the freejoint in between.
        robot_zmq_attach_body(self.robot, body, self.ee_body, snap_pos=grasp)
        ok, _ = self._plan_and_execute_ee(lift)
        if not ok:
            robot_zmq_detach_body(self.robot, body)
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, self._stage_failure('lift'))
        self._sleep(0.35)
        # Re-glue at the lift pose: Molmo freejoint children can lag the EE during actuator
        # streaming even when attach was registered (seen as attach_verify_failed with ~2cm dz).
        robot_zmq_attach_body(self.robot, body, self.ee_body, snap_pos=lift)
        self._sleep(0.2)
        if not self._verify_grasp_lift(body, lift, pre_pos=pre_pos):
            robot_zmq_detach_body(self.robot, body)
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, "attach_verify_failed")
        return KinematicPickPlaceResult(True, body, self.ee_body, g_err, None, "ok")

    def _placement_geometry(self, body, receptacle):
        """Fresh world-frame scene, EE-local payload and explicit support bounds.

        Observed providers return (PlacementScene, HeldObject, support_bounds).
        They must segment robot/payload from occupancy; no automatic cropping.
        """
        from emet.motion.placement_geometry import HeldObject, PlacementScene

        provider = getattr(self, "placement_geometry_provider", None)
        if provider is not None:
            return provider(self, body, receptacle)
        if self.manip_collision == "voxel":
            raise ValueError("observed_placement_geometry_provider_required")
        placements = self._placements()
        if not placements or body not in placements or receptacle not in placements:
            raise ValueError("placement_geometry_missing")
        scene = PlacementScene.from_placements(placements, held_object=body)
        ee = self._data.body(self.ee_body)
        payload = HeldObject.from_world_bounds(placements[body]["bounds"],
                                              ee_position=ee.xpos, ee_rotation=ee.xmat)
        # A semantic appliance AABB is not a support surface. New simulators
        # publish grounded horizontal collision faces; explicit providers can
        # supply observed patches or interior shelf regions.
        from emet.motion.placement_surfaces import support_patches

        patches = placements[receptacle].get("support_surfaces")
        if patches is None:
            raise ValueError("placement_support_geometry_missing")
        return scene, payload, support_patches(patches)

    def _search_placement(self, body, receptacle, *, approach_base):
        from emet.motion.placement import (
            placement_base_candidates,
            plan_placement_paths,
        )
        from emet.motion.placement_geometry import PlacementCollisionChecker

        if not self._sync_qpos_from_robot():
            raise ValueError("placement_" + (self._last_motion_failure or "missing_joint_state"))
        scene, payload, support = self._placement_geometry(body, receptacle)
        rotation = self._data.body(self.ee_body).xmat.reshape(3, 3).copy()
        from emet.motion.placement_surfaces import free_surface_centers, support_patches

        support = support_patches(support)
        surface_search = free_surface_centers(support, scene=scene, payload=payload, ee_rotation=rotation,
                                              clearance_m=max(.02, self.place_z_offset_m))
        self.last_surface_search = surface_search
        centers = surface_search.centers
        current = self._world_base_xyt()
        if current is None:
            raise ValueError("placement_base_pose_missing")
        spec = self.robot._spec
        mode = str(getattr(spec, "tamp_approach", "front") or "front")
        offset = (np.pi / 2 if self.arm == "left" else -np.pi / 2) if mode == "side" else 0.
        poses = placement_base_candidates(np.asarray(support).mean(axis=(0, 1))[:2], current_xyt=current,
                                           yaw_offset=offset) if approach_base else [current]
        state = getattr(self.robot, "_state", {})
        clear = None
        if state.get("sim_base_pose_query") is True:
            clear = []
            # The command protocol bounds each query to 32 endpoints; preserve
            # candidate order and validate every batch before trusting any result.
            for start in range(0, len(poses), 32):
                batch = poses[start:start + 32]
                evidence = self.robot.check_base_poses(batch)["clear"]
                if len(evidence) != len(batch) or any(type(value) is not bool for value in evidence):
                    raise ValueError("invalid_placement_clearance_response")
                clear.extend(evidence)
        poses = [pose for index, pose in enumerate(poses) if clear is None or clear[index]]

        def set_base(model, data, pose):
            return write_offline_mjcf_base_xyt(model, data, pose,
                planar_joint_names=self._planar_joint_names(), freejoint_name=self.profile.base_freejoint_name)

        contacts = (self.ee_body, *self.profile.gripper_contact_bodies())
        snapshot_root = os.environ.get("EMET_PLACEMENT_DIAGNOSTICS_DIR")
        if snapshot_root:
            from emet.motion.placement_replay import save_snapshot

            snapshot = save_snapshot(Path(snapshot_root) / uuid.uuid4().hex, self._model, self._data,
                scene=scene, payload=payload, object_centers=centers, base_candidates=poses,
                joint_names=self.joint_names, ee_body=self.ee_body, robot_body=spec.base_link_name,
                contact_bodies=contacts, rrt_max_iter=self.rrt_max_iter,
                base_writer={"planar_joint_names": self._planar_joint_names(),
                             "freejoint_name": self.profile.base_freejoint_name})
            logger.info(f"Placement snapshot: {snapshot}")
        result = plan_placement_paths(self._model, self._data, joint_names=self.joint_names,
            ee_body=self.ee_body, robot_body=spec.base_link_name, scene=scene, payload=payload,
            object_centers=centers, base_candidates=poses, set_base=set_base, contact_bodies=contacts,
            rrt_max_iter=self.rrt_max_iter)
        if clear is not None and not all(clear):
            result.rejections["base_endpoint_rejected"] = sum(not value for value in clear)
        if not centers:
            result.rejections["surface_search_budget_exhausted" if surface_search.budget_exhausted
                              else "no_accepted_surface_candidate"] = 1
        self.last_placement_search = result
        self._placement_support_bounds = np.array(support, dtype=float, copy=True)
        logger.info(f"Placement search source={result.geometry_source} paths={len(result.paths)} "
                    f"rejections={result.rejections} scope={result.collision_scope}")
        checker = PlacementCollisionChecker(self._model, robot_body=spec.base_link_name, ee_body=self.ee_body,
            scene=scene, payload=payload, contact_bodies=contacts)
        return result, checker

    def _refresh_placement_checker(self, body, receptacle, checker):
        """Refresh obstacles before each segment; changed support/attachment needs replanning."""
        from emet.motion.placement_geometry import PlacementCollisionChecker

        if not self._sync_qpos_from_robot():
            raise ValueError("placement_" + (self._last_motion_failure or "missing_joint_state"))
        scene, payload, support = self._placement_geometry(body, receptacle)
        if scene.geometry_digest != checker.scene.geometry_digest:
            logger.info("Placement scene changed; revalidating the full segment against refreshed occupancy")
        if scene.source != checker.scene.source:
            raise ValueError("placement_geometry_source_changed")
        from emet.motion.placement_surfaces import support_patches

        support = support_patches(support)
        expected_support = support_patches(self._placement_support_bounds)
        if support.shape != expected_support.shape or not np.allclose(support, expected_support, atol=.01, rtol=0):
            raise ValueError("placement_support_moved")
        if (payload.vertices_ee.shape != checker.payload.vertices_ee.shape or
            not np.allclose(payload.vertices_ee, checker.payload.vertices_ee, atol=.01, rtol=0)):
            raise ValueError("placement_attachment_changed")
        return PlacementCollisionChecker(self._model, robot_body=self.robot._spec.base_link_name,
            ee_body=self.ee_body, scene=scene, payload=payload,
            contact_bodies=(self.ee_body, *self.profile.gripper_contact_bodies()))

    def _execute_placement_segment(self, path, target, rotation, checker):
        from scipy.spatial.transform import Rotation

        from emet.motion.mujoco_arm_ik import joint_qpos_addrs
        from emet.motion.placement import validated_dense_path

        self._last_motion_failure = None
        self.last_ee_verification = None
        if not self._sync_qpos_from_robot():
            self._last_motion_failure = "placement_" + (self._last_motion_failure or "missing_joint_state")
            return False, float("inf")
        q = self._data.qpos[joint_qpos_addrs(self._model, self.joint_names)].copy()
        # Validate the measured-start connector as well as every planned edge.
        dense = validated_dense_path(self._model, self._data, self.joint_names, [q, *path], checker)
        if dense is None:
            self._last_motion_failure = "placement_path_invalidated"
            return False, float("inf")
        for point in dense:
            self._stream_arm_q(point)
            self._sleep(self.traj_dt)
        self._last_cmd_q = dense[-1].copy()
        ok, error = self._wait_measured_ee(target)
        actual = self._data.body(self.ee_body).xmat.reshape(3, 3)
        orientation_error = float(Rotation.from_matrix(rotation @ actual.T).magnitude())
        if not isinstance(self.last_ee_verification, dict):
            self._last_motion_failure = "missing_pose_evidence"
            return False, error
        self.last_ee_verification.update({"orientation_error_rad": orientation_error,
            "orientation_tolerance_rad": .1, "accepted": bool(ok and orientation_error <= .1)})
        logger.info("Placement measured pose: " + json.dumps(self.last_ee_verification))
        if not ok or orientation_error > .1:
            self._last_motion_failure = "tracking_failed"
            return False, error
        return True, error

    def place_only(
        self,
        receptacle_query: str,
        *,
        object_gt_body: str | None = None,
        receptacle_gt_body: str | None = None,
        approach_base: bool = True,
    ) -> KinematicPickPlaceResult:
        if not self._ensure_model():
            return KinematicPickPlaceResult(False, None, self.ee_body, None, None, "mjcf_missing")
        pl = self._placements()
        if not pl:
            return KinematicPickPlaceResult(False, object_gt_body, self.ee_body, None, None, "no_placements")
        body = object_gt_body
        if not body or body not in pl:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, None, "object_not_in_gt")
        from emet.eval.ovmm_find_phase import bodies_matching_category

        if receptacle_gt_body:
            if receptacle_gt_body not in pl:
                return KinematicPickPlaceResult(False, body, self.ee_body, None, None, "recep_not_in_gt")
            receps = [receptacle_gt_body]
        else:
            receps = bodies_matching_category(pl, receptacle_query)
        if not receps:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, None, "recep_not_in_gt")
        # Prefer farthest matching recep from the held object (same idea as OVMM sim place).
        # Among far candidates, prefer lower Z — tall appliance body COMs are not shelves.
        obj_xy = np.asarray(pl[body]["pos"], dtype=np.float64).reshape(3)[:2]
        scored: list[tuple[float, float, str]] = []
        for cand in receps:
            cpos = np.asarray(pl[cand]["pos"], dtype=np.float64).reshape(3)
            d = float(np.linalg.norm(obj_xy - cpos[:2]))
            prefer = 0.05 if str(cand).endswith("_main") else 0.0
            scored.append((d + prefer, float(cpos[2]), cand))
        scored.sort(key=lambda t: (-t[0], t[1]))
        recep_body = scored[0][2]
        recep_pos = np.asarray(pl[recep_body]["pos"], dtype=np.float64).reshape(3)
        logger.info(
            f"KinematicPickPlace: place target recep={recep_body!r} pos={recep_pos.tolist()} (n_receps={len(receps)})"
        )
        try:
            search, checker = self._search_placement(body, recep_body, approach_base=approach_base)
            if not search.paths:
                return KinematicPickPlaceResult(False, body, self.ee_body, None, None, "no_collision_free_placement")
            selected = search.paths[0]
            if approach_base:
                transport_offset = checker.payload.vertices_ee.mean(axis=0)
                moved = self.robot.move_base_to(selected.base_xyt, blocking=True, world_frame=True)
                success = getattr(moved, "success", moved)
                if not isinstance(success, (bool, np.bool_)) or not success:
                    return KinematicPickPlaceResult(False, body, self.ee_body, None, None, "place_approach_failed")
                self._sleep(.4)
                # Refresh geometry and measured arm/base/attachment after transport.
                search, checker = self._search_placement(body, recep_body, approach_base=False)
                if not search.paths:
                    return KinematicPickPlaceResult(False, body, self.ee_body, None, None, "placement_invalidated")
                if not np.allclose(checker.payload.vertices_ee.mean(axis=0), transport_offset, atol=.01, rtol=0):
                    return KinematicPickPlaceResult(False, body, self.ee_body, None, None, "placement_attachment_changed")
                selected = search.paths[0]
        except (ValueError, KeyError, RuntimeError, TimeoutError) as exc:
            logger.warning(f"Placement geometry unavailable: {exc}")
            return KinematicPickPlaceResult(False, body, self.ee_body, None, None,
                str(exc) if str(exc) in PLACEMENT_STATE_FAILURES else "placement_geometry_unavailable")
        preplace, place_ee = selected.ee_targets
        place = selected.object_center
        self.last_targets = {"preplace": preplace, "place": place_ee, "object_place": place, "recep": recep_pos}
        for stage, path, target in zip(("preplace", "place"), selected.segments, selected.ee_targets, strict=True):
            try:
                checker = self._refresh_placement_checker(body, recep_body, checker)
            except (ValueError, KeyError, RuntimeError, TimeoutError) as exc:
                logger.warning(f"Placement snapshot invalidated: {exc}")
                return KinematicPickPlaceResult(False, body, self.ee_body, None, None,
                    str(exc) if str(exc) in PLACEMENT_STATE_FAILURES else "placement_invalidated")
            ok, p_err = self._execute_placement_segment(path, target, selected.ee_rotation, checker)
            if not ok:
                return KinematicPickPlaceResult(False, body, self.ee_body, None, p_err, self._stage_failure(stage))
        # Detach first so per-step kinematic snap cannot pull the freejoint back to the EE,
        # then oracle-snap like OVMM manip_mode=sim and score before physics drops a mid-air COM.
        robot_zmq_detach_body(self.robot, body)
        robot_zmq_set_body_pose(self.robot, body, place)
        self._sleep(0.25)
        ok_place, p_err = self._verify_place_xy(body, place[:2])
        if not ok_place:
            # Freejoint children can lag one publish step after detach+snap.
            robot_zmq_set_body_pose(self.robot, body, place)
            self._sleep(0.2)
            ok_place, p_err = self._verify_place_xy(body, place[:2])
        try:
            self._set_gripper(open_=True)
        except Exception:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, p_err, "release_execution_error")
        vertices = checker.payload.world_vertices(place_ee, selected.ee_rotation)
        checker.released_bounds = np.stack((vertices.min(axis=0) - checker.margin,
                                           vertices.max(axis=0) + checker.margin))
        previous_collision = self._collision
        self._collision = checker
        try:
            retracted, retract_error = self._plan_and_execute_ee(place_ee + np.array([0.0, 0.0, 0.15]))
        except Exception:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, p_err, "retract_execution_error")
        finally:
            self._collision = previous_collision
        if not retracted:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, retract_error,
                                            self._stage_failure("retract"))
        if not ok_place:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, p_err, "place_verify_failed")
        return KinematicPickPlaceResult(True, body, self.ee_body, None, p_err, "ok")

    def pick_and_place(
        self,
        object_query: str,
        receptacle_query: str,
        *,
        object_gt_body: str | None = None,
        grasp_T_world: np.ndarray | None = None,
    ) -> KinematicPickPlaceResult:
        grasp = self.grasp_only(object_query, object_gt_body=object_gt_body, grasp_T_world=grasp_T_world)
        if not grasp.success:
            return grasp
        place = self.place_only(receptacle_query, object_gt_body=grasp.object_body)
        return KinematicPickPlaceResult(
            place.success,
            grasp.object_body,
            self.ee_body,
            grasp.grasp_err_m,
            place.place_err_m,
            place.message if not place.success else "ok",
        )
