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

import re
import time
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
from emet.motion.voxel_arm_collision import VoxelMapArmCollisionChecker
from emet.simulation.sim_manipulation import (
    resolve_sim_object_body,
    robot_zmq_attach_body,
    robot_zmq_detach_body,
    robot_zmq_set_body_pose,
)
from emet.utils.logger import Logger

logger = Logger(__name__)


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
        self.ee_body = self.profile.ee_body
        self.joint_names = self.profile.joint_names
        self.link_bodies = list(self.profile.link_bodies)
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._collision: VoxelMapArmCollisionChecker | AabbArmCollisionChecker | None = None
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
        after = self._body_pos(body)
        if after is None:
            return False
        lift = np.asarray(lift_xyz, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(after - lift)) <= self.grasp_lift_verify_tol_m:
            return True
        if pre_pos is not None:
            dz = float(after[2] - pre_pos[2])
            if dz >= max(0.04, 0.4 * self.lift_m):
                return True
        return False

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

    def _sync_qpos_from_robot(self) -> None:
        assert self._model is not None and self._data is not None
        self._sync_base_freejoint()
        q, _, _ = self.robot.get_joint_state(timeout=2.0)
        names = self._actuator_names()
        if q is None or len(q) < len(names):
            mujoco.mj_forward(self._model, self._data)
            return
        for i, aname in enumerate(names):
            jname = self._actuator_to_joint_name(aname)
            if not jname:
                continue
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                self._data.qpos[int(self._model.jnt_qposadr[jid])] = float(q[i])
        mujoco.mj_forward(self._model, self._data)

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
        val = 0.05 if open_ else 0.0
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
        assert self._model is not None and self._data is not None
        from emet.motion.mujoco_arm_ik import joint_qpos_addrs

        self._sync_base_freejoint()
        qadr = joint_qpos_addrs(self._model, self.joint_names)
        live_seed: np.ndarray | None = None
        q_live, _, _ = self.robot.get_joint_state(timeout=2.0)
        names = self._actuator_names()
        if q_live is not None and len(q_live) >= len(names):
            live_map: dict[str, float] = {}
            for i, aname in enumerate(names):
                jname = self._actuator_to_joint_name(aname)
                if jname:
                    live_map[jname] = float(q_live[i])
            if all(n in live_map for n in self.joint_names):
                live_seed = np.array([live_map[n] for n in self.joint_names], dtype=np.float64)

        if self._last_cmd_q is not None and len(self._last_cmd_q) == len(qadr):
            for a, v in zip(qadr, self._last_cmd_q, strict=True):
                self._data.qpos[a] = float(v)
        elif live_seed is not None:
            for a, v in zip(qadr, live_seed, strict=True):
                self._data.qpos[a] = float(v)
        else:
            self._sync_qpos_from_robot()

        ee_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body)
        spec = getattr(self.robot, "_spec", None)
        base_name = str(getattr(spec, "base_link_name", None) or "base_link")
        base_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, base_name)
        if base_id < 0:
            base_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if ee_id >= 0 and base_id >= 0:
            mujoco.mj_forward(self._model, self._data)
            ee_z = float(self._data.body(ee_id).xpos[2])
            base_z = float(self._data.body(base_id).xpos[2])
            if ee_z < base_z - 0.2:
                for name in self.joint_names:
                    jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid >= 0 and self._model.jnt_limited[jid]:
                        lo, hi = float(self._model.jnt_range[jid][0]), float(self._model.jnt_range[jid][1])
                        self._data.qpos[int(self._model.jnt_qposadr[jid])] = 0.5 * (lo + hi)

        mujoco.mj_forward(self._model, self._data)
        q0 = np.array([float(self._data.qpos[a]) for a in qadr], dtype=np.float64)
        seeds = []
        for s in (live_seed, self._last_cmd_q):
            if s is not None and np.linalg.norm(np.asarray(s, dtype=np.float64).reshape(-1) - q0) > 1e-3:
                seeds.append(s)
        result = solve_position_ik_multiseed(
            self._model,
            self._data,
            ee_body=self.ee_body,
            joint_names=self.joint_names,
            target_pos=target_xyz_world,
            seeds=seeds,
            try_midrange=True,
            tol_m=self.ik_tol_m,
            max_iters=self.ik_max_iters,
        )
        if not result.success:
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
            logger.warning(f"KinematicPickPlace: path plan failed planner={plan.planner!r} reason={plan.reason!r}")
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
        return True, result.pos_error_m

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
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, "pregrasp_ik_failed")
        ok, g_err = self._plan_and_execute_ee(grasp)
        if not ok:
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, "grasp_ik_failed")
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
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, "lift_ik_failed")
        self._sleep(0.35)
        # Re-glue at the lift pose: Molmo freejoint children can lag the EE during actuator
        # streaming even when attach was registered (seen as attach_verify_failed with ~2cm dz).
        robot_zmq_attach_body(self.robot, body, self.ee_body, snap_pos=lift)
        self._sleep(0.2)
        if not self._verify_grasp_lift(body, lift, pre_pos=pre_pos):
            robot_zmq_detach_body(self.robot, body)
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, "attach_verify_failed")
        return KinematicPickPlaceResult(True, body, self.ee_body, g_err, None, "ok")

    def _approach_xy(
        self,
        target_xy: np.ndarray,
        *,
        standoff_m: float = 0.55,
        yaw: float | None = None,
    ) -> None:
        """Nav-teleport base near *target_xy* (MuJoCo world) so arm IK is in reach."""
        import os

        os.environ.setdefault("EMET_SIM_NAV_TELEPORT", "1")
        xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
        # Stand off along the vector from target toward current base (fallback +Y).
        cur = self._world_base_xyt()
        if cur is not None:
            delta = np.asarray(cur[:2], dtype=np.float64) - xy
            n = float(np.linalg.norm(delta))
            if n > 1e-3:
                delta = delta / n
            else:
                delta = np.array([0.0, 1.0], dtype=np.float64)
        else:
            delta = np.array([0.0, 1.0], dtype=np.float64)
        approach_xy = xy + float(standoff_m) * delta
        if yaw is None:
            # Face the receptacle from the approach pose.
            face = xy - approach_xy
            th = float(np.arctan2(face[1], face[0])) if float(np.linalg.norm(face)) > 1e-3 else -np.pi / 2
        else:
            th = float(yaw)
        approach = np.array([float(approach_xy[0]), float(approach_xy[1]), th], dtype=np.float64)
        move = getattr(self.robot, "move_base_to", None)
        if not callable(move):
            return
        logger.info(f"KinematicPickPlace: approach base -> {approach.tolist()}")
        move(approach, blocking=True, world_frame=True)
        self._sleep(0.4)

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

        if receptacle_gt_body and receptacle_gt_body in pl:
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
        if approach_base:
            self._approach_xy(recep_pos[:2])
            # Re-register attach after base teleport so offset matches the live EE.
            robot_zmq_attach_body(self.robot, body, self.ee_body)
            self._sleep(0.15)
        place = recep_pos + np.array([0.0, 0.0, self.place_z_offset_m])
        preplace = place + np.array([0.0, 0.0, 0.12])
        self.last_targets = {"preplace": preplace, "place": place, "recep": recep_pos, "recep_body": recep_body}
        ok, p_err = self._plan_and_execute_ee(preplace)
        if not ok:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, p_err, "preplace_ik_failed")
        ok, p_err = self._plan_and_execute_ee(place)
        if not ok:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, p_err, "place_ik_failed")
        # Detach first so per-step kinematic snap cannot pull the freejoint back to the EE,
        # then oracle-snap like OVMM manip_mode=sim and score before physics drops a mid-air COM.
        robot_zmq_detach_body(self.robot, body)
        robot_zmq_set_body_pose(self.robot, body, place)
        self._sleep(0.25)
        ok_place, p_err = self._verify_place_xy(body, recep_pos[:2])
        if not ok_place:
            # Freejoint children can lag one publish step after detach+snap.
            robot_zmq_set_body_pose(self.robot, body, place)
            self._sleep(0.2)
            ok_place, p_err = self._verify_place_xy(body, recep_pos[:2])
        try:
            self._set_gripper(open_=True)
        except Exception as e:
            logger.warning(f"KinematicPickPlace: open gripper after place failed: {e}")
        try:
            self._plan_and_execute_ee(place + np.array([0.0, 0.0, 0.15]))
        except Exception as e:
            logger.warning(f"KinematicPickPlace: retract after place failed: {e}")
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
