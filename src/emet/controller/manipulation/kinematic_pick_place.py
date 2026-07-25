# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Kinematic pick-and-place skill graph for registry robots (rby1) in MuJoCo.

Uses MuJoCo position IK + ZMQ joint streaming + sim kinematic attach. Optional voxel-map
collision filter (same world model as base nav). Not CuRobo / not contact physics.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from emet.motion.arm_manip_profile import (
    ArmManipProfile,
    home_arm_q_array,
    robot_id_from_client,
)
from emet.motion.arm_rrt import plan_arm_joint_path, resolve_agent_manip_planner
from emet.motion.mujoco_arm_ik import solve_position_ik_multiseed
from emet.motion.voxel_arm_collision import VoxelMapArmCollisionChecker
from emet.simulation.sim_manipulation import (
    resolve_sim_object_body,
    robot_zmq_attach_body,
    robot_zmq_detach_body,
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
        self.ee_body = self.profile.ee_body
        self.joint_names = self.profile.joint_names
        self.link_bodies = list(self.profile.link_bodies)
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._collision: VoxelMapArmCollisionChecker | None = None
        self._last_cmd_q: np.ndarray | None = None

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
        return True

    def _actuator_names(self) -> list[str]:
        spec = getattr(self.robot, "_spec", None)
        if spec is not None and getattr(spec, "actuator_names", None):
            return list(spec.actuator_names)
        return list(self.profile.actuator_names)

    def _sync_base_freejoint(self) -> None:
        """Align IK model freejoint with live robot base (world XY/Z + yaw)."""
        assert self._model is not None and self._data is not None
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, self.profile.base_freejoint_name)
        if jid < 0:
            return
        qadr = int(self._model.jnt_qposadr[jid])
        pose = np.asarray(self.robot.get_base_pose(timeout=2.0), dtype=np.float64).reshape(-1)
        if pose.size < 3:
            return
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        z = float(self._data.qpos[qadr + 2])
        state = getattr(self.robot, "_state", None)
        if isinstance(state, dict) and state.get("base_xyz") is not None:
            try:
                z = float(np.asarray(state["base_xyz"], dtype=np.float64).reshape(-1)[2])
            except Exception:
                pass
        half = 0.5 * th
        self._data.qpos[qadr : qadr + 7] = [
            x,
            y,
            z,
            float(np.cos(half)),
            0.0,
            0.0,
            float(np.sin(half)),
        ]

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
        """Drive rby1 finger actuators (ctrlrange 0..0.05); ignore Stretch-style gripper key."""
        names = self._actuator_names()
        hold = self._hold_actuator_dict()
        val = 0.05 if open_ else 0.0
        if self.arm == "right":
            keys = ("right_gripper1", "right_gripper2")
        else:
            keys = ("left_gripper1", "left_gripper2")
        for k in keys:
            if k in names:
                hold[k] = val
        if hold:
            self.robot.set_actuator_positions(hold)

    def _stream_arm_q(self, arm_q: np.ndarray) -> None:
        """Stream torso+arm waypoint (``self.joint_names`` order) into actuator holds."""
        names = self._actuator_names()
        hold = self._hold_actuator_dict()
        qq = np.asarray(arm_q, dtype=np.float64).reshape(-1)
        if len(qq) != len(self.joint_names):
            logger.warning(f"KinematicPickPlace: waypoint len {len(qq)} != joints {len(self.joint_names)}")
            return
        # Map joint_* names to actuators: torso_joint1 -> torso1, left_arm_joint1 -> left_arm1
        for jname, val in zip(self.joint_names, qq, strict=True):
            aname = jname.replace("_joint", "") if "_joint" in jname else jname
            # left_arm_joint1 -> left_arm1
            aname = aname.replace("arm_joint", "arm")
            if aname.startswith("torso_joint"):
                aname = aname.replace("torso_joint", "torso")
            if aname in names:
                hold[aname] = float(val)
        self.robot.set_actuator_positions(hold)

    def _command_home_posture(self, settle_s: float = 1.5) -> None:
        """Drive actuators toward profile home ctrl."""
        names = self._actuator_names()
        home_ctrl = self.profile.home_cmd
        cmd = {
            n: float(home_ctrl[i])
            for i, n in enumerate(self.profile.actuator_names)
            if i < len(home_ctrl) and n in names
        }
        if cmd:
            self.robot.set_actuator_positions(cmd)
            time.sleep(float(settle_s))
        self._last_cmd_q = home_arm_q_array(self.profile)

    def _plan_and_execute_ee(self, target_xyz_world: np.ndarray) -> tuple[bool, float]:
        assert self._model is not None and self._data is not None
        from emet.motion.mujoco_arm_ik import joint_qpos_addrs

        # Base from live robot; arm/torso prefer last commanded IK goal (PD lags after stream).
        self._sync_base_freejoint()
        qadr = joint_qpos_addrs(self._model, self.joint_names)
        live_seed: np.ndarray | None = None
        q_live, _, _ = self.robot.get_joint_state(timeout=2.0)
        names = self._actuator_names()
        if q_live is not None and len(q_live) >= len(names):
            # Build live torso+arm vector in joint_names order for alternate seed.
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

        # If arms look collapsed (EE far below base), force midrange via multiseed.
        ee_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body)
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
        alt_seeds = [s for s in (live_seed, self._last_cmd_q) if s is not None]
        # Drop duplicates of current q0
        seeds = []
        for s in alt_seeds:
            if np.linalg.norm(np.asarray(s, dtype=np.float64).reshape(-1) - q0) > 1e-3:
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
        for q in plan.waypoints:
            self._stream_arm_q(q)
            time.sleep(self.traj_dt)
        # Hold the goal setpoint so PD can catch up before the next IK solve.
        if plan.waypoints:
            self._stream_arm_q(plan.waypoints[-1])
            time.sleep(max(0.25, self.traj_dt * 3))
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
        """Pregrasp → grasp → attach → lift (no place)."""
        if not self._ensure_model():
            return KinematicPickPlaceResult(False, None, self.ee_body, None, None, "mjcf_missing")
        body = object_gt_body or resolve_sim_object_body(self.robot, object_query)
        pl = self._placements()
        if not body or not pl or body not in pl:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, None, "object_not_in_gt")
        self._command_home_posture()
        obj_pos = np.asarray(pl[body]["pos"], dtype=np.float64).reshape(3)
        try:
            self._set_gripper(open_=True)
        except Exception:
            pass
        if grasp_T_world is not None:
            pregrasp, grasp, lift = _targets_from_grasp_T(
                grasp_T_world, pregrasp_standoff_m=self.pregrasp_standoff_m, lift_m=self.lift_m
            )
        else:
            pregrasp = obj_pos + np.array([0.0, 0.0, 0.15])
            grasp = obj_pos + np.array([0.0, 0.0, 0.02])
            lift = grasp + np.array([0.0, 0.0, self.lift_m])
        ok, g_err = self._plan_and_execute_ee(pregrasp)
        if not ok:
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, "pregrasp_ik_failed")
        ok, g_err = self._plan_and_execute_ee(grasp)
        if not ok:
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, "grasp_ik_failed")
        try:
            self._set_gripper(open_=False)
        except Exception:
            pass
        robot_zmq_attach_body(self.robot, body, self.ee_body)
        ok, _ = self._plan_and_execute_ee(lift)
        if not ok:
            robot_zmq_detach_body(self.robot, body)
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, "lift_ik_failed")
        return KinematicPickPlaceResult(True, body, self.ee_body, g_err, None, "ok")

    def place_only(
        self,
        receptacle_query: str,
        *,
        object_gt_body: str | None = None,
    ) -> KinematicPickPlaceResult:
        """Place attached (or resolved) object onto receptacle, then detach."""
        if not self._ensure_model():
            return KinematicPickPlaceResult(False, None, self.ee_body, None, None, "mjcf_missing")
        pl = self._placements()
        if not pl:
            return KinematicPickPlaceResult(False, object_gt_body, self.ee_body, None, None, "no_placements")
        body = object_gt_body
        if not body or body not in pl:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, None, "object_not_in_gt")
        from emet.eval.ovmm_find_phase import bodies_matching_category

        receps = bodies_matching_category(pl, receptacle_query)
        if not receps:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, None, "recep_not_in_gt")
        recep_pos = np.asarray(pl[receps[0]]["pos"], dtype=np.float64).reshape(3)
        place = recep_pos + np.array([0.0, 0.0, self.place_z_offset_m])
        preplace = place + np.array([0.0, 0.0, 0.12])
        ok, p_err = self._plan_and_execute_ee(preplace)
        if not ok:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, p_err, "preplace_ik_failed")
        ok, p_err = self._plan_and_execute_ee(place)
        if not ok:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, p_err, "place_ik_failed")
        robot_zmq_detach_body(self.robot, body)
        try:
            self._set_gripper(open_=True)
        except Exception:
            pass
        self._plan_and_execute_ee(place + np.array([0.0, 0.0, 0.15]))
        pl2 = self._placements() or pl
        if body in pl2:
            final = np.asarray(pl2[body]["pos"], dtype=np.float64).reshape(3)
            p_err = float(np.linalg.norm(final[:2] - recep_pos[:2]))
        return KinematicPickPlaceResult(True, body, self.ee_body, None, p_err, "ok")

    def pick_and_place(
        self,
        object_query: str,
        receptacle_query: str,
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
        from emet.eval.ovmm_find_phase import bodies_matching_category

        receps = bodies_matching_category(pl, receptacle_query)
        if not receps:
            return KinematicPickPlaceResult(False, body, self.ee_body, None, None, "recep_not_in_gt")
        recep_pos = np.asarray(pl[receps[0]]["pos"], dtype=np.float64).reshape(3)

        try:
            self._set_gripper(open_=True)
        except Exception as e:
            logger.debug(f"open_gripper: {e}")

        if grasp_T_world is not None:
            pregrasp, grasp, lift = _targets_from_grasp_T(
                grasp_T_world, pregrasp_standoff_m=self.pregrasp_standoff_m, lift_m=self.lift_m
            )
        else:
            pregrasp = obj_pos + np.array([0.0, 0.0, 0.15])
            grasp = obj_pos + np.array([0.0, 0.0, 0.02])
            lift = grasp + np.array([0.0, 0.0, self.lift_m])
        ok, g_err = self._plan_and_execute_ee(pregrasp)
        if not ok:
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, "pregrasp_ik_failed")
        ok, g_err = self._plan_and_execute_ee(grasp)
        if not ok:
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, "grasp_ik_failed")

        try:
            self._set_gripper(open_=False)
        except Exception as e:
            logger.debug(f"close_gripper: {e}")

        robot_zmq_attach_body(self.robot, body, self.ee_body)
        ok, _ = self._plan_and_execute_ee(lift)
        if not ok:
            robot_zmq_detach_body(self.robot, body)
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, None, "lift_ik_failed")

        place = recep_pos + np.array([0.0, 0.0, self.place_z_offset_m])
        preplace = place + np.array([0.0, 0.0, 0.12])
        ok, p_err = self._plan_and_execute_ee(preplace)
        if not ok:
            robot_zmq_detach_body(self.robot, body)
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, p_err, "preplace_ik_failed")
        ok, p_err = self._plan_and_execute_ee(place)
        if not ok:
            robot_zmq_detach_body(self.robot, body)
            return KinematicPickPlaceResult(False, body, self.ee_body, g_err, p_err, "place_ik_failed")

        robot_zmq_detach_body(self.robot, body)
        try:
            self._set_gripper(open_=True)
        except Exception:
            pass
        retreat = place + np.array([0.0, 0.0, 0.15])
        self._plan_and_execute_ee(retreat)

        pl2 = self._placements() or pl
        if body in pl2:
            final = np.asarray(pl2[body]["pos"], dtype=np.float64).reshape(3)
            p_err = float(np.linalg.norm(final[:2] - recep_pos[:2]))
        return KinematicPickPlaceResult(True, body, self.ee_body, g_err, p_err, "ok")
