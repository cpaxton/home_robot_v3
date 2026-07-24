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

from emet.motion.arm_rrt import plan_arm_joint_path, resolve_agent_manip_planner
from emet.motion.mujoco_arm_ik import (
    RBY1_LEFT_ARM_JOINTS,
    RBY1_LEFT_EE_BODY,
    solve_position_ik,
)
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


class KinematicPickPlaceExecutor:
    """Navigate (optional) → IK pregrasp/grasp → attach → lift → place → detach."""

    def __init__(
        self,
        robot: Any,
        *,
        arm: str = "left",
        manip_collision: str = "none",
        manip_planner: str = "rrt_connect",
        voxel_map: Any | None = None,
        traj_dt: float = 0.04,
        traj_steps: int = 15,
        lift_m: float = 0.12,
        place_z_offset_m: float = 0.02,
        rrt_max_iter: int = 400,
    ) -> None:
        self.robot = robot
        self.arm = str(arm).lower()
        self.manip_collision = str(manip_collision).lower()
        self.manip_planner = resolve_agent_manip_planner(config_mode=manip_planner)
        self.voxel_map = voxel_map
        self.traj_dt = float(traj_dt)
        self.traj_steps = int(traj_steps)
        self.lift_m = float(lift_m)
        self.place_z_offset_m = float(place_z_offset_m)
        self.rrt_max_iter = int(rrt_max_iter)
        if self.arm == "right":
            from emet.motion.mujoco_arm_ik import RBY1_RIGHT_ARM_JOINTS, RBY1_RIGHT_EE_BODY

            self.arm_joint_names = RBY1_RIGHT_ARM_JOINTS
            self.ee_body = RBY1_RIGHT_EE_BODY
            self.link_bodies = [f"right_arm_link{i}" for i in range(3, 7)]
        else:
            self.arm_joint_names = RBY1_LEFT_ARM_JOINTS
            self.ee_body = RBY1_LEFT_EE_BODY
            self.link_bodies = [f"left_arm_link{i}" for i in range(3, 7)]
        # Include torso for reach on table / Molmo benches.
        self.joint_names = tuple(f"torso_joint{i}" for i in range(1, 5)) + tuple(self.arm_joint_names)
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._collision: VoxelMapArmCollisionChecker | None = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        spec = getattr(self.robot, "_spec", None) or getattr(self.robot, "get_robot_spec", lambda: None)()
        mjcf = getattr(spec, "mjcf_path", None) if spec is not None else None
        if not mjcf:
            # Fall back to rby1 asset
            from emet.robots.rby1 import Rby1Backend

            mjcf = Rby1Backend().get_spec().mjcf_path
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
        from emet.robots.galaxea_r1 import R1_ACTUATOR_NAMES

        return list(R1_ACTUATOR_NAMES)

    def _sync_base_freejoint(self) -> None:
        """Align IK model freejoint with live robot base (world XY/Z + yaw)."""
        assert self._model is not None and self._data is not None
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")
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
        """Drive actuators toward MJCF ``home`` keyframe (torso up, arms mid)."""
        from emet.robots.galaxea_r1 import R1_ACTUATOR_NAMES

        # Matches galaxea_r1.xml key name="home" ctrl vector (26 actuators).
        home_ctrl = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.5,
            -0.5,
            0,
            0,
            0,
            0.04,
            0.04,
            0,
            0.5,
            -0.5,
            0,
            0,
            0,
            0.04,
            0.04,
        ]
        names = self._actuator_names()
        cmd = {n: float(home_ctrl[i]) for i, n in enumerate(R1_ACTUATOR_NAMES) if i < len(home_ctrl) and n in names}
        if cmd:
            self.robot.set_actuator_positions(cmd)
            time.sleep(float(settle_s))

    def _plan_and_execute_ee(self, target_xyz_world: np.ndarray) -> tuple[bool, float]:
        assert self._model is not None and self._data is not None
        self._sync_qpos_from_robot()
        from emet.motion.mujoco_arm_ik import joint_qpos_addrs

        qadr = joint_qpos_addrs(self._model, self.joint_names)
        # If live arms look collapsed (EE far below base), seed mid-range before IK.
        ee_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body)
        base_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if ee_id >= 0 and base_id >= 0:
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
        result = solve_position_ik(
            self._model,
            self._data,
            ee_body=self.ee_body,
            joint_names=self.joint_names,
            target_pos=target_xyz_world,
            tol_m=0.025,
            max_iters=100,
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
        return True, result.pos_error_m

    def _placements(self) -> dict[str, dict[str, Any]] | None:
        from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements

        return read_sim_object_placements(self.robot.get_emet_session())

    def grasp_only(
        self,
        object_query: str,
        *,
        object_gt_body: str | None = None,
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
        pregrasp = obj_pos + np.array([0.0, 0.0, 0.15])
        grasp = obj_pos + np.array([0.0, 0.0, 0.02])
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
        lift = grasp + np.array([0.0, 0.0, self.lift_m])
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

        pregrasp = obj_pos + np.array([0.0, 0.0, 0.15])
        grasp = obj_pos + np.array([0.0, 0.0, 0.02])
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
        lift = grasp + np.array([0.0, 0.0, self.lift_m])
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

        # Final place error from session GT
        pl2 = self._placements() or pl
        if body in pl2:
            final = np.asarray(pl2[body]["pos"], dtype=np.float64).reshape(3)
            p_err = float(np.linalg.norm(final[:2] - recep_pos[:2]))
        return KinematicPickPlaceResult(True, body, self.ee_body, g_err, p_err, "ok")
