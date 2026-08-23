# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for deterministic TAMP task search (no sim)."""

from __future__ import annotations

import numpy as np

from emet.controller.task.tamp.smoke_grasps import plant_mixed_grasp_poses
from emet.controller.task.tamp.task_search import (
    TaskPlan,
    TaskPlanStep,
    approach_pose_for_object_xy,
    plan_pick_place,
    rank_grasps_by_ik,
)


class _FakeRobot:
    def __init__(self, placements: dict):
        self._pl = placements
        self.moved: list[np.ndarray] = []

    def get_emet_session(self):
        return {
            "is_simulation": True,
            "emet_robot_id": "rby1",
            "sim_object_placements": self._pl,
            "capabilities": {"kinematic_manip": True, "sim_set_body_pose": True},
        }

    def move_base_to(self, xyt, blocking=True, world_frame=True):
        self.moved.append(np.asarray(xyt, dtype=np.float64).copy())


class _FakeGrasp:
    def __init__(self, xyz):
        T = np.eye(4)
        T[:3, 3] = np.asarray(xyz, dtype=np.float64)
        self.T_world = T
        self.position = T[:3, 3].copy()


def test_approach_pose_faces_minus_y():
    p = approach_pose_for_object_xy([1.0, 2.0], standoff=0.4)
    assert abs(p[0] - 1.0) < 1e-9
    assert abs(p[1] - 2.4) < 1e-9
    assert abs(p[2] + np.pi / 2) < 1e-9


def test_plant_mixed_grasps_puts_decoys_first():
    poses = plant_mixed_grasp_poses([0.1, -0.5, 0.8], n_infeasible=2)
    assert len(poses) == 3
    assert poses[0].asset_id.startswith("decoy")
    assert poses[-1].asset_id == "reachable_topdown"
    # top_down_grasp_T adds the 0.02 m approach z-offset (grasp frame +Z into object).
    np.testing.assert_allclose(poses[-1].position, np.array([0.1, -0.5, 0.82]), atol=1e-9)


def test_rank_grasps_skips_infeasible_picks_reachable():
    """Offline IK: planted far grasps fail; near-EE grasp succeeds and is chosen."""
    import mujoco

    from emet.motion.mujoco_arm_ik import RBY1_LEFT_ARM_JOINTS, RBY1_LEFT_EE_BODY
    from emet.robots.rby1 import Rby1Backend

    mjcf = Rby1Backend().get_spec().mjcf_path
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    for name in RBY1_LEFT_ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        data.qpos[int(model.jnt_qposadr[jid])] = 0.5 * (lo + hi)
    mujoco.mj_forward(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RBY1_LEFT_EE_BODY)
    ee = np.asarray(data.body(ee_id).xpos, dtype=np.float64).copy()
    poses = plant_mixed_grasp_poses(ee + np.array([0.0, 0.0, 0.05]), n_infeasible=2)
    scores = rank_grasps_by_ik(
        model,
        data,
        ee_body=RBY1_LEFT_EE_BODY,
        joint_names=RBY1_LEFT_ARM_JOINTS,
        grasp_poses=poses,
        top_k=8,
    )
    assert any(ok for _i, _e, ok in scores), scores
    assert any(not ok for _i, _e, ok in scores), scores
    chosen = next(i for i, _e, ok in scores if ok)
    assert chosen == len(poses) - 1


def test_plan_pick_place_chooses_reachable_when_executor_ranks():
    """plan_pick_place with a stub executor that ranks via real IK model."""
    import mujoco

    from emet.motion.mujoco_arm_ik import RBY1_LEFT_ARM_JOINTS, RBY1_LEFT_EE_BODY
    from emet.robots.rby1 import Rby1Backend

    mjcf = Rby1Backend().get_spec().mjcf_path
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    for name in RBY1_LEFT_ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        data.qpos[int(model.jnt_qposadr[jid])] = 0.5 * (lo + hi)
    mujoco.mj_forward(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RBY1_LEFT_EE_BODY)
    ee = np.asarray(data.body(ee_id).xpos, dtype=np.float64).copy()
    good = ee + np.array([0.0, 0.0, 0.04])
    poses = plant_mixed_grasp_poses(good, n_infeasible=2)

    class _Exe:
        ee_body = RBY1_LEFT_EE_BODY
        joint_names = RBY1_LEFT_ARM_JOINTS
        _model = model
        _data = data

        def _ensure_model(self):
            return True

    pl = {
        "obj_a": {"cat": "red cylinder", "pos": good.tolist()},
        "cube_b": {"cat": "blue cube", "pos": [0.2, -0.4, 0.75]},
    }
    robot = _FakeRobot(pl)
    plan = plan_pick_place(
        robot,
        object_query="red cylinder",
        receptacle_query="blue cube",
        grasp_poses=poses,
        object_gt_body="obj_a",
        receptacle_gt_body="cube_b",
        executor=_Exe(),
    )
    assert plan.success
    assert plan.chosen_grasp_index == len(poses) - 1
    assert any(ok for _i, _e, ok in plan.grasp_scores)
    assert any(not ok for _i, _e, ok in plan.grasp_scores)


def test_plan_pick_place_without_executor():
    pl = {
        "obj_a": {"cat": "red cylinder", "pos": [0.1, -0.5, 0.8]},
        "cube_b": {"cat": "blue cube", "pos": [0.2, -0.4, 0.75]},
    }
    robot = _FakeRobot(pl)
    grasps = [_FakeGrasp([0.1, -0.5, 0.85]), _FakeGrasp([0.12, -0.5, 0.85])]
    plan = plan_pick_place(
        robot,
        object_query="red cylinder",
        receptacle_query="blue cube",
        grasp_poses=grasps,
        object_gt_body="obj_a",
        receptacle_gt_body="cube_b",
        executor=None,
    )
    assert plan.success
    assert plan.chosen_grasp_index == 0
    assert [s.op for s in plan.steps] == ["approach", "grasp", "place"]
    assert any("approach@" in n for n in plan.expanded_nodes)


def test_plan_missing_object():
    robot = _FakeRobot({})
    plan = plan_pick_place(
        robot,
        object_query="x",
        receptacle_query="y",
        grasp_poses=[],
        object_gt_body="missing",
        executor=None,
    )
    assert not plan.success
    assert plan.message == "object_not_in_gt"


def test_task_plan_dataclass():
    p = TaskPlan(steps=[TaskPlanStep("approach", {"xyt": [0, 0, 0]})], object_body="a", receptacle_body=None)
    assert p.steps[0].op == "approach"
