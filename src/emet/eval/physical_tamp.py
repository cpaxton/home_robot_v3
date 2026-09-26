# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Frozen physical TAMP fixtures and GT adapters for the existing motion stack."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from emet.motion.base import XYT
from emet.motion.mujoco_collision import MujocoSceneCollisionChecker, body_subtree
from emet.motion.navigation_sweep import validate_navigation_sweep
from emet.simulation.molmospaces_mobile_autoplace import write_base_freejoint_xyt


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_fixture(sim_path, scorer_path, output, *, seed):
    """Resolve exact input files and archive compiled geometry/initial state."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(Path(sim_path).expanduser().read_text())
    scene = Path(config["scene_path"]).expanduser().resolve(strict=True)
    scorer = json.loads(Path(scorer_path).expanduser().read_text())
    model = mujoco.MjModel.from_xml_path(str(scene))
    for name in (scorer["object_body"], scorer["support_body"], scorer["ee_body"], *scorer["gripper_bodies"]):
        model.body(name)  # Fail on missing identity; no category fallback.
    mujoco.mj_saveModel(model, str(output / "model.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.savez(output / "initial_state.npz", qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, act=data.act)
    (output / "scene.xml").write_bytes(scene.read_bytes())
    config.update(seed=int(seed), scene_path=str(scene), headless=True)
    (output / "sim.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    (output / "scorer.json").write_text(json.dumps(scorer, indent=2) + "\n")
    manifest = {
        "schema": 1,
        "source_scene": str(scene),
        "seed": int(seed),
        "robot": config["robot"],
        "scene_sha256": sha256(scene),
        "model_sha256": sha256(output / "model.mjb"),
        "initial_state_sha256": sha256(output / "initial_state.npz"),
        "scorer_sha256": sha256(output / "scorer.json"),
        "scorer_source_sha256": sha256(Path(__file__).with_name("manipulation_trace.py")),
        "mujoco_version": mujoco.__version__,
        "task": scorer,
        "initial_state_kind": "model_default_before_controller_startup",
        "budgets": {
            "candidates": 32,
            "mcts_iterations": 120,
            "base_rrt_iterations": 400,
            "arm_rrt_iterations": 400,
            "replans": 2,
            "wall_timeout_s": 600,
        },
        "tolerances": {
            "base_position_m": 0.05,
            "base_yaw_rad": 0.1,
            "ik_position_m": 0.01,
            "ik_orientation_rad": 0.1,
            "joint_tracking": 0.03,
            "penetration_m": 0.001,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return model, data, scorer, manifest


def base_pose(model, data, body="base_link"):
    mujoco.mj_forward(model, data)
    pose = data.body(body)
    rotation = pose.xmat.reshape(3, 3)
    return np.array([*pose.xpos[:2], np.arctan2(rotation[1, 0], rotation[0, 0])])


class SceneNavigationSpace(XYT):
    """GT scene collision adapter for the existing RRT-Connect base planner."""

    def __init__(self, model, data, checker, *, base_body="base_link", seed=0):
        self.model, self.data, self.checker, self.base_body = model, data, checker, base_body
        start = base_pose(model, data, base_body)
        super().__init__(mins=np.r_[start[:2] - 6, -np.pi], maxs=np.r_[start[:2] + 6, np.pi])
        self.step_size = 0.025
        self.rng = np.random.default_rng(seed)
        self.last_validity = {}
        self.min_clearance_m = .22
        self.environment_geoms = np.array([
            g for g in range(model.ngeom)
            if model.geom_bodyid[g] not in checker.robot_ids
            and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ], dtype=int)

    def sample(self):
        return self.rng.uniform(self.mins, self.maxs)

    def extend(self, start, goal):
        start, goal = np.asarray(start), np.asarray(goal)
        delta = np.arctan2(np.sin(goal[2] - start[2]), np.cos(goal[2] - start[2]))
        count = max(1, int(np.ceil(np.linalg.norm(goal[:2] - start[:2]) / 0.025)), int(np.ceil(abs(delta) / 0.05)))
        for t in np.linspace(0, 1, count + 1)[1:]:
            yield np.r_[start[:2] + t * (goal[:2] - start[:2]), start[2] + t * delta]

    def is_valid(self, pose):
        pose = np.asarray(pose)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            self.last_validity = {"reason": "invalid_navigation_pose"}
            return False
        if not write_base_freejoint_xyt(
            self.model,
            self.data,
            base_body_name=self.base_body,
            x=float(pose[0]),
            y=float(pose[1]),
            theta=float(pose[2]),
        ):
            self.last_validity = {"reason": "unsupported_base_model"}
            return False
        hit = self.checker.configuration_collides(self.model, self.data)
        self.last_validity = {"reason": "scene_collision" if hit else "ok", "contacts": self.checker.last_contacts}
        if hit:
            return False
        # Keep the existing 22 cm center-clearance gate separate from full-body
        # contact checks. AABBs are conservative measured geometry, not dilated maps.
        geoms = self.environment_geoms
        if self.checker.payload_body:
            payload = body_subtree(self.model, self.checker.payload_body)
            geoms = np.array([g for g in geoms if self.model.geom_bodyid[g] not in payload], dtype=int)
        if len(geoms):
            rotation = self.data.geom_xmat[geoms].reshape(-1, 3, 3)
            centers = self.data.geom_xpos[geoms] + np.einsum("nij,nj->ni", rotation, self.model.geom_aabb[geoms, :3])
            half = np.einsum("nij,nj->ni", np.abs(rotation), self.model.geom_aabb[geoms, 3:])
            height = float(self.data.body(self.base_body).xpos[2])
            relevant = (centers[:, 2] + half[:, 2] > height + .05) & (centers[:, 2] - half[:, 2] < height + .5)
            distances = np.linalg.norm(np.maximum(np.abs(centers[:, :2] - pose[:2]) - half[:, :2], 0), axis=1)
            clearance = float(distances[relevant].min()) if relevant.any() else None
            self.last_validity["min_clearance_m"] = clearance
            if clearance is not None and clearance < self.min_clearance_m:
                self.last_validity["reason"] = "below_clearance"
                return False
        return True

    def plan_route(self, start, goal):
        from emet.motion.algo import get_planner

        before = self.data.qpos.copy()
        try:
            if validate_navigation_sweep(self, start, [goal])[0]:
                # Short waypoints bound unobserved tracking divergence.
                points = list(self.extend(start, goal))
                return [q.tolist() for q in points[7::8]] + [np.asarray(goal).tolist()]
            planner = get_planner("rrt_connect", self, self.is_valid, max_iter=400, goal_tolerance=0.025)
            result = planner.plan(np.asarray(start), np.asarray(goal))
            if not result.success:
                return []
            route = [node.state for node in result.trajectory]
            route.append(np.asarray(goal))
            if not validate_navigation_sweep(self, start, route)[0]:
                return []
            return [np.asarray(q).tolist() for q in route]
        finally:
            self.data.qpos[:] = before
            mujoco.mj_forward(self.model, self.data)


def make_scene_checker(model, scorer):
    """Allow explicit wheel/floor and gripper/target contact pairs only."""
    wheels = [model.body(i).name for i in range(model.nbody) if "wheel" in model.body(i).name.lower()]
    floors = [model.body(i).name for i in range(model.nbody) if model.body(i).name.lower().startswith("floor")]
    # Plane geometry is the declared ground, including unnamed worldbody planes.
    floors.extend(
        model.body(model.geom_bodyid[g]).name
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
    )
    wheel_ids = {model.body(name).id for name in wheels}
    floor_ids = {model.body(name).id for name in floors if name != "world"}
    wheel_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in wheel_ids]
    ground_geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] in floor_ids or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
    ]
    allowed = []
    target = body_subtree(model, scorer["object_body"])
    for gripper in scorer["gripper_bodies"]:
        allowed.extend((model.body(g).name, model.body(o).name) for g in body_subtree(model, gripper) for o in target)
    return MujocoSceneCollisionChecker(
        model,
        robot_body="base_link",
        allowed_pairs=allowed,
        allowed_geom_pairs=[(wheel, floor) for wheel in wheel_geoms for floor in ground_geoms],
    )


def score_physical_acceptance(trace_path, audit_path, *, execution_completed):
    """Independent task score plus continuous retention and actuation evidence."""
    from emet.eval.manipulation_trace import score_trace

    rows = [json.loads(line) for line in Path(trace_path).read_text().splitlines()][1:]
    rows = [row for row in rows if row.get("scored_execution", False)]
    result = score_trace(rows)
    audit = (
        [json.loads(line) for line in Path(audit_path).read_text().splitlines()] if Path(audit_path).exists() else []
    )
    result["forbidden_actuation"] = any(not row.get("accepted", False) for row in audit)
    result["actuation_audit_present"] = bool(audit)
    result["execution_completed"] = bool(execution_completed)
    result["robot_collision_audited"] = bool(rows) and all(r.get("robot_collision_audited", False) for r in rows)
    result["forbidden_robot_contacts"] = rows[-1].get("forbidden_robot_contacts", []) if rows else []
    result["physical_transport_success"] = False
    if result.get("physical_pick_success") and result.get("physical_place_success"):
        start = result["pick_time"]
        release = result["first_place_time"] - 1.0
        carry = [r for r in rows if start <= r["sim_time"] < release and not r["support_contact"]]
        if carry:
            origin = np.asarray(carry[0]["relative_pos"])
            result["physical_transport_success"] = bool(
                all(r["gripper_contact"] and not r["other_contact"] for r in carry)
                and all(np.linalg.norm(np.asarray(r["relative_pos"]) - origin) <= 0.02 for r in carry)
                and all(b["sim_time"] - a["sim_time"] <= 0.25 for a, b in zip(carry, carry[1:], strict=False))
            )
    result["task_success"] = bool(
        execution_completed
        and result.get("verified")
        and result.get("physical_place_success")
        and result["physical_transport_success"]
        and audit
        and not result["forbidden_actuation"]
        and result["robot_collision_audited"]
        and not result["forbidden_robot_contacts"]
    )
    return result
