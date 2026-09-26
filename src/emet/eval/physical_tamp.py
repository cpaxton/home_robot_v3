# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Frozen physical TAMP fixtures and GT adapters for the existing motion stack."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from emet.motion.base import XYT
from emet.motion.mujoco_collision import MujocoSceneCollisionChecker, body_subtree
from emet.motion.navigation_sweep import validate_navigation_sweep
from emet.simulation.molmospaces_mobile_autoplace import base_body_free_joint_qposadr


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
    mujoco.mj_kinematics(model, data)
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
            "candidates": 48,
            "mcts_iterations": 120,
            "base_rrt_iterations": 400,
            "base_route_wall_s": 10,
            "grasp_rotations": 2,
            "place_base_candidates": 41,
            "support_surfaces": 4,
            "release_points_per_surface": 9,
            "preplace_clearances_m": [0.12, 0.06, 0.03],
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
    mujoco.mj_kinematics(model, data)
    pose = data.body(body)
    rotation = pose.xmat.reshape(3, 3)
    return np.array([*pose.xpos[:2], np.arctan2(rotation[1, 0], rotation[0, 0])])


def write_offline_base_pose(model, data, *, base_body_name, x, y, theta):
    """Set an offline planning pose, retaining measured height and base tilt.

    This does not advance dynamics or write a live simulator. Callers run the
    scene collision checker after transforming the candidate configuration.
    """
    address = base_body_free_joint_qposadr(model, base_body_name)
    if address is None:
        return False
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, data.qpos[address + 3 : address + 7])
    rotation = rotation.reshape(3, 3)
    delta = theta - np.arctan2(rotation[1, 0], rotation[0, 0])
    yaw_quaternion = np.array([np.cos(delta / 2), 0.0, 0.0, np.sin(delta / 2)])
    quaternion = np.empty(4)
    mujoco.mju_mulQuat(quaternion, yaw_quaternion, data.qpos[address + 3 : address + 7])
    data.qpos[address : address + 2] = [x, y]
    data.qpos[address + 3 : address + 7] = quaternion
    mujoco.mj_kinematics(model, data)
    return True


def kinematic_base_candidates(model, data, *, target_xy, ee_body, extension_joints, base_body="base_link"):
    """Ground base candidates from actual FK reach, then let route/pose IK certify them.

    Five fixed actuator fractions and eight yaws form a deterministic bounded set.
    This avoids inferring the arm's extension line from the stowed EE-to-base angle.
    No reach threshold, joint limit, obstacle, or observed-space rule is relaxed.
    """
    before = data.qpos.copy()
    mujoco.mj_kinematics(model, data)
    base = data.body(base_body)
    origin = base.xpos[:2].copy()
    matrix = base.xmat.reshape(3, 3)
    yaw0 = np.arctan2(matrix[1, 0], matrix[0, 0])
    candidates = []
    try:
        for fraction in (0.25, 0.5, 0.75, 0.9, 0.98):
            for name in extension_joints:
                joint = model.joint(name)
                if not model.jnt_limited[joint.id]:
                    raise ValueError("Approach generation requires bounded extension joints")
                lo, hi = model.jnt_range[joint.id]
                data.qpos[joint.qposadr[0]] = lo + fraction * (hi - lo)
            mujoco.mj_kinematics(model, data)
            offset = data.body(ee_body).xpos[:2] - origin
            for delta in np.linspace(-np.pi, np.pi, 8, endpoint=False):
                c, sn = np.cos(delta), np.sin(delta)
                xy = np.asarray(target_xy) - np.array([[c, -sn], [sn, c]]) @ offset
                yaw = np.arctan2(np.sin(yaw0 + delta), np.cos(yaw0 + delta))
                candidates.append(np.r_[xy, yaw])
        candidates.sort(
            key=lambda p: (
                float(np.linalg.norm(p[:2] - origin)),
                float(abs(np.arctan2(np.sin(p[2] - yaw0), np.cos(p[2] - yaw0)))),
            )
        )
        return candidates
    finally:
        data.qpos[:] = before
        mujoco.mj_kinematics(model, data)


class SceneNavigationSpace(XYT):
    """GT scene collision adapter for the existing RRT-Connect base planner."""

    def __init__(self, model, data, checker, *, base_body="base_link", seed=0, route_timeout_s=10.0):
        self.model, self.data, self.checker, self.base_body = model, data, checker, base_body
        start = base_pose(model, data, base_body)
        super().__init__(mins=np.r_[start[:2] - 6, -np.pi], maxs=np.r_[start[:2] + 6, np.pi])
        self.step_size = 0.025
        self.rng = np.random.default_rng(seed)
        self.last_validity = {}
        if not np.isfinite(route_timeout_s) or route_timeout_s <= 0:
            raise ValueError("Positive finite route planning timeout required")
        self.route_timeout_s = float(route_timeout_s)
        self._route_deadline = None
        self.route_stats = {}
        self.min_clearance_m = 0.22
        self.environment_geoms = np.array(
            [
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] not in checker.robot_ids
                and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
                and (model.geom_contype[g] or model.geom_conaffinity[g])
            ],
            dtype=int,
        )

    def sample(self):
        return self.rng.uniform(self.mins, self.maxs)

    def extend(self, start, goal):
        start, goal = np.asarray(start), np.asarray(goal)
        delta = np.arctan2(np.sin(goal[2] - start[2]), np.cos(goal[2] - start[2]))
        count = max(1, int(np.ceil(np.linalg.norm(goal[:2] - start[:2]) / 0.025)), int(np.ceil(abs(delta) / 0.05)))
        for t in np.linspace(0, 1, count + 1)[1:]:
            yield np.r_[start[:2] + t * (goal[:2] - start[:2]), start[2] + t * delta]

    def is_valid(self, pose):
        started = time.monotonic()
        try:
            return self._is_valid(pose)
        finally:
            self.route_stats["validity_calls"] = self.route_stats.get("validity_calls", 0) + 1
            self.route_stats["validity_wall_s"] = self.route_stats.get("validity_wall_s", 0.0) + time.monotonic() - started

    def _is_valid(self, pose):
        if self._route_deadline is not None and time.monotonic() >= self._route_deadline:
            raise TimeoutError("route_planning_budget_exhausted")
        pose = np.asarray(pose)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            self.last_validity = {"reason": "invalid_navigation_pose"}
            return False
        if not write_offline_base_pose(
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
            relevant = (centers[:, 2] + half[:, 2] > height + 0.05) & (centers[:, 2] - half[:, 2] < height + 0.5)
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
        started = time.monotonic()
        self.route_stats = {"validity_calls": 0, "validity_wall_s": 0.0, "phase": "endpoints"}
        self._route_deadline = started + self.route_timeout_s
        try:
            # Reject invalid endpoints before an expensive mesh sweep. This
            # preserves the same validity contract while reserving search time
            # for candidates that can actually be reached.
            if not self.is_valid(start) or not self.is_valid(goal):
                return []
            self.route_stats["phase"] = "direct_sweep"
            if validate_navigation_sweep(self, start, [goal])[0]:
                # Short waypoints bound unobserved tracking divergence.
                points = list(self.extend(start, goal))
                return [q.tolist() for q in points[7::8]] + [np.asarray(goal).tolist()]
            self.route_stats["phase"] = "rrt"
            planner = get_planner("rrt_connect", self, self.is_valid, max_iter=400, goal_tolerance=0.025)
            result = planner.plan(np.asarray(start), np.asarray(goal))
            if not result.success:
                return []
            route = [node.state for node in result.trajectory]
            route.append(np.asarray(goal))
            self.route_stats["phase"] = "rrt_sweep"
            if not validate_navigation_sweep(self, start, route)[0]:
                return []
            return [np.asarray(q).tolist() for q in route]
        except TimeoutError as exc:
            if str(exc) != "route_planning_budget_exhausted":
                raise
            self.last_validity = {"reason": "route_planning_budget_exhausted"}
            return []
        finally:
            self.route_stats["wall_s"] = time.monotonic() - started
            self.last_validity["route_stats"] = dict(self.route_stats)
            self._route_deadline = None
            self.data.qpos[:] = before
            mujoco.mj_kinematics(self.model, self.data)


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
    result["release_command_audited"] = bool(rows) and all(r.get("release_command_audited", False) for r in rows)
    if result.get("physical_pick_success") and result.get("physical_place_success"):
        start = result["pick_time"]
        # Intentional release precedes settling onto the support. The motor
        # command distinguishes that interval from an uncommanded transport drop.
        openings = [r["sim_time"] for r in rows if r["sim_time"] > start and r.get("release_open_command", False)]
        release = openings[0] if openings else None
        result["release_command_time"] = release
        carry = [r for r in rows if release is not None and start <= r["sim_time"] < release]
        if carry:
            origin = np.asarray(carry[0]["relative_pos"])
            rotations = np.asarray([r["relative_rot"] for r in carry]).reshape(-1, 3, 3)
            angles = np.arccos(np.clip((np.einsum("nij,ij->n", rotations, rotations[0]) - 1) / 2, -1, 1))
            result["physical_transport_success"] = bool(
                all(r["gripper_contact"] and not r["other_contact"] for r in carry)
                and all(np.linalg.norm(np.asarray(r["relative_pos"]) - origin) <= 0.02 for r in carry)
                and np.max(angles) <= 0.1
                and all(b["sim_time"] - a["sim_time"] <= 0.25 for a, b in zip(carry, carry[1:], strict=False))
            )
    result["task_success"] = bool(
        execution_completed
        and result.get("verified")
        and result.get("physical_place_success")
        and result["physical_transport_success"]
        and result["release_command_audited"]
        and audit
        and not result["forbidden_actuation"]
        and result["robot_collision_audited"]
        and not result["forbidden_robot_contacts"]
    )
    return result


def save_motion_overview(path, model, data, *, scorer, footprint, initial_pose, candidates, witness):
    """Export auditable GT geometry and sampled routes; never feeds the planner."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon, Rectangle

    figure = Figure(figsize=(9, 8))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    mujoco.mj_kinematics(model, data)
    robot_ids = body_subtree(model, "base_link")
    z = data.body("base_link").xpos[2]
    for g in range(model.ngeom):
        if model.geom_bodyid[g] in robot_ids or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        rotation = data.geom(g).xmat.reshape(3, 3)
        center = data.geom(g).xpos + rotation @ model.geom_aabb[g, :3]
        half = np.abs(rotation) @ model.geom_aabb[g, 3:]
        if center[2] + half[2] <= z + 0.05 or center[2] - half[2] >= z + 1.5:
            continue
        ax.add_patch(Rectangle(center[:2] - half[:2], *(2 * half[:2]), color="0.5", alpha=0.08))
    for row in candidates:
        pose = row["approach"]
        color = "green" if row.get("accepted") else "crimson"
        ax.plot(*pose[:2], marker="x", color=color)
        ax.arrow(*pose[:2], 0.1 * np.cos(pose[2]), 0.1 * np.sin(pose[2]), color=color, head_width=0.025)
    for key, color in [("approach_route", "royalblue"), ("place_route", "darkorange")]:
        if witness.get(key):
            route = np.asarray(witness[key])
            ax.plot(route[:, 0], route[:, 1], color=color, label=key.replace("_", " "))
    pose = np.asarray(initial_pose)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [footprint.length / 2, footprint.width / 2]
    corners += [footprint.length_offset, footprint.width_offset]
    rotation = np.array([[np.cos(pose[2]), -np.sin(pose[2])], [np.sin(pose[2]), np.cos(pose[2])]])
    ax.add_patch(Polygon(corners @ rotation.T + pose[:2], fill=False, edgecolor="royalblue", label="initial footprint"))
    points = [pose[:2]]
    for key, label, color in [("object_body", "target", "red"), ("support_body", "support", "purple")]:
        point = data.body(scorer[key]).xpos[:2]
        points.append(point.copy())
        ax.scatter(*point, color=color, label=label, zorder=5)
    points = np.asarray(points)
    ax.set(
        xlim=(points[:, 0].min() - 1, points[:, 0].max() + 1),
        ylim=(points[:, 1].min() - 1, points[:, 1].max() + 1),
        xlabel="world X (m)",
        ylabel="world Y (m)",
        title="GT geometry / candidate rejections / planned routes\nRed candidates are rejected; this is not execution evidence",
    )
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(alpha=0.2)
    figure.savefig(path, dpi=160, bbox_inches="tight")


def support_release_points(model, data, *, support_body, payload_body, ee_body, max_surfaces=4):
    """Candidate support levels from named collision geometry and payload extent.

    Group nearby geom tops and require enough XY extent for the payload. This
    excludes tiny parked fixture geoms without a scene-specific height cutoff.
    These are candidates: full payload collision and physical settling still decide.
    """

    def bounds(body):
        ids = body_subtree(model, body)
        result = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] not in ids or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            rotation = data.geom(g).xmat.reshape(3, 3)
            center = data.geom(g).xpos + rotation @ model.geom_aabb[g, :3]
            half = np.abs(rotation) @ model.geom_aabb[g, 3:]
            result.append((center - half, center + half))
        return result

    mujoco.mj_kinematics(model, data)
    payload, support = bounds(payload_body), bounds(support_body)
    if not payload or not support:
        return []
    payload_lo = np.min([lo for lo, hi in payload], axis=0)
    payload_hi = np.max([hi for lo, hi in payload], axis=0)
    ee = data.body(ee_body).xpos
    bottom_offset = float(ee[2] - payload_lo[2])
    center_offset = (payload_lo[:2] + payload_hi[:2]) / 2 - ee[:2]
    groups = []
    for lo, hi in sorted(support, key=lambda b: -b[1][2]):
        if not groups or groups[-1][0][1][2] - hi[2] > 0.03:
            groups.append([])
        groups[-1].append((lo, hi))
    points = []
    surfaces = 0
    for group in groups:
        lo = np.min([a for a, b in group], axis=0)
        hi = np.max([b for a, b in group], axis=0)
        if np.any(hi[:2] - lo[:2] < payload_hi[:2] - payload_lo[:2]):
            continue
        # A named support's center may be occupied (e.g. by an appliance).
        # Search its interior with payload-sized edge clearance. These AABB
        # candidates still require full geometry checks and physical support.
        center = (lo[:2] + hi[:2]) / 2
        inset_half = (hi[:2] - lo[:2] - (payload_hi[:2] - payload_lo[:2])) / 2
        offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for offset in offsets:
            xy = center + 0.8 * inset_half * offset - center_offset
            point = np.r_[xy, hi[2] + bottom_offset + 0.015]
            if not any(np.allclose(point, previous) for previous in points):
                points.append(point)
        surfaces += 1
        if surfaces >= max_surfaces:
            break
    return points
