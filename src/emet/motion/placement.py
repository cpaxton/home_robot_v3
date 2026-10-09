# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Multiple placement approach/arm paths against an explicit 3D geometry snapshot.

Plans on private MjData. This does not execute, attach, release, infer support
surfaces, or certify a base route. A result covers sampled arm edges and candidate
base endpoints, conditional on the supplied scene and attachment estimate.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import mujoco
import numpy as np

from emet.motion.arm_rrt import joint_limits_from_model, make_arm_validate_fn, plan_arm_joint_path
from emet.motion.mujoco_arm_ik import joint_qpos_addrs, solve_pose_ik
from emet.motion.placement_geometry import HeldObject, PlacementCollisionChecker, PlacementScene, bounds_array


@dataclass
class PlacementPath:
    base_xyt: np.ndarray
    object_center: np.ndarray
    ee_targets: tuple[np.ndarray, np.ndarray]
    ee_rotation: np.ndarray
    segments: tuple[list[np.ndarray], list[np.ndarray]]
    cost: float


@dataclass
class PlacementSearchResult:
    paths: list[PlacementPath] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    geometry_source: str = ""
    collision_scope: str = "sampled_arm_and_payload;base_endpoint_only"


def surface_placement_centers(surface_bounds, *, payload: HeldObject, ee_rotation, clearance_m=.02):
    """Candidate object-box centers above an explicitly selected horizontal support.

    Bounds must describe the support region, not the center of an appliance or an
    inferred interior shelf. The whole footprint must fit, with clearance.
    """
    lo, hi = bounds_array(surface_bounds)
    if not np.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError("Positive finite support clearance required")
    rotated = payload.vertices_ee @ np.asarray(ee_rotation).reshape(3, 3).T
    half = np.ptp(rotated, axis=0) / 2
    low, high = lo[:2] + half[:2] + clearance_m, hi[:2] - half[:2] - clearance_m
    if np.any(low > high):
        return []
    center = (low + high) / 2
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    return [np.r_[center + np.array(offset) * (high - low) / 2, hi[2] + half[2] + clearance_m]
            for offset in offsets]


def placement_base_candidates(center_xy, *, current_xyt, radii=(.55, .75, .95), count=16, yaw_offset=0.):
    """Bounded, deterministic rings; nearby endpoints first, no category-specific rules."""
    center, current = np.asarray(center_xy, dtype=float), np.asarray(current_xyt, dtype=float)
    if center.shape != (2,) or current.shape != (3,) or not np.isfinite(np.r_[center, current, yaw_offset]).all():
        raise ValueError("Finite placement center and base pose required")
    if not 1 <= count <= 64 or not radii or any(not np.isfinite(r) or r <= 0 for r in radii):
        raise ValueError("Invalid approach search budget")
    candidates = []
    for radius in radii:
        for angle in np.arange(count) * 2 * np.pi / count:
            xy = center + radius * np.array([np.cos(angle), np.sin(angle)])
            yaw = angle + np.pi + yaw_offset
            candidates.append(np.r_[xy, np.arctan2(np.sin(yaw), np.cos(yaw))])
    return [current.copy(), *sorted(candidates, key=lambda p: float(np.linalg.norm(p[:2] - current[:2])))]


def validated_dense_path(model, data, joint_names, path, collision, *, joint_step=.025):
    """Check all endpoints and interpolated edges, returning the samples to execute."""
    if not np.isfinite(joint_step) or joint_step <= 0:
        raise ValueError("Positive finite edge step required")
    if not path:
        return None
    validate = make_arm_validate_fn(model, data, joint_names, collision)
    dense = [np.asarray(path[0], dtype=float).copy()]
    if not validate(dense[0]):
        return None
    for a, b in zip(path, path[1:], strict=False):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != (len(joint_names),) or b.shape != a.shape or not np.isfinite(np.r_[a, b]).all():
            return None
        steps = max(1, int(np.ceil(np.linalg.norm(b - a) / joint_step)))
        for t in np.arange(1, steps + 1) / steps:
            q = a + t * (b - a)
            if not validate(q):
                return None
            dense.append(q.copy())
    return dense


def plan_placement_paths(
    model, data, *, joint_names: Sequence[str], ee_body: str, robot_body: str,
    scene: PlacementScene, payload: HeldObject, object_centers, base_candidates,
    set_base: Callable, contact_bodies=(), max_solutions=3, ik_attempts=3,
    max_candidates=64, max_ik_calls=96, rrt_max_iter=400, preplace_height_m=.12, margin_m=.005,
    endpoint_validator: Callable | None = None, seed=0, coupled_groups=(),
) -> PlacementSearchResult:
    """Search base endpoints, support points, IK seeds and RRT paths before execution.

    ``set_base(model, private_data, xyt)`` must preserve measured base height and
    return True on success. ``endpoint_validator`` is an optional extra live
    simulator/navigation check, not a replacement for the 3D volume checker.
    Robot model must be standalone; scene and payload are supplied separately.
    At least one path is needed; callers may request several alternatives.
    """
    if not 1 <= max_solutions <= 16 or not 1 <= ik_attempts <= 16 or not 1 <= max_candidates <= 256:
        raise ValueError("Invalid placement search budget")
    if not 1 <= max_ik_calls <= 4096:
        raise ValueError("Invalid IK call budget")
    if coupled_groups:
        raise ValueError("Placement RRT currently requires independently controlled joints")
    if not np.isfinite(preplace_height_m) or preplace_height_m <= 0:
        raise ValueError("Positive finite preplace height required")
    centers = [np.asarray(p, dtype=float) for p in object_centers]
    poses = [np.asarray(p, dtype=float) for p in base_candidates][:max_candidates]
    if any(p.shape != (3,) or not np.isfinite(p).all() for p in centers + poses):
        raise ValueError("Finite 3D placement targets and base poses required")
    if len(centers) > 64:
        raise ValueError("Too many placement targets")
    probe = mujoco.MjData(model)
    mujoco.mj_copyData(probe, model, data)
    mujoco.mj_forward(model, probe)
    original = probe.qpos.copy()
    qadr = joint_qpos_addrs(model, joint_names)
    start_q = original[qadr].copy()
    rotation = probe.body(ee_body).xmat.reshape(3, 3).copy()
    rotated = payload.vertices_ee @ rotation.T
    center_offset = (rotated.min(axis=0) + rotated.max(axis=0)) / 2
    checker = PlacementCollisionChecker(model, robot_body=robot_body, ee_body=ee_body,
                                        scene=scene, payload=payload, contact_bodies=contact_bodies,
                                        margin_m=margin_m)
    low, high = joint_limits_from_model(model, joint_names)
    rng = np.random.default_rng(seed)
    rejects = Counter()
    result = PlacementSearchResult(geometry_source=scene.source)
    ik_calls = 0
    for pose in poses:
        probe.qpos[:] = original
        if not set_base(model, probe, pose):
            rejects['unsupported_base'] += 1
            continue
        if endpoint_validator is not None and not endpoint_validator(pose.copy()):
            rejects['base_endpoint_rejected'] += 1
            continue
        base_state = probe.qpos.copy()
        if checker.configuration_collides(model, probe):
            rejects[checker.last_reason] += 1
            continue
        for center in centers:
            place = center - center_offset
            targets = (place + np.array([0., 0., preplace_height_m]), place)
            for attempt in range(ik_attempts):
                probe.qpos[:] = base_state
                q0 = start_q.copy()
                segments = []
                for target in targets:
                    if ik_calls >= max_ik_calls:
                        rejects["ik_budget_exhausted"] += 1
                        result.rejections = dict(rejects)
                        result.paths.sort(key=lambda path: path.cost)
                        return result
                    ik_calls += 1
                    if attempt:
                        probe.qpos[qadr] = rng.uniform(low, high)
                    ik = solve_pose_ik(model, probe, ee_body=ee_body, joint_names=joint_names,
                                       target_pos=target, target_rotation=rotation, tol_m=.005,
                                       tol_rad=.03, coupled_groups=coupled_groups)
                    if not ik.success:
                        rejects['pose_ik_failed'] += 1
                        break
                    goal = probe.qpos[qadr].copy()
                    path = plan_arm_joint_path(model, probe, joint_names=joint_names, q_start=q0,
                                               q_goal=goal, collision=checker, max_iter=rrt_max_iter,
                                               step_size=.025, linear_fallback=False)
                    if not path.success:
                        rejects[path.reason or 'arm_path_failed'] += 1
                        break
                    dense = validated_dense_path(model, probe, joint_names, path.waypoints, checker)
                    if dense is None:
                        rejects['arm_edge_collision'] += 1
                        break
                    segments.append(dense)
                    q0 = goal
                    probe.qpos[qadr] = goal
                if len(segments) == 2:
                    cost = sum(float(np.linalg.norm(b - a)) for segment in segments
                               for a, b in zip(segment, segment[1:], strict=False))
                    result.paths.append(PlacementPath(pose.copy(), center.copy(), targets, rotation.copy(),
                                                      tuple(segments), cost))
                    break  # seek a different target or base for the next alternative
            if len(result.paths) >= max_solutions:
                break
        if len(result.paths) >= max_solutions:
            break
    result.paths.sort(key=lambda path: path.cost)
    result.rejections = dict(rejects)
    return result
