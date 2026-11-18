# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Recertify an inert native proposal under genuinely reacquired local evidence.

Only private data is changed. No IK search, commands, coarse-grid equivalence,
renewed captures, or additional collision allowances authorize this boundary.
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.native_scene import model_signature, qpos_difference
from emet.motion.arm_rrt import plan_arm_joint_path
from emet.motion.local_scene_store import LocalSceneLease
from emet.motion.mujoco_arm_ik import joint_qpos_addrs
from emet.motion.mujoco_collision import body_subtree


def frozen(value):
    value = np.asarray(value, float)
    return np.frombuffer(value.tobytes(), dtype=float).reshape(value.shape)


@dataclass(frozen=True)
class JointProof:
    phase: str
    qpos: np.ndarray
    names: tuple[str, ...]
    waypoints: tuple[np.ndarray, ...]
    payload: tuple
    endpoint: tuple | None = None


@dataclass(frozen=True)
class BaseProof:
    qpos: np.ndarray
    payload: tuple
    start: np.ndarray
    goal: np.ndarray


@dataclass(frozen=True)
class NativeProposal:
    intent: str
    model_signature: str
    policy: str
    boot: str
    initial: np.ndarray
    joints: tuple[JointProof, ...]
    base: BaseProof | None
    source_lease: LocalSceneLease
    source_capture: object
    execution_policy: str


@dataclass(frozen=True)
class RefreshedNativeScene:
    lease: LocalSceneLease
    checks: tuple

    def __post_init__(self):
        object.__setattr__(self, "checks", tuple(self.checks))
        if not isinstance(self.lease, LocalSceneLease) or not self.checks or not all(callable(c) for c in self.checks):
            raise ValueError("invalid_native_scene_refresh")
        for check in self.checks:
            owner = getattr(check, "__self__", None)
            if getattr(owner, "lease", self.lease) is not self.lease:
                raise ValueError("invalid_native_scene_refresh")


def payload_copy(checker):
    obj, parent, transform = checker.capture_payload()
    return obj, parent, None if transform is None else frozen(transform)


def joint_proof(executor, phase, qpos, payload, path, *, names=None, endpoint=None):
    return JointProof(
        phase,
        frozen(qpos),
        tuple(names or executor.joint_names),
        tuple(frozen(q) for q in path),
        payload,
        None if endpoint is None else tuple(frozen(x) for x in endpoint),
    )


def intent_digest(plan):
    ex = plan.executor
    payload = [
        plan.object_body,
        plan.support_body,
        plan.goal.tolist(),
        plan.rotation.tolist(),
        None if plan.placement_region is None else plan.placement_region.tolist(),
        plan.transport,
        plan.boot,
        plan.planning_ik_calls,
        plan.planning_ik_limit,
        plan.planning_seconds_remaining,
        list(ex.joint_names),
        ex.gripper_open_configuration,
        ex.gripper_close_configuration,
        ex.ee_body,
        ex.input_mode,
        ex.position_tolerance_m,
        ex.orientation_tolerance_rad,
        ex.joint_tolerance,
        ex.coupled_groups,
        ex.joint_limit_margins,
        type(ex.collision).__name__,
        ex.collision.penetration_tolerance_m,
        sorted(map(int, ex.collision.robot_ids)),
        sorted(sorted(pair) for pair in ex.collision.allowed_pairs),
        sorted(sorted(map(int, pair)) for pair in ex.collision.allowed_geom_pairs),
        [[np.asarray(q).tolist() for q in path] for path in ex.grasp_paths],
        [[np.asarray(q).tolist() for q in path] for path in ex.place_paths],
        [[np.asarray(x).tolist() for x in t] for t in ex.grasp_targets],
        [[np.asarray(x).tolist() for x in t] for t in ex.place_targets],
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def policy_digest(adapter):
    names = (
        "policy",
        "payload_binding",
        "arm_extension_margin_m",
        "gripper_close_loose",
        "gripper_contact_hold_fraction",
        "checked_base_approach",
    )
    values = [getattr(adapter, n, None) for n in names]
    values.append(repr(getattr(adapter, "release_limits", None)))
    return hashlib.sha256(json.dumps(values, allow_nan=False).encode()).hexdigest()


def require_binding(adapter, plan):
    proof = plan.proposal
    if (
        not isinstance(proof, NativeProposal)
        or plan.base_approach is not None
        or plan.approach is not None
        or adapter.boot != proof.boot
        or model_signature(adapter.model) != proof.model_signature
        or policy_digest(adapter) != proof.execution_policy
        or intent_digest(plan) != proof.intent
    ):
        raise ObservationUnavailable("scene_changed_replan")

    admission = plan.admission
    if admission is None and plan.evidence_lease is not None:
        raise ObservationUnavailable("scene_changed_replan")
    if admission is not None:
        current = plan.executor.collision.additional_scene_checks
        if (
            not isinstance(admission, RefreshedNativeScene)
            or plan.evidence_lease is not admission.lease
            or plan.executor.planning_lease is not admission.lease
            or len(current) != len(admission.checks)
            or any(a is not b for a, b in zip(current, admission.checks, strict=True))
            or any(
                getattr(getattr(check, "__self__", None), "lease", admission.lease) is not admission.lease
                for check in admission.checks
            )
        ):
            raise ObservationUnavailable("scene_changed_replan")


def seal_proposal(adapter, plan, base):
    if plan.base_approach is not None or plan.approach is not None:
        raise ObservationUnavailable("native_execution_unavailable")
    ex = plan.executor
    grasp, place = getattr(ex, "_grasp_proof", ()), getattr(ex, "_place_proof", ())
    if (
        len(grasp) != 4
        or len(place) != 4
        or not isinstance(getattr(ex, "planning_lease", None), LocalSceneLease)
        or getattr(ex, "planning_capture", None) is None
    ):
        raise ObservationUnavailable("native_execution_unavailable")
    return NativeProposal(
        intent_digest(plan),
        adapter.signature,
        adapter.policy,
        adapter.boot,
        frozen(plan.initial),
        (*grasp, *place),
        base,
        ex.planning_lease,
        ex.planning_capture,
        policy_digest(adapter),
    )


def finalize_proposal(adapter, plan, reacquire, *, budget_s=240.0):
    """Acquire new evidence, recertify fixed edges, then bind the fresh lease.

    The caller has not published this proposal. The source owner invalidates
    older handles/leases before acquisition; failure publishes nothing.
    """
    proof = plan.proposal
    if (
        not isinstance(proof, NativeProposal)
        or not np.isfinite(budget_s)
        or budget_s <= 0
        or intent_digest(plan) != proof.intent
    ):
        raise ObservationUnavailable("scene_changed_replan")
    require_binding(adapter, plan)
    calls = adapter.planning_ik_calls
    adapter.plans.clear()
    proof.source_lease.retire()
    acquisition_barrier = time.monotonic()
    try:
        scene = reacquire()
    except TimeoutError as error:
        raise ObservationUnavailable("observation_unavailable") from error
    if not isinstance(scene, RefreshedNativeScene):
        raise ObservationUnavailable("native_execution_unavailable")
    with scene.lease.borrow() as fresh_scene:
        capture = fresh_scene.support.capture
        if (
            scene.lease.generation == proof.source_lease.generation
            or capture.observation_id == proof.source_capture.observation_id
            or capture.calibration_id != proof.source_capture.calibration_id
            or capture.earliest < acquisition_barrier
            or capture.earliest <= proof.source_capture.latest
            or fresh_scene.support.object_ref != plan.object_body
            or fresh_scene.support.support_ref != plan.support_body
        ):
            raise ObservationUnavailable("scene_changed_replan")
    del fresh_scene
    deadline = time.monotonic() + budget_s
    ex, model = plan.executor, adapter.model
    data = mujoco.MjData(model)
    adapter.synchronize(data)
    before = data.qpos.copy()
    if (
        adapter.boot != proof.boot
        or adapter.policy != proof.policy
        or model_signature(model) != proof.model_signature
        or intent_digest(plan) != proof.intent
        or np.max(np.abs(qpos_difference(model, before, proof.initial))) > 1e-8
    ):
        raise ObservationUnavailable("scene_changed_replan")
    checker = copy(ex.collision)
    checker.robot_ids = set(ex.collision.robot_ids)
    checker.allowed_pairs = set(ex.collision.allowed_pairs)
    checker.allowed_geom_pairs = set(ex.collision.allowed_geom_pairs)
    checker.last_contacts = []

    def require():
        scene.lease.require_current()
        if time.monotonic() >= deadline:
            raise ValueError("no_plan_within_budget")

    def checked(current, m, d):
        require()
        for check in scene.checks:
            outcome = check(current, m, d)
            if type(outcome) is not bool:
                raise ObservationUnavailable("native_execution_unavailable")
            if outcome:
                return True
        return False

    checker.additional_scene_checks = (checked,)
    robot_ids = body_subtree(model, "base_link")
    robot_addresses = [
        a
        for j in range(model.njnt)
        if int(model.jnt_bodyid[j]) in robot_ids
        for a in range(int(model.jnt_qposadr[j]), int(model.jnt_qposadr[j + 1]) if j + 1 < model.njnt else model.nq)
    ]
    root = int(model.body(plan.object_body).jntadr[0])
    if root < 0 or model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
        raise ObservationUnavailable("native_execution_unavailable")
    objaddr = int(model.jnt_qposadr[root])

    def restore(qpos, payload):
        data.qpos[:] = before
        data.qpos[robot_addresses] = qpos[robot_addresses]
        # Includes the explicitly released pose in opening/retreat segments.
        data.qpos[objaddr : objaddr + 7] = qpos[objaddr : objaddr + 7]
        checker.restore_payload(payload)
        mujoco.mj_kinematics(model, data)

    for segment in proof.joints:
        require()
        restore(segment.qpos, segment.payload)
        addresses = joint_qpos_addrs(model, segment.names)
        if not segment.waypoints:
            raise ObservationUnavailable("scene_changed_replan")
        # Recertify the bridge too, rather than trusting endpoint proximity.
        current = data.qpos[addresses].copy()
        for target in segment.waypoints:
            result = plan_arm_joint_path(
                model,
                data,
                joint_names=segment.names,
                q_start=current,
                q_goal=target,
                collision=checker,
                planner="linear",
                step_size=0.025,
                goal_tolerance=0.0001,
            )
            if not result.success:
                adapter.events.append(
                    {
                        "phase": "source_revalidation_rejected",
                        "stage": segment.phase,
                        "reason": result.reason,
                        "detail": result.detail,
                    }
                )
                raise ObservationUnavailable("scene_changed_replan")
            current = np.asarray(target).copy()
        data.qpos[addresses] = current
        if segment.endpoint is not None:
            mujoco.mj_kinematics(model, data)
            ee = data.body(ex.ee_body)
            position, rotation = segment.endpoint
            angle = np.arccos(np.clip((np.sum(ee.xmat.reshape(3, 3) * rotation) - 1) / 2, -1, 1))
            if np.linalg.norm(ee.xpos - position) > ex.position_tolerance_m or angle > ex.orientation_tolerance_rad:
                raise ObservationUnavailable("scene_changed_replan")
        adapter.events.append({"phase": "source_revalidated_segment", "stage": segment.phase})
    if proof.base is not None:
        # Deferred import avoids the native adapter/approach dependency cycle.
        from emet.controller.task.tamp.native_approach import sweep_clear

        restore(proof.base.qpos, proof.base.payload)
        if not sweep_clear(
            model,
            data,
            proof.base.start,
            proof.base.goal,
            loaded_checker=checker,
            event=lambda **r: adapter.events.append(r),
        ):
            raise ObservationUnavailable("scene_changed_replan")
    require()
    require_binding(adapter, plan)
    fresh = mujoco.MjData(model)
    adapter.synchronize(fresh)
    if (
        adapter.boot != proof.boot
        or intent_digest(plan) != proof.intent
        or np.max(np.abs(qpos_difference(model, fresh.qpos, before))) > 1e-8
        or adapter.planning_ik_calls != calls
    ):
        raise ObservationUnavailable("scene_changed_replan")
    require()
    plan.initial = fresh.qpos.copy()
    ex.collision.additional_scene_checks = scene.checks
    plan.evidence_lease = ex.planning_lease = scene.lease
    plan.admission = scene
    adapter.events.append(
        {
            "phase": "source_revalidation_complete",
            "ik_calls": 0,
            "segments": len(proof.joints),
            "base_sweep": proof.base is not None,
        }
    )
