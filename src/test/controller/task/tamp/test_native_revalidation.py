# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Fresh evidence recertifies fixed edges without commands, IK or handle reuse."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
from revalidation_fixtures import local_scene

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor
from emet.controller.task.tamp import native_revalidation as module
from emet.controller.task.tamp.native_adapter import NativePlan
from emet.controller.task.tamp.native_dispatch import dispatch_boundary
from emet.motion.local_scene_store import LocalMapUnavailable, LocalSceneStore
from emet.motion.mujoco_collision import MujocoSceneCollisionChecker


@pytest.fixture
def proposal(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="base_link" pos="0 0 .2"><joint name="arm" type="slide" axis="1 0 0" range="-.2 .2"/>
        <geom type="sphere" size=".005"/></body>
      <body name="payload" pos=".15 .1 .1"><freejoint/><geom size=".005"/></body>
      <body name="table" pos="0 0 -.03"><geom type="box" size=".2 .2 .01"/></body>
      </worldbody></mujoco>""")
    live = mujoco.MjData(model)
    mujoco.mj_forward(model, live)
    checker = MujocoSceneCollisionChecker(model, robot_body="base_link")
    ex = PhysicalPickPlaceExecutor(
        SimpleNamespace(),
        model=model,
        data=live,
        ee_body="base_link",
        joint_names=("arm",),
        collision=checker,
        synchronize=lambda d: None,
        command_joints=Mock(),
        gripper_open_configuration={"arm": 0.0},
        input_mode="privileged",
    )
    ex.plan_pose = Mock(side_effect=AssertionError("revalidation must not run IK"))
    path = [np.array([0.0]), np.array([0.08])]
    endpoint = (np.array([0.08, 0, 0.2]), np.eye(3))
    ex.grasp_paths = [path] * 3
    ex.place_paths = [path] * 3
    ex.grasp_targets = [endpoint] * 3
    ex.place_targets = [endpoint] * 3
    store = LocalSceneStore(clock=lambda: clock[0] + 0.02)
    old = store.load("source", lambda: local_scene(100.0), reserve_bytes=1_000_000)
    with old.borrow() as scene:
        oldcap = scene.support.capture
    plan = NativePlan("payload", "table", live.qpos.copy(), np.array([0.1, 0, 0.01]), np.eye(3), ex, "boot")

    def sync(data):
        data.qpos[:] = live.qpos
        data.qvel[:] = live.qvel
        mujoco.mj_kinematics(model, data)

    adapter = SimpleNamespace(
        model=model,
        signature=model_signature(model),
        policy="legacy",
        boot="boot",
        planning_ik_calls=17,
        synchronize=sync,
        plans={"old": plan},
        events=[],
    )
    segment = module.joint_proof(ex, "pregrasp", live.qpos, module.payload_copy(checker), path, endpoint=endpoint)
    plan.proposal = module.NativeProposal(
        module.intent_digest(plan),
        adapter.signature,
        "legacy",
        "boot",
        module.frozen(live.qpos),
        (segment,),
        None,
        old,
        oldcap,
        module.policy_digest(adapter),
    )
    clock[0] = 120.0
    return clock, store, old, adapter, plan, live


def acquire(proposal, *, check=lambda c, m, d: False, missing_ray=False, stamp=120.0, capture_id=None):
    _, store, _, _, _, _ = proposal
    lease = store.load(
        "source", lambda: local_scene(stamp, missing_ray=missing_ray, capture_id=capture_id), reserve_bytes=1_000_000
    )
    return module.RefreshedNativeScene(lease, (check,))


def test_fresh_evidence_recertifies_edges_without_commands_or_ik(proposal):
    _, _, old, adapter, plan, live = proposal
    before = [x.copy() for x in (live.qpos, live.qvel, live.qacc, live.ctrl, live.contact.dist)]
    module.finalize_proposal(adapter, plan, lambda: acquire(proposal))
    assert adapter.plans == {}  # no old handle may retain authority
    with pytest.raises(LocalMapUnavailable):
        old.require_current()
    assert plan.evidence_lease.generation != old.generation
    plan.evidence_lease.require_current()
    assert adapter.planning_ik_calls == 17
    plan.executor.command_joints.assert_not_called()
    plan.executor.plan_pose.assert_not_called()
    for a, b in zip(before, (live.qpos, live.qvel, live.qacc, live.ctrl, live.contact.dist), strict=True):
        np.testing.assert_array_equal(a, b)


def test_new_obstacle_in_an_edge_rejects_without_dispatch(proposal):
    *_, adapter, plan, live = proposal
    samples = []

    def obstacle(c, m, d):
        x = float(d.qpos[0])
        samples.append(x)
        return 0.03 < x < 0.05

    with pytest.raises(ObservationUnavailable, match="scene_changed_replan"):
        module.finalize_proposal(adapter, plan, lambda: acquire(proposal, check=obstacle))
    assert any(0.03 < x < 0.05 for x in samples)
    assert not adapter.plans and plan.evidence_lease is None
    plan.executor.command_joints.assert_not_called()


def test_changed_original_ray_rejects_even_with_identical_coarse_grid(proposal):
    clock, store, old, adapter, plan, live = proposal

    def get():
        fresh = acquire(proposal, missing_ray=True)

        def rays(c, m, d):
            with fresh.lease.borrow() as scene:
                center = d.geom_xpos[0]
                bounds = np.array([center - 0.005, center + 0.005])
                return not scene.support.covers_box(bounds, allow_support_material=False)

        return replace(fresh, checks=(rays,))

    with pytest.raises(ObservationUnavailable, match="scene_changed_replan"):
        module.finalize_proposal(adapter, plan, get)
    assert not adapter.plans and plan.evidence_lease is None
    plan.executor.command_joints.assert_not_called()


@pytest.mark.parametrize("failure", ["same_capture", "predates_barrier", "same_lease", "policy", "moved", "budget"])
def test_failed_refresh_never_publishes_an_authorization(proposal, failure):
    clock, store, old, adapter, plan, live = proposal

    def get():
        if failure == "same_lease":
            return module.RefreshedNativeScene(old, (lambda c, m, d: False,))
        fresh = acquire(
            proposal,
            stamp=119.0 if failure == "predates_barrier" else 120.0,
            capture_id="100.0" if failure == "same_capture" else None,
        )
        if failure == "policy":
            adapter.policy = "different"
        if failure == "moved":
            live.qpos[0] += 0.001
        if failure == "budget":

            def expired(c, m, d):
                clock[0] += 2.0
                return False

            fresh = replace(fresh, checks=(expired,))
        return fresh

    with pytest.raises((ObservationUnavailable, ValueError)):
        module.finalize_proposal(adapter, plan, get, budget_s=1.0)
    assert not adapter.plans and plan.evidence_lease is None
    with pytest.raises(LocalMapUnavailable):
        old.require_current()
    plan.executor.command_joints.assert_not_called()


def test_refreshed_callbacks_are_immutable_and_wrong_lease_owner_rejects(proposal):
    fresh = acquire(proposal)

    def check(c, m, d):
        return False

    callbacks = [check]
    copied = module.RefreshedNativeScene(fresh.lease, callbacks)
    callbacks.clear()
    assert copied.checks == (check,)

    class WrongGuard:
        lease = proposal[2]

        def check(self, c, m, d):
            return False

    with pytest.raises(ValueError, match="invalid_native_scene_refresh"):
        module.RefreshedNativeScene(fresh.lease, (WrongGuard().check,))


@pytest.mark.parametrize("mutation", ["approach", "base_approach", "checks", "lease", "owner_lease", "admission"])
def test_finalized_binding_rejects_added_motion_or_dropped_geometry(proposal, mutation):
    _, _, _, adapter, plan, _ = proposal
    module.finalize_proposal(adapter, plan, lambda: acquire(proposal))
    if mutation in {"approach", "base_approach"}:
        setattr(plan, mutation, object())
    elif mutation == "checks":
        plan.executor.collision.additional_scene_checks = ()
    elif mutation == "lease":
        plan.evidence_lease = proposal[2]
    elif mutation == "admission":
        plan.admission = None
    else:
        plan.executor.planning_lease = proposal[2]
    with pytest.raises(ObservationUnavailable, match="scene_changed_replan"):
        module.require_binding(adapter, plan)
    plan.executor.command_joints.assert_not_called()


def test_finalized_binding_rechecks_mutable_guard_lease(proposal):
    _, _, old, adapter, plan, _ = proposal
    fresh = acquire(proposal)

    class Guard:
        lease = fresh.lease

        def check(self, c, m, d):
            return False

    guard = Guard()
    module.finalize_proposal(adapter, plan, lambda: module.RefreshedNativeScene(fresh.lease, (guard.check,)))
    guard.lease = old
    with pytest.raises(ObservationUnavailable, match="scene_changed_replan"):
        module.require_binding(adapter, plan)
    plan.executor.command_joints.assert_not_called()


@pytest.mark.parametrize("transport", ["close_gripper", "gripper_to", "move_base_to"])
def test_adapter_owned_callbacks_check_lease_at_the_transport_boundary(transport):
    raw = Mock()
    owner = SimpleNamespace(robot=raw)
    ex = SimpleNamespace(robot=raw, command_joints=Mock(), command_gripper_joints=None)
    gate = Mock(side_effect=ObservationUnavailable("stale_observation"))

    # The callback resolves owner.robot after a potentially long preflight.
    def callback():
        return getattr(owner.robot, transport)(0.1)

    with pytest.raises(ObservationUnavailable, match="stale_observation"):
        with dispatch_boundary(ex, gate, owner=owner):
            callback()
    gate.assert_called_once()
    getattr(raw, transport).assert_not_called()
    assert owner.robot is raw and ex.robot is raw
