# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Target contact is exact-geom and phase scoped, never an obstacle exemption."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.native_inputs import ObjectPoseEvidence, ObservationUnavailable
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.manipulation.observed_collision import ObservedNativeCollisionChecker, PayloadGeometryEvidence
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.controller.manipulation.physical_pick_place import PhysicalPickPlaceExecutor
from emet.motion.observed_contact import ObservedTargetContact, solids_disjoint
from emet.motion.observed_solids import WorldSolid
from emet.motion.placement_geometry import PlacementScene, box_corners


def box(center=(0.0, 0.0, 0.0), half=(0.05, 0.05, 0.05)):
    pose = np.eye(4)
    pose[:3, 3] = center
    return WorldSolid("box", half, pose)


def policy(signature="model", gid=0, target=None):
    return ObservedTargetContact(
        "object",
        "reference",
        "geometry",
        CaptureInterval("capture", 10.0, 10.001, 10.01, "clock"),
        box() if target is None else target,
        signature,
        "geometry-calibration",
        frozenset((gid,)),
        0.003,
    )


@pytest.mark.parametrize("phase", ["grasp", "close", "lift", "preplace", "place", "release"])
def test_only_named_fingertip_can_make_shallow_contact(phase):
    contact = policy()
    finger = WorldSolid("sphere", (0.01,), box(center=(0.058, 0.0, 0.0)).world_from_local)
    assert not contact.collides(finger, geom_id=0, phase=phase)
    assert contact.collides(finger, geom_id=1, phase=phase)
    assert contact.collides(replace(finger, offset_m=0.004), geom_id=0, phase=phase)


@pytest.mark.parametrize("phase", ["preflight", "pregrasp", "retreat", "gripper_open"])
def test_other_phases_do_not_allow_fingertip_contact(phase):
    assert policy().collides(box(center=(0.058, 0.0, 0.0), half=(0.01, 0.01, 0.01)), geom_id=0, phase=phase)


def test_rotated_aabb_overlap_is_not_contact_and_budget_exhaustion_rejects():
    a = box(half=(0.1, 0.005, 0.005))
    pose = a.world_from_local.copy()
    pose[:3, :3] = Rotation.from_euler("z", np.pi / 4).as_matrix()
    a = replace(a, world_from_local=pose)
    b = box(center=(0.05, -0.05, 0.0), half=(0.002, 0.002, 0.002))
    assert np.all(a.bounds[1] >= b.bounds[0]) and np.all(a.bounds[0] <= b.bounds[1])
    assert solids_disjoint(a, b)
    # Small gap at an oblique corner needs subdivision. A one-node budget
    # cannot establish that gap and must not silently permit motion.
    pose = np.eye(4)
    pose[:3, 3] = [0.018, 0.018, 0.0]
    c = WorldSolid("sphere", (0.01,), np.eye(4))
    d = WorldSolid("sphere", (0.01,), pose)
    assert solids_disjoint(c, d)
    assert not solids_disjoint(c, d, max_nodes=1)


def test_thin_object_cannot_disappear_when_the_contact_core_is_eroded_away():
    target = box(half=(0.05, 0.05, 0.001))
    contact = policy(target=target)
    assert contact.collides(box(half=(0.01, 0.01, 0.01)), geom_id=0, phase="grasp")


def rig(*, free=True, obstacle=False, housing=False):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><body name="robot">
      <joint name="x" type="slide" axis="1 0 0" range="-.2 .2"/>
      <body name="ee"><geom name="finger" type="sphere" size=".01" pos=".058 0 0"/>
      <geom name="housing" type="sphere" size=".01" pos="{".058" if housing else ".3"} 0 0"/>
      </body></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    p = policy(model_signature(model), mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger"))
    evidence = PayloadGeometryEvidence(
        "object",
        "reference",
        "geometry",
        box_corners([[-0.05] * 3, [0.05] * 3]),
        ObjectPoseEvidence(np.eye(4), 10.01, "rgbd_tracking", "capture"),
        "known_model_prior",
    )
    provider = SimpleNamespace(input_mode="observed", observe_payload=lambda _: evidence)
    scene = PlacementScene(
        [[[0.055, -0.01, -0.01], [0.065, 0.01, 0.01]]] if obstacle else [],
        source="observed_voxels",
        workspace=[[-1] * 3, [1] * 3],
        known_free=lambda bounds: free,
    )
    checker = ObservedNativeCollisionChecker(
        model,
        robot_body="robot",
        ee_body="ee",
        scene=scene,
        payload_provider=provider,
        scene_revision="scene",
        current_revision=lambda: "scene",
        received_monotonic=10.01,
        clock=lambda: 10.02,
        margin_m=0,
        contact_bodies=("ee",),
        target_contact=p,
    )
    return model, data, checker


def test_real_checker_target_contact_is_active_before_attachment_and_phase_is_restored():
    model, data, checker = rig()
    before = data.qpos.copy()
    assert checker.configuration_collides(model, data)
    with checker.motion_phase("grasp"):
        assert not checker.configuration_collides(model, data)
    assert checker.phase == "preflight"
    np.testing.assert_array_equal(data.qpos, before)
    data.qpos[0] = -0.004
    mujoco.mj_kinematics(model, data)
    with checker.motion_phase("grasp"):
        assert checker.configuration_collides(model, data)


@pytest.mark.parametrize("condition", ["unknown", "obstacle", "housing"])
def test_contact_never_exempts_unknown_environment_or_wrist_housing(condition):
    model, data, checker = rig(
        free=condition != "unknown", obstacle=condition == "obstacle", housing=condition == "housing"
    )
    with checker.motion_phase("grasp"):
        assert checker.configuration_collides(model, data)


def test_loaded_contact_uses_payload_geometry_and_never_inherits_body_subtree_exemptions():
    model, data, checker = rig()
    checker.set_payload(model, data, "object", "ee")
    # Entire EE subtree was allowed by the old contract; the new policy names
    # only the fingertip, and remains active against a moving attached envelope.
    with checker.motion_phase("lift"):
        assert not checker.configuration_collides(model, data)
        checker.target_contact = replace(checker.target_contact, fingertip_geom_ids=frozenset((1,)))
        assert checker.configuration_collides(model, data)


def test_target_capture_identity_and_geometry_mismatch_reject():
    model, data, checker = rig()
    checker.target_contact = replace(
        checker.target_contact, capture=CaptureInterval("older", 9.0, 9.001, 9.01, "clock")
    )
    with pytest.raises(ObservationUnavailable, match="identity_changed"):
        checker.configuration_collides(model, data)
    with pytest.raises(ObservationUnavailable, match="stale"):
        policy().require(now=13.0, model_signature="model")
    with pytest.raises(ValueError, match="fingertip_geoms"):
        checker.set_target_contact(policy(model_signature(model), gid=200))
    checker.set_target_contact(replace(policy(model_signature(model)), object_ref="another-object"))
    with pytest.raises(ObservationUnavailable, match="identity_changed"):
        checker.set_payload(model, data, "object", "ee")


def test_released_object_moving_into_retreat_is_checked_against_fresh_visual_geometry():
    model, data, checker = rig()
    checker.set_payload(model, data, "object", "ee")
    checker.set_payload(model, data, None)
    data.qpos[0] = 0.1
    mujoco.mj_kinematics(model, data)
    with checker.motion_phase("retreat"):
        assert not checker.configuration_collides(model, data)
        # The release box stays at the old object location. The next camera
        # capture sees the object move into the housing's upcoming sweep.
        checker.set_target_contact(replace(checker.target_contact, target=box(center=(0.4, 0.0, 0.0))))
        assert checker.configuration_collides(model, data)


def test_released_contact_does_not_treat_planning_clearance_as_penetration():
    model, data, checker = rig()
    checker.margin = 0.005
    checker.set_payload(model, data, "object", "ee")
    state = checker.capture_payload()
    checker.set_payload(model, data, None)
    assert checker.released_bounds[1, 0] == pytest.approx(0.055)
    with checker.motion_phase("release"):
        assert not checker.configuration_collides(model, data)
    checker.restore_payload(state)
    assert checker.payload is not None and checker._released_target is None


def test_nonfinger_clearance_is_kept_when_contact_policy_has_no_material_admission():
    model, data, checker = rig()
    checker.margin = 0.005
    # Nominal finger is physically clear but inside the ordinary padding. Only
    # naming it as a contact fingertip in a contact phase can remove padding.
    data.qpos[0] = 0.004
    mujoco.mj_kinematics(model, data)
    with checker.motion_phase("pregrasp"):
        assert checker.configuration_collides(model, data)
    with checker.motion_phase("grasp"):
        assert not checker.configuration_collides(model, data)
        checker.set_target_contact(replace(checker.target_contact, fingertip_geom_ids=frozenset((1,))))
        assert checker.configuration_collides(model, data)


def test_executor_scopes_dense_execution_phase_and_restores_on_failure():
    _, _, checker = rig()
    executor = PhysicalPickPlaceExecutor.__new__(PhysicalPickPlaceExecutor)
    executor.collision = checker
    seen = []
    executor._execute_path_in_phase = lambda phase, path: seen.append(checker.phase)
    executor._execute_path("lift", [])
    assert seen == ["lift"] and checker.phase == "preflight"

    def fail(phase, path):
        assert checker.phase == "place"
        raise RuntimeError("transport failed")

    executor._execute_path_in_phase = fail
    with pytest.raises(RuntimeError, match="transport failed"):
        executor._execute_path("place", [])
    assert checker.phase == "preflight"


def closing_rig(*, blocked=False, moves=True):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="robot">
      <joint name="x" type="slide" axis="1 0 0" range="-.1 .1"/>
      <geom size=".01" pos="1 0 0"/>
      <body name="ee"><joint name="jaw" type="slide" axis="0 1 0" range="-.05 .05"/>
      <geom name="finger" type="sphere" size=".005"/></body>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    measured = np.array([0.0, -0.05])

    def sync(d):
        d.qpos[:] = measured
        mujoco.mj_kinematics(model, d)

    sync(data)
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger")
    contact = policy(model_signature(model), finger, target=box(center=(0.7, 0.0, 0.0)))
    scene = PlacementScene(
        [[[-0.01] * 3, [0.01] * 3]] if blocked else [],
        source="observed_voxels",
        workspace=[[-2] * 3, [2] * 3],
        known_free=lambda bounds: True,
    )
    checker = ObservedNativeCollisionChecker(
        model,
        robot_body="robot",
        ee_body="ee",
        scene=scene,
        payload_provider=SimpleNamespace(input_mode="observed", observe_payload=lambda _: None),
        scene_revision="scene",
        current_revision=lambda: "scene",
        received_monotonic=10.01,
        clock=lambda: 10.02,
        margin_m=0,
        target_contact=contact,
    )
    commands = []

    def command(target):
        commands.append(np.asarray(target).copy())
        if moves:
            measured[1] = target[0]
        return True

    def preset(**kwargs):
        raise AssertionError("Unbounded full-close preset must not be used")

    robot = SimpleNamespace(get_gripper_position=lambda: measured[1], close_gripper=preset)
    ex = PhysicalPickPlaceExecutor(
        robot,
        model=model,
        data=data,
        ee_body="ee",
        joint_names=["x"],
        collision=checker,
        synchronize=sync,
        command_joints=lambda _: True,
        object_pose_provider=checker,
        gripper_close_configuration=(("jaw", 0.05),),
        command_gripper_joints=command,
    )
    return ex, commands, measured


def test_bounded_closing_rejects_obstacle_only_in_intermediate_sweep_before_any_command():
    ex, commands, _ = closing_rig(blocked=True)
    with ex.collision.motion_phase("close"):
        assert not ex.collision.configuration_collides(ex.model, ex.data)
        ex.data.qpos[1] = 0.05
        mujoco.mj_kinematics(ex.model, ex.data)
        assert not ex.collision.configuration_collides(ex.model, ex.data)
    ex.synchronize(ex.data)
    result = ex._close_gripper_until_still()
    assert not result.success and result.message.startswith("gripper_close_path_failed")
    assert commands == [] and ex.data.qpos[1] == pytest.approx(-0.05)


def test_bounded_closing_uses_only_certified_finite_motor_targets_and_measured_feedback():
    ex, commands, measured = closing_rig()
    result = ex._close_gripper_until_still()
    assert result.success, result
    assert len(commands) > 1 and all(-0.05 <= q[0] <= 0.05 for q in commands)
    assert measured[1] == pytest.approx(0.05) and ex.collision.phase == "preflight"


def test_bounded_closing_requires_explicit_command_geometry_and_rejects_ack_without_motion():
    ex, commands, _ = closing_rig()
    ex.gripper_close_configuration = {}
    result = ex._close_gripper_until_still()
    assert not result.success and result.message == "gripper_close_geometry_unavailable" and not commands
    ex, commands, _ = closing_rig(moves=False)
    result = ex._close_gripper_until_still(timeout_s=0.3)
    assert not result.success and result.message == "gripper_tracking_failed" and commands


def test_bounded_closing_drift_after_arrival_cannot_pass_final_settling(monkeypatch):
    import emet.controller.manipulation.physical_pick_place as motion

    ex, commands, measured = closing_rig()
    clock = [0.0]

    def sleep(duration):
        clock[0] += duration
        if measured[1] == pytest.approx(0.05):
            measured[1] -= 0.005

    monkeypatch.setattr(motion, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    result = ex._close_gripper_until_still()
    assert commands and not result.success and result.message == "gripper_tracking_failed"


def test_bounded_closing_budget_expiring_during_revalidation_sends_no_next_command(monkeypatch):
    import emet.controller.manipulation.physical_pick_place as motion

    ex, commands, _ = closing_rig()
    clock = [0.0]
    monkeypatch.setattr(motion, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _: None))
    plan = motion.plan_arm_joint_path
    calls = []

    def slow(*args, **kwargs):
        calls.append(True)
        result = plan(*args, **kwargs)
        if len(calls) > 1:  # Initial full sweep passes; the next segment exhausts budget.
            clock[0] += 2.0
        return result

    monkeypatch.setattr(motion, "plan_arm_joint_path", slow)
    result = ex._close_gripper_until_still(timeout_s=1.0)
    assert not result.success and result.message == "gripper_settling_timeout" and commands == []
