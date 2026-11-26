# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""WBC v2 foundations: multi-arm contract, case library, and synthetic generator."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from emet.controller.manipulation.case_library import CaseLibrary, ScenarioRecord
from emet.controller.manipulation.data_generators import synthetic_obstacle_target
from emet.controller.manipulation.wbc_contract import (
    ArmAction,
    ArmObservation,
    BaseSpec,
    GripperCommand,
    GripperObservation,
    JointGroupObservation,
    JointGroupSpec,
    JointSpec,
    ReachSpec,
    WbcAction,
    WbcObservation,
    action_bounds,
    action_from_normalized_vector,
    action_from_vector,
    action_vector,
    active_action_mask,
    normalized_action_vector,
    reach_reward,
)
from emet.controller.manipulation.wbc_profiles import (
    apply_actuator_identification,
    example_mars_profile,
    example_rby1_profile,
    example_sourccey_profile,
    example_stretch_profile,
    load_actuator_identification,
    mars_sim_profile,
    rby1_sim_profile,
    sim_profile,
    sourccey_sim_profile,
    stretch_sim_profile,
)


def _observation(profile, *, targets=None, ee_poses=None, failure_mode="", base_target=None, torso_target=None):
    targets = targets or {}
    ee_poses = ee_poses or {}
    arms = []
    for arm_spec in profile.arms:
        group = profile.joint_group(arm_spec.joint_group)
        target = targets.get(arm_spec.name)
        pose = ee_poses.get(arm_spec.name, target or (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        gripper = None
        if arm_spec.gripper is not None:
            gripper = GripperObservation(position=(0.0,) * arm_spec.gripper.dof)
        arms.append(
            ArmObservation(
                name=arm_spec.name,
                joint_position=(0.0,) * len(group.joints),
                joint_velocity=(0.0,) * len(group.joints),
                joint_torque=(0.0,) * len(group.joints),
                ee_pose=tuple(pose),
                ee_target=None if target is None else tuple(target),
                gripper=gripper,
            )
        )
    torso = None
    if profile.torso is not None:
        torso = JointGroupObservation(
            name=profile.torso.name,
            position=(0.0,) * len(profile.torso.joints),
            velocity=(0.0,) * len(profile.torso.joints),
            target=() if torso_target is None else tuple(torso_target),
        )
    return WbcObservation(
        base_pose=(0.0, 0.0, 0.0),
        base_twist=(0.0,) * profile.base.action_dim,
        arms=tuple(arms),
        occupancy=np.zeros((8, 8, 8), dtype=np.uint8),
        torso=torso,
        failure_mode=failure_mode,
        base_target=base_target,
    )


def _action(profile, *, torque=0.0):
    arms = []
    for arm_spec in profile.arms:
        group = profile.joint_group(arm_spec.joint_group)
        gripper = None
        if arm_spec.gripper is not None:
            gripper = GripperCommand(values=(0.0,) * arm_spec.gripper.dof, mode=arm_spec.gripper.command_mode)
        arms.append(
            ArmAction(
                name=arm_spec.name,
                joint_torques=(torque,) * len(group.joints),
                gripper=gripper,
            )
        )
    torso = (0.0,) * len(profile.torso.joints) if profile.torso is not None else ()
    return WbcAction(
        base_velocity=(0.0,) * profile.base.action_dim,
        arms=tuple(arms),
        torso_torques=torso,
    )


# --- contract ---------------------------------------------------------------


def test_single_arm_observation_and_action_round_trip():
    profile = example_mars_profile()
    target = (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
    obs = _observation(profile, targets={"primary": target})
    obs.validate(profile)
    restored = WbcObservation.from_dict(obs.to_dict())
    restored.validate(profile)
    assert np.array_equal(restored.occupancy, obs.occupancy)
    assert restored.arm("primary").ee_target == target

    action = _action(profile, torque=0.25)
    action.validate(profile)
    restored_action = WbcAction.from_dict(action.to_dict())
    restored_action.validate(profile)
    assert restored_action.arm("primary").joint_torques == action.arm("primary").joint_torques
    assert restored_action.arm("primary").gripper == action.arm("primary").gripper


def test_example_profiles_have_bounded_action_channels():
    for profile in (
        example_mars_profile(),
        example_sourccey_profile(),
        example_rby1_profile(),
        example_stretch_profile(),
    ):
        lower, upper = action_bounds(profile)
        assert np.isfinite(lower).all()
        assert np.isfinite(upper).all()
        assert np.all(lower < upper)


def test_profile_serialization_round_trip_and_fingerprint():
    profiles = (
        example_mars_profile(),
        example_sourccey_profile(),
        example_rby1_profile(),
        example_stretch_profile(),
    )
    fingerprints = set()
    for profile in profiles:
        restored = profile.__class__.from_dict(profile.to_dict())
        assert restored == profile
        assert restored.fingerprint == profile.fingerprint
        fingerprints.add(profile.fingerprint)
    assert len(fingerprints) == len(profiles)

    profile = example_rby1_profile()
    changed_base = replace(profile.base, velocity_limits=(0.6, 0.6, 1.2))
    changed = replace(profile, base=changed_base)
    assert changed.fingerprint != profile.fingerprint


def test_mars_sim_profile_matches_retained_mujoco_actuator():
    from emet.rl.profile import MARS

    profile = mars_sim_profile()
    assert profile.meta("actuator_model") == "retained_mujoco_kp80_kv4_effort2"
    assert profile.arms[0].target_frame == "world"
    assert profile.arms[0].gripper.command_limits == ((-0.3491, 0.3491),)
    assert profile.base.velocity_limits == (MARS.max_v, MARS.max_w)
    for joint in profile.joint_group("primary_arm").joints:
        assert joint.kp_eff == MARS.arm_kp
        assert joint.kd_eff == MARS.arm_kv
        assert joint.torque_limit == MARS.arm_effort_limit
        assert joint.velocity_limit == MARS.max_joint_speed
        assert joint.max_lead == MARS.max_target_lead
    assert profile.fingerprint != example_mars_profile().fingerprint


def test_sim_profiles_match_packaged_robot_models():
    import mujoco

    from emet.rl.manipulation.profiles import PROFILES
    from emet.rl.model import build_model

    repo = Path(__file__).resolve().parents[3]
    sourccey = mujoco.MjModel.from_xml_path(str(repo / "src/emet/assets/robot/sourccey/sourccey.xml"))
    rby1 = mujoco.MjModel.from_xml_path(str(repo / "src/emet/assets/robot/galaxea_r1/galaxea_r1.xml"))
    sourccey_joints = {sourccey.joint(i).name for i in range(sourccey.njnt)}
    rby1_joints = {rby1.joint(i).name for i in range(rby1.njnt)}
    mars_joints = {build_model("procedural").joint(i).name for i in range(build_model("procedural").njnt)}

    for robot, profile, model_joints in (
        ("innate_mars", mars_sim_profile(), mars_joints),
        ("sourccey", sourccey_sim_profile(), sourccey_joints),
        ("rby1", rby1_sim_profile(), rby1_joints),
    ):
        assert sim_profile(robot).name == profile.name
        for group in profile.joint_groups:
            for joint in group.joints:
                assert joint.name in model_joints, (robot, joint.name)
        for arm in profile.arms:
            assert arm.gripper is not None
            for name in arm.gripper.joint_names:
                assert name in model_joints, (robot, name)

    stretch = stretch_sim_profile()
    assert tuple(joint.name for group in stretch.joint_groups for joint in group.joints) == (
        "lift",
        "arm",
        "wrist_yaw",
        "wrist_pitch",
        "wrist_roll",
    )
    assert set(PROFILES["stretch"].actuators) == {"lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll"}
    assert stretch.action_dim == 8
    assert sourccey_sim_profile().action_dim == 15
    assert rby1_sim_profile().action_dim == 23
    assert rby1_sim_profile().base.kind == "holonomic"


def test_actuator_identification_patch_is_validated_and_loaded(tmp_path):
    profile = example_mars_profile()
    identification = {
        "embodiment": profile.name,
        "source": "bench-test",
        "joints": {"joint3": {"kp_eff": 12.3, "max_lead": 0.02}},
    }
    updated = apply_actuator_identification(profile, identification)
    assert updated.joint_group("primary_arm").joints[2].kp_eff == 12.3
    assert updated.joint_group("primary_arm").joints[2].max_lead == 0.02
    assert updated.fingerprint != profile.fingerprint
    assert updated.meta("actuator_identification") == "bench-test"

    path = tmp_path / "actuator.json"
    path.write_text(json.dumps(identification), encoding="utf-8")
    assert load_actuator_identification(path, profile) == updated

    with pytest.raises(ValueError, match="unknown joints"):
        apply_actuator_identification(profile, {"joints": {"not_a_joint": {"kp_eff": 1.0}}})
    with pytest.raises(ValueError, match="unknown actuator fields"):
        apply_actuator_identification(profile, {"joints": {"joint1": {"friction": 1.0}}})
    with pytest.raises(ValueError, match="embodiment"):
        apply_actuator_identification(profile, {"embodiment": "other", "joints": {"joint1": {"kp_eff": 1.0}}})


def test_observation_round_trip_preserves_whole_body_targets_and_voxel_metadata():
    profile = example_sourccey_profile()
    target = (0.1, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
    obs = _observation(
        profile,
        targets={"left": target},
        base_target=(0.8, -0.2, 0.4),
        torso_target=(0.12,),
    )
    obs = replace(obs, occupancy_origin=(-0.6, -0.6, 0.0), occupancy_resolution=0.05)
    obs.validate(profile)

    restored = WbcObservation.from_dict(obs.to_dict())
    restored.validate(profile)
    assert restored.base_target == obs.base_target
    assert restored.base_target_mask == obs.base_target_mask
    assert restored.torso.target == (0.12,)
    assert restored.torso.effective_target_mask == (True,)
    assert restored.occupancy_origin == obs.occupancy_origin
    assert restored.occupancy_resolution == obs.occupancy_resolution


def test_observation_rejects_partial_occupancy_metadata():
    profile = example_mars_profile()
    obs = _observation(profile)
    with pytest.raises(ValueError, match="supplied together"):
        replace(obs, occupancy_origin=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="occupancy_resolution"):
        replace(obs, occupancy_origin=(0.0, 0.0, 0.0), occupancy_resolution=0.0)


def test_fixed_base_cannot_accept_a_base_target():
    profile = replace(
        example_mars_profile(),
        base=BaseSpec(name="base", kind="fixed", velocity_limits=()),
    )
    obs = replace(_observation(profile), base_target=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="fixed base"):
        obs.validate(profile)


def test_head_group_is_required_and_observed_only():
    base_profile = example_mars_profile()
    head = JointGroupSpec(
        name="head",
        role="head",
        joints=(JointSpec("head_pan", (-1.0, 1.0), 1.0, 1.0),),
    )
    profile = replace(
        base_profile,
        joint_groups=base_profile.joint_groups + (head,),
        head_group="head",
    )
    obs = _observation(profile)
    with pytest.raises(ValueError, match="missing head"):
        obs.validate(profile)

    with_head = replace(
        obs,
        head=JointGroupObservation(name="head", position=(0.0,), velocity=(0.0,)),
    )
    with_head.validate(profile)
    with_target = replace(
        with_head,
        head=JointGroupObservation(
            name="head",
            position=(0.0,),
            velocity=(0.0,),
            target=(0.2,),
        ),
    )
    with pytest.raises(ValueError, match="observed-only"):
        with_target.validate(profile)


def test_v1_payload_is_rejected_instead_of_silently_reinterpreted():
    profile = example_mars_profile()
    obs = _observation(profile)
    legacy = obs.to_dict()
    legacy.pop("schema_version")
    with pytest.raises(ValueError, match="schema version"):
        WbcObservation.from_dict(legacy)

    action = _action(profile)
    legacy_action = action.to_dict()
    legacy_action.pop("schema_version")
    with pytest.raises(ValueError, match="schema version"):
        WbcAction.from_dict(legacy_action)


def test_dual_arm_more_dof_and_action_vector_round_trip():
    profile = example_rby1_profile()
    assert profile.arm_names == ("left", "right")
    assert len(profile.torso.joints) == 4
    assert len(profile.joint_group("left_arm").joints) == 6
    assert len(profile.joint_group("right_arm").joints) == 6
    assert profile.action_dim == 3 + 4 + 6 + 2 + 6 + 2  # base + torso + arms + grippers

    targets = {
        "left": (0.1, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0),
        "right": (-0.1, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0),
    }
    obs = _observation(profile, targets=targets)
    obs.validate(profile)
    action = _action(profile, torque=1.0)
    action.validate(profile)

    vector = action_vector(action, profile)
    assert vector.shape == (profile.action_dim,)
    restored = action_from_vector(vector, profile)
    assert restored.arm("left").joint_torques == action.arm("left").joint_torques
    assert restored.arm("right").gripper.values == action.arm("right").gripper.values
    assert restored.torso_torques == action.torso_torques


def test_sourccey_has_two_arms_and_lift():
    profile = example_sourccey_profile()
    assert profile.arm_names == ("left", "right")
    assert profile.torso is not None
    assert profile.torso.name == "lift"
    assert profile.torso.joints[0].kind == "prismatic"
    assert profile.action_dim == 2 + 1 + 5 + 1 + 5 + 1


def test_profile_rejects_duplicate_joint_names():
    joint = JointSpec("joint1", (-1.0, 1.0), 1.0, 1.0)
    with pytest.raises(ValueError, match="duplicate joint names"):
        JointGroupSpec("arm", (joint, joint))


def test_action_validation_rejects_wrong_arm_count():
    profile = example_rby1_profile()
    action = _action(profile)
    bad = WbcAction(base_velocity=action.base_velocity, arms=(action.arm("left"),))
    with pytest.raises(ValueError, match="do not match profile"):
        bad.validate(profile)


def test_observation_requires_all_profile_arms():
    profile = example_rby1_profile()
    obs = _observation(profile)
    incomplete = WbcObservation(
        base_pose=obs.base_pose,
        base_twist=obs.base_twist,
        arms=(obs.arm("left"),),
        occupancy=obs.occupancy,
        torso=obs.torso,
    )
    with pytest.raises(ValueError, match="do not match profile"):
        incomplete.validate(profile)


def test_observation_rejects_nonfinite_base_twist():
    profile = example_mars_profile()
    obs = _observation(profile)
    with pytest.raises(ValueError, match="base_twist"):
        WbcObservation(
            base_pose=obs.base_pose,
            base_twist=(float("nan"), 0.0),
            arms=obs.arms,
            occupancy=obs.occupancy,
            torso=obs.torso,
        )


def test_observation_rejects_wrong_torso_name():
    profile = example_rby1_profile()
    obs = _observation(profile)
    wrong = JointGroupObservation(name="not_torso", position=(0.0,) * 4, velocity=(0.0,) * 4)
    with pytest.raises(ValueError, match="torso observation name"):
        WbcObservation(
            base_pose=obs.base_pose,
            base_twist=obs.base_twist,
            arms=obs.arms,
            occupancy=obs.occupancy,
            torso=wrong,
        ).validate(profile)


def test_mars_profile_declares_actuator_control_modes():
    profile = example_mars_profile()
    group = profile.joint_group("primary_arm")
    assert group.joints[0].control_mode == "position"
    assert group.joints[3].control_mode == "current_position"
    assert group.joints[4].control_mode == "current_position"


def test_reach_reward_single_arm_success_and_collision():
    profile = example_mars_profile()
    target = (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
    action = _action(profile)
    r, c = reach_reward(action, _observation(profile, targets={"primary": target}), profile=profile)
    assert c["success"] == 1.0
    assert c["pos_err"] < 1e-9
    assert r > 0

    r2, c2 = reach_reward(
        action,
        _observation(profile, targets={"primary": target}, ee_poses={"primary": (0.1, 0.0, 0.5, 1, 0, 0, 0)}),
        profile=profile,
        collision=True,
    )
    assert c2["collision"] == 1.0
    assert c2["success"] == 0.0
    assert r2 < r


def test_reach_reward_dual_arm_all_and_any():
    profile = example_rby1_profile()
    good = (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
    bad = (0.1, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
    obs = _observation(
        profile,
        targets={"left": good, "right": good},
        ee_poses={"left": good, "right": bad},
    )
    action = _action(profile)
    _, all_components = reach_reward(action, obs, profile=profile, spec=ReachSpec(success_mode="all"))
    _, any_components = reach_reward(action, obs, profile=profile, spec=ReachSpec(success_mode="any"))
    assert all_components["success"] == 0.0
    assert any_components["success"] == 1.0
    assert all_components["arm.right.pos_err"] > 0.05


def test_reach_reward_ignores_inactive_arm_target():
    profile = example_rby1_profile()
    target = (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
    obs = _observation(profile, targets={"left": target})
    action = _action(profile)
    _, components = reach_reward(action, obs, profile=profile)
    assert components["success"] == 1.0
    assert "arm.right.pos_err" not in components


def test_reach_reward_supports_base_only_goal():
    profile = example_mars_profile()
    action = _action(profile)
    at_goal = _observation(profile, base_target=(0.0, 0.0, 0.0))
    _, components = reach_reward(action, at_goal, profile=profile)
    assert components["success"] == 1.0
    assert components["base_xy_err"] == pytest.approx(0.0)
    assert components["base_heading_err"] == pytest.approx(0.0)

    off_goal = replace(at_goal, base_target=(0.2, 0.0, 0.3))
    _, components = reach_reward(action, off_goal, profile=profile)
    assert components["success"] == 0.0
    assert components["base_xy_err"] == pytest.approx(0.2)
    assert components["base_heading_err"] == pytest.approx(0.3)


def test_reach_reward_supports_torso_only_goal():
    profile = example_sourccey_profile()
    action = _action(profile)
    at_goal = _observation(profile, torso_target=(0.0,))
    r, components = reach_reward(action, at_goal, profile=profile)
    assert components["success"] == 1.0
    assert components["torso_err"] == pytest.approx(0.0)
    assert r > 0

    off_goal = _observation(profile, torso_target=(0.15,))
    _, components = reach_reward(action, off_goal, profile=profile)
    assert components["torso_err"] == pytest.approx(0.15)
    assert components["success"] == 0.0


def test_reach_reward_orientation_tolerance_is_radians():
    profile = example_mars_profile()
    target = (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
    quarter_turn = (0.0, 0.0, 0.5, math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    obs = _observation(profile, targets={"primary": target}, ee_poses={"primary": quarter_turn})
    _, components = reach_reward(_action(profile), obs, profile=profile)
    assert components["quat_err"] == pytest.approx(math.pi / 2.0)
    assert components["success"] == 0.0  # 90 deg exceeds the default 0.05 rad tolerance


def test_normalized_action_vector_round_trips_bounded_channels():
    profile = example_mars_profile()
    action = WbcAction(
        base_velocity=(0.2, -0.5),
        arms=(
            ArmAction(
                name="primary",
                joint_torques=(0.3, -0.2, 0.1, 0.0, -0.1),
                gripper=GripperCommand(values=(0.2,)),
            ),
        ),
    )
    lower, upper = action_bounds(profile)
    assert lower.shape == upper.shape == (profile.action_dim,)

    normalized = normalized_action_vector(action, profile)
    assert np.all(normalized >= -1.0)
    assert np.all(normalized <= 1.0)
    restored = action_from_normalized_vector(normalized, profile)
    assert np.allclose(action_vector(restored, profile), action_vector(action, profile))


def test_active_action_mask_follows_base_torso_and_arm_targets():
    profile = example_rby1_profile()
    target = (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
    obs = _observation(
        profile,
        targets={"left": target},
        base_target=(0.5, 0.0, 0.0),
        torso_target=(0.0,) * 4,
    )
    mask = active_action_mask(obs, profile)
    assert mask.shape == (profile.action_dim,)
    # base + torso + left arm/gripper are active; right arm/gripper are not.
    expected = [True] * 3 + [True] * 4 + [True] * 6 + [True] * 2 + [False] * 6 + [False] * 2
    assert mask.tolist() == expected

    # An arm-only goal leaves base and torso available for whole-body reach.
    arm_only = _observation(profile, targets={"left": target})
    arm_mask = active_action_mask(arm_only, profile)
    assert arm_mask[:7].all()
    assert arm_mask[7:15].all()
    assert not arm_mask[15:].any()

    no_targets = _observation(profile)
    assert not active_action_mask(no_targets, profile).any()


def test_reach_reward_normalizes_by_joint_torque_limits():
    profile = example_mars_profile()
    target = (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
    obs = _observation(profile, targets={"primary": target})
    action = _action(profile, torque=1.0)
    _, components = reach_reward(action, obs, profile=profile)
    expected = 5 * (1.0 / 1.5) ** 2
    assert components["action_norm_sq"] == pytest.approx(expected)


# --- case library -----------------------------------------------------------


def _record(voxel_value=1, failure_mode="tracking_lag"):
    return ScenarioRecord(
        initial_voxel=np.full((4, 4, 4), voxel_value, dtype=np.uint8),
        goal={"kind": "ee_target", "ee_target": [0.0, 0.0, 0.5, 1, 0, 0, 0]},
        robot_state={"q": [0.0] * 4, "base_xyt": [0.0, 0.0, 0.0]},
        meta={"failure_mode": failure_mode, "source": "test"},
    )


def test_case_library_add_dedupes_and_fingerprints():
    lib = CaseLibrary()
    r = _record()
    assert lib.add(r) is True
    assert lib.add(r) is False
    assert len(lib) == 1
    assert lib.add(_record()) is False
    assert lib.add(_record(voxel_value=0)) is True
    assert len(lib) == 2


def test_case_library_fingerprint_includes_embodiment():
    first = _record()
    first.meta["embodiment"] = "alpha"
    second = _record()
    second.meta["embodiment"] = "beta"
    lib = CaseLibrary()
    assert lib.add(first) is True
    assert lib.add(second) is True
    assert len(lib) == 2


def test_case_library_fingerprint_includes_embodiment_profile_revision():
    first = _record()
    first.meta["embodiment"] = "alpha"
    first.meta["embodiment_fingerprint"] = "profile-v1"
    second = _record()
    second.meta["embodiment"] = "alpha"
    second.meta["embodiment_fingerprint"] = "profile-v2"
    lib = CaseLibrary()
    assert lib.add(first) is True
    assert lib.add(second) is True
    assert len(lib) == 2


def test_case_library_by_failure_mode_and_sample():
    lib = CaseLibrary()
    for i, mode in enumerate(("tracking_lag", "ik_failed", "collision")):
        lib.add(_record(voxel_value=i + 1, failure_mode=mode))
    buckets = lib.by_failure_mode()
    assert set(buckets) == {"tracking_lag", "ik_failed", "collision"}
    rng = np.random.default_rng(0)
    assert len(lib.sample(5, rng=rng)) == 5
    assert len(lib.sample(2, rng=rng, failure_mode="ik_failed")) == 2


def test_case_library_save_load_round_trip(tmp_path):
    lib = CaseLibrary()
    lib.add(_record(failure_mode="tracking_lag"))
    lib.add(_record(voxel_value=0, failure_mode="ik_failed"))
    path = lib.save(tmp_path / "cases.jsonl")
    loaded = CaseLibrary.load(path)
    assert len(loaded) == 2
    assert {r.failure_mode for r in loaded} == {"tracking_lag", "ik_failed"}


def test_case_library_preserves_multi_arm_goal_and_state(tmp_path):
    profile = example_sourccey_profile()
    record = synthetic_obstacle_target(np.random.default_rng(3), grid_shape=(8, 8, 8), profile=profile)
    lib = CaseLibrary(tmp_path / "multi.jsonl")
    assert lib.add(record) is True
    path = lib.save()
    loaded = CaseLibrary.load(path).sample(1, rng=np.random.default_rng(0))[0]
    assert loaded.goal["kind"] == "ee_targets"
    assert set(loaded.goal["arms"]) == {"left", "right"}
    assert loaded.meta["embodiment"] == profile.name
    assert set(loaded.robot_state["arms"]) == {"left", "right"}


# --- data generator ---------------------------------------------------------


def test_synthetic_obstacle_target_produces_valid_scenario():
    rng = np.random.default_rng(42)
    rec = synthetic_obstacle_target(rng, grid_shape=(16, 16, 16), n_obstacles=(2, 4))
    assert rec.initial_voxel.shape == (16, 16, 16)
    assert rec.initial_voxel.sum() > 0
    target_pos = np.asarray(rec.goal["ee_target"][:3], dtype=np.float64)
    lo = np.array([-0.6, -0.6, 0.0])
    hi = np.array([0.6, 0.6, 1.0])
    cell = (hi - lo) / 15
    idx = np.clip(((target_pos - lo) / cell).astype(int), 0, 15)
    assert rec.initial_voxel[tuple(idx)] == 0


def test_synthetic_generator_is_reproducible_with_seed():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    r1 = synthetic_obstacle_target(rng1)
    r2 = synthetic_obstacle_target(rng2)
    assert np.array_equal(r1.initial_voxel, r2.initial_voxel)
    assert r1.goal == r2.goal


def test_synthetic_generator_multi_arm_profile():
    profile = example_rby1_profile()
    rec = synthetic_obstacle_target(
        np.random.default_rng(11),
        grid_shape=(8, 8, 8),
        profile=profile,
        target_arms=("left",),
    )
    assert rec.goal["kind"] == "ee_targets"
    assert set(rec.goal["arms"]) == {"left"}
    assert rec.robot_state["base_twist"] == [0.0, 0.0, 0.0]
    assert set(rec.robot_state["arms"]) == {"left", "right"}
    assert len(rec.robot_state["arms"]["left"]["q"]) == 6
    assert len(rec.robot_state["arms"]["left"]["gripper"]) == 2
    assert len(rec.robot_state["torso"]["q"]) == 4
    assert rec.meta["embodiment"] == profile.name


def test_synthetic_generator_adds_base_and_torso_targets():
    profile = example_sourccey_profile()
    rec = synthetic_obstacle_target(
        np.random.default_rng(12),
        grid_shape=(8, 8, 8),
        profile=profile,
        target_base=(0.7, -0.3, 0.25),
        target_torso=(0.1,),
    )
    assert rec.goal["base_target"] == [0.7, -0.3, 0.25]
    assert rec.goal["torso_target"] == [0.1]
    assert rec.meta["embodiment_fingerprint"] == profile.fingerprint

    with pytest.raises(ValueError, match="target_torso"):
        synthetic_obstacle_target(
            np.random.default_rng(13),
            profile=profile,
            target_torso=(0.5,),
        )


def test_scenario_from_attempt_and_record_scenario(tmp_path, monkeypatch):
    from emet.controller.manipulation.failure_record import record_scenario, scenario_from_attempt

    attempt = {
        "failure_mode": "tracking_failed",
        "target_xyz": [0.0, 0.0, 0.9],
        "joint_names": ["torso_joint1", "arm_joint1"],
        "trace": [{"qpos": [0.1, 0.2], "ee_xyz": [0.0, 0.0, 0.5]}],
    }
    rec = scenario_from_attempt(
        attempt, context={"object_query": "mug", "receptacle_query": "cab"}
    )
    assert rec.goal["object"] == "mug"
    assert rec.goal["receptacle"] == "cab"
    assert rec.meta["privileged"] is False
    assert "object_body" not in rec.meta
    assert rec.goal["ee_target_pos"] == [0.0, 0.0, 0.9]
    assert rec.robot_state["q"] == [0.1, 0.2]
    assert rec.failure_mode == "tracking_failed"
    assert rec.initial_voxel["kind"] == "voxel_map_2d_unavailable"

    privileged = scenario_from_attempt(
        attempt,
        context={
            "object_query": "mug",
            "receptacle_query": "cab",
            "privileged": True,
            "object_body": "obj_main",
            "receptacle_body": "cab_main",
        },
    )
    assert privileged.meta["privileged"] is True
    assert privileged.meta["object_body"] == "obj_main"
    assert privileged.meta["receptacle_body"] == "cab_main"

    monkeypatch.setenv("EMET_CASE_LIBRARY_DIR", str(tmp_path))
    assert record_scenario(rec) is True
    assert record_scenario(rec) is False


def test_record_scenario_serializes_on_the_case_library_lock(tmp_path, monkeypatch):
    import fcntl
    import threading
    import time

    from emet.controller.manipulation.failure_record import (
        default_library_path,
        record_scenario,
        scenario_from_attempt,
    )

    monkeypatch.setenv("EMET_CASE_LIBRARY_DIR", str(tmp_path))
    rec = scenario_from_attempt(
        {"failure_mode": "lock_test", "joint_names": ["j1"], "trace": [{"qpos": [0.0]}]},
        context={"object_query": "mug", "receptacle_query": "cab"},
    )
    library = default_library_path()
    lock_path = library.with_name(library.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    result: list[bool] = []
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        thread = threading.Thread(target=lambda: result.append(record_scenario(rec)))
        thread.start()
        time.sleep(0.1)
        assert thread.is_alive(), "record_scenario did not wait for the case-library lock"
        fcntl.flock(lock, fcntl.LOCK_UN)
        thread.join(timeout=2)
    assert result == [True]
    assert library.is_file()


def test_export_initial_voxel_map_from_mock():
    from emet.controller.manipulation.failure_record import export_initial_voxel_map

    class MockVoxelMap:
        grid_origin = np.array([-2.0, -3.0])
        grid_resolution = 0.1

        def get_2d_map(self):
            return (
                np.array([[0, 1], [1, 0]], dtype=np.float32),
                np.array([[1, 1], [1, 0]], dtype=np.float32),
            )

    out = export_initial_voxel_map(MockVoxelMap())
    assert out["kind"] == "voxel_map_2d"
    assert out["grid_origin"] == [-2.0, -3.0]
    assert out["grid_resolution"] == 0.1
    assert out["obstacles"] == [[0, 1], [1, 0]]
    assert out["explored"] == [[1, 1], [1, 0]]


def test_export_initial_voxel_map_never_raises():
    from emet.controller.manipulation.failure_record import export_initial_voxel_map

    class Boom:
        grid_origin = np.array([0.0, 0.0])

        def get_2d_map(self):
            raise RuntimeError("no map")

    out = export_initial_voxel_map(Boom())
    assert out["kind"] == "voxel_map_2d_unavailable"
    assert out["obstacles"] == []


def test_scenario_uses_pre_motion_initial_q_when_present():
    from emet.controller.manipulation.failure_record import scenario_from_attempt

    attempt = {
        "target_xyz": [0.0, 0.0, 0.9],
        "initial_q": [0.5, 0.4],
        "trace": [{"qpos": [0.1, 0.2]}],
        "joint_names": ["joint1", "joint2"],
        "failure_mode": "tracking_failed",
    }
    rec = scenario_from_attempt(attempt, context={"object_query": "mug"})
    assert rec.robot_state["q"] == [0.5, 0.4]
