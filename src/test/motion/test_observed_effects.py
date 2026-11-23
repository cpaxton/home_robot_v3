# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.native_inputs import ObjectPoseEvidence
from emet.controller.manipulation.observed_collision import PayloadGeometryEvidence
from emet.controller.manipulation.observed_effects import (
    CaptureInterval,
    ObservedVisualVerifier,
    RobotPoseSample,
    VisualEffectFrame,
    evaluate_lift,
    evaluate_place,
    pose_uncertainty_margin,
)
from emet.motion.placement_geometry import box_corners


@pytest.mark.parametrize('angle', [0., .03, np.pi/2, np.pi, 2*np.pi])
def test_pose_margin_encloses_rigid_motion_and_saturates_at_full_rotation(angle):
    points = box_corners([[-.02, -.03, -.04], [.02, .03, .04]])
    radius, error = np.linalg.norm(points, axis=1).max(), .004
    margin = pose_uncertainty_margin(radius, error, angle)
    rng = np.random.default_rng(226)
    for _ in range(30):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = Rotation.from_rotvec(axis*min(np.pi, angle)).as_matrix()
        moved = points @ rotation.T + axis*error
        assert np.all(np.linalg.norm(moved-points, axis=1) <= margin+1e-12)
    if angle >= np.pi:
        assert margin == pytest.approx(error+2*radius)


def frame(t, *, obj=(0, 0, .03), ee=(0, 0, .03), capture_id=None):
    p, e = np.eye(4), np.eye(4)
    p[:3, 3], e[:3, 3] = obj, ee
    capture_id = capture_id or f'camera:{t}'
    pose = ObjectPoseEvidence(p, t+.02, 'rgbd_tracking', capture_id)
    payload = PayloadGeometryEvidence('object', 'reference', 'shape',
        box_corners([[-.02]*3, [.02]*3]), pose, 'known_model_prior')
    return VisualEffectFrame(payload, CaptureInterval(capture_id, t, t+.01, t+.02, 'clock-calibration'),
        RobotPoseSample(e, np.zeros(2), t-.01, np.array([[e[:3, 3]-.001, e[:3, 3]+.001]])),
        RobotPoseSample(e, np.zeros(2), t+.03, np.array([[e[:3, 3]-.001, e[:3, 3]+.001]])), .001, .005, .0001, .0001)


@pytest.mark.parametrize('fresh', [False, True])
def test_begin_grasp_requests_post_barrier_acquisition_instead_of_cached_snapshot(fresh):
    seen = []
    cached, new = frame(10.), frame(10.7 if fresh else 10.3, capture_id='new-id')
    def observe_after(obj, *, after):
        seen.append((obj, after))
        return new
    verifier = ObservedVisualVerifier(object_ref='object', support_bounds=[[-.5, -.5, -.1], [.5, .5, 0.]],
        observe=lambda _: cached, observe_after=observe_after, collect=lambda *a, **k: (), clock=lambda: 11.)
    result = verifier.begin_grasp(after=10.5)
    assert result.satisfied is fresh and seen == [('object', 10.5)]
    assert (verifier.reference is new) is fresh


def test_lift_requires_repeated_comotion_and_separation():
    before = frame(10.)
    after = [frame(10.5, obj=(0, 0, .13), ee=(0, 0, .13)),
             frame(10.8, obj=(0, 0, .13), ee=(0, 0, .13))]
    assert evaluate_lift(before, after, support_top_z=0, after=10.4, now=11.).satisfied
    # Height alone cannot substitute for matching measured gripper motion.
    wrong = [replace(f, robot_before=before.robot_before, robot_after=RobotPoseSample(
        before.robot_after.world_from_ee, np.zeros(2), f.capture.latest+.02)) for f in after]
    assert not evaluate_lift(before, wrong, support_top_z=0, after=10.4, now=11.).satisfied


def test_visual_verifier_checks_lift_against_pickup_height():
    before = frame(10., obj=(0, 0, .25), ee=(0, 0, .25))
    lifted = [frame(t, obj=(0, 0, .35), ee=(0, 0, .35)) for t in (10.5, 10.8)]
    verifier = ObservedVisualVerifier(
        object_ref='object',
        support_bounds=[[-.5, -.5, .5], [.5, .5, .6]],
        source_support_bounds=[[-.5, -.5, .1], [.5, .5, .22]],
        observe=lambda _: before,
        collect=lambda *args, **kwargs: lifted,
        clock=lambda: 11.,
    )
    assert verifier.begin_grasp(after=9.9).satisfied
    assert verifier.verify_lift(after=10.4).satisfied


@pytest.mark.parametrize('obj', [(0, 0, .03), (.1, 0, .13)])
def test_static_or_slipped_object_does_not_pass_lift(obj):
    after = [frame(t, obj=obj, ee=(0, 0, .13)) for t in (10.5, 10.8)]
    result = evaluate_lift(frame(10.), after, support_top_z=0, after=10.4, now=11.)
    assert result.state == 'violated'


def test_stale_replayed_occluded_and_unaligned_frames_are_unknown():
    good = [frame(t, obj=(0, 0, .13), ee=(0, 0, .13)) for t in (10.5, 10.8)]
    for frames, now, after in [(good, 15., 10.4), ([good[0], good[0]], 11., 10.4),
                               ([good[0], None], 11., 10.4), (good, 11., 10.6)]:
        result = evaluate_lift(frame(10.), frames, support_top_z=0, after=after, now=now)
        assert result.state == 'unknown'
    moving = replace(good[1], robot_after=RobotPoseSample(good[1].robot_after.world_from_ee,
                                                       np.array([.01, 0]), 10.9))
    assert evaluate_lift(frame(10.), [good[0], moving], support_top_z=0, after=10.4, now=11.).state == 'unknown'


def test_successful_place_after_retreat_and_settling():
    before = frame(10., obj=(0, 0, .13), ee=(0, 0, .13))
    after = [frame(t, obj=(.1, 0, .02), ee=(.1, 0, .2)) for t in (10.5, 10.9)]
    result = evaluate_place(before, after, support_bounds=[[-.3, -.3, -.1], [.3, .3, 0]], after=10.4, now=11.)
    assert result.satisfied and result.as_dict()['source'] == 'rgbd_tracking'


@pytest.mark.parametrize('objects,ees', [
    ([(.5, 0, .02)]*2, [(0, 0, .2)]*2),
    ([(0, 0, .1)]*2, [(0, 0, .3)]*2),
    ([(0, 0, .02)]*2, [(0, 0, .03)]*2),
    ([(0, 0, .02), (.03, 0, .02)], [(0, 0, .2)]*2),
])
def test_outside_unsupported_not_retreated_or_unsettled_place_fails(objects, ees):
    after = [frame(t, obj=o, ee=e) for t, o, e in zip((10.5, 10.9), objects, ees, strict=True)]
    assert not evaluate_place(frame(10.), after, support_bounds=[[-.3, -.3, -.1], [.3, .3, 0]],
                              after=10.4, now=11.).satisfied


def test_historical_grasp_reference_is_not_renewed_or_required_to_be_live():
    before = frame(5.)
    frames = [frame(t, obj=(0, 0, .13), ee=(0, 0, .13)) for t in (10.5, 10.8)]
    assert evaluate_lift(before, frames, support_top_z=0, after=10.4, now=11.).satisfied
    assert before.payload.pose.received_monotonic == 5.02


def test_long_transport_preserves_historical_reference_but_requires_fresh_placement():
    before = frame(5., obj=(0, 0, .13), ee=(0, 0, .13))
    frames = [frame(t, obj=(.1, 0, .02), ee=(.1, 0, .2)) for t in (130.5, 130.9)]
    kwargs = {'support_bounds': [[-.3, -.3, -.1], [.3, .3, 0]], 'after': 130.4}
    assert evaluate_place(before, frames, now=131., **kwargs).satisfied
    assert evaluate_place(before, frames, now=135., **kwargs).reason == 'object_observation_stale'
    assert before.capture.earliest == 5.
    assert before.payload.pose.received_monotonic == 5.02


def test_coarse_placement_settling_tolerance_includes_both_capture_errors():
    before = frame(5., obj=(0, 0, .13), ee=(0, 0, .13))
    frames = [replace(frame(t, obj=(x, 0, .02), ee=(.1, 0, .2)), position_error_m=.005)
              for t, x in ((130.5, .1), (130.9, .101))]
    kwargs = {'support_bounds': [[-.3, -.3, -.1], [.3, .3, 0]], 'after': 130.4, 'now': 131.}
    assert not evaluate_place(before, frames, **kwargs).satisfied
    assert evaluate_place(before, frames, settling_tolerance_m=.02, **kwargs).satisfied
    moved = replace(frame(130.9, obj=(.13, 0, .02), ee=(.1, 0, .2)), position_error_m=.005)
    assert not evaluate_place(before, [frames[0], moved], settling_tolerance_m=.02, **kwargs).satisfied


@pytest.mark.parametrize('tolerance', [0., -.01, .031, float('nan')])
def test_invalid_placement_settling_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match='invalid_placement_settling_tolerance'):
        evaluate_place(None, [], support_bounds=[[-.3, -.3, -.1], [.3, .3, 0]],
                       after=0., settling_tolerance_m=tolerance)


def test_uncertain_pose_and_wrong_capture_identity_are_unknown():
    frames = [frame(t, obj=(0, 0, .13), ee=(0, 0, .13)) for t in (10.5, 10.8)]
    for changed in (replace(frames[1], rotation_error_rad=.2),
                    replace(frames[1], capture=replace(frames[1].capture, observation_id='other'))):
        assert evaluate_lift(frame(10.), [frames[0], changed], support_top_z=0,
                             after=10.4, now=11.).state == 'unknown'


def test_retreat_checks_full_gripper_volume_not_just_ee_origin():
    frames = [frame(t, obj=(0, 0, .02), ee=(0, 0, .2)) for t in (10.5, 10.9)]
    frames = [replace(f, robot_after=replace(f.robot_after,
        gripper_bounds=np.array([[[-.01, -.01, .01], [.01, .01, .2]]]))) for f in frames]
    assert not evaluate_place(frame(10.), frames, support_bounds=[[-.3, -.3, -.1], [.3, .3, 0]],
                              after=10.4, now=11.).satisfied


def test_lift_uncertainty_straddling_retention_limit_is_unknown():
    frames = [frame(t, obj=(.019, 0, .13), ee=(0, 0, .13)) for t in (10.5, 10.8)]
    assert evaluate_lift(frame(10.), frames, support_top_z=0, after=10.4, now=11.).state == 'unknown'


def robot_error(f, *, position=0., rotation=0.):
    return replace(f, robot_before=replace(f.robot_before, position_error_m=position, rotation_error_rad=rotation),
        robot_after=replace(f.robot_after, position_error_m=position, rotation_error_rad=rotation))


def test_robot_position_error_prevents_false_retention_acceptance():
    before = frame(10.)
    frames = [frame(t, obj=(.015, 0, .13), ee=(0, 0, .13)) for t in (10.5, 10.8)]
    assert evaluate_lift(before, frames, support_top_z=0, after=10.4, now=11.).satisfied
    result = evaluate_lift(robot_error(before, position=.002),
        [robot_error(f, position=.002) for f in frames], support_top_z=0, after=10.4, now=11.)
    assert result.state == 'unknown'


def test_robot_angular_error_accounts_for_ee_to_object_lever_arm():
    before = frame(10., obj=(.4, 0, .03))
    frames = [frame(t, obj=(.4, 0, .13), ee=(0, 0, .13)) for t in (10.5, 10.8)]
    assert evaluate_lift(before, frames, support_top_z=0, after=10.4, now=11.).satisfied
    result = evaluate_lift(robot_error(before, rotation=.03),
        [robot_error(f, rotation=.03) for f in frames], support_top_z=0, after=10.4, now=11.)
    assert result.state == 'unknown'


def test_ee_height_uncertainty_cannot_certify_a_marginal_lift():
    before = frame(10.)
    frames = [frame(t, obj=(0, 0, .075), ee=(0, 0, .071)) for t in (10.5, 10.8)]
    assert evaluate_lift(before, frames, support_top_z=0, after=10.4, now=11.).satisfied
    result = evaluate_lift(robot_error(before, position=.001),
        [robot_error(f, position=.001) for f in frames], support_top_z=0, after=10.4, now=11.)
    assert result.state == 'unknown'


def test_robot_uncertainty_prevents_false_retreat_acceptance():
    before = frame(10.)
    frames = [frame(t, obj=(0, 0, .02), ee=(0, 0, .2)) for t in (10.5, 10.9)]
    frames = [replace(f, robot_after=replace(f.robot_after,
        gripper_bounds=np.array([[[-.001, -.001, .054], [.001, .001, .055]]]))) for f in frames]
    assert evaluate_place(before, frames, support_bounds=[[-.3, -.3, -.1], [.3, .3, 0]], after=10.4, now=11.).satisfied
    result = evaluate_place(before, [robot_error(f, position=.005) for f in frames],
        support_bounds=[[-.3, -.3, -.1], [.3, .3, 0]], after=10.4, now=11.)
    assert result.state == 'unknown'


def test_empty_gripper_bounds_are_missing_geometry():
    with pytest.raises(ValueError, match='invalid_gripper_geometry'):
        replace(frame(10.).robot_after, gripper_bounds=np.empty((0, 2, 3)))


def test_identical_joint_endpoints_without_motion_bound_do_not_prove_alignment():
    frames = [frame(t, obj=(0, 0, .13), ee=(0, 0, .13)) for t in (10.5, 10.8)]
    frames[1] = replace(frames[1], alignment_error_m=None)
    assert evaluate_lift(frame(10.), frames, support_top_z=0, after=10.4, now=11.).state == 'unknown'


def test_public_effect_receipt_redacts_private_provider_errors_and_capture_ids():
    import json

    from emet.controller.manipulation.observed_effects import VisualEffectResult

    result = VisualEffectResult('unknown', '/private/scene/object.xml', ('/private/device/camera',))
    public = result.as_dict()
    assert public['reason'] == 'observation_unavailable'
    assert 'private' not in json.dumps(public)
    assert public['observation_ids'][0].startswith('observation:')
    assert public == result.as_dict()


def test_simulator_passive_joint_truth_cannot_qualify_observed_robot_feedback():
    current = frame(10.)
    debug = replace(current, robot_before=replace(current.robot_before, source='simulator_robot_kinematics_debug'))
    with pytest.raises(RuntimeError, match='robot_feedback_unqualified'):
        debug.require(now=10.1)


def test_explicit_lift_rotation_tolerance_preserves_height_and_comotion_checks():
    before = frame(10.)
    frames = [frame(t, obj=(0, 0, .13), ee=(0, 0, .13)) for t in (10.5, 10.8)]
    rotated = []
    for sample in frames:
        pose = sample.payload.pose.world_from_object.copy()
        pose[:3, :3] = Rotation.from_euler('y', .12).as_matrix()
        rotated.append(replace(sample, payload=replace(sample.payload,
            pose=replace(sample.payload.pose, world_from_object=pose))))
    assert not evaluate_lift(before, rotated, support_top_z=0., after=10.4, now=11.).satisfied
    assert evaluate_lift(before, rotated, support_top_z=0., after=10.4, now=11.,
                         retention_rotation_rad=.2).satisfied
    assert not evaluate_lift(before, rotated, support_top_z=.2, after=10.4, now=11.,
                             retention_rotation_rad=.2).satisfied
    with pytest.raises(ValueError, match='invalid_retention_rotation'):
        evaluate_lift(before, rotated, support_top_z=0., after=10.4, now=11., retention_rotation_rad=np.pi+.01)


def test_explicit_lift_translation_tolerance_keeps_default_and_support_gates():
    before = frame(10.)
    frames = [frame(t, obj=(.023, 0., .13), ee=(0., 0., .13)) for t in (10.5, 10.8)]
    assert not evaluate_lift(before, frames, support_top_z=0., after=10.4, now=11.).satisfied
    assert evaluate_lift(before, frames, support_top_z=0., after=10.4, now=11.,
                         retention_translation_m=.03).satisfied
    assert not evaluate_lift(before, frames, support_top_z=.2, after=10.4, now=11.,
                             retention_translation_m=.03).satisfied
    with pytest.raises(ValueError, match='invalid_retention_translation'):
        evaluate_lift(before, frames, support_top_z=0., after=10.4, now=11., retention_translation_m=.06)


def test_unconstrained_lift_orientation_does_not_disable_position_or_support_checks():
    before = frame(10.)
    samples = []
    for stamp in (10.5, 10.8):
        sample = frame(stamp, obj=(0., 0., .13), ee=(0., 0., .13))
        pose = sample.payload.pose.world_from_object.copy()
        pose[:3, :3] = Rotation.from_euler('y', 1.).as_matrix()
        samples.append(replace(sample, payload=replace(sample.payload,
            pose=replace(sample.payload.pose, world_from_object=pose))))
    assert not evaluate_lift(before, samples, support_top_z=0., after=10.4, now=11.).satisfied
    assert evaluate_lift(before, samples, support_top_z=0., after=10.4, now=11.,
                         retention_rotation_rad=np.pi).satisfied
    assert not evaluate_lift(before, samples, support_top_z=.2, after=10.4, now=11.,
                             retention_rotation_rad=np.pi).satisfied
    displaced = samples[-1].payload.pose.world_from_object.copy()
    displaced[0, 3] = .1
    samples[-1] = replace(samples[-1], payload=replace(samples[-1].payload,
        pose=replace(samples[-1].payload.pose, world_from_object=displaced)))
    assert not evaluate_lift(before, samples, support_top_z=0., after=10.4, now=11.,
                             retention_rotation_rad=np.pi).satisfied


def test_measured_support_extent_is_separate_from_padded_planning_payload():
    sample = frame(10., obj=(0, 0, .02), ee=(0, 0, .2))
    measured = np.array([[-.01, -.01, .005], [.01, .01, .035]])
    sample = replace(sample, observed_bounds_world=measured)
    assert np.allclose(sample.support_evidence_bounds(), measured + [[-.001], [.001]])
    assert not np.allclose(sample.support_evidence_bounds(), sample.bounds())
    with pytest.raises(ValueError):
        sample.observed_bounds_world[0, 0] = 100.
    # Historical callers without measured extents retain their existing checks.
    legacy = replace(sample, observed_bounds_world=None)
    assert np.array_equal(legacy.support_evidence_bounds(), legacy.bounds())
