# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
from dataclasses import replace

import numpy as np
import pytest

from emet.controller.manipulation.native_inputs import ObjectPoseEvidence
from emet.controller.manipulation.observed_collision import PayloadGeometryEvidence
from emet.controller.manipulation.observed_effects import (
    CaptureInterval,
    RobotPoseSample,
    VisualEffectFrame,
    evaluate_lift,
    evaluate_place,
)
from emet.motion.placement_geometry import box_corners


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


def test_lift_requires_repeated_comotion_and_separation():
    before = frame(10.)
    after = [frame(10.5, obj=(0, 0, .13), ee=(0, 0, .13)),
             frame(10.8, obj=(0, 0, .13), ee=(0, 0, .13))]
    assert evaluate_lift(before, after, support_top_z=0, after=10.4, now=11.).satisfied
    # Height alone cannot substitute for matching measured gripper motion.
    wrong = [replace(f, robot_before=before.robot_before, robot_after=RobotPoseSample(
        before.robot_after.world_from_ee, np.zeros(2), f.capture.latest+.02)) for f in after]
    assert not evaluate_lift(before, wrong, support_top_z=0, after=10.4, now=11.).satisfied


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
