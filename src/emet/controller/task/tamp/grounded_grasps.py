# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Fresh RGB-D realization of optional semantic grasp proposals; no object CAD."""
import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.motion.observed_pose_targets import horizontal_image_direction
from emet.perception.grasps.observed import GraspProposal

from .observed_capture import capture_arrays


def grasp_proposals_from_decision(sample, selection, plane, decision, *, position_error_m):
    if not all(key in decision for key in ('grasp_point', 'grasp_yaw_degrees', 'grasp_height_fraction')):
        return ()
    point = np.asarray(decision['grasp_point'], float)
    fraction, yaw = float(decision['grasp_height_fraction']), float(decision['grasp_yaw_degrees'])
    if (point.shape != (2,) or not np.isfinite(point).all() or np.any((point < 0) | (point > 1000))
            or not np.isfinite([fraction, yaw]).all() or not 0 <= fraction <= 1 or not 0 <= yaw <= 360):
        raise ObservationUnavailable('invalid_manipulation_hint')
    observation = sample.observations[selection.object_camera]
    points = capture_arrays(observation)[2][selection.object_mask]
    xy = np.median(points[:, :2], axis=0)
    bottom = float(xy @ plane[:2]+plane[2])
    top = float(np.quantile(points[:, 2], .98))
    if top <= bottom:
        raise ObservationUnavailable('payload_geometry_unknown')
    height = bottom+fraction*(top-bottom)
    ys, xs = np.where(selection.object_mask)
    pixel = point/1000*np.array(selection.object_mask.shape[::-1])
    pixel_distance = np.hypot(xs-pixel[0], ys-pixel[1])
    nearest = int(np.argmin(pixel_distance))
    if pixel_distance[nearest] > .05*np.hypot(*selection.object_mask.shape):
        raise ObservationUnavailable('invalid_manipulation_hint')
    seed = capture_arrays(observation)[2][ys[nearest], xs[nearest]]
    site = points[np.linalg.norm(points[:, :2]-seed[:2], axis=1) <= .03]
    if len(site) < 8:
        raise ObservationUnavailable('invalid_manipulation_hint')
    # Use the image point to select measured material, not a ray projected
    # through an unseen interior plane. At oblique angles that projection can
    # move the grasp several centimetres off the measured body section.
    center = np.r_[(np.quantile(site[:, :2], .1, axis=0)+np.quantile(site[:, :2], .9, axis=0))/2, height]
    closing = horizontal_image_direction(observation, point, yaw, height)
    x, z = np.r_[closing, 0.], np.array([0., 0., -1.])
    rotation = np.column_stack((x, np.cross(z, x), z))
    # A local observed body section estimates opening; it is not a complete
    # shape or a claim that an unseen side was measured.
    local = points[np.linalg.norm(points[:, :2]-center[:2], axis=1) <= .04]
    if len(local) < 8:
        raise ObservationUnavailable('invalid_manipulation_hint')
    projection = local @ x
    opening = float(np.quantile(projection, .95)-np.quantile(projection, .05)+2*position_error_m)
    poses = []
    for signs in (np.eye(3), np.diag([-1., -1., 1.])):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation @ signs, center
        poses.append(GraspProposal(sample.capture.observation_id, pose, max(opening, .005),
                                  provenance='vlm_site_observed_depth'))
    return tuple(poses)
