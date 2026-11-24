# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Turn current-view semantic masks into frame-bound transfer proposals."""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from emet.controller.manipulation.native_inputs import ObservationUnavailable

from .live_transfer import TransferSelection
from .observed_capture import capture_arrays


def _bounds(mask):
    if not mask.any():
        raise ObservationUnavailable("transfer_grounding_unavailable")
    y, x = np.where(mask)
    h, w = mask.shape
    return [
        int(1000 * x.min() / w), int(1000 * y.min() / h),
        int(np.ceil(1000 * (x.max() + 1) / w)), int(np.ceil(1000 * (y.max() + 1) / h)),
    ]


def _seed_from_support(mask, world, *, max_height=None):
    valid = mask & np.isfinite(world).all(axis=-1)
    if max_height is not None:
        valid &= world[..., 2] < max_height
    if valid.sum() < 30:
        raise ObservationUnavailable("support_geometry_unknown")
    heights, counts = np.unique(np.round(world[..., 2][valid] / .004), return_counts=True)
    for index in np.argsort(-counts, kind="stable")[:6]:
        plane = valid & (np.abs(world[..., 2] - heights[index] * .004) <= .0028)
        components, _ = label(binary_erosion(plane, iterations=2))
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        if sizes.max() < 25:
            continue
        ys, xs = np.where(components == sizes.argmax())
        surface = world[ys, xs]
        _, singular, axes = np.linalg.svd(surface-surface.mean(axis=0), full_matrices=False)
        if singular[1]/np.sqrt(len(surface)) < .005 or abs(axes[-1, 2]) < .94:
            continue  # A horizontal slice of a cabinet face is not a support.
        nearest = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
        h, w = mask.shape
        seed = [int(np.ceil(1000 * xs[nearest] / w)), int(np.ceil(1000 * ys[nearest] / h))]
        measured, _ = label(plane)
        component = measured[ys[nearest], xs[nearest]]
        return {"verified": True, "box": _bounds(mask), "point": seed}, float(heights[index] * .004), (
            measured == component
        )
    raise ObservationUnavailable("support_geometry_unknown")


def _box_mask(box, image_shape):
    h, w = image_shape
    limits = np.asarray(box, dtype=int)
    if limits.shape != (4,) or np.any(limits < 0) or np.any(limits > 1000) or np.any(limits[:2] >= limits[2:]):
        raise ValueError("invalid_transfer_diagnostic_box")
    x0, y0 = np.floor(limits[:2] * [w, h] / 1000).astype(int)
    x1, y1 = np.ceil(limits[2:] * [w, h] / 1000).astype(int)
    mask = np.zeros(image_shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _depth_components(mask, world, *, max_neighbor_gap_m=.02):
    """Label neighboring observed surfaces without bridging image depth jumps."""
    indices = np.full(mask.shape, -1, dtype=np.int32)
    indices[mask] = np.arange(mask.sum())
    rows, columns = [], []
    for left, right in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        adjacent = mask[left] & mask[right]
        adjacent &= np.linalg.norm(world[left]-world[right], axis=-1) <= max_neighbor_gap_m
        rows.append(indices[left][adjacent])
        columns.append(indices[right][adjacent])
    graph = coo_matrix((np.ones(sum(len(r) for r in rows)),
        (np.concatenate(rows), np.concatenate(columns))), shape=(int(mask.sum()), int(mask.sum())))
    _, components = connected_components(graph, directed=False)
    result = np.zeros(mask.shape, dtype=np.int32)
    result[mask] = components+1
    return result


def select_transfer(sample, *, profile, segment=None, grow_object_region=False, require_destination=True):
    ground = profile["grounding"]
    cameras = {
        role: ground[role]["camera"] for role in ("object", "source", "destination")
    }
    masks = {}
    world = {}
    for camera in set(cameras.values()):
        obs = sample.observations[camera]
        _, depth, world[camera] = capture_arrays(obs)
        roles = [role for role, selected in cameras.items() if selected == camera]
        vocabulary = [ground[role]["label"] for role in roles if "label" in ground[role]]
        detections = segment(obs.rgb, depth, vocabulary) if vocabulary and segment is not None else ()
        for role in roles:
            if "box" in ground[role]:
                mask = _box_mask(ground[role]["box"], depth.shape)
            else:
                query = ground[role]["label"].casefold()
                matches = [d for d in detections if d.get("label", "").casefold() == query]
                if len(matches) != 1:
                    raise ObservationUnavailable("transfer_grounding_unavailable" if not matches else "ambiguous_object")
                mask = np.asarray(matches[0]["mask"], dtype=bool)
            masks[role] = mask & (depth >= .1) & np.isfinite(world[camera]).all(axis=-1)
    source_camera, dest_camera, object_camera = (cameras[role] for role in ("source", "destination", "object"))
    source_region = masks['source']
    if grow_object_region:
        source_region = binary_dilation(source_region, iterations=12)
        if source_camera == object_camera:
            source_region &= ~binary_dilation(masks['object'], iterations=2)
    if profile.get('local_support_estimation'):
        object_points = world[object_camera][masks['object']]
        if len(object_points) < 25:
            raise ObservationUnavailable('transfer_grounding_unavailable')
        # A support cannot be above the selected object. If the semantic support
        # patch was on the robot, estimate support from measured pixels nearby.
        top = float(np.quantile(object_points[:, 2], .9))+.005
        try:
            source_seed, counter_height, source_mask = _seed_from_support(
                source_region, world[source_camera], max_height=top)
        except ObservationUnavailable:
            neighborhood = binary_dilation(masks['object'], iterations=30)
            neighborhood &= ~binary_dilation(masks['object'], iterations=2)
            source_camera = object_camera
            source_seed, counter_height, source_mask = _seed_from_support(
                neighborhood, world[source_camera], max_height=top)
    else:
        source_seed, counter_height, source_mask = _seed_from_support(source_region, world[source_camera])
    if require_destination:
        dest_seed, floor_height, dest_mask = _seed_from_support(masks["destination"], world[dest_camera], max_height=counter_height - .025)
    else:
        # Pickup diagnostics do not depend on observing a placement surface.
        dest_camera, dest_seed, dest_mask = source_camera, source_seed, source_mask.copy()
    if grow_object_region and require_destination:
        floor_points = world[dest_camera]
        plane = np.isfinite(floor_points).all(axis=-1) & (np.abs(floor_points[..., 2]-floor_height) <= .008)
        components, _ = label(plane)
        ids, counts = np.unique(components[dest_mask], return_counts=True)
        counts[ids == 0] = 0
        if counts.max() > 0:
            dest_mask = components == ids[counts.argmax()]
    # A segmented pickup candidate needs depth above the selected counter.
    object_mask = masks["object"] & (world[object_camera][..., 2] > counter_height + .008)
    if grow_object_region:
        # A semantic image hint seeds measured connected geometry; it need not
        # trace the entire silhouette. Competing seeded components stay ambiguous.
        depth = np.asarray(sample.observations[object_camera].depth)
        eligible = (np.isfinite(world[object_camera]).all(axis=-1) & (depth >= .1)
                    & (world[object_camera][..., 2] > counter_height + .008))
        # Do not grow an object hint through the entire scene merely because
        # the robot, cabinets and walls touch in the image's above-plane mask.
        # This is a local observed patch, not a complete object reconstruction.
        eligible &= binary_dilation(masks['object'], iterations=12)
        grown = _depth_components(eligible, world[object_camera])
        ids, overlaps = np.unique(grown[object_mask], return_counts=True)
        overlaps, ids = overlaps[ids != 0], ids[ids != 0]
        if len(ids) > 1:
            order = np.argsort(-overlaps)
            if overlaps[order[0]] < 1.5*overlaps[order[1]]:
                raise ObservationUnavailable('ambiguous_object')
            ids = ids[order[:1]]
        if not len(ids) and masks['object'].any():
            # VLM coordinates are coarse hints. Snap only to a nearby, clearly
            # preferred measured component, never an arbitrary scene object.
            yy, xx = np.where(masks['object'])
            center = np.array([xx.mean(), yy.mean()])
            candidates = []
            for component in range(1, int(grown.max())+1):
                cy, cx = np.where(grown == component)
                if len(cx) >= 25:
                    distance = np.linalg.norm(np.array([cx.mean(), cy.mean()])-center)
                    candidates.append((distance, component))
            candidates.sort()
            scale = np.hypot(*depth.shape)
            if (candidates and candidates[0][0] <= .1*scale
                    and (len(candidates) == 1 or candidates[1][0]-candidates[0][0] > .01*scale)):
                ids = np.array([candidates[0][1]])
        object_mask = np.isin(grown, ids) if len(ids) else np.zeros_like(object_mask)
    components, _ = label(object_mask)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    viable = np.flatnonzero(sizes >= 25)
    if len(viable) != 1:
        raise ObservationUnavailable("transfer_grounding_unavailable" if not len(viable) else "ambiguous_object")
    object_mask = binary_erosion(components == viable[0])
    if object_mask.sum() < 35:
        raise ObservationUnavailable("transfer_grounding_unavailable")
    return TransferSelection(
        object_camera, object_mask, source_camera, source_seed, dest_camera, dest_seed,
        sample.capture.observation_id, source_mask, dest_mask,
    )


def select_transfer_from_hints(sample, hints, *, refine):
    """Ground a transfer from agent-provided image hints and measured depth.

    No asset identity, object CAD or manual benchmark profile is consulted.
    ``refine(rgb, box_xyxy)`` supplies a segmentation mask for an image-space
    proposal. Subsequent depth checks remain the same as ordinary grounding.
    """
    from emet.perception.grasps.observed import GroundingHint

    roles = ('object', 'source', 'destination')
    if set(hints) != set(roles) or not callable(refine):
        raise ValueError('invalid_transfer_hints')
    ground = {}
    masks = {}
    for role in roles:
        hint = hints[role]
        if (not isinstance(hint, GroundingHint) or hint.observation_id != sample.capture.observation_id
                or hint.camera_id not in sample.observations):
            raise ObservationUnavailable('capture_identity_mismatch')
    # Validate every hint before invoking a potentially expensive model.
    for role in roles:
        hint = hints[role]
        obs = sample.observations[hint.camera_id]
        mask = np.asarray(refine(obs.rgb, hint.box_xyxy))
        if mask.dtype != bool or mask.shape != obs.rgb.shape[:2]:
            raise ValueError('unaligned_grounding_mask')
        masks[role] = mask
        ground[role] = {'camera': hint.camera_id, 'label': role}

    def segment(_rgb, _depth, vocabulary):
        return [{'label': role, 'mask': masks[role]} for role in vocabulary]

    return select_transfer(sample, profile={'grounding': ground}, segment=segment)


def propose_transfer_cloud_grasps(sample, selection, *, provider, position_error_m):
    """Generate capture-bound world-hand proposals from the selected raw RGB-D.

    The returned geometry describes visible samples only. It does not qualify a
    complete carried-object collision envelope or invent an object pose.
    """
    from emet.perception.grasps.observed import GroundingHint, geometry_from_mask

    if selection.capture_id != sample.capture.observation_id:
        raise ObservationUnavailable('capture_identity_mismatch')
    camera = selection.object_camera
    obs = sample.observations[camera]
    _, _, world = capture_arrays(obs)
    hint = GroundingHint(selection.capture_id, camera, tuple(_bounds(selection.object_mask)), 'selected-object')
    geometry = geometry_from_mask(
        hint, observation_id=sample.capture.observation_id, camera_id=camera,
        acquired_monotonic=sample.capture.earliest, mask=selection.object_mask,
        world_points=world, position_error_m=position_error_m,
    )
    return geometry, provider.propose(geometry)
