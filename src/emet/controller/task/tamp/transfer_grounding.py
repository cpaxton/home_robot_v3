# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Turn current-view semantic masks into frame-bound transfer proposals."""

import numpy as np
from scipy.ndimage import binary_erosion, label

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


def select_transfer(sample, *, profile, segment=None):
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
    source_seed, counter_height, source_mask = _seed_from_support(masks["source"], world[source_camera])
    dest_seed, _, dest_mask = _seed_from_support(masks["destination"], world[dest_camera], max_height=counter_height - .025)
    # A segmented pickup candidate needs depth above the selected counter.
    object_mask = masks["object"] & (world[object_camera][..., 2] > counter_height + .008)
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
