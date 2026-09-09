# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Class-agnostic, observation-local surface proposals; never object identities."""

import base64

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label


def surface_candidates(depth, box, *, min_depth, max_depth, proposal_masks=None):
    """Split a VLM search box into measured depth layers and connected components.

    Optional class-agnostic masks replace depth-layer proposals, not depth
    validity checks. Touching/coplanar objects need not separate. Overflow fails
    closed instead of silently discarding a possibly relevant small surface.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2 or not isinstance(box, list) or len(box) != 4:
        raise ValueError("RGB-D and normalized search box required")
    if any(type(v) is not int or not 0 <= v <= 1000 for v in box):
        raise ValueError("invalid normalized search box")
    h, w = depth.shape
    x0, y0 = np.floor(np.array(box[:2]) * [w, h] / 1000).astype(int)
    x1, y1 = np.ceil(np.array(box[2:]) * [w, h] / 1000).astype(int)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("empty search box")
    roi = depth[y0:y1, x0:x1]
    valid = np.isfinite(roi) & (roi > min_depth) & (roi < max_depth)
    if not valid.any():
        return []
    if proposal_masks is None:
        # Split only at measured gaps, not an assumed foreground or object depth.
        values = np.unique(roi[valid])
        cuts = (values[:-1] + values[1:])[np.diff(values) > 0.08] / 2
        layers = np.searchsorted(cuts, roi)
        masks = [valid & (layers == index) for index in range(len(cuts) + 1)]
    else:
        masks = []
        for mask in proposal_masks:
            mask = np.asarray(mask)
            if mask.dtype != bool or mask.shape != depth.shape:
                raise ValueError("proposal masks must be boolean and aligned with depth")
            masks.append(mask[y0:y1, x0:x1] & valid)
    regions = []
    for mask in masks:
        components, count = label(mask)
        sizes = np.bincount(components.ravel())
        for index in range(1, count + 1):
            if sizes[index] < 25:
                continue
            support = components == index
            yy, xx = np.where(support)
            left, top, right, bottom = xx.min(), yy.min(), xx.max() + 1, yy.max() + 1
            crop = support[top:bottom, left:right]
            regions.append(
                {
                    "id": len(regions),
                    "bbox_xyxy": [int(left + x0), int(top + y0), int(right + x0), int(bottom + y0)],
                    "mask_bits": base64.b64encode(np.packbits(crop).tobytes()).decode("ascii"),
                    "points": int(crop.sum()),
                }
            )
            if len(regions) > 8:
                raise ValueError("too many surface candidates; another view or segmentation is needed")
    return regions


def candidate_mask(region, shape):
    left, top, right, bottom = region["bbox_xyxy"]
    crop_shape = (bottom - top, right - left)
    packed = np.frombuffer(base64.b64decode(region["mask_bits"]), dtype=np.uint8)
    crop = np.unpackbits(packed, count=int(np.prod(crop_shape))).reshape(crop_shape).astype(bool)
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = crop
    return mask


def surface_candidate_image(rgb, regions):
    """Exact mask overlay sent to the VLM; the unmodified image is sent too."""
    colors = [(255, 190, 0), (0, 220, 255), (220, 80, 255), (80, 255, 130)]
    overlay = np.asarray(rgb).copy()
    positions = []
    for region in regions:
        mask = candidate_mask(region, rgb.shape[:2])
        overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array(colors[region["id"] % len(colors)])).astype(np.uint8)
        # Put the ID inside actual support rather than a possibly empty box center.
        y, x = np.unravel_index(
            np.argmax(distance_transform_edt(np.pad(mask, 1))), (mask.shape[0] + 2, mask.shape[1] + 2)
        )
        positions.append((x - 1, y - 1, region["id"]))
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    for x, y, index in positions:
        draw.rectangle((x - 7, y - 8, x + 8, y + 8), fill="black")
        draw.text((x - 3, y - 5), str(index), fill="white")
    return image
