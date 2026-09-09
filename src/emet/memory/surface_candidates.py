# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Class-agnostic, observation-local surface proposals; never object identities."""

import base64

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def surface_candidates(depth, box, *, min_depth, max_depth, rgb=None, proposal_masks=None):
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
    if proposal_masks is None and rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError("RGB must be uint8 and aligned with depth")
        colors = rgb[y0:y1, x0:x1].astype(float)
        indices = np.arange(roi.size).reshape(roi.shape)
        rows, cols = [], []
        for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
            compatible = valid[a] & valid[b]
            compatible &= np.abs(roi[a] - roi[b]) <= 0.08
            # Class-agnostic appearance boundary: no category/color vocabulary.
            # This avoids joining target and support through a few depth outliers.
            compatible &= np.max(np.abs(colors[a] - colors[b]), axis=-1) <= 35
            rows.append(indices[a][compatible])
            cols.append(indices[b][compatible])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        graph = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(roi.size, roi.size)).tocsr()
        _, pixel_components = connected_components(graph, directed=False)
        sizes = np.bincount(pixel_components[valid.ravel()])
        pixel_components = pixel_components.reshape(roi.shape)
        masks = (valid & (pixel_components == index) for index in np.flatnonzero(sizes >= 25))
    elif proposal_masks is None:
        # Split only at measured gaps, not an assumed foreground or object depth.
        values = np.unique(roi[valid])
        cuts = (values[:-1] + values[1:])[np.diff(values) > 0.08] / 2
        layers = np.searchsorted(cuts, roi)
        masks = (valid & (layers == index) for index in range(len(cuts) + 1))
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


def surface_candidate_panels(rgb, regions):
    """Enlarged candidate panels; retain RGB on support and dim its surroundings.

    All panels use the same context crop. Labels live outside the image, never
    over small objects. The separate original image remains unmodified.
    """
    if not regions:
        return []
    bounds = np.asarray([r["bbox_xyxy"] for r in regions])
    left, top = bounds[:, :2].min(axis=0)
    right, bottom = bounds[:, 2:].max(axis=0)
    padding = max(right - left, bottom - top) // 2
    left, top = max(0, left - padding), max(0, top - padding)
    right, bottom = min(rgb.shape[1], right + padding), min(rgb.shape[0], bottom + padding)
    crop = rgb[top:bottom, left:right]
    panels = []
    for region in regions:
        mask = candidate_mask(region, rgb.shape[:2])[top:bottom, left:right]
        pixels = (crop * 0.15).astype(np.uint8)
        pixels[mask] = crop[mask]
        panels.append((f"Candidate {region['id']}", pixels))
    width, height = 256, 280
    font = ImageFont.load_default(size=18)
    images = []
    for title, pixels in panels:
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        tile = Image.fromarray(pixels)
        scale = min(width / tile.width, (height - 24) / tile.height)
        tile = tile.resize(
            (max(1, round(tile.width * scale)), max(1, round(tile.height * scale))), Image.Resampling.NEAREST
        )
        image.paste(tile, ((width - tile.width) // 2, 24 + (height - 24 - tile.height) // 2))
        draw.text((6, 3), title, fill="white", font=font)
        images.append(image)
    return images


def surface_candidate_image(rgb, regions):
    """Human-review contact sheet; model inputs are the individual panels."""
    panels = surface_candidate_panels(rgb, regions)
    if not panels:
        return Image.fromarray(rgb)
    image = Image.new("RGB", (512, 280 * ((len(panels) + 1) // 2)))
    for index, panel in enumerate(panels):
        image.paste(panel, (256 * (index % 2), 280 * (index // 2)))
    return image
