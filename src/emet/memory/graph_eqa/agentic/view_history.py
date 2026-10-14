# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Adapt the existing voxel-frame history to the optional shared view selector."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np

from emet.llms.eqa_vl_settings import get_eqa_vl_int
from emet.memory.graph_eqa.agentic.config import _eqa_cfg
from emet.memory.graph_eqa.ingest.instance_observations import frame_rgb_hwc_uint8
from emet.memory.view_coverage import CoverageGrid, depth_surface_quality, gaussian_coverage_objective
from emet.memory.view_selection import SelectionConfig, ViewFrame, select_views


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _dump_selection(executor, selection):
    """Persist the exact ordered packet independently of graph/debug exporters."""
    root = getattr(executor.agent, "_episode_debug_dir", None)
    if not root:
        return
    from emet.memory.graph_eqa.eqa.query_images import save_rgb_png

    sequence = getattr(executor, "_view_selection_sequence", 0) + 1
    executor._view_selection_sequence = sequence
    directory = Path(root) / "selected_views" / f"{sequence:04d}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        record = selection.to_dict()
        for index, (selected, row) in enumerate(zip(selection.views, record["views"], strict=True), 1):
            pixels = selected.frame.rgb
            filename = f"image_{index}.png"
            save_rgb_png(directory / filename, pixels)
            row["image"] = filename
            row["rgb_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
            row["camera_pose"] = None if selected.frame.camera_pose is None else selected.frame.camera_pose.tolist()
        (directory / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    except OSError as error:
        executor._append_trace({"event": "view_selection_artifact_error", "reason": str(error)})


def _coverage_inputs(executor, frames, sources, target_xy, scope):
    """Bounded, query-local cache of immutable retained RGB-D surface features."""
    grid = CoverageGrid(0.2)
    previous = getattr(executor, "_view_surface_cache", {})
    cache = {}
    enriched = []
    for frame in frames:
        source = sources[frame.view_id]
        entry = previous.get(frame.view_id)
        if entry is None or entry[0] is not source:
            depth = _numpy(getattr(source, "depth", None))
            if depth.shape != frame.rgb.shape[:2]:
                raise ValueError("missing_or_unregistered_depth")
            quality = depth_surface_quality(
                depth,
                _numpy(getattr(source, "camera_K", None)),
                frame.camera_pose,
                grid=grid,
                max_depth_m=4.0,
                pixel_stride=4,
                reference_pixels=1024,
                context_credit=0.5,
            )
            entry = (source, quality)
        cache[frame.view_id] = entry
        enriched.append(replace(frame, visible_cells=frozenset(entry[1]), cell_quality=entry[1]))
    executor._view_surface_cache = cache
    centers = grid.centers(cell for frame in enriched for cell in frame.visible_cells)
    # Match the offline support: a square ROI extending three sigma from target.
    centers = {k: xy for k, xy in centers.items() if np.all(np.abs(np.asarray(xy) - target_xy) <= 4.5)}
    if not centers:
        raise ValueError("no_observed_target_local_cells")
    objective = gaussian_coverage_objective(
        centers,
        target_xy=tuple(target_xy),
        sigma_m=1.5,
        scope_id=scope,
        visibility_source="retained_rgbd_endpoints_stride4_range4_quality1024",
    )
    return enriched, objective


def select_assessment_history(executor, rgb, *, target_obs_id=None):
    settings = dict(_eqa_cfg(executor.agent).get("view_selection") or {})
    if os.environ.get("EMET_EQA_VIEW_SELECTION_POLICY"):
        settings["policy"] = os.environ["EMET_EQA_VIEW_SELECTION_POLICY"]
    if not settings:
        return None
    limit = min(4, get_eqa_vl_int(executor.agent.parameters, "eqa_max_images", 4))
    if limit <= 0:
        return None
    settings["max_images"] = min(settings.get("max_images", 4), limit)
    config = SelectionConfig(**settings)
    history = getattr(getattr(executor.agent, "voxel_map", None), "observations", ())
    # Clock names alone are not world identities. Reset on replacement/clear of
    # the history, including a clear followed by a refill between assessments.
    first = history[0] if len(history) else None
    if getattr(executor, "_view_history_first", None) is not first:
        executor._view_history_first = first
        executor._view_history_scope = uuid4().hex
        executor._view_surface_cache = {}
    scope = getattr(executor, "_view_history_scope", "empty")
    frames = []
    sources = {}
    for index in range(max(0, len(history) - config.max_candidates), len(history)):
        frame = history[index]
        timing = ((getattr(frame, "info", None) or {}).get("image_timing") or {}).get("head_cam_left") or {}
        stamp, clock = timing.get("timestamp_ns"), timing.get("clock_domain")
        if not isinstance(stamp, (int, float)) or not np.isfinite(stamp) or stamp <= 0 or not clock:
            continue
        pixels = frame_rgb_hwc_uint8(frame)
        if pixels is None:
            continue
        pose = getattr(frame, "camera_pose", None)
        if hasattr(pose, "detach"):
            pose = pose.detach().cpu().numpy()
        # The executor owns a single query in one world; clocks must not mix.
        view_id = str(index + 1)
        frames.append(ViewFrame(view_id, f"{scope}:{clock}", stamp / 1e9, pixels, camera_pose=pose))
        sources[view_id] = frame
    if not frames or frames[-1].view_id != str(len(history)) or not np.array_equal(frames[-1].rgb, rgb):
        executor._append_trace({"event": "view_selection_unavailable", "reason": "current_acquisition_not_retained"})
        return None
    current_id = frames[-1].view_id
    current_scope = frames[-1].scope_id
    approach = getattr(executor.agent.robot, "navigation_view_history", None)
    for view in approach.views if approach is not None else ():
        frames.append(
            ViewFrame(
                view.view_id,
                f"{scope}:{view.clock_domain}",
                view.timestamp_ns / 1e9,
                view.rgb,
                camera_pose=view.camera_pose,
            )
        )
        sources[view.view_id] = view
    pre_turn_id = None
    before = getattr(executor.agent.robot, "pre_turn_observation", None)
    if before is not None:
        timing = (getattr(before, "image_timing", None) or {}).get("head_cam_left") or {}
        stamp, clock = timing.get("timestamp_ns"), timing.get("clock_domain")
        if (
            isinstance(stamp, (int, float))
            and np.isfinite(stamp)
            and f"{scope}:{clock}" == current_scope
            and 0 <= next(f.timestamp_s for f in frames if f.view_id == current_id) - stamp / 1e9 <= config.max_age_s
        ):
            pre_turn_id = f"pre-turn:{stamp}"
            frames.append(
                ViewFrame(pre_turn_id, f"{scope}:{clock}", stamp / 1e9, before.rgb, camera_pose=before.camera_pose)
            )
            sources[pre_turn_id] = before
    objective = None
    requested_policy = config.policy
    if requested_policy in {"weighted_coverage", "quality_coverage"}:
        # Only explicitly associated investigation geometry is a target. A
        # general observation anchor may just be the camera/viewpoint position.
        anchor = getattr(executor, "_inspection_target_by_obs", {}).get(target_obs_id)
        try:
            if anchor is None or np.asarray(anchor).size < 2 or not np.isfinite(np.asarray(anchor)[:2]).all():
                raise ValueError("missing_target_geometry")
            # Scope is the current acquisition, not the oldest retained frame.
            current_scope = next(f.scope_id for f in frames if f.view_id == current_id)
            eligible = [f for f in frames if f.scope_id == current_scope]
            frames, objective = _coverage_inputs(executor, eligible, sources, np.asarray(anchor)[:2], current_scope)
        except (ValueError, TypeError, np.linalg.LinAlgError) as error:
            executor._append_trace(
                {
                    "event": "view_selection_fallback",
                    "requested_policy": requested_policy,
                    "actual_policy": "pose",
                    "reason": str(error),
                }
            )
            config = replace(config, policy="pose")
    selection = select_views(
        frames, current_id=current_id, pre_turn_id=pre_turn_id, config=config, coverage_objective=objective
    )
    executor._append_trace({"event": "view_selection", **selection.to_dict()})
    _dump_selection(executor, selection)
    return selection
