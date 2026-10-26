# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Preserve qualified before-turn evidence; optionally select extra history."""

import os
from uuid import uuid4

import numpy as np

from emet.llms.eqa_vl_settings import get_eqa_vl_int
from emet.memory.graph_eqa.agentic.config import _eqa_cfg
from emet.memory.graph_eqa.ingest.instance_observations import frame_rgb_hwc_uint8
from emet.memory.view_selection import SelectionConfig, ViewFrame, select_views


def select_assessment_history(executor, rgb, *, current_observation=None):
    settings = dict(_eqa_cfg(executor.agent).get("view_selection") or {})
    if os.environ.get("EMET_EQA_VIEW_SELECTION_POLICY"):
        settings["policy"] = os.environ["EMET_EQA_VIEW_SELECTION_POLICY"]
    rank_history = bool(settings)
    before = getattr(executor.agent.robot, "pre_turn_observation", None)
    if not rank_history and before is None:
        return None
    limit = min(4, get_eqa_vl_int(executor.agent.parameters, "eqa_max_images", 4))
    if limit <= 0:
        return None
    settings["max_images"] = min(settings.get("max_images", 4 if rank_history else 2), limit)
    config = SelectionConfig(**settings)
    voxel_map = getattr(executor.agent, "voxel_map", None)
    history = getattr(voxel_map, "observations", ())
    # Production maps mint an epoch on reset and stamp it into each frame.
    # Legacy adapters without that contract use the retained-history identity.
    epoch = getattr(voxel_map, "observation_scope_id", None)
    if epoch is None:
        first = history[0] if len(history) else None
        if getattr(executor, "_view_history_first", None) is not first:
            executor._view_history_first = first
            executor._view_history_scope = uuid4().hex
        epoch = getattr(executor, "_view_history_scope", "empty")
    frames = []
    for index in range(max(0, len(history) - config.max_candidates), len(history)):
        frame = history[index]
        frame_epoch = (getattr(frame, "info", None) or {}).get("observation_scope_id")
        if getattr(voxel_map, "observation_scope_id", None) is not None and frame_epoch != epoch:
            continue
        timing = ((getattr(frame, "info", None) or {}).get("image_timing") or {}).get("head_cam_left") or {}
        stamp, clock = timing.get("timestamp_ns"), timing.get("clock_domain")
        if (
            isinstance(stamp, bool)
            or not isinstance(stamp, (int, float))
            or not np.isfinite(stamp)
            or stamp <= 0
            or not clock
        ):
            continue
        pixels = frame_rgb_hwc_uint8(frame)
        if pixels is None:
            continue
        pose = getattr(frame, "camera_pose", None)
        if hasattr(pose, "detach"):
            pose = pose.detach().cpu().numpy()
        # The executor owns a single query in one world; clocks must not mix.
        frames.append(ViewFrame(str(index + 1), f"{epoch}:{clock}", stamp / 1e9, pixels, camera_pose=pose))
    current_id = None
    if current_observation is not None and np.array_equal(current_observation.rgb, rgb):
        timing = (getattr(current_observation, "image_timing", None) or {}).get("head_cam_left") or {}
        stamp, clock = timing.get("timestamp_ns"), timing.get("clock_domain")
        observation_epoch = getattr(current_observation, "observation_scope_id", None)
        if (
            isinstance(stamp, (int, float))
            and not isinstance(stamp, bool)
            and np.isfinite(stamp)
            and stamp > 0
            and clock
            and (
                observation_epoch == epoch
                or (observation_epoch is None and getattr(voxel_map, "observation_scope_id", None) is None)
            )
        ):
            current_id = f"assessment:{stamp}"
            frames.append(
                ViewFrame(current_id, f"{epoch}:{clock}", stamp / 1e9, rgb, camera_pose=current_observation.camera_pose)
            )
    if (
        current_id is None
        and frames
        and frames[-1].view_id == str(len(history))
        and np.array_equal(frames[-1].rgb, rgb)
    ):
        current_id = frames[-1].view_id
    if current_id is None:
        executor._append_trace(
            {
                "event": "view_selection_unavailable" if rank_history else "assessment_handoff_unavailable",
                "reason": "current_acquisition_not_retained",
            }
        )
        return None
    current = frames[-1]
    pre_turn_id = None
    if before is not None:
        timing = (getattr(before, "image_timing", None) or {}).get("head_cam_left") or {}
        stamp, clock = timing.get("timestamp_ns"), timing.get("clock_domain")
        if (
            isinstance(stamp, (int, float))
            and not isinstance(stamp, bool)
            and np.isfinite(stamp)
            and (getattr(before, "observation_scope_id", None) in (None, epoch))
            and f"{epoch}:{clock}" == current.scope_id
            and 0 < current.timestamp_s - stamp / 1e9 <= config.max_age_s
            and stamp / 1e9 >= min(f.timestamp_s for f in frames if f.scope_id == current.scope_id)
        ):
            pre_turn_id = f"pre-turn:{stamp}"
            frames.append(
                ViewFrame(pre_turn_id, current.scope_id, stamp / 1e9, before.rgb, camera_pose=before.camera_pose)
            )
    if not rank_history:
        if pre_turn_id is None:
            # Keep the pre-approach fallback when the adapter has no qualified
            # before-turn capture; never substitute unrelated history frames.
            executor._append_trace(
                {"event": "assessment_handoff_unavailable", "reason": "pre_turn_acquisition_unqualified"}
            )
            return None
        frames = [frame for frame in frames if frame.view_id in (current_id, pre_turn_id)]
    selection = select_views(frames, current_id=current_id, pre_turn_id=pre_turn_id, config=config)
    event = "view_selection" if rank_history else "assessment_handoff"
    executor._append_trace({"event": event, **selection.to_dict()})
    return selection
