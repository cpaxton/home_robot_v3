# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Adapt the existing voxel-frame history to the optional shared view selector."""

import os

import numpy as np

from emet.llms.eqa_vl_settings import get_eqa_vl_int
from emet.memory.graph_eqa.agentic.config import _eqa_cfg
from emet.memory.graph_eqa.ingest.instance_observations import frame_rgb_hwc_uint8
from emet.memory.view_selection import SelectionConfig, ViewFrame, select_views


def select_assessment_history(executor, rgb):
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
    frames = []
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
        frames.append(ViewFrame(str(index + 1), str(clock), stamp / 1e9, pixels, camera_pose=pose))
    if not frames or frames[-1].view_id != str(len(history)) or not np.array_equal(frames[-1].rgb, rgb):
        executor._append_trace({"event": "view_selection_unavailable", "reason": "current_acquisition_not_retained"})
        return None
    current_id = frames[-1].view_id
    pre_turn_id = None
    before = getattr(executor.agent.robot, "pre_turn_observation", None)
    if before is not None:
        timing = (getattr(before, "image_timing", None) or {}).get("head_cam_left") or {}
        stamp, clock = timing.get("timestamp_ns"), timing.get("clock_domain")
        if (
            isinstance(stamp, (int, float))
            and np.isfinite(stamp)
            and clock == frames[-1].scope_id
            and 0 <= frames[-1].timestamp_s - stamp / 1e9 <= config.max_age_s
        ):
            pre_turn_id = f"pre-turn:{stamp}"
            frames.append(ViewFrame(pre_turn_id, str(clock), stamp / 1e9, before.rgb, camera_pose=before.camera_pose))
    selection = select_views(frames, current_id=current_id, pre_turn_id=pre_turn_id, config=config)
    executor._append_trace({"event": "view_selection", **selection.to_dict()})
    return selection
