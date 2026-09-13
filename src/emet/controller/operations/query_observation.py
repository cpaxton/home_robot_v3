# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Ephemeral visual geometry for manipulation; does not create memory instances."""

import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emet.controller.dynamem.look import wait_post_motion_obs
from emet.memory.query_grounding import cache_grounding_record


def observe_query_points(agent, robot, query, *, stage):
    before = getattr(robot, "_seq_id", None)
    wait_post_motion_obs(robot, timeout=2.0)
    if not isinstance(before, int) or getattr(robot, "_seq_id", before) <= before:
        raise ValueError("Fresh manipulation image required")
    obs = robot.get_observation()
    world = obs.get_xyz_in_world_frame()
    if world is None or obs.rgb is None or obs.depth is None:
        raise ValueError("Calibrated manipulation RGB-D required")
    frame = SimpleNamespace(rgb=obs.rgb, depth=obs.depth, full_world_xyz=world)
    detected, matching, verification = None, [], {}
    error = None
    try:
        detected, _, matching, verification = agent.ground_vlm_frame(frame, query, query, min_depth=0.0)
        if not verification.get("valid") or len(matching) != 1:
            raise ValueError("Manipulation target identity absent or ambiguous")
        mask = (detected.instance == matching[0]) & np.isfinite(world).all(axis=-1)
        mask &= np.isfinite(obs.depth) & (obs.depth > 0)
        points = world[mask]
        if len(points) < 10:
            raise ValueError("Insufficient observed manipulation geometry")
        return obs, points
    except ValueError as exc:
        error = str(exc)
        raise
    finally:
        output = os.environ.get("EMET_EQA_EPISODE_DIR")
        if output:
            cache_grounding_record(
                Path(output) / stage,
                query=query,
                revision=len(agent.voxel_map.observations),
                source_obs_id=None,
                detections=[],
                matching_ids=matching,
                verification={
                    **verification,
                    "valid": error is None and bool(verification.get("valid")),
                    "stage": stage,
                    "reason": error,
                },
                rgb=obs.rgb,
                depth=obs.depth,
                masks=None if detected is None else detected.instance,
                metadata={
                    "camera_K": obs.camera_K.tolist(),
                    "camera_pose": obs.camera_pose.tolist(),
                    "recorded_at": time.time(),
                    "joint": obs.joint.tolist(),
                    "ee_pose": None if obs.ee_pose is None else obs.ee_pose.tolist(),
                },
            )
