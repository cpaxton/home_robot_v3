# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Proposal bearings are source-view facts, not object or route authorization."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from emet.memory.query_grounding import proposal_view_context


def fixture():
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    frame = SimpleNamespace(
        depth=np.ones((100, 100)),
        camera_K=np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]]),
        camera_pose=pose,
        base_pose=np.zeros(3),
        info={"observation_scope_id": "episode-a", "image_timing": {"clock_domain": "sim"}},
    )
    support = {
        "semantics": "unverified proposals, not localized targets",
        "proposals": [
            {"proposal_index": 0, "bbox_xyxy": [0, 40, 40, 60], "median_depth_m": 3.34, "supported_pixels": 0},
        ],
    }
    return frame, support


def test_signed_bearing_links_rejected_far_proposal_to_capture_without_xyz():
    frame, support = fixture()
    before = deepcopy(support)
    result = proposal_view_context(support, frame, source_obs_id=17)
    row = result["proposals"][0]
    assert row["base_yaw_delta_deg"] == pytest.approx(np.degrees(np.arctan(0.3)))
    assert row["optical_offset_deg"]["right"] < 0 and row["touches_image_edge"]
    assert row["image_center_normalized_xy"] == [200, 500]
    assert row["supported_pixels"] == 0 and row["median_depth_m"] == 3.34
    assert result["view"]["source_obs_id"] == 17
    assert result["view"]["observation_scope_id"] == "episode-a"
    assert "not object geometry" in result["view"]["bearing_semantics"]
    assert "xyz" not in row and support == before
    result["view"]["image_timing"]["clock_domain"] = "changed"
    assert frame.info["image_timing"]["clock_domain"] == "sim"


def test_bearing_is_invariant_to_world_yaw_and_translation():
    frame, support = fixture()
    reference = proposal_view_context(support, frame, source_obs_id=1)["proposals"][0]["base_yaw_delta_deg"]
    theta = 1.2
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    frame.camera_pose[:3, :3] = rotation @ frame.camera_pose[:3, :3]
    frame.camera_pose[:3, 3] = [5, -8, 2]
    frame.base_pose = np.array([5, -8, theta])
    assert proposal_view_context(support, frame, source_obs_id=2)["proposals"][0][
        "base_yaw_delta_deg"
    ] == pytest.approx(reference)


@pytest.mark.parametrize("missing", ["camera_K", "camera_pose", "base_pose"])
def test_missing_calibration_never_invents_base_bearing(missing):
    frame, support = fixture()
    setattr(frame, missing, None)
    result = proposal_view_context(support, frame, source_obs_id=1)
    assert result["proposals"][0]["base_yaw_delta_deg"] is None


def test_reflected_basis_and_bad_box_do_not_generate_motion_cues():
    frame, support = fixture()
    frame.camera_pose[:3, 0] *= -1
    assert proposal_view_context(support, frame, source_obs_id=1)["proposals"][0]["base_yaw_delta_deg"] is None
    support["proposals"][0]["bbox_xyxy"] = [-1, 2, 30, 50]
    row = proposal_view_context(support, frame, source_obs_id=1)["proposals"][0]
    assert row["image_geometry_status"] == "invalid_box" and row["base_yaw_delta_deg"] is None
