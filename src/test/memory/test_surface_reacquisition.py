# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
from dataclasses import dataclass

import numpy as np
import pytest

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.task.tamp.observed_discovery import RgbdTaskProvider
from emet.memory.surface_reacquisition import reacquire_surface


def scene():
    y, x = np.mgrid[:60, :60]
    xyz = np.stack([x * 0.005, y * 0.005, np.full(x.shape, 0.5)], axis=-1)
    rgb = np.zeros((60, 60, 3), dtype=np.uint8)
    depth = np.full((60, 60), np.nan)
    table = np.zeros(depth.shape, dtype=bool)
    table[5:55, 5:55] = True
    obj = np.zeros(depth.shape, dtype=bool)
    obj[20:32, 20:32] = True
    depth[table] = 1.0
    depth[obj] = 0.8
    rgb[table] = [90, 60, 20]
    rgb[obj] = [200, 30, 30]
    xyz[obj, 2] = 0.6
    return rgb, depth, xyz, obj, table & ~obj


def associate(current=None):
    rgb, depth, xyz, mask, _ = scene()
    if current is None:
        current = rgb, depth, xyz
    return reacquire_surface(rgb, xyz, mask, *current, min_depth=0.2, max_depth=3.0)


def test_current_points_come_from_new_depth_not_reference_or_query_time():
    rgb, depth, xyz, mask, _ = scene()
    xyz[mask, 2] += 0.01
    result = associate((rgb, depth, xyz))
    assert np.all(result.points[:, 2] == 0.61)
    # Normal discontinuities can omit boundary pixels; no background may enter.
    assert not (result.mask & ~mask).any()
    assert result.mask.sum() >= 0.6 * mask.sum()
    assert result.reference_support_fraction == 1.0


@pytest.mark.parametrize("change", ["removed", "appearance", "far", "missing_depth", "mixed"])
def test_absent_changed_moved_or_mixed_surface_rejects(change):
    rgb, depth, xyz, mask, _ = scene()
    if change == "removed":
        xyz[mask, 2] = 0.5
        depth[mask] = 1.0
        rgb[mask] = [90, 60, 20]
    elif change == "appearance":
        rgb[mask] = [20, 200, 30]
    elif change == "far":
        xyz[mask, 0] += 0.1
    elif change == "missing_depth":
        depth[mask] = np.nan
    else:
        # Merge a large coplanar, same-appearance patch with the target. Whole
        # candidate support must reject it; do not trim to the old reference.
        rgb[10:45, 10:45] = [200, 30, 30]
        depth[10:45, 10:45] = 0.8
        xyz[10:45, 10:45, 2] = 0.6
    expected = "surface_not_reacquired|too many surface candidates" if change == "mixed" else "surface_not_reacquired"
    with pytest.raises(ValueError, match=expected):
        associate((rgb, depth, xyz))


def test_two_plausible_surfaces_reject_instead_of_nearest_wins():
    rgb, depth, xyz, mask, _ = scene()
    # Distinct image components at the same world bounds are deliberately
    # ambiguous (e.g. bad calibration or duplicated projected candidates).
    rgb[35:47, 20:32] = rgb[20:32, 20:32]
    depth[35:47, 20:32] = depth[20:32, 20:32]
    xyz[35:47, 20:32] = xyz[20:32, 20:32]
    with pytest.raises(ValueError, match="surface_association_ambiguous"):
        associate((rgb, depth, xyz))


@dataclass
class Observation:
    mask: np.ndarray
    points_3d: np.ndarray
    label: str
    timestep: int = 0


def timing(stamp):
    return {
        "head_cam_left": {"available": True, "timestamp_ns": stamp, "source": "camera", "clock_domain": "camera_clock"}
    }


def test_delayed_semantics_refresh_read_only_proposals_from_actual_current_capture():
    provider = RgbdTaskProvider()
    rgb, depth, xyz, obj, table = scene()
    observations = [Observation(obj, xyz[obj], "object"), Observation(table, xyz[table], "table")]
    reference = provider.begin_frame(0.0, timing(1))
    current = provider.begin_frame(19.0, timing(2))
    rows = provider.update_reacquired(
        observations,
        reference_ticket=reference,
        current_ticket=current,
        reference_rgb=rgb,
        reference_world_xyz=xyz,
        rgb=rgb,
        depth=depth,
        world_xyz=xyz,
        min_depth=0.2,
        max_depth=3.0,
        now=19.1,
    )
    assert rows and rows[0].object_pose.received_monotonic == 19.0
    assert rows[0].object_pose.source == "rgbd_pointcloud"
    rows[0].object_pose.require("observed", now=20.0)
    with pytest.raises(ObservationUnavailable, match="object_observation_stale"):
        rows[0].object_pose.require("observed", now=22.0)


@pytest.mark.parametrize("change", ["motion", "replayed", "stale_current", "expired_reference", "untimed"])
def test_reacquisition_cannot_renew_cached_frames_or_cross_motion(change):
    provider = RgbdTaskProvider()
    rgb, depth, xyz, obj, _ = scene()
    reference = provider.begin_frame(0.0, None if change == "untimed" else timing(1))
    if change == "motion":
        provider.invalidate()
    current = provider.begin_frame(19.0, timing(1 if change == "replayed" else 2))
    now = 22.0 if change == "stale_current" else (40.0 if change == "expired_reference" else 19.1)
    with pytest.raises(ObservationUnavailable, match="surface_reacquisition_stale"):
        provider.update_reacquired(
            [Observation(obj, xyz[obj], "object")],
            reference_ticket=reference,
            current_ticket=current,
            reference_rgb=rgb,
            reference_world_xyz=xyz,
            rgb=rgb,
            depth=depth,
            world_xyz=xyz,
            min_depth=0.2,
            max_depth=3.0,
            now=now,
        )
    assert provider.observe_tasks() == ()


def test_reacquired_snapshot_reaches_registry_without_granting_execution():
    import json
    import time
    from unittest.mock import Mock

    from emet.agent.tools import get_tools

    provider = RgbdTaskProvider()
    rgb, depth, xyz, obj, table = scene()
    now = time.monotonic()
    reference = provider.begin_frame(now - 19.0, timing(1))
    current = provider.begin_frame(now, timing(2))
    provider.update_reacquired(
        [Observation(obj, xyz[obj], "object"), Observation(table, xyz[table], "table")],
        reference_ticket=reference,
        current_ticket=current,
        reference_rgb=rgb,
        reference_world_xyz=xyz,
        rgb=rgb,
        depth=depth,
        world_xyz=xyz,
        min_depth=0.2,
        max_depth=3.0,
    )
    robot = Mock()
    tools = {
        t.name: t
        for t in get_tools(
            {"robot": robot, "manip_mode": "physical", "tamp_inputs": "observed", "observed_tamp_provider": provider}
        )
    }
    result = json.loads(tools["scene_tasks"].func())
    assert result["status"] == "ok" and result["data"]["tasks"]
    assert result["data"]["discovery_only"] and not result["data"]["execution_available"]
    task = result["data"]["tasks"][0]
    assert task["sources"]["object"] == "rgbd_pointcloud"
    assert task["observation_age_s"]["object"] < 2.0
    blocked = json.loads(tools["plan_pick_place"].func(task_ref=task["task_ref"]))
    assert blocked["status"] == "error"
    assert robot.mock_calls == []


def test_motion_during_association_cannot_publish_geometry(monkeypatch):
    import emet.memory.surface_reacquisition as association

    provider = RgbdTaskProvider()
    rgb, depth, xyz, obj, table = scene()
    reference = provider.begin_frame(0.0, timing(1))
    current = provider.begin_frame(19.0, timing(2))
    original = association.reacquire_surface

    def invalidate_during_computation(*args, **kwargs):
        surface = original(*args, **kwargs)
        provider.invalidate()
        return surface

    monkeypatch.setattr(association, "reacquire_surface", invalidate_during_computation)
    assert (
        provider.update_reacquired(
            [Observation(obj, xyz[obj], "object"), Observation(table, xyz[table], "table")],
            reference_ticket=reference,
            current_ticket=current,
            reference_rgb=rgb,
            reference_world_xyz=xyz,
            rgb=rgb,
            depth=depth,
            world_xyz=xyz,
            min_depth=0.2,
            max_depth=3.0,
            now=19.1,
        )
        == ()
    )


def test_failed_support_association_does_not_publish_partial_object_snapshot():
    provider = RgbdTaskProvider()
    rgb, depth, xyz, obj, table = scene()
    observations = [Observation(obj, xyz[obj], "object"), Observation(table, xyz[table], "table")]
    reference = provider.begin_frame(0.0, timing(1))
    provider.update(observations, frame_ticket=reference)
    assert provider.observe_tasks()
    current = provider.begin_frame(19.0, timing(2))
    depth[table] = np.nan
    with pytest.raises(ValueError, match="surface_not_reacquired"):
        provider.update_reacquired(
            observations,
            reference_ticket=reference,
            current_ticket=current,
            reference_rgb=rgb,
            reference_world_xyz=xyz,
            rgb=rgb,
            depth=depth,
            world_xyz=xyz,
            min_depth=0.2,
            max_depth=3.0,
            now=19.1,
        )
    assert provider.observe_tasks() == ()
