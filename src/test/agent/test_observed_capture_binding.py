# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Real discovery tool over new captures; no model or simulator metadata."""

import json
import time
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.agent.tools import get_tools
from emet.controller.task.tamp.observed_capture import bind_observed_capture_source, capture_arrays
from emet.controller.task.tamp.observed_discovery import RgbdTaskProvider


@dataclass
class Observation:
    mask: np.ndarray
    points_3d: np.ndarray
    label: str
    timestep: int = 1


def capture(stamp, received):
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    depth = np.full((80, 80), np.nan)
    table = np.zeros(depth.shape, dtype=bool)
    table[10:70, 10:70] = True
    obj = np.zeros_like(table)
    obj[30:43, 30:43] = True
    depth[table] = 1.0
    depth[obj] = 0.9
    rgb[table] = [90, 60, 20]
    rgb[obj] = [200, 30, 30]
    # Optical axis points down onto a horizontal table in world coordinates.
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[:3, 3] = [0, 0, 1.5]
    obs = SimpleNamespace(
        rgb=rgb,
        depth=depth,
        camera_K=np.array([[100.0, 0, 40], [0, 100.0, 40], [0, 0, 1]]),
        camera_pose=pose,
        received_monotonic=received,
        image_timing={
            "head_cam_left": {
                "timestamp_ns": stamp,
                "available": True,
                "source": "camera",
                "clock_domain": "camera_clock",
            }
        },
    )
    _, _, world = capture_arrays(obs)
    return obs, world, obj, table & ~obj


def setup():
    now = time.monotonic()
    ref, world, obj, table = capture(1, now - 10)
    p = RgbdTaskProvider()
    ticket = p.begin_frame(ref.received_monotonic, ref.image_timing)
    p.update(
        [Observation(obj, world[obj], "object"), Observation(table, world[table], "table")],
        frame_ticket=ticket,
        reference_rgb=ref.rgb,
        reference_world_xyz=world,
    )
    robot = Mock()
    robot.get_observation.return_value = capture(2, now)[0]
    assert bind_observed_capture_source(p, robot, "observed")
    tools = {
        t.name: t
        for t in get_tools(
            {"robot": robot, "manip_mode": "physical", "tamp_inputs": "observed", "observed_tamp_provider": p}
        )
    }
    return p, robot, tools


def test_live_tool_reacquires_new_depth_without_semantic_calls_or_truth():
    p, robot, tools = setup()
    result = json.loads(tools["scene_tasks"].func())
    assert result["status"] == "ok" and result["data"]["tasks"]
    assert result["data"]["discovery_only"]
    assert result["data"]["tasks"][0]["observation_age_s"]["object"] < 2
    assert [c[0] for c in robot.mock_calls] == ["get_observation"]
    # Another query reads the existing stream, not a VLM or expensive mapper.
    result = json.loads(tools["scene_tasks"].func())
    assert result["status"] == "ok"
    assert robot.get_observation.call_count == 2


@pytest.mark.parametrize("failure", ["stale", "missing", "calibration", "replayed", "motion"])
def test_binding_never_falls_back_to_old_snapshot(failure):
    p, robot, tools = setup()
    if failure == "stale":
        robot.get_observation.return_value.received_monotonic = time.monotonic() - 3
    elif failure == "missing":
        robot.get_observation.return_value.depth = None
    elif failure == "calibration":
        robot.get_observation.return_value.camera_pose[:3, :3] *= 2
    elif failure == "replayed":
        robot.get_observation.return_value.image_timing["head_cam_left"]["timestamp_ns"] = 1
    else:
        p.invalidate()
    result = json.loads(tools["scene_tasks"].func())
    if failure == "motion":
        assert result["data"]["tasks"] == []
    else:
        assert result["code"] == "observed_scene_unavailable"
    assert all(c[0] == "get_observation" for c in robot.mock_calls)


def test_privileged_and_custom_providers_are_not_rebound():
    robot = Mock()
    assert not bind_observed_capture_source(RgbdTaskProvider(), robot, "privileged")
    assert not bind_observed_capture_source(SimpleNamespace(input_mode="observed"), robot, "observed")
    assert not bind_observed_capture_source(None, robot, "observed")
    assert robot.mock_calls == []


def test_delayed_semantics_can_seed_without_publishing_outdated_points():
    p = RgbdTaskProvider()
    now = time.monotonic()
    ref, world, obj, table = capture(1, now - 10)
    ticket = p.begin_frame(ref.received_monotonic, ref.image_timing)
    p.begin_frame(now, capture(2, now)[0].image_timing)
    p.update(
        [Observation(obj, world[obj], "object"), Observation(table, world[table], "table")],
        frame_ticket=ticket,
        reference_rgb=ref.rgb,
        reference_world_xyz=world,
    )
    assert p.observe_tasks() == ()
    robot = Mock(get_observation=Mock(return_value=capture(3, now)[0]))
    bind_observed_capture_source(p, robot, "observed")
    assert p.observe_tasks()


def test_same_verified_fresh_capture_keeps_original_evidence_age():
    now = time.monotonic()
    ref, world, obj, table = capture(1, now - 0.5)
    p = RgbdTaskProvider()
    ticket = p.begin_frame(ref.received_monotonic, ref.image_timing)
    p.update(
        [Observation(obj, world[obj], "object"), Observation(table, world[table], "table")],
        frame_ticket=ticket,
        reference_rgb=ref.rgb,
        reference_world_xyz=world,
    )
    # Client republishes the same capture with a later local receive timestamp.
    ref.received_monotonic = now
    robot = Mock(get_observation=Mock(return_value=ref))
    bind_observed_capture_source(p, robot, "observed")
    rows = p.observe_tasks()
    assert rows and rows[0].object_pose.received_monotonic == now - 0.5


def test_actual_processor_seeds_stream_binding_and_negative_segmentation_clears_it(monkeypatch):
    from emet.mapping.scene_graph.processor import SceneGraphProcessor

    now = time.monotonic()
    obs, world, obj, table = capture(1, now - 10)
    processor = SceneGraphProcessor(config={}, device="cpu")
    monkeypatch.setattr(
        processor,
        "_run_segmentation",
        lambda *args: [{"mask": obj, "label": "object"}, {"mask": table, "label": "table"}],
    )
    monkeypatch.setattr(processor, "_encode_siglip", lambda *args: None)
    monkeypatch.setattr(processor, "_encode_dinov3", lambda *args: None)
    processor.process_frame(
        obs.rgb,
        obs.depth,
        obs.camera_K,
        obs.camera_pose,
        received_monotonic=obs.received_monotonic,
        image_timing=obs.image_timing,
    )
    robot = Mock(get_observation=Mock(return_value=capture(2, now)[0]))
    p = processor.observed_tamp_provider
    bind_observed_capture_source(p, robot, "observed")
    assert p.observe_tasks()
    monkeypatch.setattr(processor, "_run_segmentation", lambda *args: [])
    next_obs = capture(3, now)[0]
    processor.process_frame(
        next_obs.rgb,
        next_obs.depth,
        next_obs.camera_K,
        next_obs.camera_pose,
        received_monotonic=next_obs.received_monotonic,
        image_timing=next_obs.image_timing,
    )
    robot.get_observation.return_value = capture(4, now)[0]
    assert p.observe_tasks() == ()


@pytest.mark.parametrize("missing_depth", [False, True])
def test_real_dispatch_stops_after_discovery_or_failed_live_refresh(missing_depth):
    from emet.agent.loop import _dispatch_tool_calls

    provider, robot, tools = setup()
    if missing_depth:
        robot.get_observation.return_value.depth = None
    motion = Mock(side_effect=AssertionError("must not dispatch motion"))
    tools["unsafe_motion"] = SimpleNamespace(func=motion, returns_info=True, to_executor=lambda args: [])
    executor = Mock()
    _dispatch_tool_calls(
        [{"name": "scene_tasks", "arguments": {}}, {"name": "unsafe_motion", "arguments": {}}],
        tools,
        executor,
        tamp_context={"observed_tamp_provider": provider},
    )
    motion.assert_not_called()
    executor.assert_not_called()
    assert [call[0] for call in robot.mock_calls] == ["get_observation"]
