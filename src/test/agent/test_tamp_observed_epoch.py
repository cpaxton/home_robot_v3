# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Observed frame replay and navigation invalidation through real CHAT dispatch."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.agent.loop import _dispatch_tool_calls
from emet.controller.task.tamp.observed_discovery import RgbdTaskProvider, discover_observed_tasks
from emet.controller.task.tamp.observed_frames import ObservedFrameGate
from emet.controller.task.tamp.task_contracts import GoalSpec, SceneBinding, TaskState
from emet.controller.task.tamp.task_session import TaskSession


def timing(stamp):
    return {'head_cam_left': {'timestamp_ns': stamp, 'source': 'renderer_scene_capture',
                             'clock_domain': 'mujoco_render_monotonic', 'available': True}}


def points():
    x = np.linspace(-.04, .04, 8)
    cube = np.stack(np.meshgrid(x, x, x), axis=-1).reshape(-1, 3) + [0, 0, .6]
    x = np.linspace(-.3, .3, 24)
    xy = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)
    return [SimpleNamespace(label='cup', points_3d=cube, timestep=1),
            SimpleNamespace(label='counter', points_3d=np.c_[xy, np.full(len(xy), .55)], timestep=1)]


def test_cached_capture_cannot_renew_local_freshness_or_rewind_stream():
    provider = RgbdTaskProvider()
    provider.update(points(), received_monotonic=100., image_timing=timing(5000000000000))
    provider.update(points(), received_monotonic=103., image_timing=timing(5000000000000))
    assert discover_observed_tasks(provider, now=103.)['code'] == 'object_observation_stale'
    provider.update(points(), received_monotonic=103., image_timing=timing(4999999999999))
    assert provider.observe_tasks() == ()
    provider.update(points(), received_monotonic=104., image_timing=timing(5000000000001))
    assert discover_observed_tasks(provider, now=104.)['status'] == 'ok'


def test_inflight_inference_and_legacy_pixels_cannot_cross_motion_barrier():
    provider = RgbdTaskProvider()
    ticket = provider.begin_frame(100., timing(10))
    provider.invalidate()
    provider.update(points(), frame_ticket=ticket)
    assert provider.observe_tasks() == ()
    provider.update(points(), received_monotonic=101., image_timing=timing(10))
    assert provider.observe_tasks() == ()
    provider.update(points(), received_monotonic=102.)
    assert provider.observe_tasks() == ()
    provider.update(points(), received_monotonic=103., image_timing=timing(11))
    assert len(provider.observe_tasks()) == 1


@pytest.mark.parametrize('bad', [{'head_cam_left': {}}, {'ee_cam': timing(10)['head_cam_left']},
                               {'head_cam_left': {**timing(10)['head_cam_left'], 'timestamp_ns': True}}])
def test_malformed_capture_identity_never_becomes_evidence(bad):
    gate = ObservedFrameGate()
    assert gate.begin(100., bad) is None


def test_clock_domain_change_requires_explicit_provider_reconnect():
    gate = ObservedFrameGate()
    assert gate.begin(100., timing(10)) is not None
    changed = timing(11)
    changed['head_cam_left']['clock_domain'] = 'other_host'
    assert gate.begin(101., changed) is None


@pytest.mark.parametrize('executor_mapped', [False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_navigation_dispatch_discards_old_and_during_motion_plans(executor_mapped, fails):
    provider = RgbdTaskProvider()
    provider.update(points(), received_monotonic=100., image_timing=timing(10))
    state = TaskState(frozenset(), SceneBinding('a'*64, 'b'*64, 'boot', 'reset', 0))
    session = TaskSession(SimpleNamespace(snapshot=lambda: state, compile_goal=lambda _: GoalSpec(frozenset())))
    ref = session.plan('already satisfied')['data']['plan_ref']
    primitive = {'old': object()}
    native = SimpleNamespace(lock=threading.RLock(), plans={'old': object()})
    context = {'observed_tamp_provider': provider, 'tamp_task_session': session,
               '_tamp_plans': primitive, '_native_tamp_adapter': native}
    def move(*args):
        assert provider.observe_tasks() == () and not session._plans
        assert primitive == {} and native.plans == {}
        provider.update(points(), received_monotonic=101., image_timing=timing(11))
        assert provider.observe_tasks()
        if fails:
            raise TimeoutError('motion uncertain')
        return True
    tool = SimpleNamespace(func=move, returns_info=True,
                           to_executor=lambda args: [('explore', '')] if executor_mapped else [])
    if fails and executor_mapped:
        with pytest.raises(TimeoutError):
            _dispatch_tool_calls([{'name': 'explore'}], {'explore': tool}, move, tamp_context=context)
    else:
        _dispatch_tool_calls([{'name': 'explore'}], {'explore': tool}, move, tamp_context=context)
    assert provider.observe_tasks() == ()
    assert session.execute(ref)['code'] == 'unknown_plan'
    provider.update(points(), received_monotonic=102., image_timing=timing(11))
    assert provider.observe_tasks() == ()  # Same capture, new publication.
    provider.update(points(), received_monotonic=103., image_timing=timing(12))
    assert provider.observe_tasks()


def test_read_only_tool_does_not_invalidate_evidence():
    provider = RgbdTaskProvider()
    provider.update(points(), received_monotonic=100.)
    tool = SimpleNamespace(func=lambda: 'view', returns_info=True, to_executor=lambda args: [])
    _dispatch_tool_calls([{'name': 'describe_scene'}], {'describe_scene': tool}, Mock(),
                         tamp_context={'observed_tamp_provider': provider})
    assert provider.observe_tasks()


def test_actual_processor_carries_capture_identity_and_preserves_cached_age(monkeypatch):
    from emet.mapping.scene_graph.processor import SceneGraphProcessor
    processor = SceneGraphProcessor(config={}, device='cpu')
    rgb = np.zeros((40, 40, 3), np.uint8)
    mask = np.zeros((40, 40), bool)
    mask[16:24, 16:24] = True
    depth = np.full((40, 40), .5, np.float32)
    depth[mask] = .4
    K = np.array([[50., 0., 20.], [0., 50., 20.], [0., 0., 1.]])
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 1.
    monkeypatch.setattr(processor, '_run_segmentation', lambda *args: [
        {'mask': mask, 'label': 'cup'}, {'mask': ~mask, 'label': 'counter'}])
    monkeypatch.setattr(processor, '_encode_siglip', lambda *args: None)
    monkeypatch.setattr(processor, '_encode_dinov3', lambda *args: None)
    provider = processor.observed_tamp_provider
    processor.process_frame(rgb, depth, K, pose, received_monotonic=100., image_timing=timing(10))
    assert provider.observe_tasks()
    processor.process_frame(rgb, depth, K, pose, received_monotonic=103., image_timing=timing(10))
    assert discover_observed_tasks(provider, now=103.)['code'] == 'object_observation_stale'
    provider.invalidate()
    processor.process_frame(rgb, depth, K, pose, received_monotonic=104., image_timing=timing(10))
    assert provider.observe_tasks() == ()
    processor.process_frame(rgb, depth, K, pose, received_monotonic=105., image_timing=timing(11))
    assert provider.observe_tasks()
