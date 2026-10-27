# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Live audit must not turn reference features or carved cells into qualification."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from emet.agent.tools import get_tools
from emet.eval.observed_readiness import assess_live_readiness, capture_metadata


@pytest.mark.parametrize('missing_depth', [False, True])
def test_sensor_diagnostic_keeps_missing_execution_gates_explicit(missing_depth):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), bool)
    mask[8:56, 8:56] = True
    received = time.monotonic()
    timing = {'head_cam_left': {'available': True, 'source': 'sensor', 'clock_domain': 'device', 'timestamp_ns': 10}}
    observation = SimpleNamespace(rgb=rgb, depth=np.full((64, 64), 0. if missing_depth else 1.),
        camera_K=np.array([[30, 0, 32], [0, 30, 32], [0, 0, 1]]), camera_pose=np.eye(4),
        received_monotonic=received, image_timing=timing)
    tools = {t.name: t for t in get_tools({'robot': SimpleNamespace(), 'manip_mode': 'physical', 'tamp_inputs': 'observed'})}
    result = assess_live_readiness(observation=observation, reference_rgb=rgb, reference_mask=mask,
                                  tools=tools, refreshes=[True, True])
    assert result['qualification'] == 'not_qualified' and not result['task_success_assessed']
    assert result['gates']['reference_features']['state'] == 'passed'
    assert result['gates']['capture_encoder_alignment']['state'] == 'not_configured'
    assert result['gates']['complete_payload_geometry']['state'] == 'not_configured'
    assert result['gates']['default_execution_boundary']['state'] == 'passed'
    assert result['gates']['live_rgbd']['state'] == ('failed' if missing_depth else 'passed')
    assert result['gates']['live_rgbd']['capture']['received_monotonic'] == received
    assert capture_metadata(observation)['image_timing'] is timing
    if missing_depth:
        assert result['gates']['depth_workspace']['cells']['free'] == 0
