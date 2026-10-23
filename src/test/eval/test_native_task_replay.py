# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Real captured failure: changing goal geometry is not an IK solver fix."""
import json
from pathlib import Path

import numpy as np
import pytest

from emet.eval.native_task_replay import RecordedNativeTablePlanner

pytestmark = pytest.mark.sim


@pytest.fixture(scope='module')
def captured():
    return json.loads((Path(__file__).parents[1]/'fixtures/tamp/recorded_post_buffer_20261022.json').read_text())


def test_exact_failed_state_separates_region_feasibility_without_execution(captured):
    planner = RecordedNativeTablePlanner(captured['snapshot'])
    before = planner.data.qpos.copy(), planner.data.qvel.copy(), planner.data.ctrl.copy()
    narrow = planner.plan_transfer(captured['object_body'], captured['support_body'], captured['narrow_region'])
    assert narrow['code'] == 'no_plan_within_budget'
    assert narrow['data']['planning_rejections']['surface:no_collision_free_placement'] > 0
    wider = planner.plan_transfer(captured['object_body'], captured['support_body'], captured['wider_region'])
    assert wider['status'] == 'ok'
    assert wider['data']['mode'] == 'offline_planning'
    assert wider['data']['execution_available'] is False
    assert wider['data']['physical_acceptance'] is None
    assert 0 < planner.planning_ik_calls <= planner.max_planning_ik_calls
    for actual, original in zip((planner.data.qpos, planner.data.qvel, planner.data.ctrl), before, strict=True):
        np.testing.assert_array_equal(actual, original)
    with pytest.raises(RuntimeError, match='cannot_execute'):
        planner.execute(wider['data']['plan_ref'])
    with pytest.raises(RuntimeError, match='cannot_actuate'):
        planner.robot.arm_to([])


def test_recorded_state_rejects_wrong_model_or_nonfinite_values(captured):
    wrong = {**captured['snapshot'], 'model_signature': '0'*64}
    with pytest.raises(ValueError, match='model_mismatch'):
        RecordedNativeTablePlanner(wrong)
    invalid = {**captured['snapshot'], 'qpos': [float('nan')]*len(captured['snapshot']['qpos'])}
    with pytest.raises(ValueError, match='invalid_recorded_native_state'):
        RecordedNativeTablePlanner(invalid)
