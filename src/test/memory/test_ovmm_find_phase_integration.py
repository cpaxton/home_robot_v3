# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Integration smoke: OVMM find-phase on S0 default table (gated).

from __future__ import annotations

import os

import pytest

RUN_OVMM = os.environ.get("RUN_OVMM_FIND_TESTS", "").strip() in ("1", "true", "yes")


@pytest.mark.skipif(not RUN_OVMM, reason="Set RUN_OVMM_FIND_TESTS=1 to run sim integration")
@pytest.mark.timeout(300)
def test_ovmm_find_phase_s0_ground_truth():
    """Ground-truth backend should achieve perfect FindObj/FindRec on default table."""
    from emet.eval.ovmm_find_phase import FindPhaseRunConfig, load_find_phase_episodes, run_episode_find_phase

    episodes = load_find_phase_episodes("configs/ovmm/find_phase_episodes.yaml")
    ep = next(e for e in episodes if e.id == "default_table_s0")
    run_cfg = FindPhaseRunConfig(
        backend="ground_truth",
        cpu_only=True,
        not_rotate=True,
        port_offset=int(os.getpid() % 400 + 200),
    )
    metrics = run_episode_find_phase(ep, run_cfg)
    assert metrics.get("find_object_success") is True, metrics
    assert metrics.get("find_recep_success") is True, metrics
    assert metrics.get("find_partial_success") == 1.0


@pytest.mark.skipif(not RUN_OVMM, reason="Set RUN_OVMM_FIND_TESTS=1 to run sim integration")
@pytest.mark.timeout(720)
def test_ovmm_find_phase_s0_dynagraph_perception():
    """Dynagraph with rotate + perfect sim depth should find table objects on S0."""
    from emet.eval.ovmm_find_phase import FindPhaseRunConfig, load_find_phase_episodes, run_episode_find_phase

    episodes = load_find_phase_episodes("configs/ovmm/find_phase_episodes.yaml")
    ep = next(e for e in episodes if e.id == "default_table_s0")
    run_cfg = FindPhaseRunConfig(
        backend="dynagraph",
        cpu_only=True,
        perfect_depth=True,
        agentic_find=False,
        port_offset=int(os.getpid() % 400 + 210),
    )
    metrics = run_episode_find_phase(ep, run_cfg)
    assert metrics.get("find_partial_success", 0) >= 1.0, metrics
    assert metrics.get("find_object_success") is True, metrics
    assert metrics.get("find_recep_success") is True, metrics
    obj_err = metrics.get("localization_err_obj_m")
    recep_err = metrics.get("localization_err_recep_m")
    if obj_err is not None:
        assert float(obj_err) <= 0.30, metrics
    if recep_err is not None:
        assert float(recep_err) <= 0.30, metrics
