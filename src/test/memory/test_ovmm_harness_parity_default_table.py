# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""S0 default-table harness parity: oneshot find matches pytest-style localize."""

from __future__ import annotations

import os

import pytest

RUN_OVMM = os.environ.get("RUN_OVMM_FIND_TESTS", "").strip() in ("1", "true", "yes")


@pytest.mark.skipif(not RUN_OVMM, reason="Set RUN_OVMM_FIND_TESTS=1 to run sim integration")
@pytest.mark.timeout(900)
def test_ovmm_s0_distinct_recep_oneshot_parity():
    """Oneshot harness on distinct-recep S0 should score FindObj + FindRec within 0.30 m."""
    from emet.eval.ovmm_find_phase import FindPhaseRunConfig, load_find_phase_episodes, run_episode_find_phase

    episodes = load_find_phase_episodes("configs/ovmm/find_phase_episodes.yaml")
    ep = next(e for e in episodes if e.id == "default_table_s0_distinct_recep")
    run_cfg = FindPhaseRunConfig(
        backend="dynagraph",
        cpu_only=True,
        perfect_depth=True,
        agentic_find=False,
        port_offset=int(os.getpid() % 400 + 220),
    )
    metrics = run_episode_find_phase(ep, run_cfg)
    assert metrics.get("s0_parity") is True, metrics
    assert metrics.get("s0_phrase_only") is True, metrics
    assert metrics.get("s0_oneshot_pytest") is True, metrics
    assert metrics.get("obj_query_used") == "red cylinder", metrics
    assert metrics.get("find_object_success") is True, metrics
    assert metrics.get("find_recep_success") is True, metrics
    obj_err = metrics.get("localization_err_obj_m")
    recep_err = metrics.get("localization_err_recep_m")
    assert obj_err is not None and float(obj_err) <= 0.30, metrics
    assert recep_err is not None and float(recep_err) <= 0.30, metrics
