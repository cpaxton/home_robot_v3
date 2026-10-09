# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace

import pytest

from emet.memory.graph_eqa.agentic.run import redirect_rejected_candidates
from emet.memory.query_candidates import QueryCandidates


@pytest.mark.parametrize("tool", ["investigate", "navigate_to_obs"])
def test_rejected_handle_redirects_without_reviving_it(tool):
    store = QueryCandidates()
    candidate = store.propose("lamp", 1, 0, [1, 2, 3])
    agent = SimpleNamespace(query_driven_memory=True, query_candidates=store)
    calls = [(tool, {"obs_id": candidate.handle})]
    assert redirect_rejected_candidates(agent, calls) == calls
    store.reject(candidate.handle, observation_revision=2, reason="absent")
    for _ in range(8):  # Recorded pilot repeated the rejected selection eight times.
        assert redirect_rejected_candidates(agent, calls) == [("explore_frontier", {})]
    assert candidate.rejected_revision == 2
    fresh = store.propose("lamp", 3, 0, [1, 2, 3])
    fresh_call = [(tool, {"obs_id": fresh.handle})]
    assert redirect_rejected_candidates(agent, fresh_call) == fresh_call


def test_non_query_and_non_navigation_actions_are_unchanged():
    calls = [("investigate", {"obs_id": -3000000})]
    assert redirect_rejected_candidates(SimpleNamespace(query_driven_memory=False), calls) == calls
    agent = SimpleNamespace(query_driven_memory=True, query_candidates=QueryCandidates())
    for calls in ([("submit_answer", {})], [("investigate", {"obs_id": "bad"})]):
        assert redirect_rejected_candidates(agent, calls) == calls
