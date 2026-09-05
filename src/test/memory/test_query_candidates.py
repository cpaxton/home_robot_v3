# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import json

import pytest

from emet.memory.query_candidates import QueryCandidates


def test_queries_reuse_evidence_but_keep_distinct_locations():
    store = QueryCandidates()
    first = store.propose(" Red Mug ", 1, 0, [1, 2, 3])
    assert store.propose("red mug", 1, 0, [1, 2, 3]) is first
    second = store.propose("red mug", 1, 0, [2, 2, 3])
    assert second.handle != first.handle
    with pytest.raises(ValueError, match="fresh"):
        first.require_grounding(0)


def test_moved_instance_invalidates_all_query_references():
    store = QueryCandidates()
    first = store.propose("mug", 1, 0, [1, 2, 3])
    second = store.propose("cup", 1, 0, [1, 2, 3])
    for record in (first, second):
        store.ground(record.handle, instance_id=10, observation_revision=1)
        assert record.require_grounding(1) == 10
        with pytest.raises(ValueError, match="fresh"):
            record.require_grounding(2)
    store.invalidate_instance(10, "object picked")
    for record in (first, second):
        with pytest.raises(ValueError, match="fresh"):
            record.require_grounding(1)


def test_checkpoint_retains_identity_but_requires_reacquisition():
    store = QueryCandidates()
    record = store.propose("mug", 1, 0, [1, 2, 3])
    store.ground(record.handle, instance_id=10, observation_revision=1)
    loaded = QueryCandidates.from_dict(json.loads(json.dumps(store.to_dict())))
    restored = loaded.records[record.handle]
    assert restored.instance_id == 10
    with pytest.raises(ValueError, match="fresh"):
        restored.require_grounding(1)


def test_budget_never_silently_evicts_an_active_reference():
    store = QueryCandidates(max_candidates=1)
    record = store.propose("mug", 1, 0, [1, 2, 3])
    with pytest.raises(ValueError, match="budget"):
        store.propose("bowl", 2, 0, [2, 2, 3])
    store.release(record.handle)
    assert store.propose("bowl", 2, 0, [2, 2, 3]).handle != record.handle
