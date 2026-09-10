# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from audit_candidate_context import selected_id, verify

from emet.memory.surface_candidates import surface_candidates


def test_blind_identification_does_not_receive_target_and_rejects_mixed_support():
    rgb = np.full((30, 30, 3), 120, dtype=np.uint8)
    regions = surface_candidates(np.ones((30, 30)), [0, 0, 1000, 1000], min_depth=0.25, max_depth=4)
    client = Mock(
        side_effect=[
            json.dumps({"candidates": [{"id": 0, "identity": "mug", "unambiguous": True, "mixed": True}]}),
            '{"selected_id":0,"target_unambiguous":true}',
        ]
    )
    chosen, requests, panels = verify(client, rgb, regions, "paper towel", "blind_context")
    assert chosen is None
    assert "paper towel" not in requests[0]["prompt"]
    assert "paper towel" in requests[1]["prompt"]
    assert requests[1]["image_count"] == 0
    assert len(panels) == 2


def test_selection_is_strict_and_ambiguous_identity_cannot_be_overridden():
    regions = [{"id": 0}]
    assert selected_id({"selected_id": True, "target_unambiguous": True}, regions) is None
    assert selected_id({"selected_id": 2, "target_unambiguous": True}, regions) is None
    assert selected_id({"selected_id": 0, "target_unambiguous": True}, regions, []) is None
    assert selected_id({"selected_id": 0, "target_unambiguous": True}, regions) == 0


def test_missing_candidates_do_not_call_model():
    client = Mock()
    assert verify(client, np.zeros((20, 20, 3), dtype=np.uint8), [], "cup", "context")[0] is None
    client.assert_not_called()
