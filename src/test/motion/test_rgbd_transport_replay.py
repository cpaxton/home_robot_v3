# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""A bounded recording tail cannot reuse an earlier frame's reference mask."""

import json

import pytest
from eval_rgbd_transport import replay


def test_replay_rejects_a_recording_that_no_longer_contains_the_reference_capture(tmp_path):
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "input_captures": [{"observation_id": "later-1"}, {"observation_id": "later-2"}],
                "captures": [{"observation_id": "original-reference"}],
            }
        )
    )
    with pytest.raises(ValueError, match="reference_capture_not_retained"):
        replay(tmp_path)
