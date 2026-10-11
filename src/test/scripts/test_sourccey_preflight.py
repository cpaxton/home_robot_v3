# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Arrival diagnostics must reject absent protobuf state and incomplete images."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

path = Path(__file__).resolve().parents[3] / "scripts/sourccey_preflight.py"
spec = importlib.util.spec_from_file_location("sourccey_preflight", path)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def observation():
    obs = dict.fromkeys(preflight.STATE_KEYS, 0.0)
    obs.update({name: np.arange(144, dtype=np.uint8).reshape(6, 8, 3) for name in preflight.CAMERAS})
    return obs


def test_complete_observation():
    report = preflight.validate_observation(observation())
    assert len(report["state"]) == 16
    assert report["camera_shapes"]["bottom"] == [6, 8, 3]


@pytest.mark.parametrize(
    "key,value",
    [("bottom", None), ("front_left", np.zeros((6, 8, 3))), ("z.pos", float("nan")), ("left_gripper.pos", None)],
)
def test_incomplete_observation_fails(key, value):
    obs = observation()
    obs[key] = value
    with pytest.raises(ValueError, match=key):
        preflight.validate_observation(obs)


def test_protocol_mismatch_never_decodes():
    message = Mock(protocol_version=0)
    converter = Mock()
    with pytest.raises(ValueError, match="protocol"):
        preflight.decode_packet(b"", lambda: message, converter, 1)
    converter.protobuf_to_observation.assert_not_called()


def test_absent_state_cannot_pass_as_zero():
    message = Mock(protocol_version=1)
    message.HasField.side_effect = lambda field: field != "left_arm_joints"
    converter = Mock()
    with pytest.raises(ValueError, match="omitted left_arm_joints"):
        preflight.decode_packet(b"", lambda: message, converter, 1)
    converter.protobuf_to_observation.assert_not_called()
