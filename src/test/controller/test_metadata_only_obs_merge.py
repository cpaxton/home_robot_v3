# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for metadata-only obs merge locking in GenericZmqClient."""

from __future__ import annotations

import threading

import numpy as np

from emet.controller.generic_zmq_client import GenericZmqClient
from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY


def _metadata_only_session() -> dict:
    return {
        EMET_ZMQ_SESSION_KEY: {
            "capabilities": {"zmq_obs_metadata_only": True},
        }
    }


def _jpeg_rgb() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_obs_ready_for_mapping_merges_under_lock(monkeypatch) -> None:
    client = GenericZmqClient.__new__(GenericZmqClient)
    client._obs_lock = threading.Lock()
    client._obs = {**_metadata_only_session(), "step": 1}
    client._servo = {"head_cam_left/color_image": _jpeg_rgb()}

    merge_calls: list[tuple[int, bool]] = []

    def _track_merge(full_obs, servo):
        merge_calls.append((full_obs["step"], threading.current_thread() is threading.main_thread()))
        from emet.core.zmq_obs_codec import merge_servo_images_into_full_obs as real_merge

        return real_merge(full_obs, servo)

    monkeypatch.setattr(
        "emet.controller.generic_zmq_client.merge_servo_images_into_full_obs",
        _track_merge,
    )

    assert client._obs_ready_for_mapping()
    assert merge_calls
    assert client._obs.get("rgb") is not None


def test_get_observation_snapshot_matches_live_obs_after_merge() -> None:
    client = GenericZmqClient.__new__(GenericZmqClient)
    client._obs_lock = threading.Lock()
    client._allow_missing_depth = True
    client._seq_id = 7
    client._obs = {**_metadata_only_session(), "step": 3, "gps": [0.0, 0.0], "compass": [0.0]}
    client._servo = {"head_cam_left/color_image": _jpeg_rgb()}

    obs = client.get_observation()
    assert obs is not None
    assert obs.rgb.shape == (4, 4, 3)
    assert obs.seq_id == 7
    assert client._obs.get("rgb") is not None


def test_generic_sequence_advances_only_when_decoded_observation_is_installed(monkeypatch):
    from unittest.mock import Mock

    client = GenericZmqClient.__new__(GenericZmqClient)
    client._obs_lock = threading.Lock()
    client._seq_id = 7
    client._last_step = 2
    client._emet_session_cache = None
    client._emet_session_cache_step = -1
    client._allow_missing_depth = True
    client._servo = None
    client._wait_if_streams_paused = lambda: True
    output = {"rgb": _jpeg_rgb(), "depth": None}

    def receive(**kwargs):
        client._finish = True  # Process exactly one frame.
        return output

    client.recv_socket = Mock(recv_pyobj=receive)

    def decode(message):
        assert client._seq_id == 7, "Receiving an undecoded frame is not fresh observation availability"
        return message is output

    monkeypatch.setattr("emet.controller.generic_zmq_client.decode_zmq_obs_images_inplace", decode)
    client._finish = False
    client._recv_loop()
    assert client._obs is output
    assert client._seq_id == 8
