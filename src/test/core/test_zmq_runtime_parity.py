# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Deployed and workstation wire helpers must have identical behavior."""

from pathlib import Path

import pytest

from emet.core.zmq_server_env import resolve_zmq_image_scaling, zmq_h264_port, zmq_send_period_s


@pytest.mark.parametrize("name", ["core/zmq_obs_codec.py", "core/zmq_server_env.py", "utils/compression.py"])
def test_deployed_runtime_parity(name):
    root = Path(__file__).resolve().parents[2]
    assert (root / "emet" / name).read_bytes() == (root / "emet_core/emet" / name).read_bytes()


@pytest.mark.parametrize("raw", ["abc", "nan", "inf", "-inf"])
def test_invalid_float_falls_back(monkeypatch, raw):
    monkeypatch.setenv("EMET_ZMQ_FULL_HZ", raw)
    with pytest.warns(RuntimeWarning):
        assert zmq_send_period_s("EMET_ZMQ_FULL_HZ") == 0
    monkeypatch.setenv("EMET_ZMQ_IMAGE_SCALING", raw)
    with pytest.warns(RuntimeWarning):
        assert resolve_zmq_image_scaling() == 0.5


@pytest.mark.parametrize("raw", ["abc", "nan", "0", "65536", "2.5"])
def test_invalid_port_falls_back(monkeypatch, raw):
    monkeypatch.setenv("EMET_ZMQ_H264_PORT", raw)
    with pytest.warns(RuntimeWarning):
        assert zmq_h264_port() == 4405
