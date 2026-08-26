# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import patch

import numpy as np
import pytest
import torch

from emet.perception.encoders.remote_dinov3_encoder import (
    RemoteDinov3Encoder,
    build_dinov3_encoder,
    resolve_dinov3_endpoint,
)


class _FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            body = json.dumps({"ready": True, "feature_dim": 4}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/embed":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        body = json.dumps({"embedding": [1.0, 0.0, 0.0, 0.0], "dim": 4}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def dinov3_server():
    httpd = HTTPServer(("127.0.0.1", 0), _FakeHandler)
    port = httpd.server_address[1]
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def test_resolve_dinov3_endpoint_host_only():
    with patch.dict("os.environ", {"EMET_DINOV3_HOST": "caliban:8002"}, clear=True):
        assert resolve_dinov3_endpoint() == "http://caliban:8002"


def test_remote_dinov3_encoder_roundtrip(dinov3_server):
    enc = RemoteDinov3Encoder(endpoint=dinov3_server, version="vits16", timeout_s=5.0)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = enc.encode_image(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 4)
    assert abs(float(out.norm()) - 1.0) < 1e-4


def test_remote_dinov3_encoder_circuit_breaker(dinov3_server):
    enc = RemoteDinov3Encoder(endpoint=dinov3_server, version="vits16", timeout_s=5.0)
    enc._last_failure_at = time.monotonic()
    with pytest.raises(RuntimeError, match="circuit open"):
        enc.encode_image(np.zeros((8, 8, 3), dtype=np.uint8))


def test_build_dinov3_encoder_uses_remote_when_env_set(dinov3_server):
    with patch.dict("os.environ", {"EMET_DINOV3_ENDPOINT": dinov3_server}, clear=True):
        enc = build_dinov3_encoder(version="vits16", timeout_s=5.0)
    assert enc.__class__.__name__ == "RemoteDinov3Encoder"
