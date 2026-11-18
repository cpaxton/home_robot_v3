# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Recorder redaction must not alter requests delivered to the model API."""

import hashlib
import json
import stat
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emet.eval.chat_gateway import _recording_body, recorded_chat_gateway


@pytest.mark.parametrize("body", [b"not json api_key=secret", b"\xff"])
def test_unparsable_bodies_are_not_persisted(body):
    recorded, status = _recording_body(body)
    assert status == "non_json_body_omitted"
    assert b"secret" not in recorded


def test_nonsecret_request_recording_preserves_exact_bytes():
    body = b'{ "messages": [{"role":"user", "content":"inspect this image"}] }'
    assert _recording_body(body) == (body, "exact")


def test_gateway_redacts_credentials_but_forwards_original_bytes(tmp_path, monkeypatch):
    output = tmp_path / "private-artifacts"
    body = json.dumps(
        {
            "api_key": "body-secret",
            "messages": [{"content": "question", "metadata": {"access_token": "nested-secret"}}],
        }
    ).encode()
    reply = b'{"answer":"ok", "token":"response-secret"}'
    forwarded = []
    response = Mock()
    response.__enter__ = Mock(return_value=SimpleNamespace(status=200, read=lambda: reply))
    response.__exit__ = Mock(return_value=False)
    original_urlopen = urllib.request.urlopen

    def upstream(request, **kwargs):
        if request.full_url.startswith("http://upstream.invalid/"):
            forwarded.append(request)
            return response
        return original_urlopen(request, **kwargs)

    monkeypatch.setattr("emet.eval.chat_gateway.urllib.request.urlopen", upstream)
    with recorded_chat_gateway("http://upstream.invalid/v1", output) as endpoint:
        request = urllib.request.Request(
            endpoint + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer header-secret"},
        )
        with original_urlopen(request, timeout=5) as result:
            assert result.read() == reply
    assert forwarded[0].data == body
    assert forwarded[0].get_header("Authorization") == "Bearer header-secret"
    case = output / "request-001"
    artifacts = b"\n".join(path.read_bytes() for path in case.iterdir())
    for secret in (b"body-secret", b"nested-secret", b"header-secret", b"response-secret"):
        assert secret not in artifacts
    receipt = json.loads((case / "receipt.json").read_text())
    assert receipt["request_sha256"] == hashlib.sha256(body).hexdigest()
    assert receipt["recorded_request_sha256"] == hashlib.sha256((case / "request.json").read_bytes()).hexdigest()
    assert receipt["request_recording"] == receipt["response_recording"] == "credential_fields_redacted"
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
