# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for remote LLM health/smoke helpers (no network)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from emet.llms.remote_ops import (
    DEFAULT_LLM_PORT,
    DEFAULT_VL_PORT,
    apply_llm_host,
    fetch_health,
    health_url_from_base,
    normalize_openai_base,
    openai_base_for_host,
    resolve_llm_host,
    smoke_chat_completions,
)


def test_openai_base_for_host_unified_ports() -> None:
    assert openai_base_for_host("orin", DEFAULT_LLM_PORT) == "http://orin:8000/v1"
    assert DEFAULT_VL_PORT == DEFAULT_LLM_PORT
    assert openai_base_for_host("orin", 8001) == "http://orin:8001/v1"


def test_resolve_llm_host_prefers_arg_then_env(monkeypatch) -> None:
    monkeypatch.delenv("EMET_LLM_HOST", raising=False)
    monkeypatch.delenv("EMET_CALIBAN_HOST", raising=False)
    assert resolve_llm_host(None) is None
    assert resolve_llm_host("  my-orin ") == "my-orin"
    monkeypatch.setenv("EMET_CALIBAN_HOST", "legacy")
    assert resolve_llm_host(None) == "legacy"
    monkeypatch.setenv("EMET_LLM_HOST", "preferred")
    assert resolve_llm_host(None) == "preferred"
    assert resolve_llm_host("explicit") == "explicit"


def test_apply_llm_host_sets_env(monkeypatch) -> None:
    import os

    for key in (
        "EMET_LLM_HOST",
        "EMET_CALIBAN_HOST",
        "EMET_OPENAI_BASE_URL",
        "EMET_VL_ENDPOINT",
    ):
        monkeypatch.delenv(key, raising=False)
    assert apply_llm_host(None) is None
    specs = apply_llm_host("orin", port=8000, vl_port=8001)
    assert specs == (
        "openai@http://orin:8000/v1",
        "openai@http://orin:8001/v1",
    )
    assert os.environ["EMET_LLM_HOST"] == "orin"
    assert os.environ["EMET_OPENAI_BASE_URL"] == "http://orin:8000/v1"
    assert os.environ["EMET_VL_ENDPOINT"] == "openai@http://orin:8001/v1"
    # unified default: same port for VL
    apply_llm_host("orin2")
    assert os.environ["EMET_VL_ENDPOINT"] == "openai@http://orin2:8000/v1"
    # Drop process env so other tests in this process stay isolated (monkeypatch
    # would restore these values on teardown if we only delenv'd them).
    for key in ("EMET_LLM_HOST", "EMET_OPENAI_BASE_URL", "EMET_VL_ENDPOINT"):
        os.environ.pop(key, None)


def test_normalize_openai_base() -> None:
    assert normalize_openai_base("openai@http://orin:8001/v1#m") == "http://orin:8001/v1"
    assert normalize_openai_base("http://h:9") == "http://h:9/v1"
    assert health_url_from_base("http://h:9/v1") == "http://h:9/health"


def test_fetch_health_and_smoke_local_http() -> None:
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({"status": "ok", "ready": True, "model": "t"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.dumps(
                {"choices": [{"message": {"content": "pong"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = HTTPServer(("127.0.0.1", 0), H)
    port = httpd.server_address[1]
    t = Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}/v1"
        h = fetch_health(base)
        assert h.ok is True
        assert smoke_chat_completions(base).strip() == "pong"
    finally:
        httpd.shutdown()


def test_chat_help_lists_host_vl() -> None:
    from click.testing import CliRunner

    from emet.app.chat import main as chat_main

    r = CliRunner().invoke(chat_main, ["--help"])
    assert r.exit_code == 0
    assert "--host" in r.output
    assert "--caliban" not in r.output
    assert "--vl" in r.output
    assert "--once" in r.output


def test_emet_llm_help() -> None:
    from click.testing import CliRunner

    from emet.cli import main

    r = CliRunner().invoke(main, ["llm", "--help"])
    assert r.exit_code == 0
    assert "health" in r.output
    assert "smoke" in r.output

    h = CliRunner().invoke(main, ["llm", "health", "--help"])
    assert h.exit_code == 0
    assert "--host" in h.output
