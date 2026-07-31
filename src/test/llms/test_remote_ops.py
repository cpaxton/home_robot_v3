# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for remote LLM health/smoke helpers (no network)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from emet.llms.remote_ops import (
    fetch_health,
    health_url_from_base,
    normalize_openai_base,
    smoke_chat_completions,
)


def test_normalize_openai_base() -> None:
    assert normalize_openai_base("openai@http://caliban:8001/v1#m") == "http://caliban:8001/v1"
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


def test_chat_help_lists_caliban_vl() -> None:
    from click.testing import CliRunner

    from emet.app.chat import main as chat_main

    r = CliRunner().invoke(chat_main, ["--help"])
    assert r.exit_code == 0
    assert "--caliban" in r.output
    assert "--vl" in r.output
    assert "--once" in r.output


def test_emet_llm_help() -> None:
    from click.testing import CliRunner

    from emet.cli import main

    r = CliRunner().invoke(main, ["llm", "--help"])
    assert r.exit_code == 0
    assert "health" in r.output
    assert "smoke" in r.output
