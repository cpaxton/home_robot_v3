# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Local black-box request recorder for an unchanged OpenAI-compatible agent.

Bodies contain actual image data URLs and may contain private task text. Store
artifacts outside the checkout or under ignored logs/ or runs/. Output
directories are private. Headers are not persisted; common JSON credential
fields are redacted on disk. Arbitrary secrets inside free text are not detected.
Upstream requests/responses are forwarded unchanged; redaction affects only logs.
"""

import hashlib
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_CREDENTIAL_FIELDS = frozenset(
    {
        "apikey",
        "authorization",
        "proxyauthorization",
        "accesstoken",
        "refreshtoken",
        "clientsecret",
        "password",
        "token",
    }
)


def _recording_body(body):
    """Retain exact nonsecret JSON bytes; never store unparsable raw bodies."""
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return b'{"recording_error":"non_json_body_omitted"}', "non_json_body_omitted"

    def redact(item):
        if isinstance(item, dict):
            return {
                key: "[REDACTED]"
                if key.lower().replace("_", "").replace("-", "") in _CREDENTIAL_FIELDS
                else redact(val)
                for key, val in item.items()
            }
        return [redact(val) for val in item] if isinstance(item, list) else item

    recorded = redact(value)
    if recorded == value:
        return body, "exact"
    return json.dumps(recorded).encode(), "credential_fields_redacted"


@contextmanager
def recorded_chat_gateway(upstream, output):
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    upstream = upstream.rstrip("/")
    counter = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            nonlocal counter
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            counter += 1
            case = output / f"request-{counter:03d}"
            case.mkdir(mode=0o700)
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            recorded_body, request_recording = _recording_body(body)
            (case / "request.json").write_bytes(recorded_body)
            headers = {"Content-Type": "application/json"}
            if self.headers.get("Authorization"):
                headers["Authorization"] = self.headers["Authorization"]
            request = urllib.request.Request(upstream + "/chat/completions", data=body, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    status, result = response.status, response.read()
            except urllib.error.HTTPError as error:
                status, result = error.code, error.read()
            except (OSError, TimeoutError) as error:
                status, result = 502, json.dumps({"error": str(error)}).encode()
            recorded_result, response_recording = _recording_body(result)
            (case / "response.json").write_bytes(recorded_result)
            (case / "receipt.json").write_text(
                json.dumps(
                    {
                        "http_status": status,
                        "request_sha256": hashlib.sha256(body).hexdigest(),
                        "response_sha256": hashlib.sha256(result).hexdigest(),
                        "recorded_request_sha256": hashlib.sha256(recorded_body).hexdigest(),
                        "recorded_response_sha256": hashlib.sha256(recorded_result).hexdigest(),
                        "request_recording": request_recording,
                        "response_recording": response_recording,
                    }
                )
                + "\n"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
