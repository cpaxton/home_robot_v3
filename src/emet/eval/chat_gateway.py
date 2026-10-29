# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Local black-box request recorder for an unchanged OpenAI-compatible agent.

Bodies contain actual image data URLs. Treat these as experiment artifacts;
headers, including credentials, are never persisted. No prompts are rewritten.
"""

import hashlib
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


@contextmanager
def recorded_chat_gateway(upstream, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
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
            case.mkdir()
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            (case / "request.json").write_bytes(body)
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
            (case / "response.json").write_bytes(result)
            (case / "receipt.json").write_text(
                json.dumps(
                    {
                        "http_status": status,
                        "request_sha256": hashlib.sha256(body).hexdigest(),
                        "response_sha256": hashlib.sha256(result).hexdigest(),
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
