#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""DINOv3 embedding server for Jetson Orin (Python 3.8 + Tegra CUDA torch).

Standalone from emet so it runs inside dustynv L4T images without emet's venv.

  python3 jetson_dinov3_server.py --host 0.0.0.0 --port 8002 --model facebook/dinov3-vits16-pretrain-lvd1689m
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _log(msg: str) -> None:
    print(msg, flush=True)


class _ModelState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model_id = DEFAULT_MODEL
        self.device = "cpu"
        self.ready = False
        self.error: str | None = None
        self.feature_dim = 0
        self.processor: Any = None
        self.model: Any = None

    def load(self, model_id: str, device: str) -> None:
        with self.lock:
            self.model_id = model_id
            self.device = device
            self.ready = False
            self.error = None
            try:
                import torch
                from transformers import AutoImageProcessor, AutoModel

                if device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("cuda requested but torch.cuda.is_available() is False")
                self.processor = AutoImageProcessor.from_pretrained(model_id)
                self.model = AutoModel.from_pretrained(model_id).to(device).eval()
                self.feature_dim = int(self.model.config.hidden_size)
                self.ready = True
                _log(f"DINOv3 ready model={model_id} device={device} dim={self.feature_dim}")
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                _log(f"DINOv3 load failed: {self.error}")

    def embed_b64_jpeg(self, image_b64: str) -> list[float]:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image

        with self.lock:
            if not self.ready or self.model is None or self.processor is None:
                raise RuntimeError(self.error or "model not ready")
            blob = base64.b64decode(image_b64)
            img = Image.open(BytesIO(blob)).convert("RGB")
            inputs = self.processor(images=np.asarray(img), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                feat = outputs.pooler_output
                feat = F.normalize(feat, dim=-1)
            return feat.squeeze(0).float().cpu().tolist()


STATE = _ModelState()


class Handler(BaseHTTPRequestHandler):
    server_version = "emet-jetson-dinov3/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        _log(f"{self.address_string()} - {fmt % args}")

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            import torch

            self._send_json(
                200,
                {
                    "status": "ok" if STATE.ready else "loading",
                    "ready": STATE.ready,
                    "model": STATE.model_id,
                    "device": STATE.device,
                    "cuda": bool(torch.cuda.is_available()),
                    "feature_dim": STATE.feature_dim,
                    "error": STATE.error,
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/embed":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return
        image_b64 = data.get("image_b64") or data.get("image")
        if not image_b64:
            self._send_json(400, {"error": "missing image_b64"})
            return
        try:
            embedding = STATE.embed_b64_jpeg(str(image_b64))
            self._send_json(200, {"embedding": embedding, "dim": len(embedding)})
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Jetson DINOv3 embedding HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("EMET_DINOV3_SERVE_PORT", "8002")))
    parser.add_argument("--model", default=os.environ.get("EMET_DINOV3_MODEL_ID", DEFAULT_MODEL))
    parser.add_argument("--device", default=os.environ.get("EMET_DINOV3_SERVE_DEVICE", "cuda"))
    args = parser.parse_args()

    def _load_bg() -> None:
        STATE.load(args.model, args.device)

    threading.Thread(target=_load_bg, daemon=True).start()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    _log(f"Listening on http://{args.host}:{args.port} (model loading in background)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
