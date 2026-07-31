#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""OpenAI-compatible chat server for Jetson containers (Python 3.8 + Tegra CUDA torch).

Kept separate from ``emet.llms.openai_server`` so it runs inside dustynv L4T images
without requiring emet's Python >=3.10 package.

  python3 jetson_llm_server.py --model Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


def _log(msg: str) -> None:
    print(msg, flush=True)


def _messages_to_chat(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if content is None:
            continue
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif isinstance(block, str):
                    parts.append(block)
            content = "\n".join(p for p in parts if p)
        else:
            content = str(content)
        out.append({"role": role, "content": content})
    return out


class ModelBundle(object):
    def __init__(self, model_id: str, device: str, max_tokens: int, dtype: str) -> None:
        self.model_id = model_id
        self.device = device
        self.max_tokens = max_tokens
        self.dtype = dtype
        self.tokenizer = None
        self.model = None
        self._lock = threading.Lock()
        self.ready = False
        self.load_error = None  # type: Optional[str]

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _log(
            "Loading %s on device=%s cuda=%s torch=%s"
            % (self.model_id, self.device, torch.cuda.is_available(), torch.__version__)
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.float16
        if self.dtype == "float32":
            torch_dtype = torch.float32
        elif self.dtype == "bfloat16" and hasattr(torch, "bfloat16"):
            torch_dtype = torch.bfloat16

        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        if self.device == "cuda":
            kwargs["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        self.ready = True
        self.load_error = None
        if self.device == "cuda":
            free, total = torch.cuda.mem_get_info(0)
            _log(
                "CUDA ready: %.1f / %.1f GiB free on device 0"
                % (free / (1024**3), total / (1024**3))
            )
        _log("Model ready: %s" % self.model_id)

    def generate(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None) -> str:
        import torch

        if not self.ready or self.model is None or self.tokenizer is None:
            raise RuntimeError(self.load_error or "model not loaded")
        chat = _messages_to_chat(messages)
        if not chat:
            raise ValueError("messages must be non-empty")
        mt = int(max_tokens or self.max_tokens)
        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda(0) for k, v in inputs.items()}
        with self._lock:
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=mt,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def make_handler(bundle: ModelBundle, api_key: Optional[str]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            _log("%s - %s" % (self.address_string(), fmt % args))

        def _auth_ok(self):
            if not api_key:
                return True
            auth = self.headers.get("Authorization", "")
            if auth == "Bearer %s" % api_key:
                return True
            return self.headers.get("X-Api-Key", "") == api_key

        def _send(self, code, payload):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Api-Key")
            self.end_headers()

        def do_GET(self):
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in ("/health", "/v1/health"):
                import torch

                self._send(
                    200,
                    {
                        "status": "ok" if bundle.ready else "loading",
                        "ready": bundle.ready,
                        "model": bundle.model_id,
                        "device": bundle.device,
                        "cuda": bool(torch.cuda.is_available()),
                        "torch": torch.__version__,
                        "error": bundle.load_error,
                    },
                )
                return
            if path in ("/v1/models", "/models"):
                if not self._auth_ok():
                    self._send(401, {"error": {"message": "unauthorized", "type": "auth_error"}})
                    return
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": bundle.model_id,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "emet-jetson",
                            }
                        ],
                    },
                )
                return
            self._send(404, {"error": {"message": "unknown path %s" % path, "type": "not_found"}})

        def do_POST(self):
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._send(404, {"error": {"message": "unknown path %s" % path, "type": "not_found"}})
                return
            if not self._auth_ok():
                self._send(401, {"error": {"message": "unauthorized", "type": "auth_error"}})
                return
            try:
                body = self._read_json()
                messages = body.get("messages")
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
                mt = body.get("max_tokens") or body.get("max_completion_tokens")
                t0 = time.time()
                text = bundle.generate(messages, max_tokens=int(mt) if mt is not None else None)
                dt = time.time() - t0
                model = str(body.get("model") or bundle.model_id)
                self._send(
                    200,
                    {
                        "id": "chatcmpl-%s" % uuid.uuid4().hex[:24],
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        "emet": {
                            "latency_s": round(dt, 3),
                            "device": bundle.device,
                            "backend": "jetson-container",
                        },
                    },
                )
            except Exception as exc:
                _log("chat.completions failed: %s" % exc)
                self._send(500, {"error": {"message": str(exc), "type": type(exc).__name__}})

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description="Jetson Tegra CUDA OpenAI-compatible LLM server")
    p.add_argument("--model", default=os.environ.get("EMET_LLM_SERVE_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    p.add_argument("--host", default=os.environ.get("EMET_LLM_SERVE_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("EMET_LLM_SERVE_PORT", "8000")))
    p.add_argument(
        "--device",
        default=os.environ.get("EMET_LLM_SERVE_DEVICE", "cuda"),
        choices=("cuda", "cpu"),
    )
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("EMET_LLM_SERVE_MAX_TOKENS", "512")))
    p.add_argument(
        "--dtype",
        default=os.environ.get("EMET_LLM_SERVE_DTYPE", "float16"),
        choices=("float16", "float32", "bfloat16"),
    )
    p.add_argument("--api-key", default=os.environ.get("EMET_LLM_SERVE_API_KEY") or None)
    args = p.parse_args(argv)

    bundle = ModelBundle(args.model, args.device, args.max_tokens, args.dtype)
    try:
        bundle.load()
    except Exception as exc:
        bundle.load_error = "%s: %s" % (type(exc).__name__, exc)
        _log("FATAL: %s" % bundle.load_error)
        return 1

    httpd = ThreadingHTTPServer((args.host, int(args.port)), make_handler(bundle, args.api_key))
    _log("Serving OpenAI API on http://%s:%s/v1  model=%s device=%s" % (args.host, args.port, args.model, args.device))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
