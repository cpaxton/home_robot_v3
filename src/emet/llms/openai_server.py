# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""OpenAI-compatible HTTP LLM server backed by ``emet.llms.get_llm_client``.

Exposes a minimal subset of the OpenAI Chat Completions API so a workstation (or
any OpenAI SDK client) can call models loaded on this host (e.g. Jetson Orin)::

    emet serve llm --llm qwen25-14B --host 0.0.0.0 --port 8000

    # on another machine:
    export EMET_OPENAI_BASE_URL=http://caliban:8000/v1
    emet run agent --llm openai ...

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from emet.utils.logger import Logger

logger = Logger(__name__)

DEFAULT_LLM_SERVE_HOST = "0.0.0.0"
DEFAULT_LLM_SERVE_PORT = 8000
DEFAULT_LLM_SERVE_MODEL = "qwen25-7B"


def ensure_aarch64_gomp_preload() -> str | None:
    """Preload system libgomp before sklearn/transformers on Jetson aarch64 (static TLS).

    Without this, importing ``transformers`` can fail with::
      cannot allocate memory in static TLS block
    Returns the path that should be in ``LD_PRELOAD``, or None.
    """
    if platform.machine().lower() not in ("aarch64", "arm64"):
        return None
    if os.environ.get("EMET_SKIP_GOMP_PRELOAD", "").strip().lower() in ("1", "true", "yes"):
        return None
    candidates = [
        Path("/usr/lib/aarch64-linux-gnu/libgomp.so.1"),
        Path("/lib/aarch64-linux-gnu/libgomp.so.1"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def maybe_reexec_with_gomp_preload() -> None:
    """Re-exec this process once with libgomp preloaded (must be before sklearn loads)."""
    if os.environ.get("EMET_GOMP_PRELOAD_DONE", "").strip() == "1":
        return
    path = ensure_aarch64_gomp_preload()
    if path is None:
        return
    preload = os.environ.get("LD_PRELOAD", "")
    parts = [p for p in preload.split(":") if p]
    if parts and parts[0] == path:
        os.environ["EMET_GOMP_PRELOAD_DONE"] = "1"
        return
    os.environ["EMET_GOMP_PRELOAD_DONE"] = "1"
    os.environ["LD_PRELOAD"] = path if not parts else f"{path}:{preload}"
    logger.info(f"Re-exec with LD_PRELOAD={path} (aarch64 OpenMP / transformers TLS workaround)")
    os.execv(sys.executable, [sys.executable, *sys.argv])


def resolve_serve_device(device: str | None = None) -> str:
    """Pick cuda when torch reports it, else cpu (Jetson emet venv is often CPU-only)."""
    raw = (device or os.environ.get("EMET_LLM_SERVE_DEVICE", "auto")).strip().lower()
    if raw in ("", "auto"):
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if raw not in ("cuda", "cpu", "mps"):
        raise ValueError(f"Invalid device {device!r}; use auto, cuda, cpu, or mps")
    return raw


def _openai_messages_to_chat(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Normalize OpenAI chat messages to role/content string dicts (text only)."""
    out: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if content is None:
            continue
        if isinstance(content, list):
            parts: list[str] = []
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


def generate_chat(
    client: Any,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None = None,
) -> str:
    """Run one chat completion against an emet LLM client (Qwen pipe path preferred)."""
    chat = _openai_messages_to_chat(messages)
    if not chat:
        raise ValueError("messages must be a non-empty list")

    tokenizer = getattr(client, "tokenizer", None)
    pipe = getattr(client, "pipe", None)
    if tokenizer is not None and pipe is not None:
        template_kwargs: dict[str, Any] = {}
        if getattr(client, "_version", None) == "3.5":
            template_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        mt = int(max_tokens or getattr(client, "max_tokens", 512) or 512)
        # Avoid importing transformers.GenerationConfig here — on Jetson aarch64 that
        # pull can load sklearn/libgomp and hit "cannot allocate memory in static TLS block".
        pipe_kwargs: dict[str, Any] = {"max_new_tokens": mt}
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            pipe_kwargs["pad_token_id"] = pad_id
        outputs = pipe(text, **pipe_kwargs)
        generated = outputs[0]["generated_text"]
        if "<|im_start|>assistant" in generated:
            assistant = generated.rsplit("<|im_start|>assistant", 1)[-1]
            assistant = assistant.lstrip("\n").rstrip()
            if assistant.endswith("<|im_end|>"):
                assistant = assistant[: -len("<|im_end|>")].rstrip()
            return assistant
        if "assistant" in generated:
            return generated.split("assistant")[-1].strip()
        return str(generated)

    # Fallback: single-shot AbstractLLMClient (loses multi-turn fidelity).
    system = ""
    user_parts: list[str] = []
    for m in chat:
        if m["role"] == "system" and not system:
            system = m["content"]
        elif m["role"] == "user":
            user_parts.append(m["content"])
        elif m["role"] == "assistant":
            user_parts.append(f"(assistant previously said: {m['content']})")
    if system:
        client._prompt = system  # noqa: SLF001 — serve path reuses loaded client
    client.reset()
    if not user_parts:
        raise ValueError("no user message in messages")
    return str(client(user_parts[-1], verbose=False))


class LLMServeState:
    """Process-wide model handle for the HTTP handlers."""

    def __init__(
        self,
        *,
        llm_key: str,
        device: str,
        max_tokens: int,
        api_key: str | None = None,
    ) -> None:
        self.llm_key = llm_key
        self.device = device
        self.max_tokens = max_tokens
        self.api_key = (api_key or os.environ.get("EMET_LLM_SERVE_API_KEY") or "").strip() or None
        self.model_id = llm_key
        self.client: Any = None
        self._lock = threading.Lock()
        self.ready = False
        self.load_error: str | None = None

    def load(self) -> None:
        from emet.llms import get_llm_client

        logger.info(f"Loading LLM {self.llm_key!r} on device={self.device} (max_tokens={self.max_tokens})")
        kwargs: dict[str, Any] = {
            "device": self.device,
            "max_tokens": self.max_tokens,
        }
        # Jetson / aarch64 often has no bitsandbytes; leave quantization to the llm key
        # (e.g. qwen25-14B → no quant). Explicit Int4 keys will fail loudly if bnb missing.
        try:
            self.client = get_llm_client(self.llm_key, prompt="", **kwargs)
            self.model_id = self.llm_key
            self.ready = True
            self.load_error = None
            logger.alert(f"LLM ready: {self.llm_key} on {self.device}")
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.ready = False
            logger.error(f"LLM load failed: {self.load_error}")
            raise

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.ready or self.client is None:
            raise RuntimeError(self.load_error or "model not loaded")
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        mt = int(max_tokens) if max_tokens is not None else self.max_tokens
        t0 = time.perf_counter()
        with self._lock:
            text = generate_chat(self.client, messages, max_tokens=mt)
        dt = time.perf_counter() - t0
        model = str(body.get("model") or self.model_id)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
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
            "emet": {"latency_s": round(dt, 3), "device": self.device, "llm_key": self.llm_key},
        }


def _make_handler(state: LLMServeState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s" % (self.address_string(), fmt % args))

        def _check_auth(self) -> bool:
            if not state.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            if auth == f"Bearer {state.api_key}":
                return True
            key = self.headers.get("X-Api-Key", "")
            return key == state.api_key

        def _send(self, code: int, payload: dict[str, Any] | list[Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Api-Key")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in ("/health", "/v1/health"):
                self._send(
                    200,
                    {
                        "status": "ok" if state.ready else "loading",
                        "ready": state.ready,
                        "model": state.model_id,
                        "device": state.device,
                        "error": state.load_error,
                    },
                )
                return
            if path in ("/v1/models", "/models"):
                if not self._check_auth():
                    self._send(401, {"error": {"message": "unauthorized", "type": "auth_error"}})
                    return
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": state.model_id,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "emet",
                            }
                        ],
                    },
                )
                return
            self._send(404, {"error": {"message": f"unknown path {path}", "type": "not_found"}})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._send(404, {"error": {"message": f"unknown path {path}", "type": "not_found"}})
                return
            if not self._check_auth():
                self._send(401, {"error": {"message": "unauthorized", "type": "auth_error"}})
                return
            try:
                body = self._read_json()
                result = state.complete(body)
                self._send(200, result)
            except Exception as exc:
                logger.error(f"chat.completions failed: {exc}")
                self._send(
                    500,
                    {"error": {"message": str(exc), "type": type(exc).__name__}},
                )

    return Handler


def serve_openai_llm(
    *,
    llm: str = DEFAULT_LLM_SERVE_MODEL,
    host: str = DEFAULT_LLM_SERVE_HOST,
    port: int = DEFAULT_LLM_SERVE_PORT,
    device: str | None = None,
    max_tokens: int = 512,
    api_key: str | None = None,
    load_model: bool = True,
) -> None:
    """Load an emet LLM and serve OpenAI-compatible HTTP until interrupted."""
    maybe_reexec_with_gomp_preload()
    resolved = resolve_serve_device(device)
    state = LLMServeState(llm_key=llm, device=resolved, max_tokens=max_tokens, api_key=api_key)
    if load_model:
        state.load()
    handler = _make_handler(state)
    httpd = ThreadingHTTPServer((host, int(port)), handler)
    logger.alert(f"OpenAI-compatible LLM server on http://{host}:{port}/v1  model={llm} device={resolved}")
    logger.info("Clients: export EMET_OPENAI_BASE_URL=http://<this-host>:%s/v1" % port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down LLM server")
    finally:
        httpd.server_close()
