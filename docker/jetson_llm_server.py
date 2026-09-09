#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""OpenAI-compatible chat server for Jetson containers (Python 3.8 + Tegra CUDA torch).

Kept separate from ``emet.llms.openai_server`` so it runs inside dustynv L4T images
without requiring emet's Python >=3.10 package.

  # Text (CausalLM) on :8000
  python3 jetson_llm_server.py --model Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000

  # Vision-language (Qwen3-VL on JP7 native torch, or Qwen2-VL on JP5 dustynv)
  python3 jetson_llm_server.py --vl --model Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

_DATA_URL_RE = re.compile(r"^data:(image/[^;]+);base64,(.+)$", re.DOTALL | re.IGNORECASE)

DEFAULT_TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _messages_have_images(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                return True
    return False


def _decode_data_url_pil(url: str):
    from PIL import Image

    raw = (url or "").strip()
    m = _DATA_URL_RE.match(raw)
    if not m:
        raise ValueError("image_url must be a data:image/...;base64,... URL")
    blob = base64.b64decode(m.group(2))
    return Image.open(BytesIO(blob)).convert("RGB")


def _messages_to_chat(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
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
                elif isinstance(block, dict) and block.get("type") == "image_url":
                    # Detected for clear error in text generate(); pixels dropped here.
                    continue
            content = "\n".join(p for p in parts if p)
        else:
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def _messages_to_qwen_vl(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Convert OpenAI multimodal messages → Qwen2-VL chat + ordered PIL images."""
    qwen_msgs: list[dict[str, Any]] = []
    images: list[Any] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if content is None:
            continue
        if role == "system":
            if isinstance(content, list):
                texts = [str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                text = "\n".join(t for t in texts if t)
            else:
                text = str(content)
            if text.strip():
                qwen_msgs.append({"role": "system", "content": text})
            continue
        if role == "assistant":
            if isinstance(content, list):
                texts = [str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                text = "\n".join(t for t in texts if t)
            else:
                text = str(content)
            if text.strip():
                qwen_msgs.append({"role": "assistant", "content": text})
            continue
        # user
        if isinstance(content, str):
            qwen_msgs.append({"role": "user", "content": [{"type": "text", "text": content}]})
            continue
        if not isinstance(content, list):
            qwen_msgs.append({"role": "user", "content": [{"type": "text", "text": str(content)}]})
            continue
        parts: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, str):
                if block.strip():
                    parts.append({"type": "text", "text": block})
                continue
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = str(block.get("text") or "")
                if text.strip():
                    parts.append({"type": "text", "text": text})
            elif btype == "image_url":
                image_url = block.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if not url:
                    raise ValueError("image_url block missing url")
                pil = _decode_data_url_pil(str(url))
                images.append(pil)
                parts.append({"type": "image"})
        if parts:
            qwen_msgs.append({"role": "user", "content": parts})
    return qwen_msgs, images


class ModelBundle:
    def __init__(
        self,
        model_id: str,
        device: str,
        max_tokens: int,
        dtype: str,
        multimodal: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_tokens = max_tokens
        self.dtype = dtype
        self.multimodal = bool(multimodal)
        self.tokenizer = None
        self.processor = None
        self.model = None
        self._lock = threading.Lock()
        self.ready = False
        self.load_error = None  # type: Optional[str]

    def load(self) -> None:
        import torch

        _log(
            f"Loading {self.model_id} on device={self.device} "
            f"cuda={torch.cuda.is_available()} torch={torch.__version__} multimodal={self.multimodal}"
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        torch_dtype = torch.float16
        if self.dtype == "float32":
            torch_dtype = torch.float32
        elif self.dtype == "bfloat16" and hasattr(torch, "bfloat16"):
            torch_dtype = torch.bfloat16

        if self.multimodal:
            self._load_vl(torch, torch_dtype)
        else:
            self._load_text(torch, torch_dtype)

        self.model.eval()
        self.ready = True
        self.load_error = None
        if self.device == "cuda":
            free, total = torch.cuda.mem_get_info(0)
            _log(f"CUDA ready: {free / (1024**3):.1f} / {total / (1024**3):.1f} GiB free on device 0")
        _log(f"Model ready: {self.model_id} multimodal={self.multimodal}")

    def _load_text(self, torch, torch_dtype) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        if self.device == "cuda":
            kwargs["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.device == "cpu":
            self.model = self.model.to("cpu")

    def _load_vl(self, torch, torch_dtype) -> None:
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        if self.device == "cuda":
            kwargs["device_map"] = {"": 0}
        # JP7 native: AutoModelForImageTextToText (Qwen3-VL / Qwen3.5). JP5 dustynv
        # pins transformers<4.46 and only ships Qwen2VLForConditionalGeneration.
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            try:
                from transformers import Qwen3VLForConditionalGeneration as AutoModelForImageTextToText
            except ImportError:
                from transformers import Qwen2VLForConditionalGeneration as AutoModelForImageTextToText
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
        if self.device == "cpu":
            self.model = self.model.to("cpu")

    def generate(self, messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
        if not self.ready or self.model is None:
            raise RuntimeError(self.load_error or "model not loaded")
        if self.multimodal:
            return self._generate_vl(messages, max_tokens=max_tokens)
        return self._generate_text(messages, max_tokens=max_tokens)

    def _generate_text(self, messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
        import torch

        if self.tokenizer is None:
            raise RuntimeError("tokenizer not loaded")
        if _messages_have_images(messages):
            raise ValueError(
                "Request includes images but this Jetson container is text CausalLM only. "
                "Start a VL container: "
                "`./scripts/run_jetson_llm_container.sh --vl --detach --port 8001 "
                "--name emet-jetson-vl --model Qwen/Qwen2-VL-2B-Instruct`."
            )
        chat = _messages_to_chat(messages)
        if not chat:
            raise ValueError("messages must be non-empty")
        mt = int(max_tokens or self.max_tokens)
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
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

    def _generate_vl(self, messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
        import torch

        if self.processor is None:
            raise RuntimeError("processor not loaded")
        qwen_msgs, images = _messages_to_qwen_vl(messages)
        if not qwen_msgs:
            raise ValueError("messages must be non-empty")
        mt = int(max_tokens or self.max_tokens)
        prompt = self.processor.apply_chat_template(qwen_msgs, tokenize=False, add_generation_prompt=True)
        # Prefer processor(images=...) path; empty image list → text-only VL call.
        proc_kwargs = {
            "text": [prompt],
            "padding": True,
            "return_tensors": "pt",
        }
        if images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs)
        if self.device == "cuda":
            inputs = {k: (v.cuda(0) if hasattr(v, "cuda") else v) for k, v in inputs.items()}
        with self._lock:
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=mt, do_sample=False)
        trim = inputs["input_ids"].shape[-1]
        gen = out[0][trim:]
        return self.processor.batch_decode([gen], skip_special_tokens=True)[0].strip()


def make_handler(bundle: ModelBundle, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            _log(f"{self.address_string()} - {fmt % args}")

        def _auth_ok(self):
            if not api_key:
                return True
            auth = self.headers.get("Authorization", "")
            if auth == f"Bearer {api_key}":
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
                        "multimodal": bundle.multimodal,
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
            self._send(404, {"error": {"message": f"unknown path {path}", "type": "not_found"}})

        def do_POST(self):
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._send(404, {"error": {"message": f"unknown path {path}", "type": "not_found"}})
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
                        "emet": {
                            "latency_s": round(dt, 3),
                            "device": bundle.device,
                            "backend": "jetson-container",
                            "multimodal": bundle.multimodal,
                        },
                    },
                )
            except Exception as exc:
                _log(f"chat.completions failed: {exc}")
                self._send(500, {"error": {"message": str(exc), "type": type(exc).__name__}})

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description="Jetson Tegra CUDA OpenAI-compatible LLM/VLM server")
    p.add_argument(
        "--vl",
        action="store_true",
        help="Load Qwen3-VL / Qwen2-VL multimodal weights (image_url data URLs).",
    )
    p.add_argument(
        "--quant",
        default=os.environ.get("EMET_LLM_SERVE_QUANT", "fp16"),
        choices=("fp16", "none", "awq", "int4", "int8", "bnb"),
        help=(
            "Weight format. This server stays on fp16: bitsandbytes/AutoAWQ on Jetson "
            "often replace working CUDA torch with CPU wheels. Use vLLM for AWQ."
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help="HF model id (text default 7B; --vl default Qwen3-VL-8B-Instruct).",
    )
    p.add_argument("--host", default=os.environ.get("EMET_LLM_SERVE_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=None)
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

    quant = (args.quant or "fp16").strip().lower()
    if quant in ("none",):
        quant = "fp16"
    if quant not in ("fp16",):
        _log(
            f"FATAL: --quant {quant} is not supported by jetson_llm_server.py. "
            "Stay on fp16 (AGX Orin ~64 GiB unified memory fits Qwen3-VL-8B), or use "
            "vLLM for AWQ. See docs/llm_serve.md."
        )
        return 1

    multimodal = bool(args.vl) or os.environ.get("EMET_LLM_SERVE_VL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    default_model = DEFAULT_VL_MODEL if multimodal else DEFAULT_TEXT_MODEL
    model = args.model or os.environ.get("EMET_LLM_SERVE_MODEL") or default_model
    if args.port is not None:
        port = int(args.port)
    else:
        env_port = os.environ.get("EMET_LLM_SERVE_PORT")
        if env_port:
            port = int(env_port)
        else:
            port = 8001 if multimodal else 8000

    bundle = ModelBundle(model, args.device, args.max_tokens, args.dtype, multimodal=multimodal)
    try:
        bundle.load()
    except Exception as exc:
        bundle.load_error = f"{type(exc).__name__}: {exc}"
        _log(f"FATAL: {bundle.load_error}")
        return 1

    httpd = ThreadingHTTPServer((args.host, int(port)), make_handler(bundle, args.api_key))
    kind = "VLM" if multimodal else "LLM"
    _log(f"Serving OpenAI {kind} API on http://{args.host}:{port}/v1  model={model} device={args.device}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
