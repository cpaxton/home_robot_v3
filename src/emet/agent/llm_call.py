# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Bind one CHAT inference call without using model exceptions as capability probes."""

from __future__ import annotations

from collections.abc import Callable
from inspect import Parameter, ismethod, signature
from typing import Any

from emet.llms.base import AbstractVLLMClient


class LLMCallAdapter:
    """Inspect a client's Python signature once and preserve requested tools/images.

    Optional telemetry and legacy history keywords are sent only when accepted.
    A requested image or native tool catalog requires explicit signature support
    (or **kwargs). Exceptions raised inside inference propagate without retries.
    Clients accepting **kwargs own validation of those arguments.
    """

    def __init__(self, client: Any):
        self.client = client
        self.multimodal = isinstance(client, AbstractVLLMClient)
        self.generate = client.generate_multimodal if self.multimodal else client
        parameters = signature(self.generate).parameters.values()
        self.accepts_kwargs = any(p.kind is Parameter.VAR_KEYWORD for p in parameters)
        self.keywords = frozenset(
            p.name for p in parameters if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        )

    def __call__(
        self,
        text: str,
        *,
        tools: list | None,
        image: Any,
        reset_context: bool,
        verbose: bool,
        progress_callback: Callable[[str], None] | None,
    ) -> str:
        kwargs: dict[str, Any] = {"reset_context": reset_context, "verbose": verbose}
        if self.multimodal:
            kwargs.update(
                system_prompt=getattr(self.client, "system_prompt", None) or None,
                max_new_tokens=getattr(self.client, "max_tokens", None),
                image=image,
            )
            if progress_callback is not None:
                kwargs["progress_callback"] = progress_callback
        elif image is not None:
            kwargs["image"] = image
        if tools is not None:
            kwargs["tools"] = tools
        for name, requested in (("image", image is not None), ("tools", tools is not None)):
            if requested and not self.accepts_kwargs and name not in self.keywords:
                raise TypeError(f"{type(self.client).__name__} does not support requested {name}")
        if not self.accepts_kwargs:
            kwargs = {name: value for name, value in kwargs.items() if name in self.keywords}
        return self.generate(text, **kwargs)


def call_adapter(client: Any) -> LLMCallAdapter:
    """Cache on the exact client; bound methods and slot-only clients bind on demand.

    A bound method exposes its function's shared dictionary, so caching there
    would retain and reuse another receiver. Copied clients can also inherit an
    adapter bound to the original instance; check identity before reusing it.
    """
    state = None if ismethod(client) else getattr(client, "__dict__", None)
    if not isinstance(state, dict):
        return LLMCallAdapter(client)
    adapter = state.get("_agent_call_adapter")
    if not isinstance(adapter, LLMCallAdapter) or adapter.client is not client:
        adapter = LLMCallAdapter(client)
        state["_agent_call_adapter"] = adapter
    return adapter
