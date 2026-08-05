# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared helpers for extracting a JSON object from LLM text."""

from __future__ import annotations

import json
import re
from typing import Any


def fence_inner_json(text: str) -> str | None:
    """Return inner JSON string from first ```json ... ``` or ``` ... ``` fence, or None."""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    return m.group(1).strip() if m else None


def first_json_dict(text: str) -> dict[str, Any] | None:
    """Parse the first balanced JSON object from *text* using :meth:`json.JSONDecoder.raw_decode`.

    Avoids greedy ``\\{[\\s\\S]*\\}`` bugs when there is trailing prose or multiple ``{`` tokens.
    """
    if not text:
        return None
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, _end = dec.raw_decode(text, j)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = j + 1
    return None


def first_json_dict_lenient(text: str, *, prefill: str | None = None) -> dict[str, Any] | None:
    """Like :func:`first_json_dict`, also trying fenced bodies and optional assistant prefill repair.

    Remote OpenAI servers often return only the continuation after ``assistant_prefill``, so
    ``prefill + continuation`` may be required to recover a balanced object.
    """
    blob = (text or "").strip()
    if not blob:
        return None
    candidates: list[str] = [blob]
    fenced = fence_inner_json(blob)
    if fenced:
        candidates.append(fenced)
    pf = (prefill or "").strip()
    if pf:
        candidates.append(pf + blob)
        candidates.append(pf + "\n" + blob)
    for cand in candidates:
        obj = first_json_dict(cand)
        if obj is not None:
            return obj
    return None
