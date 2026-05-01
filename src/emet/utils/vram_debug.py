# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the
# root directory of this source tree.

"""VRAM / CUDA memory helpers for ``EMET_VRAM_DEBUG=1`` and agent model-debug milestones."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from termcolor import colored

_TRUE = frozenset({"1", "true", "yes", "on"})


def vram_debug_enabled() -> bool:
    """True when ``EMET_VRAM_DEBUG`` requests CUDA / nvidia-smi snapshots (see ``print_embodied_model_report``)."""
    v = os.environ.get("EMET_VRAM_DEBUG", "").strip().lower()
    return v in _TRUE


def _agent_model_debug() -> bool:
    v = os.environ.get("EMET_AGENT_MODEL_DEBUG", "").strip().lower()
    return v in _TRUE


def _milestone_snapshots_enabled() -> bool:
    """VRAM milestones (nvidia-smi / torch) when VRAM debug or agent model debug is on."""
    return vram_debug_enabled() or _agent_model_debug()


def client_quantization_hint(client: Any) -> str | None:
    """Best-effort quantization label for logging (HF configs, bitsandbytes, etc.)."""
    if client is None:
        return None
    for attr in ("quantization_config", "_quantization_config", "bnb_quantization_config"):
        qc = getattr(client, attr, None)
        if qc is not None:
            return type(qc).__name__
    pipe = getattr(client, "pipe", None)
    if pipe is not None:
        m = getattr(pipe, "model", None)
        cfg = getattr(m, "config", None) if m is not None else None
        if cfg is not None:
            qc = getattr(cfg, "quantization_config", None)
            if qc is not None:
                return str(qc)
    m = getattr(client, "model", None)
    if m is not None:
        cfg = getattr(m, "config", None)
        if cfg is not None:
            qc = getattr(cfg, "quantization_config", None)
            if qc is not None:
                return str(qc)
    return None


def format_vram_snapshot(header: str) -> str:
    """Return a multi-line string: optional ``nvidia-smi`` plus ``torch`` CUDA memory if available."""
    lines: list[str] = [f"--- VRAM snapshot: {header} ---"]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            lines.append("nvidia-smi (MiB used / total per GPU):")
            for row in out.stdout.strip().splitlines():
                lines.append(f"  {row.strip()}")
        elif out.stderr:
            lines.append(f"nvidia-smi: ({out.stderr.strip()[:200]})")
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as e:
        lines.append(f"nvidia-smi: unavailable ({e})")

    try:
        import torch

        if torch.cuda.is_available():
            lines.append(torch.cuda.memory_summary(device=None, abbreviated=False))
        else:
            lines.append("torch.cuda: not available")
    except Exception as e:
        lines.append(f"torch CUDA memory: ({e})")

    return "\n".join(lines)


def print_vram_snapshot(label: str, *, extra: str | None = None) -> None:
    """Print nvidia-smi + torch memory when VRAM or agent model debug is enabled."""
    if not _milestone_snapshots_enabled():
        return
    suffix = f"  |  {extra}" if extra else ""
    print(colored(f"[vram] {label}{suffix}", "magenta"), flush=True)
    for ln in format_vram_snapshot(label).splitlines():
        print(colored(ln, "magenta"), flush=True)


def cuda_pre_llm_memory_notice(*, device: str) -> str | None:
    """If CUDA is tight before a large LLM load, return a one-line warning."""
    if device != "cuda":
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        if total <= 0:
            return None
        frac = free / float(total)
        if frac >= 0.15:
            return None
        return (
            f"[VRAM] Low free GPU memory before LLM load: {free / 1e9:.2f} GiB free of {total / 1e9:.2f} GiB "
            f"({frac:.0%}); consider a smaller --llm or --device cpu."
        )
    except Exception:
        return None


def format_cuda_torch_state_line(*, label: str, device_index: int = 0) -> str | None:
    """One-line CUDA allocator state after an error."""
    try:
        import torch

        if not torch.cuda.is_available():
            return f"{label}: CUDA not available"
        dev = torch.device(f"cuda:{device_index}")
        alloc = torch.cuda.memory_allocated(dev) / 1e9
        reserved = torch.cuda.memory_reserved(dev) / 1e9
        return f"{label}: cuda:{device_index} allocated={alloc:.2f} GiB reserved={reserved:.2f} GiB"
    except Exception as e:
        return f"{label}: ({e})"


def cuda_oom_followup_hint(*, llm_key: str) -> str:
    """Actionable hints after a CUDA OOM during LLM load."""
    return (
        f"CUDA OOM with --llm {llm_key!r}: try a smaller model, ``--device cpu``, ``--no-eqa``, "
        "``--no-share-memory-vllm``, or free GPU memory and retry."
    )
