# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

"""GPU VRAM snapshots for diagnosing multi-model agent startup (SigLIP + detectors + VLMs + chat LLM)."""

from __future__ import annotations

import os
import subprocess
from typing import Any

_TRUE = frozenset({"1", "true", "yes", "on"})


def vram_debug_enabled() -> bool:
    """True when ``EMET_VRAM_DEBUG=1`` or ``EMET_AGENT_MODEL_DEBUG=1`` (``emet run agent --debug-models``)."""
    for key in ("EMET_VRAM_DEBUG", "EMET_AGENT_MODEL_DEBUG"):
        if os.environ.get(key, "").strip().lower() in _TRUE:
            return True
    return False


def nvidia_smi_gpu_rows() -> list[tuple[int, str, float, float, float]]:
    """Return [(index, name, used_mib, total_mib, free_mib), ...] or empty if unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if out.returncode != 0 or not (out.stdout or "").strip():
        return []
    rows: list[tuple[int, str, float, float, float]] = []
    for line in (out.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            used = float(parts[2])
            total = float(parts[3])
            free = float(parts[4])
        except ValueError:
            continue
        rows.append((idx, parts[1], used, total, free))
    return rows


def torch_cuda_mem_lines() -> list[str]:
    """Per-CUDA-device PyTorch allocated / reserved (MiB), if CUDA is available."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    lines: list[str] = []
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.synchronize(i)
        except Exception:
            pass
        alloc = torch.cuda.memory_allocated(i) / (1024 * 1024)
        rsrv = torch.cuda.memory_reserved(i) / (1024 * 1024)
        name = torch.cuda.get_device_name(i) if hasattr(torch.cuda, "get_device_name") else f"cuda:{i}"
        lines.append(f"torch cuda:{i} ({name}) allocated={alloc:.0f} MiB reserved={rsrv:.0f} MiB")
    return lines


def format_vram_snapshot(stage: str, *, extra: str | None = None) -> str:
    """Single multi-line block for logs / TTY (no trailing newline on last line handled by caller)."""
    lines: list[str] = [f"[vram] {stage}"]
    smi = nvidia_smi_gpu_rows()
    if smi:
        for idx, name, used, total, free in smi:
            pct = (100.0 * used / total) if total > 0 else 0.0
            lines.append(
                f"  nvidia-smi gpu{idx} {name!r}: used={used:.0f} / total={total:.0f} MiB "
                f"free={free:.0f} MiB ({pct:.1f}% used)"
            )
    else:
        lines.append("  nvidia-smi: (no data — not NVIDIA, nvidia-smi missing, or query failed)")
    for tline in torch_cuda_mem_lines():
        lines.append(f"  {tline}")
    if extra:
        lines.append(f"  note: {extra}")
    return "\n".join(lines)


def print_vram_snapshot(stage: str, *, extra: str | None = None, force: bool = False) -> None:
    """TTY: colored block when debug enabled, or when *force* (e.g. one-shot warnings)."""
    if not force and not vram_debug_enabled():
        return
    try:
        from termcolor import colored
    except ImportError:
        print(format_vram_snapshot(stage, extra=extra), flush=True)
        return
    text = format_vram_snapshot(stage, extra=extra)
    for ln in text.splitlines():
        print(colored(ln, "magenta"), flush=True)


def client_quantization_hint(client: Any) -> str | None:
    """Best-effort ``_quantization`` / config label for debug lines."""
    q = getattr(client, "_quantization", None)
    if q is not None:
        return str(q)
    return None


def torch_cuda_alloc_reserved_gib(device_index: int = 0) -> tuple[float | None, float | None]:
    """Return (allocated_gib, reserved_gib) for one device, or (None, None) if unavailable."""
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    try:
        torch.cuda.synchronize(device_index)
        a = torch.cuda.memory_allocated(device_index) / (1024**3)
        r = torch.cuda.memory_reserved(device_index) / (1024**3)
        return a, r
    except Exception:
        return None, None


def cuda_pre_llm_memory_notice(*, device: str = "cuda", device_index: int = 0) -> str | None:
    """One TTY line: VRAM already used in this process before Transformers prints ``Loading weights`` for --llm."""
    if device != "cuda":
        return None
    alloc_gib, rsrv_gib = torch_cuda_alloc_reserved_gib(device_index)
    if alloc_gib is None:
        return None

    smi_line = ""
    for row in nvidia_smi_gpu_rows():
        if row[0] == device_index:
            _idx, name, used, total, free = row
            smi_line = f" nvidia-smi gpu{device_index} ({name}): {used:.0f}/{total:.0f} MiB used, {free:.0f} MiB free."
            break
    if not smi_line and nvidia_smi_gpu_rows():
        idx, name, used, total, free = nvidia_smi_gpu_rows()[0]
        smi_line = f" nvidia-smi gpu{idx} ({name}): {used:.0f}/{total:.0f} MiB used, {free:.0f} MiB free."

    return (
        f"Before ``--llm`` weights: this process ~{alloc_gib:.1f} GiB torch allocated, ~{rsrv_gib:.1f} GiB reserved "
        f"on cuda:{device_index} (SigLIP, detector, embodied stack — little tqdm spam)."
        f"{smi_line}"
        " The **next** ``Loading weights`` bar adds most of the chat model on top (watch ``nvidia-smi`` climb). "
        "Full snapshot: EMET_VRAM_DEBUG=1 or ``emet run agent --debug-vram``."
    )


def cuda_oom_followup_hint(*, llm_key: str) -> str:
    """Short hints after chat LLM CUDA OOM."""
    return (
        f"OOM loading {llm_key!r}: the yellow line was **before** the chat checkpoint streamed in; during "
        "``Loading weights`` this process grew until the last allocation failed (often ~12–20 GiB combined "
        "for qwen35-9B int4 plus SigLIP/detector, plus short-lived load peaks). "
        "``nvidia-smi`` is **total GPU** at one instant—use ``watch -n0.5 nvidia-smi`` while loading to see it rise. "
        "Other PIDs in the PyTorch message are **other programs** on the same GPU, not hidden emet usage. "
        "Try: qwen35-4B, shrink embodied_agent graph paths in YAML, PYTORCH_ALLOC_CONF=expandable_segments:True, "
        "or ``--device cpu`` for chat only."
    )


def format_cuda_torch_state_line(*, label: str, device_index: int = 0) -> str | None:
    """One line after errors: current torch alloc/reserve (helps compare to nvidia-smi)."""
    a, r = torch_cuda_alloc_reserved_gib(device_index)
    if a is None:
        return None
    return f"[cuda] {label}: torch allocated={a:.2f} GiB reserved={r:.2f} GiB (cuda:{device_index})"
