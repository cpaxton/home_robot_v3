# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Deploy OpenAI-compatible LLM/VLM containers to a Jetson LAN host."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# AGX Orin unified memory (~64 GiB); leave headroom for the L4T stack.
CALIBAN_ORIN_VRAM_GIB = 64

LLM_PROFILES = ("dual-2b", "unified-7b")


def resolve_deploy_llm_host(host: str | None = None) -> str | None:
    """``--host``, else ``EMET_LLM_HOST``, else ``EMET_CALIBAN_HOST`` (compat)."""
    for candidate in (
        host,
        os.environ.get("EMET_LLM_HOST"),
        os.environ.get("EMET_CALIBAN_HOST"),
    ):
        s = (candidate or "").strip()
        if s:
            return s
    return None


def deploy_llm(
    *,
    host: str | None = None,
    profile: str = "unified-7b",
    model: str | None = None,
    port: int | None = None,
    name: str | None = None,
    root: Path | None = None,
) -> int:
    """Run ``scripts/deploy_caliban_vl.sh`` (rsync weights + Jetson Docker --vl).

    Default ``unified-7b`` serves one Qwen2-VL-7B on ``:8000`` for text + captions,
    which fits AGX Orin (~60–64 GiB unified memory) without holding two 7B weight
    trees on the eMMC.
    """
    repo = root or Path(__file__).resolve().parent.parent.parent
    script = repo / "scripts" / "deploy_caliban_vl.sh"
    if not script.is_file():
        print(f"ERROR: missing {script}", file=sys.stderr)
        return 1
    prof = (profile or "unified-7b").strip().lower()
    if prof in ("7b", "big"):
        prof = "unified-7b"
    if prof in ("2b",):
        prof = "dual-2b"
    if prof not in LLM_PROFILES:
        print(f"ERROR: unknown profile {profile!r}; use {LLM_PROFILES}", file=sys.stderr)
        return 1

    host_s = resolve_deploy_llm_host(host)
    if not host_s:
        print(
            "ERROR: pass --host HOST (or set EMET_LLM_HOST). Example: --host caliban",
            file=sys.stderr,
        )
        return 1
    cmd: list[str] = ["bash", str(script), "--profile", prof, "--host", host_s]
    if model:
        cmd.extend(["--model", model])
    if port is not None:
        cmd.extend(["--port", str(int(port))])
    if name:
        cmd.extend(["--name", name])

    print(f"emet deploy llm: host={host_s} profile={prof} (Orin ~{CALIBAN_ORIN_VRAM_GIB} GiB unified memory)")
    return int(subprocess.call(cmd))
