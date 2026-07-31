# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Host platform helpers (Jetson / Tegra detection for install and runtime hints)."""

from __future__ import annotations

import os
import platform
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def is_tegra() -> bool:
    """True when running on NVIDIA Tegra / Jetson (L4T)."""
    if os.environ.get("EMET_FORCE_TEGRA", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return Path("/etc/nv_tegra_release").is_file()


@lru_cache(maxsize=1)
def is_aarch64() -> bool:
    machine = platform.machine().lower()
    return machine in ("aarch64", "arm64")


def jetson_install_hints() -> list[str]:
    """Short operator hints when on Jetson (empty otherwise)."""
    if not is_tegra() and not (is_aarch64() and os.environ.get("EMET_JETSON_HINTS")):
        return []
    return [
        "Lean install: ./scripts/install_jetson.sh -y  (or: ./install.sh --profile=jetson -y)",
        "Set EMET_ALLOW_SDPA_ATTN=1 (default on jetson profile) — no Triton / flash-attn on Tegra.",
        "PyPI torch on aarch64 is not Tegra-CUDA; use NVIDIA Jetson wheels or build-from-source for GPU.",
    ]
