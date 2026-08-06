# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

import builtins

import pytest

from emet.llms import attn_impl as m
from emet.llms.attn_impl import resolve_attn_implementation


def _block_flash(monkeypatch):
    real_import = builtins.__import__

    def _no_flash(name, *args, **kwargs):
        if name == "flash_attn" or name.startswith("flash_attn."):
            raise ImportError("no flash")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_flash)
    monkeypatch.setattr(m, "flash_attn_2_available", lambda: False)


def test_resolve_attn_cpu_is_eager():
    assert resolve_attn_implementation(prefer_flash=True, device="cpu") == "eager"


def test_resolve_attn_cuda_requires_flash_by_default(monkeypatch):
    _block_flash(monkeypatch)
    monkeypatch.delenv("EMET_ALLOW_SDPA_ATTN", raising=False)
    monkeypatch.delenv("EMET_REQUIRE_FLASH_ATTN", raising=False)
    with pytest.raises(RuntimeError, match="Flash-Attn 2 is required"):
        resolve_attn_implementation(prefer_flash=True, device="cuda")


def test_resolve_attn_cuda_sdpa_when_allowed(monkeypatch):
    _block_flash(monkeypatch)
    monkeypatch.setenv("EMET_ALLOW_SDPA_ATTN", "1")
    monkeypatch.delenv("EMET_REQUIRE_FLASH_ATTN", raising=False)
    assert resolve_attn_implementation(prefer_flash=True, device="cuda") == "sdpa"
    assert resolve_attn_implementation(prefer_flash=False, device="cuda") == "sdpa"


def test_resolve_attn_allow_sdpa_overrides_installed_flash(monkeypatch):
    """Escape hatch must force SDPA even when flash-attn is importable (stuck-FA2)."""
    monkeypatch.setattr(m, "flash_attn_2_available", lambda: True)
    monkeypatch.setenv("EMET_ALLOW_SDPA_ATTN", "1")
    monkeypatch.delenv("EMET_REQUIRE_FLASH_ATTN", raising=False)
    assert resolve_attn_implementation(prefer_flash=True, device="cuda") == "sdpa"


def test_resolve_attn_flash_when_available_by_default(monkeypatch):
    monkeypatch.setattr(m, "flash_attn_2_available", lambda: True)
    monkeypatch.delenv("EMET_ALLOW_SDPA_ATTN", raising=False)
    monkeypatch.delenv("EMET_REQUIRE_FLASH_ATTN", raising=False)
    assert resolve_attn_implementation(prefer_flash=True, device="cuda") == "flash_attention_2"


def test_resolve_attn_require_flash_false(monkeypatch):
    _block_flash(monkeypatch)
    monkeypatch.delenv("EMET_ALLOW_SDPA_ATTN", raising=False)
    assert resolve_attn_implementation(prefer_flash=True, device="cuda", require_flash=False) == "sdpa"


def test_resolve_attn_env_require_off(monkeypatch):
    _block_flash(monkeypatch)
    monkeypatch.setenv("EMET_REQUIRE_FLASH_ATTN", "0")
    monkeypatch.delenv("EMET_ALLOW_SDPA_ATTN", raising=False)
    assert resolve_attn_implementation(prefer_flash=True, device="cuda") == "sdpa"
