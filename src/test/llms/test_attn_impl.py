# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from emet.llms.attn_impl import resolve_attn_implementation


def test_resolve_attn_cpu_is_eager():
    assert resolve_attn_implementation(prefer_flash=True, device="cpu") == "eager"


def test_resolve_attn_cuda_prefers_sdpa_without_flash(monkeypatch):
    import emet.llms.attn_impl as m

    monkeypatch.setattr(
        m,
        "resolve_attn_implementation",
        m.resolve_attn_implementation,
    )

    # Force flash import path to fail by patching the helpers used inside.
    import builtins

    real_import = builtins.__import__

    def _no_flash(name, *args, **kwargs):
        if name == "flash_attn" or name.startswith("flash_attn."):
            raise ImportError("no flash")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_flash)

    # transformers.utils.is_flash_attn_2_available may exist — patch if imported path runs
    try:
        import transformers.utils as tu

        monkeypatch.setattr(tu, "is_flash_attn_2_available", lambda: False, raising=False)
    except Exception:
        pass

    assert resolve_attn_implementation(prefer_flash=True, device="cuda") == "sdpa"
    assert resolve_attn_implementation(prefer_flash=False, device="cuda") == "sdpa"
