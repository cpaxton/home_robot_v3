# Copyright (c) Hello Robot, Inc. All rights reserved.

from emet.utils.vram_debug import (
    cuda_pre_llm_memory_notice,
    format_vram_snapshot,
    vram_debug_enabled,
)


def test_format_vram_snapshot_has_stage():
    s = format_vram_snapshot("unit_test_stage")
    assert "unit_test_stage" in s
    assert "[vram]" in s


def test_vram_debug_disabled_by_default(monkeypatch):
    monkeypatch.delenv("EMET_VRAM_DEBUG", raising=False)
    monkeypatch.delenv("EMET_AGENT_MODEL_DEBUG", raising=False)
    assert vram_debug_enabled() is False


def test_cuda_pre_llm_notice_none_on_cpu_device():
    assert cuda_pre_llm_memory_notice(device="cpu") is None


def test_cuda_oom_hint_mentions_llm():
    from emet.utils.vram_debug import cuda_oom_followup_hint

    s = cuda_oom_followup_hint(llm_key="qwen35-9B")
    assert "qwen35-9B" in s
    assert "OOM loading" in s
    from emet.llms import get_llm_choices

    assert "qwen3-vl-eqa" in get_llm_choices()
