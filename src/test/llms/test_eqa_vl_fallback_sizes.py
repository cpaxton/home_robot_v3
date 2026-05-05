# Copyright (c) Hello Robot, Inc. All rights reserved.

"""EQA VL multimodal checkpoint fallback order (OOM → smaller Qwen3.5-VL)."""


def test_eqa_vl_sizes_to_try_from_9b():
    from emet.llms import eqa_qwen

    assert eqa_qwen._eqa_vl_sizes_to_try("9B") == ["9B", "4B", "2B", "0.8B"]


def test_eqa_vl_sizes_to_try_from_4b():
    from emet.llms import eqa_qwen

    assert eqa_qwen._eqa_vl_sizes_to_try("4B") == ["4B", "2B", "0.8B"]


def test_eqa_vl_sizes_to_try_smallest():
    from emet.llms import eqa_qwen

    assert eqa_qwen._eqa_vl_sizes_to_try("0.8B") == ["0.8B"]
