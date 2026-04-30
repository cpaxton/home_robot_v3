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

"""Regression tests for Qwen variant string parsing (VRAM / quantization defaults)."""

from emet.llms import process_incoming_qwen_types


def test_qwen35_two_token_defaults_to_int4():
    """qwen35-9B must not parse to quantization=None (full FP model)."""
    model_size, typing, finetune, quant = process_incoming_qwen_types("qwen35-9B")
    assert model_size == "9B"
    assert typing is None
    assert finetune is None
    assert quant == "int4"


def test_qwen25_two_token_still_no_quant_by_default():
    """qwen25-7B keeps prior behavior (small enough unquantized for dev)."""
    model_size, typing, finetune, quant = process_incoming_qwen_types("qwen25-7B")
    assert model_size == "7B"
    assert quant is None
