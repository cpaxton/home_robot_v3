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

from emet.llms import QWEN_VL_PRESETS, get_llm_choices


def test_qwen_vl_keys_in_llm_choices():
    choices = get_llm_choices()
    for k in QWEN_VL_PRESETS:
        assert k in choices
