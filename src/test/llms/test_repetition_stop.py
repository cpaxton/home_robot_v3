# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Unit tests for the shared decoding repetition guard."""

from __future__ import annotations

import torch

from emet.llms.repetition_stop import RepetitionStop, repetition_stopping_criteria


def _ids(prompt, gen):
    return torch.tensor([prompt + gen])


def test_stops_on_single_token_cycle():
    stop = RepetitionStop(prompt_len=3)
    # tail = six identical tokens (period 1, >= 5 reps)
    assert stop(_ids([99, 99, 99], [7, 7, 7, 7, 7, 7]), None) is True


def test_stops_on_two_token_cycle():
    stop = RepetitionStop(prompt_len=3)
    # tail = "5 6" repeated 5x (period 2)
    assert stop(_ids([99, 99, 99], [5, 6] * 5), None) is True


def test_does_not_stop_on_normal_text():
    stop = RepetitionStop(prompt_len=3)
    assert stop(_ids([99, 99, 99], [1, 2, 3, 4, 5, 6, 7, 8]), None) is False


def test_below_threshold_does_not_stop():
    stop = RepetitionStop(prompt_len=3)
    # only 2 reps of a 2-token block (< reps=5)
    assert stop(_ids([99, 99, 99], [5, 6, 5, 6]), None) is False


def test_prompt_repetition_is_ignored():
    """Repetition inside the prompt (HISTORY) must not trip the guard."""
    stop = RepetitionStop(prompt_len=10)
    # All repetition is in the prompt region; the generated suffix is short/non-repeating.
    ids = _ids([7, 7, 7, 7, 7, 7, 7, 7, 7, 7], [1, 2, 3])
    assert stop(ids, None) is False


def test_helper_returns_stopping_criteria_list():
    crit = repetition_stopping_criteria(5)
    assert len(crit) == 1
    assert isinstance(crit[0], RepetitionStop)
