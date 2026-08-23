# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Shared decoding guard for small (V)LM degeneration.

Small (vision-)language models used for GraphEQA/DynaMem planning frequently loop on a short
phrase (e.g. the Action line: "navigate to image navigate to image ...") until ``max_new_tokens``,
wasting most of the budget every planning step. A ``repetition_penalty`` is not used because it
also penalizes the prompt's HISTORY tokens and destabilizes the MCQ answer letter; instead we
halt generation when the *generated* tail degenerates into a short repeating cycle. The valid
structured fields (answer/confidence/first action) are emitted before the degenerate tail, so
parsing is unaffected.
"""

from __future__ import annotations

import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class DecodeProgressStop(StoppingCriteria):
    """Never stops; logs when decode leaves prefill and every ``every_n`` new tokens.

    Prefill hangs never call StoppingCriteria — pair with the thread heartbeat in
    ``qwen3_vl_client`` which covers that case.
    """

    def __init__(self, prompt_len: int, *, every_n: int = 32) -> None:
        self._prompt_len = int(prompt_len)
        self._every_n = max(1, int(every_n))
        self._last_logged = 0
        self._t0 = time.monotonic()
        self._prefill_logged = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        n = int(input_ids.shape[-1]) - self._prompt_len
        if n <= 0:
            return False
        if not self._prefill_logged:
            self._prefill_logged = True
            print(
                f"[vl] decode started after prefill elapsed={time.monotonic() - self._t0:.1f}s",
                flush=True,
            )
        if n - self._last_logged >= self._every_n or n == 1:
            self._last_logged = n
            print(
                f"[vl] decode tokens={n} elapsed={time.monotonic() - self._t0:.1f}s",
                flush=True,
            )
        return False


class HardTimeStop(StoppingCriteria):
    """Stop decode when wall-clock ``max_time_s`` is exceeded; sets ``fired``.

    Only runs during autoregressive steps (not vision/text prefill). Pair with the
    thread watchdog in ``qwen3_vl_client._generate_with_heartbeat`` for prefill hangs.
    """

    def __init__(self, max_time_s: float) -> None:
        self.max_time_s = float(max_time_s)
        self._t0 = time.monotonic()
        self.fired = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.max_time_s <= 0:
            return False
        if time.monotonic() - self._t0 >= self.max_time_s:
            self.fired = True
            return True
        return False


class RepetitionStop(StoppingCriteria):
    """Stop generation when the generated tail is a short repeating cycle.

    For each period ``1..max_period``, halts if the last ``period * reps`` generated tokens are
    ``reps`` identical consecutive blocks. Only the generated suffix (``input_ids[:, prompt_len:]``)
    is inspected, so repetition already present in the prompt never trips it.
    """

    def __init__(self, prompt_len: int, max_period: int = 6, reps: int = 5) -> None:
        self._prompt_len = int(prompt_len)
        self._max_period = int(max_period)
        self._reps = int(reps)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen = input_ids[0, self._prompt_len :]
        n = gen.numel()
        for period in range(1, self._max_period + 1):
            if n < period * self._reps:
                continue
            block = gen[-period:]
            ok = True
            for k in range(1, self._reps):
                if not torch.equal(gen[-period * (k + 1) : -period * k], block):
                    ok = False
                    break
            if ok:
                return True
        return False


def repetition_stopping_criteria(prompt_len: int, *, max_period: int = 6, reps: int = 5) -> StoppingCriteriaList:
    """``StoppingCriteriaList`` with decode progress + :class:`RepetitionStop` for ``model.generate``."""
    return StoppingCriteriaList(
        [
            DecodeProgressStop(int(prompt_len)),
            RepetitionStop(int(prompt_len), max_period=max_period, reps=reps),
        ]
    )
