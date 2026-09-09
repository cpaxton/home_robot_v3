# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Episode state for :class:`AgenticEQAExecutor`.

Init fields and loop counters live here. The executor facade forwards instance
attribute reads/writes onto the session so tool bodies can keep using ``self._tried``.
"""

from __future__ import annotations


class AgenticSession:
    """Mutable per-episode state (hypotheses, verify flags, traces, policy knobs)."""

    # Fields are assigned by ``init_executor`` via the executor facade.
