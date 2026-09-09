# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""No-op visualizer used when live Rerun is off (no rerun-sdk import)."""

from __future__ import annotations

from typing import Any


def _null_noop(*args, **kwargs):
    return None


def visualizer_is_enabled(visualizer: Any | None) -> bool:
    """True only when ``enabled is True`` (a live ``RerunVisualizer``).

    ``is True`` rejects ``MagicMock.enabled`` so GraphEQA tests do not import
    rerun-sdk native extensions.
    """
    return visualizer is not None and getattr(visualizer, "enabled", False) is True


class NullVisualizer:
    """Drop-in replacement for RerunVisualizer that silently ignores all calls.

    Used when Rerun is disabled so callers do not need null-checks at every site.
    Kept in this module so CPU tests can import it without loading rerun-sdk
    native extensions (``_enums`` / ``_functions``).
    """

    enabled = False

    def __getattr__(self, name):
        return _null_noop
