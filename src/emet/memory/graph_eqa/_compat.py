# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Copy an implementation module onto a compatibility-shim globals dict."""

from __future__ import annotations

import sys
from types import ModuleType

_SKIP = frozenset(
    {
        "__name__",
        "__package__",
        "__spec__",
        "__loader__",
        "__cached__",
        "__builtins__",
    }
)


def install_shim(globals_dict: dict[str, object], impl: ModuleType) -> None:
    """Re-export ``impl`` so old import paths and ``mock.patch`` still work."""
    for key, value in vars(impl).items():
        if key not in _SKIP:
            globals_dict[key] = value
    name = globals_dict.get("__name__")
    if isinstance(name, str):
        sys.modules[name] = impl
