# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Bind implementation functions from modules onto a facade class."""

from __future__ import annotations

import inspect
from types import ModuleType

_SKIP_NAMES = frozenset(
    {
        "init_memory",
        "init_executor",
        "_logger",
        "Logger",
        "Any",
        "Callable",
        "Path",
        "Image",
        "np",
        "os",
        "re",
        "replace",
        "field",
        "dataclass",
        "Parameters",
    }
)


def bind_module_methods(cls: type, mod: ModuleType, *, skip: frozenset[str] | None = None) -> None:
    """Attach ``self``/``cls`` functions and descriptors defined in ``mod`` onto ``cls``."""
    deny = _SKIP_NAMES | (skip or frozenset())
    for name, obj in vars(mod).items():
        if name in deny or name.startswith("__"):
            continue
        owner = inspect.getmodule(obj)
        if isinstance(obj, (property, staticmethod, classmethod)):
            setattr(cls, name, obj)
            continue
        if inspect.isfunction(obj):
            if owner is not None and owner is not mod:
                continue
            params = list(inspect.signature(obj).parameters)
            if params and params[0] in ("self", "cls"):
                setattr(cls, name, obj)
            continue
        if name == "_FIXTURE_LABEL_TOKENS":
            setattr(cls, name, obj)
