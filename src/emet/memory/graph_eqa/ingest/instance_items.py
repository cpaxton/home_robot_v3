# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Normalize live RGB-D instance records into ``(label, xyz, bbox, identity_key)``."""

from __future__ import annotations

from typing import Any


def unpack_instance_item(item: Any) -> tuple[Any, Any, Any, Any]:
    """Return ``(label, xyz, bbox_xyxy, identity_key)`` from tuples, dicts, or objects."""
    if hasattr(item, "label") and hasattr(item, "xyz"):
        return (
            item.label,
            item.xyz,
            getattr(item, "bbox_xyxy", None),
            getattr(item, "identity_key", None),
        )
    if isinstance(item, dict):
        return (
            item.get("label", "object"),
            item["xyz"],
            item.get("bbox_xyxy"),
            item.get("identity_key"),
        )
    if len(item) >= 4:
        return item[0], item[1], item[2], item[3]
    if len(item) >= 3:
        return item[0], item[1], item[2], None
    return item[0], item[1], None, None
