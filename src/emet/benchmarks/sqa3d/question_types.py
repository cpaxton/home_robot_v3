# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""SQA3D question-type breakdown (first-word heuristic)."""

from __future__ import annotations

QUESTION_TYPE_NAMES = ("what", "is", "how", "can", "which", "other")


def question_type_index(question: str) -> int:
    """Map question first word to SQA3D breakdown index (0=what … 5=other)."""
    parts = question.strip().split()
    if not parts:
        return 5
    first = parts[0].lower().rstrip("?,.")
    mapping = {"what": 0, "is": 1, "how": 2, "can": 3, "which": 4}
    return mapping.get(first, 5)
