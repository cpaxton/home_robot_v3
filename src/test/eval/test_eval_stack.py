# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for shared memory-agent stack helpers."""

from __future__ import annotations

from emet.eval.stack import compose_eqa_question


def test_compose_eqa_question_empty_extra():
    assert compose_eqa_question("Where is the lamp?", None) == "Where is the lamp?"
    assert compose_eqa_question("Where is the lamp?", "  ") == "Where is the lamp?"


def test_compose_eqa_question_appends_additional_instructions():
    out = compose_eqa_question("Q?", "Answer with a single letter.")
    assert out.startswith("Q?")
    assert "Additional instructions:" in out
    assert "Answer with a single letter." in out


def test_compose_eqa_question_identical_for_same_inputs():
    a = compose_eqa_question("What color?", "Be concise.")
    b = compose_eqa_question("What color?", "Be concise.")
    assert a == b
