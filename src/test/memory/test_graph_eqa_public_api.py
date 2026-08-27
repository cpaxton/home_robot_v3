# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Public graph-memory API lives on ``emet.memory.graph_eqa``."""

from __future__ import annotations

import emet.memory.graph_eqa as graph_eqa
from emet.memory.graph_eqa import (
    AgenticEQAExecutor,
    GraphEQAMemory,
    NavHypothesis,
    question_is_locate,
    run_agentic_eqa,
)


def test_product_names_are_the_implementation_objects():
    from emet.memory.graph_eqa.agentic_config import question_is_locate as locate_impl
    from emet.memory.graph_eqa.agentic_eqa import (
        AgenticEQAExecutor as ExecutorImpl,
    )
    from emet.memory.graph_eqa.agentic_eqa import (
        run_agentic_eqa as run_impl,
    )
    from emet.memory.graph_eqa.graph_memory import (
        GraphEQAMemory as MemoryImpl,
    )
    from emet.memory.graph_eqa.graph_memory import (
        NavHypothesis as HypImpl,
    )

    assert GraphEQAMemory is MemoryImpl
    assert AgenticEQAExecutor is ExecutorImpl
    assert NavHypothesis is HypImpl
    assert run_agentic_eqa is run_impl
    assert question_is_locate is locate_impl


def test_all_public_names_import():
    for name in graph_eqa.__all__:
        assert getattr(graph_eqa, name) is not None


def test_question_shape_helpers_on_package():
    assert question_is_locate("Where is the microwave?")
    assert not question_is_locate("Where is the sink? A) kitchen B) bath")
    from emet.memory.graph_eqa import question_has_mcq_options

    assert question_has_mcq_options("Where is the sink? A) kitchen B) bath")


def test_patch_on_implementation_module_still_wins():
    from unittest.mock import patch

    with patch("emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa_result", return_value="patched"):
        from emet.memory.graph_eqa import run_agentic_eqa_result

        assert run_agentic_eqa_result() == "patched"
