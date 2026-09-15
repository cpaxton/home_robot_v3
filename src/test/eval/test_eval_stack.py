# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for shared memory-agent stack helpers."""

from __future__ import annotations

from emet.eval.stack import compose_eqa_question


def test_lazy_backend_cannot_be_overridden_to_streaming_instances():
    from unittest.mock import Mock, patch

    from emet.config.embodied_agent_config import EmbodiedAgentConfig
    from emet.eval.stack import build_memory_agent

    config = EmbodiedAgentConfig()
    config.graph_eqa_memory.enabled = True
    config.graph_eqa_memory.use_instance_graph = True
    with patch("emet.controller.controller_lazy_graph.LazyGraphController") as controller:
        build_memory_agent(
            robot=Mock(), parameters={}, backend="lazy_graph", embodied_agent=config, apply_harness_profile=False
        )
    assert controller.call_args.kwargs["use_instance_graph"] is False


def test_shared_query_preset_has_required_grounding_contract():
    from emet.core.parameters import get_parameters

    params = get_parameters("configs/emet/query_surface_pilot.yaml")
    assert params.get("query_driven_memory") is True
    assert params.get("query_memory")["grounding_backend"] == "vlm"
    assert params.get("query_memory")["region_strategy"] == "depth_candidates"
    assert params.get("eqa")["agentic_verify"] is True


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
