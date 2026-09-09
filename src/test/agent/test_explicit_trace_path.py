# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from unittest.mock import MagicMock

import pytest

from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor


@pytest.mark.parametrize("explicit,expected", [(None, True), (True, True), (False, False)])
def test_trace_destination_enables_collection_unless_explicitly_disabled(tmp_path, explicit, expected):
    agent = MagicMock()
    agent.parameters = {"eqa": {"collect_agentic_trace": False}}
    agent.graph_memory.get_nodes.return_value = []
    executor = AgenticEQAExecutor(
        agent, "Where is the cup?", router=False, trace_path=tmp_path / "trace.jsonl", collect_trace=explicit
    )
    assert executor._collect_trace is expected
