# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emet.agent.skills.eqa import bind_eqa_skill
from emet.agent.skills.specs import CHAT_SKILL_SPECS
from emet.agent.tool_outcome import ToolOutcome
from emet.controller.task.tamp.api import response


@pytest.mark.parametrize("tool", ["move_forward", "rotate_base"])
@pytest.mark.parametrize("encoding", [lambda x: x, json.dumps, str])
def test_motion_failure_never_becomes_success_after_encoding(tool, encoding):
    receipt = response(tool, code="motion_rejected")
    result = ToolOutcome.coerce(tool, encoding(receipt))
    assert not result.ok
    assert result.status in {"motion_rejected", "invalid_tool_result"}


def test_bound_eqa_skill_preserves_structured_result():
    receipt = response("move_forward", code="motion_rejected")
    executor = SimpleNamespace(handle_tool=Mock(return_value=receipt))
    spec = next(spec for spec in CHAT_SKILL_SPECS if spec.name == "move_forward")
    result = bind_eqa_skill(executor, spec).func(meters=0.2)
    assert result is receipt
    assert not ToolOutcome.coerce("move_forward", result).ok


def test_valid_json_motion_success_still_parses():
    assert ToolOutcome.coerce("move_forward", json.dumps(response("move_forward"))).ok
