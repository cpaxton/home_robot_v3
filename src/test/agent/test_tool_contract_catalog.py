# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Keep the human tool inventory complete as the executable registry changes."""

import re
from pathlib import Path

from emet.agent.skills.specs import CHAT_SKILL_SPECS, EQA_SKILL_SPECS


def test_contract_catalog_covers_every_registry_name_once():
    document = (Path(__file__).resolve().parents[3] / "docs/agent_tool_contracts.md").read_text()
    inventory = document.split("<!-- tool-inventory:start -->", 1)[1].split("<!-- tool-inventory:end -->", 1)[0]
    names = re.findall(r"^\| `([a-z_]+)` \|", inventory, flags=re.MULTILINE)
    assert len(names) == len(set(names))
    assert set(names) == {spec.name for spec in (*CHAT_SKILL_SPECS, *EQA_SKILL_SPECS)}
