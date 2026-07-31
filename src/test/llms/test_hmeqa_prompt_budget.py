# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Guards on the HM-EQA decode budget.

In the 2026-07-29 bal-32 run 31 of 32 EQA generations ran into the 256-token cap and
21 never emitted ``answer:``, because a caption block re-listing IMAGE_DESCRIPTIONS
consumed ~40% of the output. These tests pin the two things that fixed it so a future
prompt edit cannot quietly reintroduce the caption or drop the cap back down.
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from emet.core.parameters import get_parameters
from emet.llms.eqa_vl_settings import resolve_eqa_answer_max_new_tokens
from emet.llms.prompts.eqa_prompt import EQA_PROMPT, without_caption
from emet.llms.prompts.hmeqa_eqa_prompt import HMEQA_EQA_PROMPT

# A demonstrated output field, i.e. "Caption:" alone on its line. Prose that merely
# names the field (as the override does) is not a demonstration.
_CAPTION_FIELD = re.compile(r"^\s*Caption:\s*$", re.MULTILINE)


def test_no_example_anywhere_demonstrates_a_caption():
    """
    Overriding the examples is not enough.

    The 2026-07-30 q2 probe appended "do not caption" after them and still spent 48% of
    the output on a caption block, because the shared prompt demonstrates one five times.
    """
    assert not _CAPTION_FIELD.search(HMEQA_EQA_PROMPT)
    assert "first caption each image" not in HMEQA_EQA_PROMPT
    assert "IMAGE_DESCRIPTIONS:" not in HMEQA_EQA_PROMPT


def test_hmeqa_prompt_forbids_a_caption_block():
    lowered = HMEQA_EQA_PROMPT.lower()
    assert "do not output a caption: field" in lowered
    assert "do not write a caption: block" in lowered
    # Examples use RGB + SCENE_GRAPH, not the legacy IMAGE_DESCRIPTIONS dump.
    assert "image_descriptions:" not in lowered


def test_hmeqa_examples_still_model_the_answer_fields():
    examples = HMEQA_EQA_PROMPT.split("Example (multiple choice):", 1)[1]
    assert examples.count("Answer:") >= 2
    assert "Confidence_reasoning:" in examples


def test_hmeqa_answer_call_prefills_reasoning():
    """Prompt edits alone left a 26% caption share; the decode must open on Reasoning:."""
    from emet.llms.base import AbstractVLLMClient
    from emet.llms.graph_eqa_vlm import GraphEQAVLMClient

    captured: dict = {}

    class _FakeVL(AbstractVLLMClient):
        def generate_multimodal(
            self,
            user_content,
            *,
            system_prompt=None,
            max_new_tokens=None,
            reset_context=True,
            verbose=False,
            image=None,
            assistant_prefill=None,
        ):
            captured["assistant_prefill"] = assistant_prefill
            captured["max_new_tokens"] = max_new_tokens
            return "the lamp is lit.\nanswer:\na\n"

    client = GraphEQAVLMClient(_FakeVL(""), system_prompt="sys", max_tokens=64)
    out = client(["Question: color?"], max_new_tokens=128, assistant_prefill="Reasoning:")
    assert captured.get("assistant_prefill") == "Reasoning:"
    assert captured.get("max_new_tokens") == 128
    assert "answer" in out.lower()


def test_without_caption_keeps_the_other_output_fields():
    stripped = without_caption(EQA_PROMPT)
    for field in ("Reasoning:", "Answer:", "Confidence:", "Action:", "Confidence_reasoning:"):
        assert stripped.count(field) >= EQA_PROMPT.count(field), field
    assert "Image 1 is one view of the table and there is no mug." not in stripped


def test_without_caption_leaves_other_variants_alone():
    """SQA3D and plain EQA still caption; only budget-capped variants strip it."""
    assert _CAPTION_FIELD.search(EQA_PROMPT)


def test_without_caption_fails_loudly_when_the_shared_prompt_changes():
    with pytest.raises(ValueError):
        without_caption("Some other prompt with no caption instruction.")


def test_hmeqa_prompt_still_requires_a_letter():
    lowered = HMEQA_EQA_PROMPT.lower()
    assert "never leave answer: blank" in lowered
    assert "exactly one letter" in lowered


def test_answer_decode_cap_default_leaves_room_for_an_answer(monkeypatch):
    """256 truncated 31 of 32 generations; the default must fit reasoning + answer."""
    monkeypatch.delenv("EMET_EQA_ANSWER_MAX_NEW_TOKENS", raising=False)
    assert resolve_eqa_answer_max_new_tokens(None) >= 384


def test_include_image_descriptions_defaults_off(monkeypatch):
    """RGB + SCENE_GRAPH are enough; the label dump is legacy and invites Caption: loops."""
    from emet.llms.eqa_vl_settings import resolve_eqa_include_image_descriptions

    monkeypatch.delenv("EMET_EQA_INCLUDE_IMAGE_DESCRIPTIONS", raising=False)
    assert resolve_eqa_include_image_descriptions(None) is False
    assert resolve_eqa_include_image_descriptions({"eqa_vl": {}}) is False
    assert resolve_eqa_include_image_descriptions({"eqa_vl": {"include_image_descriptions": True}}) is True
    monkeypatch.setenv("EMET_EQA_INCLUDE_IMAGE_DESCRIPTIONS", "1")
    assert resolve_eqa_include_image_descriptions({"eqa_vl": {"include_image_descriptions": False}}) is True
    monkeypatch.setenv("EMET_EQA_INCLUDE_IMAGE_DESCRIPTIONS", "0")
    assert resolve_eqa_include_image_descriptions({"eqa_vl": {"include_image_descriptions": True}}) is False


def test_shipped_config_disables_image_descriptions():
    params = get_parameters("dynav_config.yaml")
    assert params.get("eqa_vl/include_image_descriptions") in (False, "false", 0, "0")


def test_strip_caption_block_from_history():
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    raw = "Caption:\nImage 1 shows a sofa.\nImage 2 shows a rug.\nReasoning:\nthe lamp is lit.\nAnswer:\nA\n"
    stripped = GraphEQAMemory.strip_caption_block_from_history(raw)
    assert "Caption:" not in stripped
    assert stripped.lstrip().startswith("Reasoning:")
    assert "Answer:" in stripped
    # Already caption-free text is unchanged aside from leading whitespace.
    clean = "Answer:A\nReasoning:ok\n"
    assert GraphEQAMemory.strip_caption_block_from_history(clean) == clean


def test_query_answer_omits_image_descriptions_by_default():
    import numpy as np

    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    captured: dict = {}

    def fake_eqa(cmds, **kwargs):
        captured["cmds"] = cmds
        return "reasoning: r\nanswer: B\nconfidence: false\naction:\nconfidence_reasoning: x"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "wall",
        parameters={"eqa": {"prompt_variant": "hmeqa"}, "eqa_vl": {"include_image_descriptions": False}},
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["wall"])
    mem._relevant_objects = ["wall"]
    mem.query_answer("Is the wall blue? A) Yes B) No")
    text_cmds = [c for c in captured["cmds"] if isinstance(c, str)]
    assert not any(c.startswith("IMAGE_DESCRIPTIONS:") for c in text_cmds)
    assert any("Attached images:" in c for c in text_cmds)


def test_query_answer_can_restore_image_descriptions(monkeypatch):
    import numpy as np

    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    monkeypatch.setenv("EMET_EQA_INCLUDE_IMAGE_DESCRIPTIONS", "1")
    captured: dict = {}

    def fake_eqa(cmds, **kwargs):
        captured["cmds"] = cmds
        return "reasoning: r\nanswer: B\nconfidence: false\naction:\nconfidence_reasoning: x"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "wall",
        parameters={"eqa": {"prompt_variant": "hmeqa"}},
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["wall"])
    mem._relevant_objects = ["wall"]
    mem.query_answer("Is the wall blue? A) Yes B) No")
    text_cmds = [c for c in captured["cmds"] if isinstance(c, str)]
    assert any(c.startswith("IMAGE_DESCRIPTIONS:") for c in text_cmds)


def test_query_answer_prefills_reasoning_when_hmeqa_variant_on_parameters():
    """Regression: Habitat left prompt_variant unset, so Caption: kept winning the first token."""
    import numpy as np

    from emet.core.parameters import Parameters
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    captured: dict = {}

    def fake_eqa(cmds, **kwargs):
        captured.update(kwargs)
        return "the lamp is lit.\nanswer:\na\nconfidence:\ntrue\naction:\nconfidence_reasoning:\nclear.\n"

    params = Parameters(eqa={"prompt_variant": "hmeqa"}, eqa_vl={"include_image_descriptions": False})
    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "wall",
        parameters=params,
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["wall"])
    mem._relevant_objects = ["wall"]
    mem.query_answer("Is the wall blue? A) Yes B) No")
    assert captured.get("assistant_prefill") == "Reasoning:"


def test_append_eqa_history_strips_caption():
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    mem = GraphEQAMemory(eqa_client=lambda _c: "", image_description_client=lambda _x: "")
    mem._append_eqa_history("Caption:\nImage 1 is a chair.\nAnswer:A\nReasoning:seen\n")
    assert len(mem._history_outputs) == 1
    assert "Caption:" not in mem._history_outputs[0]
    assert "Answer:A" in mem._history_outputs[0]


def test_answer_decode_cap_is_tunable_per_model(monkeypatch):
    """A different VLM must be able to change the budget from config, not a code edit."""
    monkeypatch.delenv("EMET_EQA_ANSWER_MAX_NEW_TOKENS", raising=False)
    assert resolve_eqa_answer_max_new_tokens({"eqa_vl": {"answer_max_new_tokens": 768}}) == 768

    monkeypatch.setenv("EMET_EQA_ANSWER_MAX_NEW_TOKENS", "512")
    assert resolve_eqa_answer_max_new_tokens({"eqa_vl": {"answer_max_new_tokens": 768}}) == 512


def test_shipped_config_sets_the_answer_decode_cap():
    """The default config the agent actually loads must carry the tuned value."""
    params = get_parameters("dynav_config.yaml")
    assert int(params.get("eqa_vl/answer_max_new_tokens")) >= 384


def test_parse_answer_tolerates_a_missing_caption():
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    raw = "reasoning:\nthe lamp shade is lit.\nanswer:\na\nconfidence:\ntrue\naction:\nconfidence_reasoning:\nclear view.\n"
    with patch.object(GraphEQAMemory, "__init__", lambda self: None):
        gm = GraphEQAMemory()
    reasoning, answer, confidence, action, conf_reason = gm.parse_answer(raw)

    assert answer.strip().upper() == "A"
    assert confidence is True
    assert "lamp shade is lit" in reasoning
