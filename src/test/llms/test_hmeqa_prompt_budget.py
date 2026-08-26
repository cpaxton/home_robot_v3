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
from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

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
    assert "do not write a caption" in lowered or "do not output a caption" in lowered
    # Examples use RGB + SCENE_GRAPH, not the legacy IMAGE_DESCRIPTIONS dump.
    assert "image_descriptions:" not in lowered


def test_hmeqa_prompt_treats_graph_as_finder_not_answer():
    lowered = HMEQA_EQA_PROMPT.lower()
    assert "not the answer" in lowered
    assert "never answer a count by counting" in lowered
    assert "scene_graph labels" in lowered and "do not decide the answer" in lowered
    assert "never answer where from" in lowered
    assert "look" in lowered and "image n" in lowered
    assert "read n" in lowered


def test_hmeqa_prompt_teaches_read_action():
    lowered = HMEQA_EQA_PROMPT.lower()
    assert '"action": "read 2"' in lowered
    assert "not legible" in lowered


def test_hmeqa_prompt_separates_attached_slots_from_graph_obs():
    lowered = HMEQA_EQA_PROMPT.lower()
    assert "attached_index" in lowered
    assert "[graph obs" in lowered
    assert '"action": "37"' in lowered
    assert "read n" in lowered
    assert "graph obs id" in lowered


def test_hmeqa_prompt_teaches_view_status_risk():
    lowered = HMEQA_EQA_PROMPT.lower()
    assert "view_status" in lowered
    assert "risky" in lowered
    assert "spent=yes" in lowered or "spent" in lowered


def test_hmeqa_examples_still_model_the_answer_fields():
    examples = HMEQA_EQA_PROMPT.split("Example (multiple choice):", 1)[1]
    assert examples.count('"answer"') >= 2
    assert '"confidence_reasoning"' in examples
    assert '"reasoning"' in examples


def test_hmeqa_answer_call_prefills_reasoning():
    """Prompt edits alone left a 26% caption share; the decode must open on the JSON seed."""
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
            return '{"reasoning": "the lamp is lit.", "answer": "The lamp is on.", "confidence": true, "action": "", "confidence_reasoning": "clear"}'

    client = GraphEQAVLMClient(_FakeVL(""), system_prompt="sys", max_tokens=64)
    out = client(["Question: color?"], max_new_tokens=128, assistant_prefill='{"reasoning":')
    assert captured.get("assistant_prefill") == '{"reasoning":'
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


def test_hmeqa_prompt_requires_semantic_answer_text():
    lowered = HMEQA_EQA_PROMPT.lower()
    assert "never leave" in lowered and "answer" in lowered
    assert "semantic option text" in lowered
    assert "without an a/b/c/d label" in lowered
    assert "exactly one letter" not in lowered
    assert '"answer": "a"' not in lowered
    assert '"answer": "d"' not in lowered


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

    captured: dict = {}

    def fake_eqa(cmds, **kwargs):
        captured.update(kwargs)
        return (
            '{"reasoning": "the lamp is lit.", "answer": "Yes", "confidence": true, '
            '"action": "", "confidence_reasoning": "clear."}'
        )

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
    assert captured.get("assistant_prefill") == '{"reasoning":'


def test_append_eqa_history_strips_caption():
    mem = GraphEQAMemory(eqa_client=lambda _c: "", image_description_client=lambda _x: "")
    mem._append_eqa_history("Caption:\nImage 1 is a chair.\nAnswer:A\nReasoning:seen\n")
    assert len(mem._history_outputs) == 1
    assert "Caption:" not in mem._history_outputs[0]
    assert "Answer:A" in mem._history_outputs[0]


def test_query_answer_stores_history_outcome_line():
    import numpy as np

    def fake_eqa(cmds, **kwargs):
        return (
            '{"reasoning": "wall looks white", "answer": "B", "confidence": false, '
            '"action": "2", "confidence_reasoning": "need closer look"}'
        )

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "wall",
        parameters={"eqa": {"prompt_variant": "hmeqa"}},
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["wall"])
    mem._relevant_objects = ["wall"]
    mem.query_answer("Is the wall blue? A) Yes B) No")
    assert len(mem._history_outputs) == 1
    line = mem._history_outputs[0]
    assert line.startswith("Iter:")
    assert "answer=B" in line
    assert "salvage=0" in line
    assert "Caption:" not in line


def test_answer_format_defaults_json_for_hmeqa(monkeypatch):
    from emet.llms.eqa_vl_settings import resolve_eqa_answer_format, resolve_eqa_answer_prefill

    monkeypatch.delenv("EMET_EQA_ANSWER_FORMAT", raising=False)
    assert resolve_eqa_answer_format({"eqa": {"prompt_variant": "hmeqa"}}) == "json"
    assert resolve_eqa_answer_prefill({"eqa": {"prompt_variant": "hmeqa"}}) == '{"reasoning":'
    assert resolve_eqa_answer_format({"eqa": {"prompt_variant": "sqa3d"}}) == "labeled"
    assert resolve_eqa_answer_prefill({"eqa": {"prompt_variant": "sqa3d"}}) is None
    monkeypatch.setenv("EMET_EQA_ANSWER_FORMAT", "labeled")
    assert resolve_eqa_answer_format({"eqa": {"prompt_variant": "hmeqa"}}) == "labeled"
    assert resolve_eqa_answer_prefill({"eqa": {"prompt_variant": "hmeqa"}}) == "Reasoning:"


def test_build_eqa_prompt_text_truncates_history_first():
    history = [f"Iter: answer=A conf=false action=- salvage=0 | long reason {i} " + ("x" * 200) for i in range(6)]
    graph = "SCENE_GRAPH:\n" + "\n".join(
        [f"Node {i}: object{i} at (0.00, 0.00, 0.00) [Image {i}]" for i in range(1, 20)]
        + [f"  near({i}, {i + 1})" for i in range(1, 15)]
    )
    parts = GraphEQAMemory.build_eqa_prompt_text(
        question_line="Question: where?",
        history_entries=history,
        history_start_index=0,
        graph_str=graph,
        img_desc_str="Attached images: none",
        max_tokens=80,
    )
    joined = "\n".join(parts)
    assert GraphEQAMemory.estimate_eqa_prompt_tokens(joined) <= 80
    iters = [p for p in parts if p.startswith("Iteration_")]
    assert len(iters) < len(history)


def test_build_eqa_prompt_text_preserves_graph_count_after_merged_tail_trim():
    graph = (
        "SCENE_GRAPH:\n"
        "Node 1: umbrella at (0.00, 0.00, 0.50) [Image 1]\n"
        "CONFIRMED_MEMORY (present):\n"
        + "\n".join(f"- remembered fact {i} " + ("x" * 80) for i in range(8))
        + "\nRooms: unknown\n"
        "GRAPH_COUNT: candidate views for 'umbrellas' "
        "(close-look Qwen names when available, otherwise Image ids; not an exact count): "
        "[Image 1] at (0.0, 0.0)."
    )
    parts = GraphEQAMemory.build_eqa_prompt_text(
        question_line="Question: How many umbrellas?",
        graph_str=graph,
        img_desc_str="Attached images: none",
        max_tokens=80,
    )
    joined = "\n".join(parts)
    assert "GRAPH_COUNT:" in joined
    assert "GRAPH_COUNT: 1" not in joined
    assert GraphEQAMemory.estimate_eqa_prompt_tokens(joined) <= 80


def test_parse_answer_json_and_labeled_fallback():
    mem = GraphEQAMemory(eqa_client=lambda _x: "", image_description_client=lambda _x: "")
    raw = (
        '{"reasoning": "lamp is lit", "answer": "A", "confidence": true, '
        '"action": "", "confidence_reasoning": "clear view"}'
    )
    r, a, c, act, cr = mem.parse_answer(raw)
    assert a == "A"
    assert c is True
    assert "lamp" in r
    assert "clear" in cr

    fenced = "```json\n" + raw + "\n```"
    _, a2, _, _, _ = mem.parse_answer(fenced)
    assert a2 == "A"

    # Continuation after assistant prefill (remote OpenAI often omits the seed).
    cont = '"shade glows", "answer": "B", "confidence": false, "action": "3", "confidence_reasoning": "blurry"}'
    r3, a3, c3, act3, _ = mem.parse_answer(cont, prefer_json=True, json_prefill='{"reasoning":')
    assert a3 == "B"
    assert c3 is False
    assert act3.strip() == "3"

    # Truncated / non-JSON falls back to labeled scrape.
    labeled = "reasoning: need more\nanswer: C\nconfidence: FALSE\naction: 1\nconfidence_reasoning: unsure"
    _, a4, c4, act4, _ = mem.parse_answer(labeled)
    assert a4.strip().upper() == "C"
    assert c4 is False
    assert act4.strip() == "1"


def test_parse_answer_preserves_semantic_choice_text():
    mem = GraphEQAMemory(eqa_client=lambda _x: "", image_description_client=lambda _x: "")
    raw = (
        '{"reasoning": "The trash can is beside the refrigerator.", '
        '"answer": "Next to the refrigerator", "confidence": true, '
        '"action": "", "confidence_reasoning": "clear view"}'
    )

    _, answer, confident, _, _ = mem.parse_answer(raw)

    assert answer == "Next to the refrigerator"
    assert confident is True


def test_parse_answer_terse_letter_under_answer_cue():
    """Remote Qwen2-VL collapses under trailing ``Answer:`` cue → bare letter + terminator."""
    mem = GraphEQAMemory(eqa_client=lambda _x: "", image_description_client=lambda _x: "")

    # Prefill arm: emits ``A}``; JSON repair fails (unquoted letter) → terse fallback.
    _, a, c, act, _ = mem.parse_answer("A}", prefer_json=True, json_prefill='{"reasoning":')
    assert a == "A"
    assert c is False
    assert act == ""

    # No-prefill arm: emits ``A) Living room`` → terse fallback.
    _, a2, _, _, _ = mem.parse_answer("A) Living room")
    assert a2 == "A"


def test_parse_answer_tolerates_a_missing_caption():
    raw = "reasoning:\nthe lamp shade is lit.\nanswer:\na\nconfidence:\ntrue\naction:\nconfidence_reasoning:\nclear view.\n"
    with patch.object(GraphEQAMemory, "__init__", lambda self: None):
        gm = GraphEQAMemory()
    reasoning, answer, confidence, action, conf_reason = gm.parse_answer(raw, prefer_json=False)

    assert answer.strip().upper() == "A"
    assert confidence is True
    assert "lamp shade is lit" in reasoning


def test_image_descriptions_omit_labels_already_on_graph():
    import numpy as np

    mem = GraphEQAMemory(defer_llm_clients=True)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([1.0, 2.0, 0.5]), ["red pillow"])
    oid = int(mem._observations[0].obs_id)
    full = mem._get_image_descriptions_str([oid])
    assert "red pillow" in full
    slim = mem._get_image_descriptions_str([oid], omit_labels_for_obs={oid})
    assert "red pillow" not in slim
    assert "at (1.00, 2.00)" in slim


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
    assert int(params.get("eqa_vl/eqa_prompt_max_tokens")) >= 2500


def test_prompt_max_tokens_resolver(monkeypatch):
    from emet.llms.eqa_vl_settings import resolve_eqa_prompt_max_tokens

    monkeypatch.delenv("EMET_EQA_PROMPT_MAX_TOKENS", raising=False)
    assert resolve_eqa_prompt_max_tokens(None) == 2500
    assert resolve_eqa_prompt_max_tokens({"eqa_vl": {"eqa_prompt_max_tokens": 1000}}) == 1000
    monkeypatch.setenv("EMET_EQA_PROMPT_MAX_TOKENS", "500")
    assert resolve_eqa_prompt_max_tokens({"eqa_vl": {"eqa_prompt_max_tokens": 1000}}) == 500
