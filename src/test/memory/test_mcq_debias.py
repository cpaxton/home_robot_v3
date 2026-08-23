# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for MCQ choice-rotation debiasing (graph_eqa/mcq_debias.py)."""

import numpy as np

from emet.memory.graph_eqa import GraphEQAMemory
from emet.memory.graph_eqa.mcq_debias import (
    answer_is_unknownish,
    extract_single_letter,
    format_rotated_question,
    letter_to_original_index,
    match_freeform_to_choice,
    rotated_choice_order,
    tally_choice_votes,
)

CHOICES = ["1-3pm", "7-9am", "10-12pm", "5-7pm"]


def test_rotated_choice_order_covers_all_positions():
    # Across the 4 cyclic rotations, each original index appears at each letter exactly once.
    for pos in range(4):
        seen = {rotated_choice_order(4, r)[pos] for r in range(4)}
        assert seen == {0, 1, 2, 3}


def test_format_rotated_question():
    q = format_rotated_question("What time is it now?", CHOICES, 0)
    assert q == "What time is it now? A) 1-3pm B) 7-9am C) 10-12pm D) 5-7pm. Answer:"
    q1 = format_rotated_question("What time is it now?", CHOICES, 1)
    assert q1 == "What time is it now? A) 7-9am B) 10-12pm C) 5-7pm D) 1-3pm. Answer:"


def test_letter_to_original_index_round_trip():
    # In rotation r, letter position i shows original index (i + r) % 4.
    for r in range(4):
        order = rotated_choice_order(4, r)
        for i, letter in enumerate("ABCD"):
            assert letter_to_original_index(letter, r, 4) == order[i]
    assert letter_to_original_index("E", 0, 4) is None
    assert letter_to_original_index("", 2, 4) is None


def test_extract_single_letter():
    assert extract_single_letter("B") == "B"
    assert extract_single_letter("Answer: c") == "C"
    assert extract_single_letter("The answer is (D).") == "D"
    assert extract_single_letter("I cannot tell.") == ""


def test_tally_choice_votes_majority():
    assert tally_choice_votes([0, 0, 1, None], CHOICES) == 0
    # Tie prefers the prior (main un-rotated answer).
    assert tally_choice_votes([0, 0, 3, 3], CHOICES, prior_index=3) == 3
    assert tally_choice_votes([None, None, None, None], CHOICES) is None


def test_tally_choice_votes_tie_without_prior_is_no_signal():
    # A position-locked model un-rotates to a uniform split: must NOT default to A.
    assert tally_choice_votes([0, 1, 2, 3], CHOICES) is None
    assert tally_choice_votes([0, 0, 3, 3], CHOICES, prior_index=1) is None
    assert tally_choice_votes([0, 0, 3, 3], CHOICES, prior_index=None) is None


def test_tally_choice_votes_drops_placeholders():
    choices = ["(Do not choose this option)", "No", "Yes", "(Do not choose this option)"]
    # Placeholder votes (0, 3) are discarded; "No" wins.
    assert tally_choice_votes([0, 0, 0, 1], choices) == 1
    assert tally_choice_votes([0, 3, 0, 3], choices) is None


def test_match_freeform_to_choice():
    loc = [
        "It is above the kitchen cabinets",
        "It is above the dining table",
        "It is above the bed",
        "It is above the TV",
    ]
    assert match_freeform_to_choice("The clock is above the kitchen cabinets.", loc) == 0
    assert match_freeform_to_choice("over the dining table", loc) == 1
    # Ambiguous ("above" matches everything after stopword stripping) -> None.
    assert match_freeform_to_choice("it is above", loc) is None
    assert match_freeform_to_choice("", loc) is None
    assert match_freeform_to_choice("I cannot tell from here", loc) is None


def test_match_q11_semantic_answer_to_refrigerator_choice():
    choices = [
        "Next to the dining table",
        "Next to the sofa",
        "Next to the kitchen sink",
        "Next to the refrigerator",
    ]

    assert match_freeform_to_choice("Next to the refrigerator", choices) == 3
    assert match_freeform_to_choice("It is next to the refrigerator.", choices) == 3
    assert match_freeform_to_choice("Next to the", choices) is None


def test_match_freeform_to_choice_yes_no_and_placeholders():
    yn = ["(Do not choose this option)", "No", "Yes", "(Do not choose this option)"]
    assert match_freeform_to_choice("No, the blanket is not folded.", yn) == 1
    assert match_freeform_to_choice("Yes.", yn) == 2
    # Never matches a placeholder even when the words overlap.
    assert match_freeform_to_choice("do not choose", yn) is None


def test_match_freeform_to_choice_times():
    assert match_freeform_to_choice("It looks like 1-3pm.", CHOICES) == 0
    assert match_freeform_to_choice("morning, around 7-9am", CHOICES) == 1


def test_answer_unknownish_distinguishes_count_none_from_abstention():
    counts = ["Three", "One", "None", "Two"]
    states = ["On", "Off", "Unknown", "(Do not choose this option)"]

    assert not answer_is_unknownish("None", counts)
    assert answer_is_unknownish("None")
    assert answer_is_unknownish("Unknown", states)
    assert answer_is_unknownish("N/A", states)


class _BiasedClient:
    """Mock VLM: knows the right *choice text* in MCQ form, unhelpful free-form."""

    def __init__(self, gold_text: str, freeform_reply: str = "hmm, unclear"):
        self.gold_text = gold_text
        self.freeform_reply = freeform_reply
        self.calls: list[str] = []

    def __call__(self, commands):
        prompt = next(c for c in commands if isinstance(c, str))
        self.calls.append(prompt)
        if "few words" in prompt:
            return self.freeform_reply
        return self.gold_text if self.gold_text in prompt else "unclear"


def _memory_with_obs(client):
    mem = GraphEQAMemory(eqa_client=client, image_description_client=lambda x: "clock")
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["clock"])
    mem.last_eqa_obs_ids = [1]
    mem.last_eqa_parsed = ("", "b", False, "", "")
    return mem


def test_vote_mcq_letter_freeform_short_circuits():
    """A decisive free-form answer wins without any rotation calls."""
    client = _BiasedClient(gold_text="1-3pm", freeform_reply="It looks like 1-3pm")
    mem = _memory_with_obs(client)
    letter = mem.vote_mcq_letter("What time is it now?", CHOICES)
    assert letter == "A"
    assert len(client.calls) == 1
    assert mem.last_mcq_debias["freeform_match"] == "A"
    assert mem.last_mcq_debias["votes"] == []


def test_vote_mcq_letter_recovers_gold_a_via_rotations():
    """Unhelpful free-form falls back to semantic rotation voting."""
    client = _BiasedClient(gold_text="1-3pm")
    mem = _memory_with_obs(client)
    letter = mem.vote_mcq_letter("What time is it now?", CHOICES)
    assert letter == "A"
    assert len(client.calls) == 5  # 1 freeform + 4 rotations
    assert mem.last_mcq_debias["letter"] == "A"
    assert mem.last_mcq_debias["freeform_match"] is None
    assert mem.last_mcq_debias["votes"] == ["A", "A", "A", "A"]
    assert mem.last_mcq_debias["prior"] == "B"
    assert all("Do not output an option letter" in prompt for prompt in client.calls[1:])


def test_vote_mcq_letter_ignores_legacy_letter_only_replies():
    """A semantic re-ask does not reinterpret bare letter tokens as answer text."""

    class _PositionLocked:
        def __call__(self, commands):
            prompt = next(c for c in commands if isinstance(c, str))
            return "unclear" if "few words" in prompt else "B"

    mem = _memory_with_obs(_PositionLocked())
    mem.last_eqa_parsed = ("", "", False, "", "")  # no prior either
    letter = mem.vote_mcq_letter("What time is it now?", CHOICES)
    assert letter == ""


def test_vote_mcq_letter_no_client():
    mem = GraphEQAMemory(eqa_client=lambda x: "", image_description_client=lambda x: "", defer_llm_clients=True)
    mem.eqa_client = None
    assert mem.vote_mcq_letter("q", CHOICES) == ""
    assert mem.last_mcq_debias == {}
