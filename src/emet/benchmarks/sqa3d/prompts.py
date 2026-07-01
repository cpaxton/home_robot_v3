# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Prompt helpers for SQA3D situated QA."""

from __future__ import annotations

from emet.llms.prompts.eqa_prompt import EQA_PROMPT

_SQA3D_HEADER = """
        SQA3D questions are situated: you are given a natural-language description of where
        you stand in a 3D scene (the situation) and a follow-up question about your surroundings.
        Answer with a short phrase (color, count, yes/no, object name, etc.) — not multiple choice.
        Always output these fields in order with lowercase labels on their own lines:
        caption:, reasoning:, answer:, confidence:, action:, confidence_reasoning:
        The answer: line must be a concise phrase matching the question type.
"""

_SQA3D_EXAMPLE = """
        Example (situated open answer):
            Input:
                Situation: I am facing a window and there is a desk on my right and a chair behind me.
                Question: What color is the desk to my right?
                IMAGE: <2 images>
                IMAGE_DESCRIPTIONS: <10 image descriptions>
            Output:
                Caption:
                    Image 1 shows a wooden desk to the right of the agent near a window.
                Reasoning:
                    The desk surface appears brown wood in the nearest view.
                Answer:
                    brown
                Confidence:
                    TRUE
                Action:
                Confidence_reasoning:
                    The desk color is visible in Image 1.
"""

SQA3D_EQA_PROMPT = _SQA3D_HEADER + EQA_PROMPT + _SQA3D_EXAMPLE


def format_sqa3d_prompt(situation: str, question: str) -> str:
    """Format situation + question for GraphEQA / Dynagraph ``query_answer``."""
    situation = situation.strip()
    question = question.strip()
    if situation and question:
        return f"Situation: {situation}\nQuestion: {question}"
    return situation or question
