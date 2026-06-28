# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""EQA system prompt for HM-EQA multiple-choice (A–D) questions in Habitat."""

from emet.llms.prompts.eqa_prompt import EQA_PROMPT

_HMEQA_HEADER = """
        HM-EQA questions are multiple-choice. The question text includes options A, B, C, and D.
        Your final Answer must be exactly one letter: A, B, C, or D (not yes/no prose).
        Always output these fields in order with lowercase labels on their own lines:
        caption:, reasoning:, answer:, confidence:, action:, confidence_reasoning:
        Caption only the attached images (Image 1 .. Image N in the user message). One short phrase per image.
        Do not caption scene-graph nodes or IMAGE_DESCRIPTIONS entries that are not attached as images.
        After a brief caption, you must output reasoning:, answer:, confidence:, action:, and confidence_reasoning:.
        The answer: line must contain only a single letter (A, B, C, or D). Never leave answer: blank.
        If uncertain, still output your best-guess letter on answer: and set confidence: FALSE.
"""

_HMEQA_MCQ_EXAMPLE = """
        Example (multiple choice):
            Input:
                Question: Is the lamp in the living room on or off?
                A. The lamp is on.
                B. The lamp is off.
                C. There is no lamp.
                D. The lamp is broken.
                IMAGE: <2 images>
                IMAGE_DESCRIPTIONS: <10 image descriptions>
            Output:
                Caption:
                    Image 1 shows a living room with a floor lamp; the shade is lit.
                Reasoning:
                    The lamp shade is glowing, so the lamp is on. That matches option A.
                Answer:
                    A
                Confidence:
                    TRUE
                Action:
                Confidence_reasoning:
                    I can see the lit lamp clearly in Image 1.

        Example (visibility + location — pick WHERE, not yes/no):
            Input:
                Question: Did you see the woven basket anywhere?
                A) By the kitchen counter
                B) Between TV and living room sofas
                C) Next to the dining table
                D) Next to the living room armchairs
                CONFIRMED_MEMORY: woven basket: PRESENT — 1 graph node(s) at (-6.5, 3.6)
                IMAGE: <2 images>
            Output:
                Caption:
                    Image 1 shows a living room seating area; the basket is not in frame.
                Reasoning:
                    CONFIRMED_MEMORY shows the woven basket was observed near (-6.5, 3.6).
                    That location is closest to the living room armchairs (option D).
                Answer:
                    D
                Confidence:
                    TRUE
                Action:
                Confidence_reasoning:
                    Graph memory confirms the basket; option D matches that area.
"""

HMEQA_EQA_PROMPT = _HMEQA_HEADER + EQA_PROMPT + _HMEQA_MCQ_EXAMPLE
