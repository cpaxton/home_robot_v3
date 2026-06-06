# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""EQA system prompt for HM-EQA multiple-choice (A–D) questions in Habitat."""

from emet.llms.prompts.eqa_prompt import EQA_PROMPT

_HMEQA_HEADER = """
        HM-EQA questions are multiple-choice. The question text includes options A, B, C, and D.
        Your final Answer must be exactly one letter: A, B, C, or D (not yes/no prose).
        Always output these fields in order with lowercase labels on their own lines:
        caption:, reasoning:, answer:, confidence:, action:, confidence_reasoning:
        The answer: line must contain only a single letter (A, B, C, or D).
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
"""

HMEQA_EQA_PROMPT = _HMEQA_HEADER + EQA_PROMPT + _HMEQA_MCQ_EXAMPLE
