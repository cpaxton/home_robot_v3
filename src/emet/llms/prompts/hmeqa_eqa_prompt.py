# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""EQA system prompt for HM-EQA multiple-choice questions in Habitat.

The decode budget, not the reasoning, was the binding constraint here. In the
2026-07-29 bal-32 run 31 of 32 answer generations ran into the 256-token cap and 21
never emitted ``answer:`` at all, so a terse re-ask had to salvage the answer. The
cause was the caption: it took ~40% of the generated text on average (88% on q2) to
re-list IMAGE_DESCRIPTIONS that were already in the prompt.

Instructing the model not to caption does not work. The 2026-07-30 q2 probe appended
an override after the examples and still got a full caption block using 48% of the
output, because the shared :data:`EQA_PROMPT` asks for a caption once and demonstrates
one five times. So the caption is removed at the source with
:func:`without_caption`, and HM-EQA answers use a JSON contract (same shape as the
chat/router tool JSON) so field scrape cannot lose the semantic answer to truncation.
"""

from emet.llms.prompts.eqa_prompt import EQA_PROMPT, without_caption

_HMEQA_HEADER = """
        HM-EQA questions are multiple-choice. The question text includes options A, B, C, and D.
        Answer with the meaning of the selected option, not its A/B/C/D label.
        Reply with ONLY a single JSON object (no markdown fences, no caption field) with keys:
        reasoning (string), answer (short semantic answer text), confidence (boolean),
        action (image id string or ""),
        confidence_reasoning (string).
        Copy the selected option text into "answer" without its letter label. Never leave
        "answer" blank. If uncertain, still give your best semantic answer and set
        "confidence" to false.
"""

_HMEQA_MCQ_EXAMPLE = """
        Example (multiple choice):
            Input:
                Question: Is the lamp in the living room on or off?
                A. The lamp is on.
                B. The lamp is off.
                C. There is no lamp.
                D. The lamp is broken.
                SCENE_GRAPH: Node 3: floor lamp at (1.2, -0.4) [Image 1]
                IMAGE: <2 RGB frames>
            Output:
                {"reasoning": "Image 1 shows the floor lamp with a glowing shade, so the lamp is on.", "answer": "The lamp is on.", "confidence": true, "action": "", "confidence_reasoning": "The lit lamp is clearly visible in Image 1."}

        Example (visibility + location — pick WHERE, not yes/no):
            Input:
                Question: Did you see the woven basket anywhere?
                A) By the kitchen counter
                B) Between TV and living room sofas
                C) Next to the dining table
                D) Next to the living room armchairs
                CONFIRMED_MEMORY: woven basket: PRESENT — graph nodes: woven basket [Image 2] at (-6.5, 3.6); list length is not a count; verify in attached images
                SCENE_GRAPH: Node 7: woven basket at (-6.5, 3.6) [Image 2]
                IMAGE: <2 RGB frames>
            Output:
                {"reasoning": "Image 2 shows the basket next to the armchairs.", "answer": "Next to the living room armchairs", "confidence": true, "action": "", "confidence_reasoning": "The basket is visible next to the armchairs in Image 2."}

        Example (count — count from images; detector class names are not the answer):
            Input:
                Question: How many table lamps are there in the bedroom?
                A) Three B) Four C) One D) Two
                CONFIRMED_MEMORY: table lamps: PRESENT — graph nodes: [Image 3] at (4.8, 5.1); [Image 4] at (3.1, 5.0); list length is not a count; verify in attached images
                IMAGE: <2 RGB frames of bedside lamps>
            Output:
                {"reasoning": "Images 3 and 4 each show one bedside lamp.", "answer": "Two", "confidence": true, "action": "", "confidence_reasoning": "Two close views of table lamps, not furniture."}
"""

_HMEQA_FORMAT_OVERRIDE = """
        OUTPUT FORMAT — this governs every example above:

        1. Do NOT write a caption field or Caption: block. Your first token continues a JSON object.
        2. "reasoning" is at most three sentences. Do not re-list objects from the attached RGB
           frames, and do not copy scene-graph coordinates unless they decide the answer.
           Cite an Image N or SCENE_GRAPH node only when it decides the answer.
        3. "answer" is the exact semantic option text without an A/B/C/D label.
           Emit it before elaborating further and never leave it blank.
        4. "confidence" is true or false. "action" is an image id or "". "confidence_reasoning"
           is one short sentence.

        You have a limited output budget. Spending it describing images instead of answering
        counts as a wrong answer.
"""

HMEQA_EQA_PROMPT = _HMEQA_HEADER + without_caption(EQA_PROMPT) + _HMEQA_MCQ_EXAMPLE + _HMEQA_FORMAT_OVERRIDE
