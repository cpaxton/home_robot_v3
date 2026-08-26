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
        action ("" if done, an attached Image slot 1..K, a graph obs id to navigate, or
        "read N" to re-inspect attached Image N),
        confidence_reasoning (string).
        Copy the selected option text into "answer" without its letter label. Never leave
        "answer" blank. If uncertain, still give your best semantic answer and set
        "confidence" to false.

        ATTACHED_INDEX maps Image 1..K (the RGB frames in this prompt) to graph obs ids.
        SCENE_GRAPH, CONFIRMED_MEMORY, GRAPH_COUNT, and FIND_QUEUE are an index of views
        to inspect, not the answer. They list other views as [graph obs N] / obsN — those
        numbers are navigation targets, not attached slots.
        Identify and count from attached RGB. Never answer a count by counting SCENE_GRAPH
        nodes, CONFIRMED_MEMORY rows, GRAPH_COUNT list length, or FIND_QUEUE length.
        Never answer WHERE from a SCENE_GRAPH node index or xyz. Detector class names are
        proposals for WHERE to look.

        Action contract:
        - "" — done, or explore a new room when attached views are spent/risky.
        - N or "Image N" — if N is 1..K, inspect that attached slot; if N is a graph obs id
          from FIND_QUEUE / GRAPH_COUNT, navigate to that view (example: action: 37).
        - "read N" — N is an attached slot 1..K only. Re-inspect that frame. Do not
          "read" a graph obs id.

        VIEW_STATUS lists per-Image investigation counters (visits, look/read picks, Unknown
        answers, spent/risky flags). More than three visits on one Image without progress is
        risky — pick a different Image, set action to an unattached FIND obs id, or leave
        action empty so the robot can explore. Do not keep answering Unknown from a view
        marked spent=yes or risky=yes.
"""

_HMEQA_MCQ_EXAMPLE = """
        Example (multiple choice):
            Input:
                Question: Is the lamp in the living room on or off?
                A. The lamp is on.
                B. The lamp is off.
                C. There is no lamp.
                D. The lamp is broken.
                ATTACHED_INDEX: Image 1=obs12
                SCENE_GRAPH: Node 3: floor lamp at (1.2, -0.4) [graph obs 12]
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
                ATTACHED_INDEX: Image 1=obs8, Image 2=obs22
                CONFIRMED_MEMORY: woven basket: LOOK — candidate views: woven basket [graph obs 22] at (-6.5, 3.6); list length is not a count; verify in attached images
                SCENE_GRAPH: Node 7: woven basket at (-6.5, 3.6) [graph obs 22]
                IMAGE: <2 RGB frames>
            Output:
                {"reasoning": "Image 2 shows the basket next to the armchairs.", "answer": "Next to the living room armchairs", "confidence": true, "action": "", "confidence_reasoning": "Visible next to the armchairs in Image 2, not from Node 7's coordinates."}

        Example (count — navigate to a FIND view, then count from attached RGB):
            Input:
                Question: How many bedside tables are there in the bedroom?
                A) Three B) One C) None D) Two
                ATTACHED_INDEX: Image 1=obs4
                FIND_QUEUE: obs37 bedside table spent=no attached=no
                GRAPH_COUNT: views to look at: obs37 (navigate via action=<obs id>; not an exact count)
                IMAGE: <1 RGB frame of a living room>
            Output:
                {"reasoning": "Attached Image 1 is a living room, not the bedroom. FIND_QUEUE lists unattached obs 37.", "answer": "Unknown", "confidence": false, "action": "37", "confidence_reasoning": "Navigate to graph obs 37 before answering None or a count."}

        Example (count — look at the listed attached views; do not copy a graph count):
            Input:
                Question: How many table lamps are there in the bedroom?
                A) Three B) Four C) One D) Two
                ATTACHED_INDEX: Image 1=obs3, Image 2=obs4
                CONFIRMED_MEMORY: table lamps: LOOK — candidate views: [graph obs 3] at (4.8, 5.1); [graph obs 4] at (3.1, 5.0); list length is not a count; verify in attached images
                IMAGE: <2 RGB frames of bedside lamps>
            Output:
                {"reasoning": "Image 1 and Image 2 each show one bedside lamp.", "answer": "Two", "confidence": true, "action": "", "confidence_reasoning": "Two close views of table lamps, not furniture."}

        Example (read — target is in the attached frame but not legible):
            Input:
                Question: What does the sign on the door say?
                A) Exit B) Open C) Closed D) Unknown
                ATTACHED_INDEX: Image 1=obs10, Image 2=obs11
                IMAGE: <2 RGB frames; Image 2 shows a distant sign>
            Output:
                {"reasoning": "Image 2 shows a sign but the letters are too small to read.", "answer": "Unknown", "confidence": false, "action": "read 2", "confidence_reasoning": "Need a closer view of attached Image 2; do not explore a new room."}
"""

_HMEQA_FORMAT_OVERRIDE = """
        OUTPUT FORMAT — this governs every example above:

        1. Do NOT write a caption field or Caption: block. Your first token continues a JSON object.
        2. "reasoning" is at most three sentences. Do not re-list objects from the attached RGB
           frames, and do not copy scene-graph coordinates. Cite an Image N when that frame
           shows the evidence. SCENE_GRAPH labels and node indices do not decide the answer.
        3. "answer" is the exact semantic option text without an A/B/C/D label.
           Emit it before elaborating further and never leave it blank.
        4. "confidence" is true or false. "action" is "", an attached slot 1..K, a graph
           obs id from FIND_QUEUE, or "read N" (attached slot only). Use "read N" when
           attached Image N shows the thing to read but the text or digits are not large
           and unambiguous. Do not guess. "confidence_reasoning" is one short sentence.
        5. For "What time is it now?" and other clock/time MCQs: do not guess a time bucket
           from a tiny clock in a wide frame. Use "read N" on the attached slot that shows
           the clock, or set action to a FIND_QUEUE obs id for a closer view. Leave "action"
           empty only when the hands or digits are clearly legible.

        You have a limited output budget. Spending it describing images instead of answering
        counts as a wrong answer.
"""

HMEQA_EQA_PROMPT = _HMEQA_HEADER + without_caption(EQA_PROMPT) + _HMEQA_MCQ_EXAMPLE + _HMEQA_FORMAT_OVERRIDE
