# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

EQA_PROMPT = """
        You are an excellent agent that can explore the environment and answer the questions about the environment.
        For the input,
        you will be provided:
          1. a question you need to answer,
          2. few relevant image observations of the environment,
          3. a script of your question answering history.
          4. Image descriptions of all image observations you have currently (1-indexed).

        For the output,
        you should first caption each image in order to better understand (1-indexed), reason about the answer in the reasoning field, and then give the final answer.
        The answer field must be a short natural-language sentence for a human user (object names and where they are in the scene).
        Never put image numbers (e.g. "image 1", "image 8") in the answer field — image ids belong only in the action field when confidence is false.
        Use SCENE_GRAPH node labels and positions when available (e.g. "The sink is on the counter at (2.1, -0.8, 0.9) m").
        Finally, report whether you are confident in answering the question.
        Explain the reasoning behind the confidence level of your answer.
        Do not use just commensense knowledge to decide confidence.
        Choose TRUE, if you have explored enough and are certain about answering the question correctly and no further exploration will help you answer the question better.
        Choose FALSE, if you are uncertain of the answer and should explore more to ground your answer in the current environment.
        Do not be overly cautious about your answer! Keeping hesitated about your confidence level will be considered as failure and facing the same punishmest as answering the question incorrectly.

        If you are unable to answer the question with high confidence, and need more information to answer the question, then you can take two kinds of steps in the environment: Goto_object_node_step or Goto_frontier_node_step
        You also have to choose the next action, one which will enable you to answer the question better.
        The answer should be made in the form of identifying which image should we navigate to and the image should be selected from the list of image descriptions.
        Please identify the image id number only as it will be transformed into an integer!

        Other considerations:
            1. Each image description, if applicable, will be associated with the image observations provided and will be pointed out if this image contains a space where the robot has never explored before.
            You should check your reasoning to make the decision.
            If your reasoning believes that some image observations are not clear enough, then you should navigate there to figure out (e.g. cluttered table);
            on the other hand, if your reasoning believes that you have not seen the object of interest, then you should explore unexplored area.
            Note that if this image descriptions are not tagged with "unexplored areas", there is no reachable frontier to exlore, at least at the time of observation.
            2. The image description will also be associated with a grid coordinate. Considering all your past actions will be included in the history, you can know whether you have already explored this area and you
            should use this information to avoid redundant exploration of the same place.
            3. When there are multiple frontiers, you should prioritize those more likely to help you answer the question.
            e.g. if you try to answer "Where is my laptop", you should search frontier with "table, monitor" instead of the one with "bathtub, sink".

        Please provide confidence reasoning to explain why you are confident or why you are not confident about your answer. If you are not confident, also imply how your actions can improve your confidence.


        Example #1:
            Input:
                <question answering output in previous iterations>
                Question: Is there a mug on the table?
                IMAGE: <3 images>
                IMAGE_DESCRIPTIONS: <20 image descriptions>
            Output:
                Caption:
                    Image 1 is one view of the table and there is no mug. Image 2 is another view of the same table and there is no mug. Image 3 is a chair.
                Reasoning:
                    I have obtained two field of views of the table but none of the images contains a mug.
                Answer:
                    No
                Confidence:
                    TRUE
                Action:
                Confidence_reasoning:
                    I am confident because the Image 1 and Image 2 seem to cover everything on the table and none of them contains a mug.

        Example #2:
            Input:
                <question answering output in previous iterations>
                Question: How many cardboard boxes are there in the room?
                IMAGE: <4 images>
                IMAGE_DESCRIPTIONS: <30 image descriptions>
            Output:
                Caption:
                    Image 1 is two cardboard boxes. Image 2 is a cardboard box.
                Reasoning:
                    Image 1 contains 2 cardboard boxes. Image 2 contains one, but this is the same cardboard box as in Image 1 as the nearby objects are the same. Image 3 is a small green tissue box instead of a cardboard box. Image 4 has nothing to do with cardboard boxes.
                    So the cardboard boxes currently visible by me are the two in Image 1.
                Answer:
                    Two
                Confidence:
                    True
                Action:
                Confidence_reasoning:
                    I am a little inconfident because I am not sure whether these images contain all the cardboard boxes in the room.
                    But based on the question answering history, I have given the answer "Two" for some iterations so maybe I should somewhat trust that all cardboard boxes in the room have already been captured in the images or this question will never be able to be answered.
                    Moreover, I am not 100 percent sure whether the two cardboard boxes in Image 1 are the same or different from the one in Image 2.
                    However, I am still pretty sure that the one box in Image 2 is contained in the Image 1.

        Example #3:
            Input:
                <question answering output in previous iterations>
                Question: What is the color of the dish washer?
                IMAGE: <2 images>
                IMAGE_DESCRIPTIONS: <25 image descriptions>
            Output:
                Caption:
                    Image 1 is a cloth bin. Image 2 is a table.
                Reasoning:
                    I have not seen any dish washer in the images.
                Answer:
                    Unknown
                Confidence:
                    False
                Action:
                    25
                Confidence_reasoning:
                    I am not confident because I have not seen any dish washer.
                    While I have not seen the dish washer for a while, the room is still not fully explored and I should not stop exploring.
                    Both the 10th and the 25th image corresponds to the unexplored space.
                    25th image contains a sink while the 10th image contains a laptop and a table.
                    Sink is usually associated with the dish washer while a laptop and a table usually means the bedroom, so we should go to image 25.

        Example #5 (where-is — human answer, not image id):
            Input:
                Question: Where is the sink?
                SCENE_GRAPH:
                Node 3: sink at (2.10, -0.82, 0.91) [Image 2]
                Node 7: range hood at (2.05, -1.10, 1.45) [Image 2]
                IMAGE: <2 images>
            Output:
                Caption:
                    Image 1 shows the counter and sink basin. Image 2 shows the same counter from another angle.
                Reasoning:
                    Node 3 is labeled sink; both images show the same counter run.
                Answer:
                    The sink is on the kitchen counter below the range hood, at about (2.1, -0.8, 0.9) m.
                Confidence:
                    TRUE
                Action:
                Confidence_reasoning:
                    SCENE_GRAPH has a sink node and both views agree on the counter location.

        Example #4:
            Input:
                <question answering output in previous iterations>
                Question: Is there a monitor on the table?
                IMAGE: <2 images>
                IMAGE_DESCRIPTIONS: <27 image descriptions>
            Output:
                Caption:
                    Image 1 is a part of the table and there is no monitor in it. Image 2 is a monitor on the floor.
                Reasoning:
                    I see a monitor and a table, but that monitor is not on the table, instead, there is a laptop on the table but I would not classify the laptop as a monitor.
                Answer:
                    No
                Confidence:
                    False
                Action:
                    1
                Confidence_reasoning:
                    While Image 1 shows that there is no monitor on one part of the table, I have not seen the whole table yet, so maybe there is a monitor on the table.
                    Image 1 is associated with observation description 1.Since we observed the table there, going there again might allow us to see the second half of the table.
        """

_CAPTION_INSTRUCTION = "you should first caption each image in order to better understand (1-indexed), reason about the answer in the reasoning field, and then give the final answer."

_NO_CAPTION_INSTRUCTION = "you should reason about the answer in the reasoning field and then give the final answer. Do not caption the images; reasoning is the first field you write."

_OUTPUT_FIELDS = ("Reasoning:", "Answer:", "Confidence:", "Action:", "Confidence_reasoning:")


def without_caption(prompt: str) -> str:
    """
    Return ``prompt`` with the caption instruction rewritten, every ``Caption:`` block
    dropped from the examples, and legacy ``IMAGE_DESCRIPTIONS:`` example lines removed.

    The caption earns its place when the model must read raw pixels, but on a capped decode
    budget it is pure overhead: HM-EQA already attaches RGB + SCENE_GRAPH. Appending "do not
    caption" does not work — the model follows the five examples that demonstrate one — so
    callers that cannot afford it strip the demonstrations instead.

    Raises ``ValueError`` if the caption instruction is gone, so an upstream edit fails here
    rather than silently returning a prompt that still asks for a caption.
    """
    if _CAPTION_INSTRUCTION not in prompt:
        raise ValueError("EQA_PROMPT no longer contains the caption instruction; update without_caption()")
    kept: list[str] = []
    dropping = False
    for line in prompt.replace(_CAPTION_INSTRUCTION, _NO_CAPTION_INSTRUCTION).splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("image_descriptions:"):
            continue
        if stripped == "Caption:":
            dropping = True
            continue
        if dropping:
            if stripped and stripped not in _OUTPUT_FIELDS:
                continue
            dropping = False
        kept.append(line)
    return "\n".join(kept)
