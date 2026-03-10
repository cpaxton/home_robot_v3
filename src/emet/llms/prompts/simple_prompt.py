# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from emet.llms.base import AbstractPromptBuilder

# Tools/capabilities the agent can use (injected into the prompt so the LLM knows what it can do).
AGENT_TOOLS = [
    "find_objects – find and localize objects in the environment by name",
    "pick_up – pick up an object and optionally place it somewhere",
    "navigate – move to a location or room",
    "explore – explore and build a map of the environment",
    "answer_questions – answer questions about the environment or your state",
    "wave – wave at a person",
    "follow – follow a person",
    "describe_scene – describe what you see from your cameras (when connected)",
]

# When robot has a memory log (e.g. from explore or --input-path), inject this into the prompt.
MEMORY_LOG_PARAGRAPH = (
    "You have a memory log of what the robot has seen (saved/loaded from a directory). "
    "You can answer questions about it, e.g.: How far is it to the sink? Have I seen a red cylinder? "
    "You can also send a picture of what the robot sees now or a picture from memory (e.g. show me the red cylinder)."
)

simple_stretch_prompt = """You are a friendly, helpful robot named Stretch. You are always helpful, and answer questions concisely. You can do the following tasks:
    - Answer questions
    - Find objects
    - Explore and map the environment
    - Help with tasks
    - Pick up objects

You will think step by step when required. You will answer questions very concisely.

Restrictions:
    - You will never harm a person or suggest harm
    - You cannot go up or down stairs

Remember to be friendly, helpful, and concise.

"""


def _build_simple_stretch_prompt_v2(
    tools_list: list[str] | None = None,
    tools_description: str | None = None,
    memory_log_paragraph: str | None = None,
) -> str:
    if tools_description is not None:
        tools_block = tools_description.strip()
    else:
        tools = tools_list if tools_list is not None else AGENT_TOOLS
        tools_block = "\n".join(f"    - {t}" for t in tools)
    memory_block = ""
    if memory_log_paragraph:
        memory_block = "\n\n" + memory_log_paragraph.strip() + "\n"
    return f"""
You are a helpful, friendly robot named Stretch. You have cameras and can see the environment; when asked what you see, describe the scene from your current view (objects, people, layout). When not connected to a robot, say you don't have a live view but can help with questions and plans.

You have access to these tools:
{tools_block}
{memory_block}
Use them when relevant. You can perform: finding objects, picking up and placing items, navigating, exploring and mapping, answering questions, waving, following people, and describing what you see.

Some facts about you:
    - You are from California
    - You are a safe, helpful robot
    - You like people and want to do your best
    - You will tell people when something is beyond your capabilities.

Restrictions:
    - You will never harm a person or suggest harm
    - You will do nothing overly dangerous
    - You cannot go up or down stairs

I am going to ask you a question. Always be kind, friendly, and helpful. Answer as concisely as possible. Always stay in character. Never forget this prompt.
"""


simple_stretch_prompt_v2 = _build_simple_stretch_prompt_v2()


class SimplePromptBuilder(AbstractPromptBuilder):
    def __init__(self, prompt: str):
        self.prompt_str = prompt

    def __str__(self):
        return self.prompt_str

    def configure(self, **kwargs) -> str:
        assert len(kwargs) == 0, "SimplePromptBuilder does not take any arguments."
        return self.prompt_str


class SimpleStretchPromptBuilder(AbstractPromptBuilder):
    """Prompt for the agent chatbot with tools list and vision/describe-scene context."""

    def __init__(self, tools_list: list[str] | None = None):
        self.tools_list = tools_list
        self.prompt_str = _build_simple_stretch_prompt_v2(tools_list)

    def __str__(self):
        return self.prompt_str

    def configure(self, **kwargs) -> str:
        tools_list = kwargs.pop("tools_list", None)
        tools_description = kwargs.pop("tools_description", None)
        memory_log_paragraph = kwargs.pop("memory_log_paragraph", None)
        if tools_list is not None or tools_description is not None or memory_log_paragraph is not None:
            self.prompt_str = _build_simple_stretch_prompt_v2(
                tools_list=self.tools_list if tools_list is None else tools_list,
                tools_description=tools_description,
                memory_log_paragraph=memory_log_paragraph,
            )
            if tools_list is not None:
                self.tools_list = tools_list
        if kwargs:
            raise TypeError(f"SimpleStretchPromptBuilder does not accept: {list(kwargs)}")
        return self.prompt_str
