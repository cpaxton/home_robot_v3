# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class VLInferenceKind:
    """Caller intent hints for DynaMem / EQA (documentation only; not enforced by clients).

    Pass at call sites (e.g. comments or kwargs) so engineers know expected token budget and prompt style;
    :class:`AbstractVLLMClient` implementations stay general-purpose.
    """

    SHORT_CAPTION = "short_caption"
    KEYWORDS_FROM_TEXT = "keywords_from_text"
    EQA_MULTI_IMAGE = "eqa_multi_image"


class AbstractPromptBuilder(ABC):
    """Abstract base class for a prompt generator."""

    def __init__(self, **kwargs):
        self.prompt_str = self.configure(**kwargs)

    def configure(self, **kwargs) -> str:
        """Configure the prompt with the given parameters, then return the prompt string."""

    def __str__(self) -> str:
        """Return the system prompt string for an LLM."""
        return self.prompt_str

    def __call__(self, kwargs: dict[str, Any] | None = None) -> str:
        """Return the system prompt string for an LLM."""
        if kwargs is not None:
            self.prompt_str = self.configure(**kwargs)
        return self.prompt_str

    def parse_response(self, response: str) -> Any:
        """Parse the response from the LLM. Usually does nothing."""
        return response

    def get_available_actions(self) -> list[str]:
        """Return a list of available actions."""
        return []


class AbstractLLMClient(ABC):
    """Abstract base class for a **text-centric** language client.

    Subclasses implement :meth:`__call__` for string (and optionally one image) turns, often with
    conversation history. For **vision-heavy**, multi-image, or per-turn system prompts, prefer
    :class:`AbstractVLLMClient` and :meth:`AbstractVLLMClient.generate_multimodal`.
    """

    def __init__(
        self,
        prompt: str | AbstractPromptBuilder,
        prompt_kwargs: dict[str, Any] | None = None,
    ):
        self.prompt_kwargs = prompt_kwargs
        self.reset()

        # If the prompt is a string, use it as the prompt. Otherwise, generate the prompt string.
        if prompt is None:
            self._prompt = ""
        elif isinstance(prompt, str):
            self._prompt = prompt
        else:
            if prompt_kwargs is None:
                prompt_kwargs = {}
            self._prompt = prompt(**prompt_kwargs)

    @property
    def system_prompt(self) -> str:
        """Return the system prompt string for an LLM."""
        return self._prompt

    def reset(self) -> None:
        """Reset the client state."""
        self.conversation_history: list[str | dict[str, str]] = []
        self._iterations = 0

    def is_first_message(self) -> bool:
        """Return True if the client has not yet sent a message."""
        return len(self.conversation_history) == 0

    @property
    def steps(self) -> int:
        """Return the number of steps taken by the client."""
        return self._iterations

    def add_history(self, message: Any) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append(message)

    def get_history(self) -> list[Any]:
        """Get the conversation history."""
        return self.conversation_history.copy()

    def get_history_as_str(self) -> str:
        """Return the conversation history as a string."""
        history = self.get_history()
        history_str = ""
        for item in history:
            if isinstance(item, str):
                history_str += item
            else:
                history_str += f"\n{item['role']}: {item['content']}"
        return history_str

    @abstractmethod
    def __call__(self, command: str, image: Image.Image | None = None, verbose: bool = False):
        """Interact with the language model to generate a plan."""

    def parse(self, content: str) -> list[tuple[str, str]]:
        """parse into list"""
        plan = []
        for command in content.split("\n"):
            action, target = command.split("=")
            plan.append((action, target))
        return plan


class AbstractVLLMClient(AbstractLLMClient):
    """Vision-language client for **stateless or reset-per-turn** multimodal inference.

    **Contract vs** :class:`AbstractLLMClient`: DynaMem and tools should call
    :meth:`generate_multimodal` with explicit ``system_prompt`` and ``max_new_tokens`` per turn.
    Implementations **must** honor ``reset_context=True`` (default) by clearing conversational state
    before each generation so one physical model can serve caption, keyword, and EQA roles safely.

    **Dedup identity**: :meth:`canonical_model_key` encodes ``family``, resolved HF id, ``device``, and
    (where applicable) quantization so :func:`emet.llms.vllm_registry.should_share_vllm` can compare
    run configs. :meth:`vllm_id` is an alias for the same string for readability at call sites.
    """

    @property
    def canonical_model_key(self) -> str:
        """Stable key for registry / dedup (subclasses override with ``family:hf:device:quant`` style)."""
        return f"{type(self).__name__}"

    @property
    def vllm_id(self) -> str:
        """Same as :meth:`canonical_model_key` (explicit name for registry consumers)."""
        return self.canonical_model_key

    @abstractmethod
    def generate_multimodal(
        self,
        user_content: str | list[Any],
        *,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        reset_context: bool = True,
        verbose: bool = False,
        image: Any | None = None,
        assistant_prefill: str | None = None,
    ) -> str:
        """One VL turn.

        ``user_content`` is a string or a list of strings / images (``PIL.Image`` or ``ndarray``) in
        model-specific order. When ``image`` is set, it is combined with text per subclass rules.
        If ``reset_context`` is True, clear history before this turn (default for DynaMem).

        ``assistant_prefill`` (optional) seeds the assistant turn so decode cannot open with a
        different field — used by HM-EQA to force ``Reasoning:`` and kill the caption habit.
        Subclasses that cannot honor it may ignore it.
        """

    def __call__(
        self,
        command: str | list[Any],
        image: Image.Image | None = None,
        verbose: bool = False,
        reset_context: bool = True,
        **kwargs: Any,
    ) -> str:
        max_tok = getattr(self, "max_tokens", None)
        if image is not None:
            import numpy as np

            return self.generate_multimodal(
                command,
                system_prompt=self.system_prompt or None,
                max_new_tokens=max_tok,
                reset_context=reset_context,
                verbose=verbose,
                image=np.asarray(image),
            )
        return self.generate_multimodal(
            command,
            system_prompt=self.system_prompt or None,
            max_new_tokens=max_tok,
            reset_context=reset_context,
            verbose=verbose,
            image=None,
        )
