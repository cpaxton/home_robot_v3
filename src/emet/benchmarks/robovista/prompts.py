# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""RoboVista standard letter-only MCQ prompt (not Habitat explore/EQA)."""

from __future__ import annotations

from emet.benchmarks.robovista.datasets import RoboVistaQuestion
from emet.habitat.metrics import MCQ_LETTERS


def format_choices_block(choices: list[str]) -> str:
    lines: list[str] = []
    for idx, choice in enumerate(choices[: len(MCQ_LETTERS)]):
        letter = MCQ_LETTERS[idx]
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def build_robovista_prompt(question: RoboVistaQuestion) -> str:
    """Build a zero-shot prompt matching RoboVista's standard (letter-only) setup."""
    n_img = len(question.images)
    media = (
        "Use the attached robot-centric image(s) as the only visual context."
        if n_img
        else "No image was attached; answer from the question text if possible."
    )
    if n_img > 1:
        media += f" There are {n_img} images; consider all of them."
    choices = format_choices_block(question.choices)
    letters = ", ".join(MCQ_LETTERS[: max(1, min(len(question.choices), len(MCQ_LETTERS)))])
    return (
        f"{media}\n\n"
        f"Question:\n{question.question.strip()}\n\n"
        f"Choices:\n{choices}\n\n"
        f"Reply with exactly one letter ({letters}). "
        "Put the letter on its own line after 'Answer:'.\n"
        "Answer:"
    )
