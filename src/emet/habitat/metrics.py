# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Episode metrics for Habitat EQA evaluation.

These helpers grade multiple-choice answers and serialize per-episode JSONL rows
compatible with GraphEQA-style Habitat benchmarks.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EpisodeMetrics:
    """One graded HM-EQA / Habitat EQA episode.

    Attributes:
        dataset: Dataset name (e.g. ``hmeqa``).
        method: Method label (e.g. ``graph_eqa``, ``dynagraph``).
        question_id: Zero-based question index in the CSV.
        scene: HM3D scene id.
        floor: Floor index for the episode.
        question: Question text (for logging).
        gold_answer_letter: Gold MCQ letter ``A``–``D``.
        predicted_answer: Raw model output string.
        correct: Whether the prediction matches gold (see :func:`grade_mcq_answer`).
        confident: Harness-level confidence flag (planner reached an answer).
        planning_steps: Number of navigation / EQA steps taken.
        success: Episode completed without fatal error.
        parsed_answer_letter: Letter extracted from ``predicted_answer``.
        model_confident: Optional model-reported confidence.
        raw_eqa_output: Full EQA trace / reasoning text when available.
    """

    dataset: str
    method: str
    question_id: int
    scene: str
    floor: int
    question: str
    gold_answer_letter: str
    predicted_answer: str
    correct: bool
    confident: bool
    planning_steps: int
    success: bool
    parsed_answer_letter: str = ""
    model_confident: bool = False
    raw_eqa_output: str = ""
    # Debug / reproducibility (full raw EQA also in per-episode bundle ``raw_eqa.txt``).
    choices: list[str] = field(default_factory=list)
    formatted_answer: str = ""
    eqa_action: str = ""
    eqa_confidence_reasoning: str = ""
    eqa_iterations: int = 0
    # MCQ choice-rotation debias (dynagraph): letter before the vote + JSON vote detail.
    predebias_letter: str = ""
    debias_votes: str = ""
    frontier_nodes: int = 0
    graph_nodes: int = 0
    observations: int = 0
    # Shared Dynagraph health snapshot (obj/vp/fr, singletons, prompt size, …).
    graph_health: dict = field(default_factory=dict)
    vl_family: str = ""
    vl_hf_model_id: str = ""
    debug_bundle_dir: str = ""
    topdown_map_path: str = ""
    diagnostics_manifest_path: str = ""
    error: str = ""
    # Harness fingerprint for reproducibility (merge / explore / git, etc.).
    harness: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (for JSONL export)."""
        return asdict(self)


_YES_NO_CHOICE_HINTS = frozenset(
    {
        "yes",
        "no",
        "true",
        "false",
        "partially",
        "cannot tell",
        "unknown",
        "on",
        "off",
    }
)


def parse_mcq_choices_from_question(question: str) -> list[str]:
    """Parse A–D option strings from HM-EQA ``question_formatted`` text."""
    text = (question or "").strip()
    if not text:
        return []
    choices = re.findall(
        r"[A-D]\)\s*(.+?)(?=\s[A-D]\)|\.\s*Answer:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return [c.strip().rstrip(".") for c in choices if c.strip()]


def question_is_visibility_location(question: str) -> bool:
    """True for stems like ``Did you see the woven basket anywhere?``."""
    head = (question or "").strip().split("?")[0].lower()
    return bool(
        re.search(
            r"\b(did you see|have you seen|do you see|can you see|did i see)\b",
            head,
        )
    )


def question_is_attribute_state(question: str) -> bool:
    """True for on/off, up/down, open/closed, light-state questions (image-first, not memory)."""
    head = (question or "").strip().split("?")[0].lower()
    return bool(
        re.search(
            r"\b("
            r"on or off|turned on|turned off|light on|lights? on|"
            r"pulled (down|up)|open or closed|opened or closed|"
            r"is the .+ (on|off)\b"
            r")",
            head,
        )
    )


def choices_are_attribute_state(choices: list[str] | None) -> bool:
    """True when options are mainly on/off / up/down style rather than places.

    Do not treat location prepositions as attributes: ``On the kitchen island``
    must stay a location MCQ (holdout q105), not trigger Unknown→letter salvage.
    """
    if not choices:
        return False
    cleaned = [(c or "").strip().lower() for c in choices[:4] if (c or "").strip()]
    if len(cleaned) < 2:
        return False
    real = [c for c in cleaned if not c.startswith("(do not choose")]
    if len(real) < 2:
        return False
    # Whole-choice / short state phrases only — not "on the …" place options.
    attr_exact = frozenset({"on", "off", "up", "down", "open", "closed", "pulled"})
    attr_phrase = re.compile(
        r"^(?:turned\s+)?(?:on|off)$|"
        r"^pulled\s+(?:up|down)$|"
        r"^(?:wide\s+)?(?:open|closed)$|"
        r"^lights?\s+(?:on|off)$"
    )
    hits = sum(1 for c in real if c in attr_exact or bool(attr_phrase.match(c)))
    return hits >= max(1, len(real) // 2)


def choices_are_count_mcq(choices: list[str] | None) -> bool:
    """True when options are cardinalities (``One``/``Two``/``None``/…)."""
    if not choices:
        return False
    cleaned = [(c or "").strip().lower() for c in choices[:4] if (c or "").strip()]
    if len(cleaned) < 2:
        return False
    real = [c for c in cleaned if not c.startswith("(do not choose")]
    if len(real) < 2:
        return False
    count_exact = frozenset(
        {
            "none",
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
        }
    )
    hits = sum(1 for c in real if c in count_exact or re.fullmatch(r"\d+", c))
    return hits >= max(2, (len(real) + 1) // 2)


def choices_are_location_mcq(choices: list[str] | None) -> bool:
    """True when MCQ options are places/things, not yes/no or count answers."""
    if not choices:
        return False
    cleaned = [(c or "").strip().lower() for c in choices[:4] if (c or "").strip()]
    if len(cleaned) < 2:
        return False
    if all(c.startswith("(do not choose") for c in cleaned):
        return False
    if choices_are_count_mcq(choices):
        return False
    yes_no_like = sum(
        1
        for c in cleaned
        if c in _YES_NO_CHOICE_HINTS or any(h in c.split() for h in ("yes", "no", "true", "false"))
    )
    return yes_no_like < max(1, len(cleaned) // 2)


def answer_is_visibility_abstain(text: str) -> bool:
    """Free-form answer that declines to pick a location (``no``, ``not seen``, …)."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if lowered in {"no", "yes", "unknown", "none", "n/a", "na", "not seen", "not visible"}:
        return True
    return bool(
        re.match(
            r"^(no|yes|not\s+seen|not\s+visible|didn'?t\s+see|have\s+not\s+seen|haven'?t\s+seen)\b",
            lowered,
        )
    )


def should_abstain_location_mcq(raw: str, choices: list[str] | None) -> bool:
    """Location MCQ + visibility-style ``answer: No`` → do not map to A–D."""
    if not choices_are_location_mcq(choices):
        return False
    fields = _answer_field_lines(raw)
    if not fields:
        return False
    return answer_is_visibility_abstain(fields[-1])


# HM-EQA uses A–D; RoboVista and similar 5-way MCQs use A–E.
MCQ_LETTER_CLASS = "A-E"
MCQ_LETTERS = "ABCDE"


def _match_choice_text_to_letter(text: str, choices: list[str]) -> str:
    """Map free-text (e.g. ``no``, ``off``) to A–E via choice strings."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return ""
    for idx, choice in enumerate(choices[: len(MCQ_LETTERS)]):
        choice_l = (choice or "").strip().lower()
        if not choice_l:
            continue
        if lowered == choice_l:
            return chr(ord("A") + idx)
        if re.search(rf"\b{re.escape(choice_l)}\b", lowered):
            return chr(ord("A") + idx)
    return ""


def extract_mcq_letter(predicted: str, choices: list[str] | None = None) -> str:
    """Extract A–E letter from free-form model output.

    Args:
        predicted: Raw EQA / VLM answer text.
        choices: Optional choice strings; used to match embedded choice text.

    Returns:
        Uppercase letter ``A``–``E``, or ``""`` if none found.
    """
    text = (predicted or "").strip()
    if not text:
        return ""
    compact = text.replace(" ", "").upper()
    if len(compact) == 1 and compact in MCQ_LETTERS:
        return compact
    m = re.search(
        rf"^answer\s*:\s*([{MCQ_LETTER_CLASS}])\b",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if m:
        return m.group(1).upper()
    m = re.search(
        rf"(?:^|\n)\s*([{MCQ_LETTER_CLASS}])\s*(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    if choices:
        letter = _match_choice_text_to_letter(text, choices)
        if letter:
            return letter
    return ""


def _answer_field_lines(raw: str) -> list[str]:
    """Capture text after each line-start ``answer:`` (ignore prose like ``cannot answer``)."""
    return [
        m.group(1).strip() for m in re.finditer(r"(?:^|\n)\s*answer\s*:\s*([^\n]*)", raw or "", flags=re.IGNORECASE)
    ]


def extract_mcq_letter_from_raw_eqa(raw: str, choices: list[str] | None = None) -> str:
    """Parse MCQ letter from raw mLLM EQA output (before human-facing reformatting)."""
    from emet.memory.graph_eqa.mcq_debias import match_freeform_to_choice

    text = raw or ""
    if not text.strip():
        return ""
    fields = _answer_field_lines(text)
    if not fields:
        return ""
    answer_field = fields[-1]
    if not answer_field:
        return ""
    if should_abstain_location_mcq(text, choices):
        return ""
    letter = extract_mcq_letter(answer_field, choices)
    if letter:
        return letter
    m = re.search(
        rf"(?:^|\n)\s*answer\s*:\s*([{MCQ_LETTER_CLASS.lower()}])\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    if choices:
        idx = match_freeform_to_choice(answer_field, choices)
        if idx is not None and 0 <= idx < len(MCQ_LETTERS):
            return chr(ord("A") + idx)
    return ""


def grade_mcq_answer(
    predicted: str,
    gold_letter: str,
    *,
    choices: list[str] | None = None,
) -> bool:
    """Return True if ``predicted`` matches MCQ letter ``gold_letter`` (A–E).

    Args:
        predicted: Model output (letter or prose containing a letter / choice text).
        gold_letter: Gold answer letter from the dataset.
        choices: Optional choice list for substring matching fallback.
    """
    gold = gold_letter.strip().upper()
    if not gold or gold not in MCQ_LETTERS:
        return False
    letter = extract_mcq_letter(predicted, choices)
    if letter:
        return letter == gold
    text = predicted.strip()
    if not text:
        return False
    if len(text) == 1 and text.upper() == gold:
        return True
    return text.upper().startswith(gold)


def write_episode_jsonl(path: Path, episodes: list[EpisodeMetrics], *, append: bool = False) -> None:
    """Write episode metrics as JSONL (one :class:`EpisodeMetrics` per line).

    Args:
        path: Output file path (parent directories are created).
        episodes: Rows to write.
        append: When True, append to an existing file instead of truncating.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep.to_dict()) + "\n")


def append_episode_jsonl(path: Path, episode: EpisodeMetrics) -> None:
    """Append one episode row to an existing JSONL file."""
    write_episode_jsonl(path, [episode], append=True)


def _eqa_produced_output(row: dict) -> bool:
    """True when the VLM / EQA loop produced at least one answer artifact."""
    if isinstance(row.get("eqa_iterations"), int) and row["eqa_iterations"] > 0:
        return True
    if (row.get("raw_eqa_output") or "").strip():
        return True
    if (row.get("parsed_answer_letter") or "").strip():
        return True
    return False


def episode_run_completed(row: dict) -> bool:
    """True when an episode row represents a finished run (not a crash/OOM stub)."""
    if row.get("error"):
        return False
    steps = row.get("planning_steps")
    if isinstance(steps, int) and steps > 0:
        # Navigation-only rows (OOM during VLM load) must be retried on --resume.
        return _eqa_produced_output(row)
    # Legacy rows without planning_steps: treat as done when no error field.
    return "planning_steps" not in row


def read_completed_question_ids(path: Path) -> set[int]:
    """Question ids with completed (non-error) runs in a JSONL file (for ``--resume``).

    OOM / startup failures are excluded so ``--resume`` can retry them.

    Args:
        path: Existing results JSONL from :func:`write_episode_jsonl`.

    Returns:
        Set of ``question_id`` integers for completed episodes (ignores malformed lines).
    """
    if not path.is_file():
        return set()
    done: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = row.get("question_id")
        if isinstance(qid, int) and episode_run_completed(row):
            done.add(qid)
    return done


def summarize_episodes(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    """Aggregate accuracy, mean planning steps, and success rate.

    Args:
        episodes: Graded episodes (may be empty).

    Returns:
        Dict with keys ``accuracy``, ``mean_steps``, ``success_rate``, ``n``.
    """
    if not episodes:
        return {"accuracy": 0.0, "mean_steps": 0.0, "success_rate": 0.0, "n": 0.0}
    n = len(episodes)
    return {
        "accuracy": sum(1 for e in episodes if e.correct) / n,
        "mean_steps": sum(e.planning_steps for e in episodes) / n,
        "success_rate": sum(1 for e in episodes if e.success) / n,
        "n": float(n),
    }


def compare_method_results(
    graph_eqa: list[EpisodeMetrics],
    dynagraph: list[EpisodeMetrics],
) -> dict:
    """Side-by-side summary for GraphEQA vs Dynagraph on the same question ids.

    Args:
        graph_eqa: Episodes from a GraphEQA Habitat run.
        dynagraph: Episodes from a Dynagraph Habitat run.

    Returns:
        Dict with per-method summaries, disagreement counts, and ``per_question`` rows.
    """
    by_q_graph = {e.question_id: e for e in graph_eqa}
    by_q_dyna = {e.question_id: e for e in dynagraph}
    qids = sorted(set(by_q_graph) | set(by_q_dyna))
    rows: list[dict] = []
    for qid in qids:
        g = by_q_graph.get(qid)
        d = by_q_dyna.get(qid)
        rows.append(
            {
                "question_id": qid,
                "gold": (g or d).gold_answer_letter if (g or d) else "",
                "graph_eqa_pred": g.parsed_answer_letter or g.predicted_answer[:1] if g else "",
                "graph_eqa_correct": g.correct if g else False,
                "dynagraph_pred": d.parsed_answer_letter or d.predicted_answer[:1] if d else "",
                "dynagraph_correct": d.correct if d else False,
                "graph_eqa_steps": g.planning_steps if g else 0,
                "dynagraph_steps": d.planning_steps if d else 0,
            }
        )
    return {
        "graph_eqa": summarize_episodes(graph_eqa),
        "dynagraph": summarize_episodes(dynagraph),
        "both_correct": sum(1 for r in rows if r["graph_eqa_correct"] and r["dynagraph_correct"]),
        "graph_only": sum(1 for r in rows if r["graph_eqa_correct"] and not r["dynagraph_correct"]),
        "dynagraph_only": sum(1 for r in rows if r["dynagraph_correct"] and not r["graph_eqa_correct"]),
        "neither": sum(1 for r in rows if not r["graph_eqa_correct"] and not r["dynagraph_correct"]),
        "per_question": rows,
    }
