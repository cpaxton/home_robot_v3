# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""HM-EQA / Explore-EQA dataset loaders (CSV compatible with GraphEQA).

These loaders read the same CSV layout used by the GraphEQA Habitat benchmark:
``questions.csv`` (multiple-choice EQA items) and ``scene_init_poses.csv`` (agent
spawn poses per scene/floor). Paths default to :func:`~emet.habitat.config.default_habitat_eqa_data_dir`
unless overridden.

Typical usage::

    from emet.habitat.datasets import load_hmeqa_questions, load_scene_init_poses

    questions = load_hmeqa_questions()
    poses = load_scene_init_poses()
    q = questions[0]
    pose = poses[(q.scene, q.floor)]
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

from emet.habitat.config import questions_csv_path, scene_init_poses_csv_path


@dataclass(frozen=True)
class SceneInitPose:
    """Explore-EQA agent spawn pose for one HM3D scene floor.

    Attributes:
        scene: HM3D scene id (e.g. ``00004-VqCaAuuoeWk``).
        floor: Floor index within the scene (Explore-EQA convention).
        x: Habitat world X (meters).
        y: Habitat world Y (meters, vertical/up).
        z: Habitat world Z (meters).
        heading: Yaw in radians (Y-up frame; positive CCW when viewed from above).
    """

    scene: str
    floor: int
    x: float
    y: float
    z: float
    heading: float


@dataclass(frozen=True)
class HMEQAQuestion:
    """One HM-EQA multiple-choice question row from ``questions.csv``.

    Attributes:
        index: Zero-based row index in the loaded CSV (stable ordering).
        scene: HM3D scene id this question is tied to.
        floor: Floor index (pairs with :class:`SceneInitPose`).
        question: Raw question text.
        choices: List of choice strings (typically four options).
        question_formatted: Prompt-ready text (falls back to ``question``).
        answer_letter: Gold MCQ letter (``A``–``D``).
        label: Optional semantic label / category from the dataset.
    """

    index: int
    scene: str
    floor: int
    question: str
    choices: list[str]
    question_formatted: str
    answer_letter: str
    label: str

    @property
    def answer_index(self) -> int:
        """Zero-based index of :attr:`answer_letter` in ``choices`` (``A`` → 0)."""
        letter = self.answer_letter.strip().upper()
        if len(letter) != 1 or not letter.isalpha():
            raise ValueError(f"Invalid MCQ answer letter {self.answer_letter!r}")
        return ord(letter) - ord("A")


def _parse_choices(raw: str) -> list[str]:
    """Parse the ``choices`` column (Python list literal) from ``questions.csv``."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse choices list: {raw!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"choices must be a list, got {type(parsed)}")
    return [str(x) for x in parsed]


def load_hmeqa_questions(path: Path | None = None) -> list[HMEQAQuestion]:
    """Load Explore-EQA / HM-EQA ``questions.csv``.

    Args:
        path: Optional CSV path. When ``None``, uses
            :func:`~emet.habitat.config.questions_csv_path`.

    Returns:
        All questions in file order.

    Raises:
        FileNotFoundError: If the CSV is missing.
        ValueError: If a ``choices`` cell is not a valid list literal.
    """
    csv_path = path or questions_csv_path()
    if not csv_path.is_file():
        raise FileNotFoundError(f"HM-EQA questions not found: {csv_path}")

    out: list[HMEQAQuestion] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            out.append(
                HMEQAQuestion(
                    index=i,
                    scene=row["scene"].strip(),
                    floor=int(row["floor"]),
                    question=row["question"].strip(),
                    choices=_parse_choices(row["choices"]),
                    question_formatted=row.get("question_formatted", row["question"]).strip(),
                    answer_letter=row["answer"].strip(),
                    label=row.get("label", "").strip(),
                )
            )
    return out


def _float_field(row: dict[str, str], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return float(row[name])
    return default


def _parse_scene_floor_row(row: dict[str, str]) -> tuple[str, int]:
    if "scene" in row and "floor" in row:
        return row["scene"].strip(), int(row["floor"])
    if "scene_floor" in row:
        scene_floor = row["scene_floor"].strip()
        scene, floor_s = scene_floor.rsplit("_", 1)
        return scene, int(floor_s)
    raise ValueError(f"scene_init_poses row missing scene/floor columns: {list(row.keys())}")


def load_scene_init_poses(path: Path | None = None) -> dict[tuple[str, int], SceneInitPose]:
    """Load Explore-EQA ``scene_init_poses.csv`` keyed by ``(scene, floor)``.

    Args:
        path: Optional CSV path. When ``None``, uses
            :func:`~emet.habitat.config.scene_init_poses_csv_path`.

    Returns:
        Mapping from ``(scene_id, floor)`` to spawn pose. Column aliases
        ``init_x`` / ``position_x`` / ``heading`` / ``theta`` are accepted.

    Raises:
        FileNotFoundError: If the CSV is missing.
        ValueError: If a row lacks scene/floor identifiers.
    """
    csv_path = path or scene_init_poses_csv_path()
    if not csv_path.is_file():
        raise FileNotFoundError(f"scene_init_poses not found: {csv_path}")

    poses: dict[tuple[str, int], SceneInitPose] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene, floor = _parse_scene_floor_row(row)
            pose = SceneInitPose(
                scene=scene,
                floor=floor,
                x=_float_field(row, "x", "init_x", "position_x"),
                y=_float_field(row, "y", "init_y", "position_y"),
                z=_float_field(row, "z", "init_z", "position_z"),
                heading=_float_field(row, "heading", "init_heading", "init_angle", "rotation", "theta"),
            )
            poses[(scene, floor)] = pose
    return poses


def get_question(
    questions: list[HMEQAQuestion],
    *,
    question_id: int | None = None,
    scene_id: str | None = None,
) -> HMEQAQuestion:
    """Select one question by index or by scene id.

    Args:
        questions: Loaded question list (e.g. from :func:`load_hmeqa_questions`).
        question_id: Zero-based index into ``questions``.
        scene_id: Return the first question whose :attr:`HMEQAQuestion.scene` matches.

    Returns:
        The matching :class:`HMEQAQuestion`.

    Raises:
        ValueError: If neither selector is provided.
        IndexError: If ``question_id`` is out of range.
        KeyError: If ``scene_id`` has no matching questions.
    """
    if question_id is not None:
        if question_id < 0 or question_id >= len(questions):
            raise IndexError(f"question_id {question_id} out of range (n={len(questions)})")
        return questions[question_id]
    if scene_id is not None:
        matches = [q for q in questions if q.scene == scene_id]
        if not matches:
            raise KeyError(f"No questions for scene {scene_id!r}")
        return matches[0]
    raise ValueError("Provide question_id or scene_id")
