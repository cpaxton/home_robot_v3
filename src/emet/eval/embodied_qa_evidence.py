# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Explicit recorded-evidence bindings; no dataset filename guessing or scoring."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from emet.eval.embodied_qa_datasets import EvaluationQuestion


@dataclass(frozen=True)
class EvidenceFrame:
    index: int
    rgb: Path
    sha256: str

    def load_rgb(self):
        if hashlib.sha256(self.rgb.read_bytes()).hexdigest() != self.sha256:
            raise ValueError(f"Evidence content changed: {self.rgb}")
        with Image.open(self.rgb) as image:
            image.load()
            return image.convert("RGB")


@dataclass(frozen=True)
class RecordedEvidence:
    question: str
    frames: tuple[EvidenceFrame, ...]

    def model_input(self, *, max_images: int):
        """Question + sensor RGB only. Selection must be a separate declared policy."""
        if type(max_images) is not int or max_images < 1 or len(self.frames) > max_images:
            raise ValueError("Image budget cannot contain evidence sequence; declare a separate selection protocol")
        return [self.question, *[frame.load_rgb() for frame in self.frames]]


def bind_recorded_evidence(question: EvaluationQuestion, manifest: dict, *, root: Path) -> RecordedEvidence:
    """Bind official identity to local RGB, preserving metadata order and pair identity.

    Manifest schema: version 1, benchmark, episodes keyed by question_id. Each
    entry declares evidence_id, scene, and frames [{index,rgb,sha256}]. The same
    ID binds both AbstainEQA variants; only their question text differs.
    EXPRESS frame indices come from the explicit supplied-trajectory manifest,
    not oracle action labels. Binding is recorded replay, never active navigation.
    """
    if manifest.get("schema_version") != 1 or manifest.get("benchmark") != question.benchmark:
        raise ValueError("Evidence manifest version/benchmark mismatch")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, dict) or question.question_id not in episodes:
        raise ValueError(f"Missing evidence binding: {question.question_id}")
    entry = episodes[question.question_id]
    if (
        not isinstance(entry, dict)
        or not {"evidence_id", "scene", "frames"}.issubset(entry)
        or entry.get("evidence_id") != question.evidence_id
        or entry.get("scene") != question.scene
    ):
        raise ValueError("Evidence identity/scene mismatch")
    rows = entry.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Nonempty evidence frames required")
    by_index = {}
    root = root.expanduser().resolve()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Frame record required")
        index, name, digest = row.get("index"), row.get("rgb"), row.get("sha256")
        if type(index) is not int or index < 0 or index in by_index:
            raise ValueError("Invalid/duplicate frame index")
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError("Relative RGB path required")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing/outside-root RGB: {name}")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("SHA256 required for each RGB frame")
        frame = EvidenceFrame(index, path, digest)
        frame.load_rgb()  # Decode and hash-check before any inference starts.
        by_index[index] = frame
    indices = question.frame_indices or tuple(by_index)
    if len(set(indices)) != len(indices) or any(i not in by_index for i in indices):
        raise ValueError("Missing/duplicate requested evidence frame")
    return RecordedEvidence(question.question, tuple(by_index[i] for i in indices))
