# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Minimal Habitat Magnum EGL / WindowlessContext probe (no VLM).

Empty ``nvidia-smi`` does not prove Habitat can create a headless context.
This opens one HM-EQA scene, grabs a single RGB frame, and closes — enough to
catch ``unable to find CUDA device 0 among N EGL devices`` before a long H2H.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EglProbeResult:
    ok: bool
    message: str
    scene_glb: str | None = None
    rgb_shape: tuple[int, ...] | None = None
    error: str | None = None


def resolve_probe_scene_glb(*, question_id: int = 0) -> Path:
    """Pick an HM-EQA scene GLB for the EGL smoke (question 0 by default)."""
    from emet.habitat.config import default_hm3d_scene_dir, hm3d_scene_glb_path, questions_csv_path
    from emet.habitat.datasets import load_hmeqa_questions

    qs = load_hmeqa_questions(questions_csv_path())
    if not qs:
        raise FileNotFoundError("No HM-EQA questions loaded; download habitat EQA data first.")
    qid = int(question_id)
    if qid < 0 or qid >= len(qs):
        qid = 0
    q = qs[qid]
    root = default_hm3d_scene_dir()
    glb = hm3d_scene_glb_path(q.scene, root)
    if not glb.is_file():
        raise FileNotFoundError(f"HM3D scene GLB missing for probe: {glb}")
    return glb


def run_egl_probe(*, question_id: int = 0) -> EglProbeResult:
    """Create Habitat-Sim once, read one frame, tear down. Never loads a VLM."""
    try:
        glb = resolve_probe_scene_glb(question_id=question_id)
    except Exception as exc:
        return EglProbeResult(ok=False, message="scene resolve failed", error=str(exc))

    sim: Any = None
    try:
        from emet_habitat.simulator import HabitatEQASimulator

        sim = HabitatEQASimulator(glb, scene_id=glb.parent.name)
        frame = sim.get_frame()
        rgb = getattr(frame, "rgb", None)
        shape = tuple(getattr(rgb, "shape", ())) if rgb is not None else None
        if shape is None or len(shape) < 2:
            return EglProbeResult(
                ok=False,
                message="EGL context opened but no RGB frame",
                scene_glb=str(glb),
                error="empty rgb",
            )
        return EglProbeResult(
            ok=True,
            message="Habitat EGL OK (WindowlessContext + one RGB frame)",
            scene_glb=str(glb),
            rgb_shape=shape,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc(limit=6)
        return EglProbeResult(
            ok=False,
            message="Habitat EGL probe failed",
            scene_glb=str(glb),
            error=f"{err}\n{tb}",
        )
    finally:
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass
