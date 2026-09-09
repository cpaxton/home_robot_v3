# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Persist each GraphEQA ``query_answer`` VLM call: prompt text + attached images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


def _prompt_text_from_blocks(text_blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in text_blocks:
        if isinstance(block, str):
            parts.append(block)
        else:
            parts.append(f"[IMAGE slot {len(parts) + 1}]")
    return "\n\n".join(parts)


def record_eqa_decision_iteration(
    trace_root: str | Path,
    iteration: int,
    *,
    question: str,
    text_blocks: list[Any],
    obs_ids: list[int],
    crop_obs_id: int | None,
    nav_fallback_count: int,
    relevant_images: list[Any],
    view_status: str = "",
    close_look_status: str = "",
    vlm_raw: str | None = None,
    parsed: dict[str, Any] | None = None,
) -> Path:
    """Write ``eqa_decisions/iter_NNN/`` under the episode bundle."""
    root = Path(trace_root)
    root.mkdir(parents=True, exist_ok=True)
    iter_dir = root / f"iter_{int(iteration):03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    image_entries: list[dict[str, Any]] = []
    for idx, im in enumerate(relevant_images):
        slot = idx + 1
        obs_id: int | None = None
        if idx < len(obs_ids):
            obs_id = int(obs_ids[idx])
            kind = "scene"
        elif crop_obs_id is not None and idx == len(obs_ids):
            obs_id = int(crop_obs_id)
            kind = "closeup"
        else:
            kind = "nav_fallback"
        fname = f"image_{slot}"
        if obs_id is not None:
            fname += f"_obs{obs_id}"
        fname += ".png"
        out_path = iter_dir / fname
        if isinstance(im, Image.Image):
            im.save(out_path)
        else:
            Image.fromarray(im).save(out_path)
        image_entries.append(
            {
                "slot": slot,
                "obs_id": obs_id,
                "kind": kind,
                "path": fname,
            }
        )

    meta: dict[str, Any] = {
        "iteration": int(iteration),
        "question": question,
        "obs_ids": [int(x) for x in obs_ids],
        "crop_obs_id": int(crop_obs_id) if crop_obs_id is not None else None,
        "nav_fallback_count": int(nav_fallback_count),
        "images": image_entries,
        "view_status": view_status or None,
        "close_look_status": close_look_status or None,
    }
    if vlm_raw is not None:
        meta["vlm_raw"] = vlm_raw
    if parsed is not None:
        meta["parsed"] = parsed
    (iter_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (iter_dir / "prompt.txt").write_text(
        _prompt_text_from_blocks(text_blocks) + "\n",
        encoding="utf-8",
    )
    if vlm_raw is not None:
        (iter_dir / "vlm_raw.txt").write_text(vlm_raw + "\n", encoding="utf-8")

    index_path = root / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "iteration": int(iteration),
                    "dir": iter_dir.name,
                    "obs_ids": meta["obs_ids"],
                    "answer": (parsed or {}).get("answer"),
                    "confidence": (parsed or {}).get("confidence"),
                },
                sort_keys=True,
            )
            + "\n"
        )
    return iter_dir


def finalize_eqa_decision_trace(trace_root: str | Path, *, n_iterations: int) -> None:
    """Write a small README pointer at bundle close."""
    root = Path(trace_root)
    if not root.is_dir():
        return
    readme = root / "README.md"
    readme.write_text(
        "# EQA decision trace\n\n"
        f"{n_iterations} ``query_answer`` iteration(s). Each ``iter_NNN/`` holds:\n\n"
        "- ``prompt.txt`` — text blocks sent to the VLM (images replaced by placeholders)\n"
        "- ``image_*_obs*.png`` — RGB frames attached in slot order\n"
        "- ``meta.json`` — obs ids, view/close-look status, parsed answer\n"
        "- ``vlm_raw.txt`` — model output (when recorded post-call)\n\n"
        "See ``index.jsonl`` for a compact timeline.\n",
        encoding="utf-8",
    )
