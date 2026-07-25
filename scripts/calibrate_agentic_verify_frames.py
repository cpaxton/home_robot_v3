#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Offline SigLIP verify calibration on saved Habitat episode RGB frames.

Compares full-frame pool cosine vs dense (MaskSigLIP) max-patch cosine against
DynaMem's 0.21 PRESENT bar — without re-running Habitat.

Example:
  uv run python scripts/calibrate_agentic_verify_frames.py \\
    --episodes ~/.cache/habitat_eqa/episodes/h2h_agentic_q0017 \\
    --phrase 'woven basket' -o /tmp/verify_calib.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _list_frames(episode_root: Path, max_frames: int) -> list[Path]:
    candidates: list[Path] = []
    for sub in ("images", "frames"):
        for d in sorted(episode_root.rglob(sub)):
            if not d.is_dir():
                continue
            for pat in ("rgb_*.png", "frame_*.png", "*.png"):
                found = sorted(d.glob(pat))
                if found:
                    candidates.extend(found)
                    break
            if candidates:
                break
        if candidates:
            break
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= max_frames:
            break
    return out


def _phrase_from_metrics(episode_root: Path) -> str | None:
    for p in episode_root.rglob("metrics.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        q = str(m.get("question") or "")
        # crude noun-ish: strip MCQ; take quoted or last noun phrase heuristics left to caller
        return q or None
    return None


def _default_phrases(question: str) -> list[str]:
    """Small set of query phrases derived from the question stem."""
    stem = re.split(r"\bA\)|\bAnswer:", question, maxsplit=1)[0]
    stem = re.sub(r"[?!.]+$", "", stem).strip()
    # pull common patterns
    phrases: list[str] = []
    for pat in (
        r"the ([a-z ]+?)(?:\s+anywhere|\s+at\b|\s*\?|$)",
        r"leave the ([a-z ]+?)(?:\s+at\b|\s*\?|$)",
        r"see the ([a-z ]+?)(?:\s+anywhere|\s*\?|$)",
        r"has the ([a-z ]+?)(?:\s*\?|$)",
        r"looking for the ([a-z ]+?)(?:\s*\?|$)",
        r"many ([a-z ]+?)(?:\s+are|\s+in|\s*\?|$)",
    ):
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            phrases.append(m.group(1).strip().lower())
    if not phrases and stem:
        phrases.append(stem.lower()[:80])
    # unique
    out: list[str] = []
    for p in phrases:
        p = re.sub(r"\s+", " ", p).strip()
        if p and p not in out:
            out.append(p)
    return out


def score_episode(
    enc: Any,
    frames: list[Path],
    phrases: list[str],
    *,
    dense: bool,
    stride: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    rows: list[dict[str, Any]] = []
    for phrase in phrases:
        text_t = enc.encode_text(phrase)
        text = text_t.detach().float().cpu().numpy().reshape(-1)
        text = text / (np.linalg.norm(text) + 1e-12)
        best_ff = -1.0
        best_dense = -1.0
        best_ff_path = None
        best_dense_path = None
        for i, fp in enumerate(frames):
            if stride > 1 and (i % stride) != 0:
                continue
            rgb = _load_rgb(fp)
            img_t = enc.encode_image(rgb)
            img = img_t.detach().float().cpu().numpy().reshape(-1)
            img = img / (np.linalg.norm(img) + 1e-12)
            ff = float(img @ text)
            if ff > best_ff:
                best_ff, best_ff_path = ff, str(fp)
            dens = None
            if dense:
                # Patch tokens → max cosine (same feature space DynaMem voxels use).
                try:
                    inputs = enc._to_model_inputs(enc.processor(images=rgb, return_tensors="pt"))
                    with torch.no_grad():
                        out = enc.model.vision_model(inputs["pixel_values"], output_hidden_states=True)
                        feat = out.last_hidden_state  # [1, L, C]
                        feat = F.normalize(feat.float(), dim=-1)
                        tt = torch.as_tensor(text, device=feat.device, dtype=feat.dtype).reshape(1, -1)
                        sims = (feat @ tt.T).squeeze(-1).squeeze(0)  # [L]
                        dens = float(sims.max().item())
                except Exception as e:
                    dens = None
                    dens_err = str(e)
                else:
                    dens_err = None
                if dens is not None and dens > best_dense:
                    best_dense, best_dense_path = dens, str(fp)
            rows.append(
                {
                    "phrase": phrase,
                    "frame": str(fp),
                    "full_frame_sim": ff,
                    "dense_max_sim": dens,
                }
            )
        rows.append(
            {
                "phrase": phrase,
                "summary": True,
                "best_full_frame_sim": best_ff,
                "best_full_frame_path": best_ff_path,
                "best_dense_max_sim": best_dense if dense else None,
                "best_dense_path": best_dense_path,
                "passes_0_21_full": best_ff >= 0.21,
                "passes_0_21_dense": (best_dense >= 0.21) if dense else None,
                "passes_0_12_full": best_ff >= 0.12,
                "passes_0_12_dense": (best_dense >= 0.12) if dense else None,
            }
        )
    return {"n_frames": len(frames), "phrases": phrases, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--episodes",
        nargs="+",
        type=Path,
        required=True,
        help="Episode root dirs (contain images/ or frames/)",
    )
    ap.add_argument("--phrase", action="append", default=[], help="Override phrase (repeatable)")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=2, help="Score every Nth frame")
    ap.add_argument("--no-dense", action="store_true", help="Skip dense max-patch scoring")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    from emet.perception.encoders.siglip_encoder import get_shared_mask_siglip_encoder

    enc = get_shared_mask_siglip_encoder()
    report: dict[str, Any] = {"episodes": []}
    for root in args.episodes:
        root = root.expanduser().resolve()
        frames = _list_frames(root, args.max_frames * max(1, args.stride))
        # subsample after list for stride applied in score
        q = _phrase_from_metrics(root) or ""
        phrases = list(args.phrase) if args.phrase else _default_phrases(q)
        if not phrases:
            phrases = ["object"]
        print(f"\n=== {root.name} frames={len(frames)} phrases={phrases} Q={q[:80]!r}")
        block = score_episode(
            enc,
            frames[: args.max_frames * args.stride],
            phrases,
            dense=not args.no_dense,
            stride=max(1, args.stride),
        )
        block["episode"] = str(root)
        block["question"] = q
        report["episodes"].append(block)
        for row in block["rows"]:
            if row.get("summary"):
                print(
                    f"  phrase={row['phrase']!r}\n"
                    f"    full_frame best={row['best_full_frame_sim']:.4f} "
                    f"(>=0.21? {row['passes_0_21_full']}, >=0.12? {row['passes_0_12_full']})\n"
                    f"    dense_max  best={row.get('best_dense_max_sim')}\n"
                    f"    (>=0.21? {row.get('passes_0_21_dense')}, >=0.12? {row.get('passes_0_12_dense')})"
                )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
