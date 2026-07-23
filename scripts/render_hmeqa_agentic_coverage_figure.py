#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Side-by-side classic vs agentic HM-EQA top-down coverage figure.

Reads snapshotted episode bundles under ``OUT/bundles/{classic,agentic}_qN/``
(produced by ``scripts/run_hmeqa_agentic_h2h.sh``) and lays out overlay maps
with planning-step / explored-cell captions.

Example::

  uv run python scripts/render_hmeqa_agentic_coverage_figure.py \\
    ~/runs/emet/hmeqa_agentic_h2h8_20260723_174307 \\
    --question-ids 15,105,68 \\
    --output paper/figs/hmeqa_agentic_coverage.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Pillow required: uv sync") from exc


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else None,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else None,
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if path and Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _load_row(out: Path, arm: str, qid: int) -> dict:
    p = out / f"{arm}_q{qid}.jsonl"
    if not p.is_file():
        p = out / f"{arm}.jsonl"
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row.get("question_id", -1)) == qid:
                    return row
        return {}
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(lines[0]) if lines else {}


def _bundle(out: Path, arm: str, qid: int) -> Path | None:
    snap = out / "bundles" / f"{arm}_q{qid}"
    if snap.is_dir() and (snap / "topdown_map_overlay.png").is_file():
        return snap
    row = _load_row(out, arm, qid)
    b = row.get("debug_bundle_dir")
    if b and Path(b).is_dir():
        return Path(b)
    return None


def _explored_cells(bundle: Path) -> int | None:
    p = bundle / "explored_2d.npy"
    if not p.is_file():
        return None
    return int(np.load(p).astype(bool).sum())


def _open_map(bundle: Path, prefer_overlay: bool = True) -> Image.Image:
    for name in (
        ("topdown_map_overlay.png", "topdown_map.png")
        if prefer_overlay
        else ("topdown_map.png", "topdown_map_overlay.png")
    ):
        p = bundle / name
        if p.is_file():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"no topdown map in {bundle}")


def _fit(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = im.size
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def render(
    out_dir: Path,
    question_ids: list[int],
    *,
    cell_w: int = 420,
    cell_h: int = 420,
    pad: int = 16,
) -> Image.Image:
    title_f = _font(22, bold=True)
    head_f = _font(16, bold=True)
    body_f = _font(13)
    arms = ("classic", "agentic")
    arm_titles = {"classic": "Classic planning", "agentic": "Agentic verify"}

    n_rows = len(question_ids)
    n_cols = 2
    header_h = 48
    caption_h = 56
    row_h = cell_h + caption_h + pad
    width = pad + n_cols * (cell_w + pad)
    height = header_h + pad + n_rows * row_h + pad
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 12), "HM-EQA coverage: classic vs agentic Dynagraph", fill=(20, 20, 20), font=title_f)

    for col, arm in enumerate(arms):
        x0 = pad + col * (cell_w + pad)
        draw.text((x0, header_h - 22), arm_titles[arm], fill=(40, 40, 40), font=head_f)

    for row_i, qid in enumerate(question_ids):
        y0 = header_h + pad + row_i * row_h
        for col, arm in enumerate(arms):
            x0 = pad + col * (cell_w + pad)
            bundle = _bundle(out_dir, arm, qid)
            row = _load_row(out_dir, arm, qid)
            box = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
            if bundle is None:
                d = ImageDraw.Draw(box)
                d.text((20, cell_h // 2), f"missing {arm} q{qid}", fill=(160, 40, 40), font=body_f)
            else:
                try:
                    m = _fit(_open_map(bundle), cell_w - 8, cell_h - 8)
                    ox = (cell_w - m.size[0]) // 2
                    oy = (cell_h - m.size[1]) // 2
                    box.paste(m, (ox, oy))
                except Exception as exc:
                    d = ImageDraw.Draw(box)
                    d.text((12, cell_h // 2), f"map err: {exc}", fill=(160, 40, 40), font=body_f)
            canvas.paste(box, (x0, y0))
            steps = row.get("planning_steps")
            ok = row.get("correct")
            pred = row.get("predicted_answer")
            gold = row.get("gold_answer_letter")
            n_exp = _explored_cells(bundle) if bundle else None
            mark = "✓" if ok else "✗" if ok is False else "?"
            line1 = f"Q{qid} {mark}  pred={pred} gold={gold}"
            line2 = f"steps={steps}  explored_cells={n_exp}"
            draw.text((x0, y0 + cell_h + 6), line1, fill=(20, 20, 20), font=body_f)
            draw.text((x0, y0 + cell_h + 26), line2, fill=(80, 80, 80), font=body_f)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path, help="H2H run directory (contains bundles/ + jsonl)")
    ap.add_argument(
        "--question-ids",
        default="15,105,68",
        help="Comma-separated question ids to panel (default: 15,105,68)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: OUT/figures/hmeqa_agentic_coverage.png)",
    )
    args = ap.parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    qids = [int(x.strip()) for x in str(args.question_ids).split(",") if x.strip()]
    output = (args.output or (out_dir / "figures" / "hmeqa_agentic_coverage.png")).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    img = render(out_dir, qids)
    img.save(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
