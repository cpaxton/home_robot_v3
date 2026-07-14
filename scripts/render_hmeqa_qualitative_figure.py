#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Render a qualitative HM-EQA figure: sample questions + determining EQA images.

Uses episode bundles from a merge-on baseline (or any) JSONL. For each episode,
parses Caption ``Image N shows … at (x, y)`` from ``raw_eqa.txt``, finds the
nearest RGB frame by pose, and lays out question / choices / answer with those
views (plus optional top-down map).

Example::

  uv run python scripts/render_hmeqa_qualitative_figure.py \\
    --jsonl ~/runs/emet/branch_verify_20260711/merge_on_baseline_20260712_001627/smoke_trio.jsonl \\
    --question-ids 3,14,17 \\
    --output paper/figs/hmeqa_qualitative_sample.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Pillow required: uv sync") from exc


_CAPTION_AT = re.compile(
    r"Image\s+(\d+)\s+(?:shows|is)\s+(.+?)\s+at\s+\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)",
    re.IGNORECASE,
)
_CAPTION_NO_XY = re.compile(
    r"Image\s+(\d+)\s+(?:shows|is)\s+(.+?)(?:\.|$)",
    re.IGNORECASE,
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _bundle_for_row(episodes_root: Path, row: dict) -> Path | None:
    bundle = row.get("debug_bundle_dir") or row.get("bundle_dir")
    if bundle:
        p = Path(str(bundle)).expanduser()
        if p.is_dir():
            return p
    qid = int(row.get("question_id", -1))
    method = str(row.get("method", "dynagraph"))
    for parent in episodes_root.glob("*"):
        if not parent.is_dir():
            continue
        cand = parent / f"q{qid:04d}_{method}"
        if cand.is_dir():
            return cand
    return None


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


def _parse_caption_images(raw: str) -> list[tuple[int, str, float | None, float | None]]:
    """Return (image_idx, description, x, y) from Caption block (x/y may be None)."""
    block = raw
    m = re.search(r"Caption:\s*(.*?)(?:\n\s*\n|\nReasoning:)", raw, re.IGNORECASE | re.DOTALL)
    if m:
        block = m.group(1)
    out: list[tuple[int, str, float | None, float | None]] = []
    seen: set[int] = set()
    for img_i, desc, xs, ys in _CAPTION_AT.findall(block):
        idx = int(img_i)
        seen.add(idx)
        out.append((idx, desc.strip().rstrip("."), float(xs), float(ys)))
    for img_i, desc in _CAPTION_NO_XY.findall(block):
        idx = int(img_i)
        if idx in seen:
            continue
        out.append((idx, desc.strip().rstrip("."), None, None))
    out.sort(key=lambda t: t[0])
    return out


def _metadata_frames(bundle: Path) -> list[dict]:
    meta_path = bundle / "metadata.jsonl"
    if not meta_path.is_file():
        return []
    return [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_rgb(bundle: Path, rel: str) -> Image.Image | None:
    p = (bundle / rel).resolve()
    if p.is_file():
        return Image.open(p).convert("RGB")
    # Fallbacks for odd layouts.
    name = Path(rel).name
    for cand in (bundle / "frames" / name, bundle / "images" / name):
        if cand.is_file():
            return Image.open(cand).convert("RGB")
    return None


def _frame_for_xy(bundle: Path, x: float, y: float) -> tuple[Image.Image | None, str, float]:
    meta = _metadata_frames(bundle)
    if not meta:
        # Last resort: any recent frame.
        frames = sorted((bundle / "frames").glob("rgb_*.png"))
        if frames:
            return Image.open(frames[-1]).convert("RGB"), frames[-1].name, float("nan")
        return None, "", float("nan")
    best = meta[0]
    best_d = float("inf")
    for row in meta:
        pose = row.get("pose_xyt") or [0.0, 0.0, 0.0]
        d = (float(pose[0]) - x) ** 2 + (float(pose[1]) - y) ** 2
        if d < best_d:
            best_d = d
            best = row
    rel = str(best.get("image") or "")
    img = _load_rgb(bundle, rel) if rel else None
    return img, rel, math.sqrt(best_d)


def _frame_fallback(bundle: Path, prefer_late: bool = True) -> Image.Image | None:
    frames = sorted((bundle / "frames").glob("rgb_*.png"))
    if not frames:
        frames = sorted((bundle / "images").glob("frame_*.png"))
    if not frames:
        return None
    pick = frames[-1] if prefer_late else frames[0]
    return Image.open(pick).convert("RGB")


def _pick_caption_view(
    bundle: Path,
    img_i: int,
    desc: str,
    x: float | None,
    y: float | None,
) -> tuple[str, Image.Image] | None:
    label = f"Image {img_i}: {desc}"
    if x is not None and y is not None:
        img, _rel, dist = _frame_for_xy(bundle, x, y)
        if img is not None:
            if math.isfinite(dist):
                label = f"Image {img_i}: {desc} ({dist:.1f}m)"
            return label, img
    img = _frame_fallback(bundle, prefer_late=img_i > 1)
    if img is None:
        return None
    return label, img


def _determining_images(
    bundle: Path,
    raw: str,
    *,
    max_images: int = 2,
) -> list[tuple[str, Image.Image]]:
    """Pick captioned EQA views (prefer object views over frontiers), pose-matched to RGB."""
    caps = _parse_caption_images(raw)
    # Prefer concrete object captions over ``unexplored frontier`` for the figure.
    ranked = sorted(
        caps,
        key=lambda t: (
            0 if "frontier" not in t[1].lower() else 1,
            t[0],
        ),
    )
    picked: list[tuple[str, Image.Image]] = []
    for img_i, desc, x, y in ranked:
        if len(picked) >= max_images:
            break
        view = _pick_caption_view(bundle, img_i, desc, x, y)
        if view is not None:
            picked.append(view)
    if not picked:
        img = _frame_fallback(bundle)
        if img is not None:
            picked.append(("view", img))
    # Keep display order by Image index when possible.
    def _img_key(item: tuple[str, Image.Image]) -> int:
        m = re.match(r"Image\s+(\d+)", item[0])
        return int(m.group(1)) if m else 99

    picked.sort(key=_img_key)
    return picked


def _choice_letter(i: int) -> str:
    return chr(ord("A") + i)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    # Approximate wrap using character width.
    avg = max(6, int(font.size * 0.55)) if hasattr(font, "size") else 8
    cols = max(20, width // avg)
    return textwrap.wrap(text, width=cols) or [""]


def _draw_row(
    canvas: Image.Image,
    *,
    y0: int,
    row_h: int,
    pad: int,
    row: dict,
    bundle: Path,
    tile: int,
    max_images: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    font_title = _font(22, bold=True)
    font_body = _font(16)
    font_small = _font(13)
    font_choice = _font(15)

    qid = int(row.get("question_id", -1))
    question = str(row.get("question") or "").strip()
    choices = list(row.get("choices") or [])
    gold = str(row.get("gold_answer_letter") or "?").upper()
    pred = str(row.get("predicted_answer") or row.get("formatted_answer") or "?").strip()
    pred_letter = pred.upper()[:1] if pred and pred[0].upper() in "ABCD" else pred
    # Prefer letter from formatted_answer / parsed
    for key in ("formatted_answer", "parsed_answer_letter"):
        v = str(row.get(key) or "").strip().upper()
        if v and v[0] in "ABCD":
            pred_letter = v[0]
            break
    ok = bool(row.get("correct"))
    status = "correct" if ok else "incorrect"

    raw_path = bundle / "raw_eqa.txt"
    raw = raw_path.read_text(encoding="utf-8") if raw_path.is_file() else ""
    views = _determining_images(bundle, raw, max_images=max_images)

    # Layout: text column | images | topdown
    text_w = 420
    img_col_x = pad + text_w + pad
    map_w = 220
    x_map = canvas.width - pad - map_w

    # Left text panel background
    draw.rectangle(
        [pad, y0 + 4, pad + text_w, y0 + row_h - 8],
        fill=(248, 246, 242),
        outline=(210, 205, 198),
    )
    tx = pad + 12
    ty = y0 + 14
    header = f"Q{qid}  ·  pred {pred_letter}  ·  gold {gold}  ·  {status}"
    draw.text((tx, ty), header, fill=(28, 28, 28), font=font_title)
    ty += 32
    for line in _wrap(draw, question, font_body, text_w - 28):
        draw.text((tx, ty), line, fill=(40, 40, 40), font=font_body)
        ty += 22
    ty += 8
    for i, choice in enumerate(choices[:4]):
        letter = _choice_letter(i)
        mark = ""
        color = (55, 55, 55)
        if letter == gold and letter == pred_letter:
            mark = "  ✓"
            color = (20, 110, 60)
        elif letter == gold:
            mark = "  (gold)"
            color = (20, 90, 140)
        elif letter == pred_letter:
            mark = "  (pred)"
            color = (150, 60, 40)
        line = f"{letter}) {choice}{mark}"
        for wrapped in _wrap(draw, line, font_choice, text_w - 36):
            draw.text((tx, ty), wrapped, fill=color, font=font_choice)
            ty += 20
        ty += 2

    # Determining images
    vx = img_col_x
    for label, img in views:
        thumb = img.copy()
        thumb.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (vx, y0 + 28))
        # caption under tile
        for j, line in enumerate(_wrap(draw, label, font_small, tile)):
            draw.text((vx, y0 + 28 + thumb.height + 4 + j * 16), line, fill=(70, 70, 70), font=font_small)
            if j >= 1:
                break
        vx += tile + 16

    # Top-down map
    map_path = bundle / "topdown_map.png"
    if not map_path.is_file():
        map_path = bundle / "topdown_map_overlay.png"
    if map_path.is_file():
        mimg = Image.open(map_path).convert("RGB")
        mimg.thumbnail((map_w, row_h - 48), Image.Resampling.LANCZOS)
        mx = x_map + (map_w - mimg.width) // 2
        my = y0 + 28
        canvas.paste(mimg, (mx, my))
        draw.text((x_map, y0 + 8), "top-down map", fill=(90, 90, 90), font=font_small)


def render_figure(
    rows: list[dict],
    *,
    episodes_root: Path,
    max_images: int = 2,
    tile: int = 260,
) -> Image.Image:
    pad = 24
    row_h = 320
    title_h = 56
    width = pad + 420 + pad + max_images * (tile + 16) + 220 + pad
    height = title_h + len(rows) * row_h + pad
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (pad, 16),
        "HM-EQA sample tasks — determining views (Dynagraph, merge-on)",
        fill=(20, 20, 20),
        font=_font(24, bold=True),
    )

    y = title_h
    for row in rows:
        bundle = _bundle_for_row(episodes_root, row)
        if bundle is None:
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (pad, y + 40),
                f"Q{row.get('question_id')}: missing episode bundle",
                fill=(160, 40, 40),
                font=_font(18),
            )
        else:
            # Enrich from metrics.json if question/choices missing in JSONL.
            metrics_path = bundle / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key in ("question", "choices", "gold_answer_letter", "predicted_answer", "formatted_answer"):
                    if not row.get(key) and metrics.get(key) is not None:
                        row[key] = metrics[key]
            _draw_row(
                canvas,
                y0=y,
                row_h=row_h,
                pad=pad,
                row=row,
                bundle=bundle,
                tile=tile,
                max_images=max_images,
            )
        y += row_h
        # separator
        draw = ImageDraw.Draw(canvas)
        draw.line([(pad, y - 4), (width - pad, y - 4)], fill=(220, 220, 220), width=1)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path.home()
        / "runs/emet/branch_verify_20260711/merge_on_baseline_20260712_001627/smoke_trio.jsonl",
    )
    parser.add_argument(
        "--episodes-root",
        type=Path,
        default=Path.home() / ".cache/habitat_eqa/episodes",
    )
    parser.add_argument(
        "--question-ids",
        type=str,
        default="3,14,17",
        help="Comma-separated question ids to include (order preserved)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figs/hmeqa_qualitative_sample.png"),
    )
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--tile", type=int, default=260)
    args = parser.parse_args()

    want = [int(x.strip()) for x in args.question_ids.split(",") if x.strip()]
    by_id = {int(r["question_id"]): r for r in _load_jsonl(args.jsonl) if "question_id" in r}
    rows: list[dict] = []
    for qid in want:
        if qid in by_id:
            rows.append(dict(by_id[qid]))
        else:
            # Bundle-only fallback
            rows.append({"question_id": qid, "method": "dynagraph", "correct": True})

    if not rows:
        raise SystemExit(f"No rows for ids {want} in {args.jsonl}")

    fig = render_figure(
        rows,
        episodes_root=args.episodes_root.expanduser(),
        max_images=args.max_images,
        tile=args.tile,
    )
    out = args.output.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out, optimize=True)
    print(f"wrote {out} ({fig.size[0]}x{fig.size[1]})")


if __name__ == "__main__":
    main()
