#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Render success/failure graph-retrieval panels from HM-EQA episode bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError as exc:
    raise SystemExit("Pillow required: uv sync") from exc


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


def _obs_images(bundle: Path, obs_ids: list[int]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for oid in obs_ids:
        found = False
        for pattern in (
            f"rgb/obs_{oid:04d}.png",
            f"frames/obs_{oid:04d}.png",
            f"rgb/{oid}.png",
            f"frames/rgb_{oid:04d}.png",
        ):
            p = bundle / pattern
            if p.is_file():
                images.append(Image.open(p).convert("RGB"))
                found = True
                break
        if found:
            continue
        crops = bundle / "dynagraph" / "crops"
        if crops.is_dir():
            for cp in crops.glob(f"*_{oid}.*"):
                if cp.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    images.append(Image.open(cp).convert("RGB"))
                    break
    return images


def _mosaic(images: list[Image.Image], labels: list[str], title: str, tile: int = 256) -> Image.Image:
    n = max(1, len(images))
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    header = 72
    canvas = Image.new("RGB", (cols * tile, rows * tile + header), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title[:200], fill=(240, 240, 240))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        thumb = img.copy()
        thumb.thumbnail((tile - 8, tile - 24), Image.Resampling.LANCZOS)
        x = c * tile + 4
        y = header + r * tile + 4
        canvas.paste(thumb, (x, y))
        if i < len(labels):
            draw.text((x, y + thumb.height + 2), labels[i][:40], fill=(200, 200, 200))
    return canvas


def _selected_obs_ids(bundle: Path) -> list[int]:
    hist_path = bundle / "eqa_history.json"
    if hist_path.is_file():
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        iters = hist.get("iterations") if isinstance(hist, dict) else None
        if isinstance(iters, list) and iters:
            last = iters[-1]
            if isinstance(last, dict):
                ids = last.get("selected_obs_ids") or last.get("obs_ids") or []
                return [int(x) for x in ids]
    obs_hist = bundle / "observations_history.jsonl"
    if obs_hist.is_file():
        ids: list[int] = []
        for line in obs_hist.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "observation" and "obs_idx" in row:
                ids.append(int(row["obs_idx"]))
        if ids:
            return ids[-6:]
    frames = sorted((bundle / "frames").glob("rgb_*.png"))
    if frames:
        return list(range(max(0, len(frames) - 6), len(frames)))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument(
        "--episodes-root",
        type=Path,
        default=Path.home() / ".cache/habitat_eqa/episodes",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-panels", type=int, default=12)
    parser.add_argument(
        "--filter",
        choices=("failures", "successes", "debias_flip", "all"),
        default="debias_flip",
    )
    args = parser.parse_args()

    rows = _load_jsonl(args.jsonl)
    picked: list[dict] = []
    for row in rows:
        ok = bool(row.get("correct"))
        pre = str(row.get("predebias_letter") or "")
        pred = str(row.get("predicted_answer") or "")
        if args.filter == "failures" and ok:
            continue
        if args.filter == "successes" and not ok:
            continue
        if args.filter == "debias_flip" and (not pre or pre == pred):
            continue
        picked.append(row)
    picked = picked[: args.max_panels]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for row in picked:
        bundle = _bundle_for_row(args.episodes_root.expanduser(), row)
        if bundle is None:
            continue
        obs_ids = _selected_obs_ids(bundle)
        imgs = _obs_images(bundle, obs_ids[:6])
        if not imgs:
            continue
        qid = int(row.get("question_id", -1))
        pred = str(row.get("predicted_answer") or "")
        pre = str(row.get("predebias_letter") or "")
        title = f"Q{qid:03d} gold={row.get('gold_answer_letter', '?')} pred={pred} pre={pre} ok={row.get('correct')}"
        labels = [f"obs {oid}" for oid in obs_ids[: len(imgs)]]
        panel = _mosaic(imgs, labels, title)
        out = args.output_dir / f"retrieval_q{qid:03d}_{row.get('method', 'x')}.png"
        panel.save(out)
        manifest.append({"question_id": qid, "panel": str(out), "bundle": str(bundle)})
        print(f"wrote {out}")

    (args.output_dir / "retrieval_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
