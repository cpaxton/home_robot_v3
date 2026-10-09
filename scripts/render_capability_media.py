#!/usr/bin/env python3
"""Build a portable HTML review gallery and paper contact sheet from real run media.

Input is JSON with entries: title, caption, run_id, image, optional video/evidence.
Paths resolve relative to the JSON file. Never synthesizes simulator frames.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    entries = data["entries"]
    if not entries:
        parser.error("manifest must contain at least one entry")
    args.output.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(entries), figsize=(5 * len(entries), 4), squeeze=False)
    cards, provenance = [], []
    for index, (entry, ax) in enumerate(zip(entries, axes[0], strict=True)):
        copied = {}
        records = {}
        for kind in ("image", "video", "evidence"):
            if not entry.get(kind):
                continue
            source = (args.manifest.parent / entry[kind]).resolve(strict=True)
            target = args.output / f"{index:02d}_{kind}{source.suffix}"
            shutil.copyfile(source, target)
            copied[kind] = target.name
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            records[kind] = {"source": str(source), "file": target.name, "sha256": digest.hexdigest()}
        image = args.output / copied["image"]
        ax.imshow(plt.imread(image))
        ax.set_title(entry["title"], fontsize=12)
        ax.axis("off")
        esc = html.escape
        links = " ".join(f'<a href="{esc(path)}">{esc(kind)}</a>' for kind, path in copied.items())
        video = f'<video controls preload="metadata" src="{esc(copied["video"])}"></video>' if "video" in copied else ""
        cards.append(f'<article><h2>{esc(entry["title"])}</h2><p>{esc(entry["caption"])}</p>'
                     f'<p>Run: <code>{esc(entry["run_id"])}</code></p>'
                     f'<img src="{esc(copied["image"])}" alt="{esc(entry["title"])}">{video}<p>{links}</p></article>')
        provenance.append({**entry, "assets": records})
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180, bbox_inches="tight")
    fig.savefig(args.figure.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Robot capability recordings</title>'
        '<style>body{font:17px sans-serif;max-width:1100px;margin:40px auto;padding:20px;background:#fafafa}'
        'article{background:white;border:1px solid #ddd;padding:20px;margin:24px 0}img,video{max-width:100%;max-height:580px}'
        'a{margin-right:20px}</style><h1>Robot capability recordings</h1>'
        '<p>Actual simulator outputs. Qualitative examples; assistance, dates and outcomes are identified per run.</p>'
        + ''.join(cards) + '<p><a href="manifest.json">Provenance and SHA-256 hashes</a></p>')
    (args.output / "manifest.json").write_text(json.dumps({"schema_version": 1, "entries": provenance}, indent=2) + "\n")
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
