# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Audit saved assessment packets without moving the robot or selecting new views.

Writes a portable HTML gallery and lossless copies of the original PNGs. Optional
inference compares image subsets with identical labels, plus an unlabeled single
image control approximating the historical input contract (not the old run).
Inventory is intentionally empty in every arm; this is not an exact live replay.
Never interpret offline answerability as physical visibility or task success.
"""

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path
from types import SimpleNamespace


def prepare(root, output):
    output.mkdir(parents=True, exist_ok=False)
    cases, sections = [], []
    for manifest_path in sorted(root.glob("s*-attempt*/q*/selected_views/*/manifest.json")):
        episode = manifest_path.parents[2]
        metadata = episode / "eqa_decisions/iter_001/meta.json"
        if not metadata.exists():
            continue
        meta = json.loads(metadata.read_text())
        packet = json.loads(manifest_path.read_text())
        name = f"{episode.parent.name}-{manifest_path.parent.name}"
        destination = output / name
        destination.mkdir()
        views, cards = [], []
        for index, row in enumerate(packet["views"], 1):
            source = manifest_path.parent / row["image"]
            filename = f"image_{index}.png"
            shutil.copy2(source, destination / filename)
            rel = f"{name}/{filename}"
            views.append({**row, "path": rel, "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
            caption = html.escape(row.get("label") or f"Image {index} · {row['role']} · view {row['view_id']}")
            age = row.get("age_s")
            timing = "" if age is None else f" · {age:.1f}s before current"
            cards.append(
                f'<figure><a href="{rel}"><img src="{rel}" loading="lazy"></a>'
                f"<figcaption>{caption}{timing}</figcaption></figure>"
            )
        prompt_labels = packet.get("prompt_labels", "")
        case = {
            "name": name,
            "source": str(manifest_path),
            "question": meta["question"],
            "prompt_labels": prompt_labels,
            "views": views,
        }
        cases.append(case)
        # The final answer is episode-level, not necessarily this assessment.
        answer = html.escape(str(meta.get("parsed", {})))
        sections.append(
            f'<section><h2>{html.escape(name)}</h2><p class="question">{html.escape(meta["question"])}</p>'
            f"<details><summary>Episode final parsed response (not necessarily this packet)</summary><pre>{answer}</pre></details>"
            f"<details><summary>Exact packet labels sent to the model</summary><pre>{html.escape(prompt_labels)}</pre></details>"
            f'<div class="grid">{"".join(cards)}</div></section>'
        )
    if not cases:
        raise ValueError("No saved assessment packets with question metadata found")
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>Frozen EQA evidence audit</title><style>"
        "body{font:18px system-ui;margin:32px;background:#f3f5f7;color:#17212b;max-width:1500px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px}"
        "figure{margin:0;background:white;padding:10px;border-radius:8px}img{width:100%;height:auto}"
        "figcaption{padding:8px}section{margin:36px 0;border-top:2px solid #cbd5df}"
        ".question{font-size:22px;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "@media print{section{break-before:page}body{margin:0;background:white}.grid{grid-template-columns:1fr 1fr}}"
        "</style><h1>Frozen EQA evidence audit</h1><p>Original, uncropped RGB. Click any image for full resolution. "
        "Current establishes current visibility; historical images supply context, not proof of present reachability. "
        "All saved assessment packets are shown; these are not necessarily final submissions.</p>"
        + "".join(sections)
        + "</html>"
    )
    return cases


def infer(cases, output, repeats, provenance_only=False):
    import numpy as np
    import torch
    from PIL import Image

    from emet.core.parameters import get_parameters
    from emet.eval.agentic_vlm_assess import assess_view_with_vlm
    from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients
    from emet.memory.view_selection import spatial_packet_labels

    params = get_parameters("configs/emet/default.yaml")
    (output / "eqa_config.json").write_text(json.dumps(dict(params.get("eqa") or {}), indent=2, default=str))
    _, client = build_graph_eqa_vlm_clients(parameters=params, device="cuda")
    with (output / "responses.jsonl").open("x") as stream:
        for case in cases:
            arrays = [np.asarray(Image.open(output / row["path"]).convert("RGB")) for row in case["views"]]
            for repeat in range(repeats):
                arms = (
                    ["labeled_full", "spatial_full"]
                    if provenance_only
                    else ["legacy_single", "labeled_single", "labeled_pair", "labeled_full"]
                )
                if repeat % 2:
                    arms.reverse()
                for arm in arms:
                    count = (
                        len(arrays)
                        if arm in {"labeled_full", "spatial_full"}
                        else (min(2, len(arrays)) if arm == "labeled_pair" else 1)
                    )
                    labels = "\n".join(row["label"] for row in case["views"][:count])
                    if arm == "spatial_full":
                        labels = spatial_packet_labels(case["views"][:count])
                    packet = SimpleNamespace(
                        views=[SimpleNamespace(frame=SimpleNamespace(rgb=rgb)) for rgb in arrays[:count]],
                        prompt_labels=lambda: labels,
                    )
                    torch.manual_seed(repeat)
                    np.random.seed(repeat)
                    result = assess_view_with_vlm(
                        client,
                        question=case["question"],
                        rgb=arrays[0],
                        is_mcq=True,
                        **({} if arm == "legacy_single" else {"selected_views": packet}),
                    )
                    if not result.raw:
                        raise RuntimeError("Empty inference response; not a valid abstention")
                    row = {"case": case["name"], "arm": arm, "repeat": repeat, "image_count": count, **result.to_dict()}
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help="Compare identical full packets with/without capture sequence and camera pose labels",
    )
    args = parser.parse_args()
    cases = prepare(args.root, args.output)
    print(f"Exported {len(cases)} packets to {args.output / 'index.html'}", flush=True)
    if args.infer:
        infer(cases, args.output, args.repeats, args.provenance_only)
