# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Fixed-image context ablation. This is not a trajectory or task-success replay.

The original system prompt is reconstructed from the frozen source/config,
not claimed to have been captured on the wire. Gold is used only for scoring.
"""

import argparse
import hashlib
import html
import json
import random
import re
import subprocess
from pathlib import Path

NEUTRAL_SYSTEM = (
    "Answer the user's question from the supplied visual evidence and context. "
    "Distinguish observations from search hypotheses and historical views from current visibility. "
    "Express uncertainty when evidence is insufficient; an unseen object is not proof of absence. "
    "Return JSON with reasoning, answer, confidence (boolean), action, and confidence_reasoning. "
    'Use a short answer or "Unknown"; action may be empty. Do not invent observations.'
)
ARMS = ("original", "neutral_user", "image_only", "neutral_system")


def user_text(original, arm):
    if arm in ("original", "neutral_system"):
        return original
    if arm == "neutral_user":
        text = re.sub(r"^QUESTION_ROOM.*\n", "", original, flags=re.MULTILINE)
        return text.replace(
            "; for location MCQs, prefer option landmarks visible in Image 1 over nearest-furniture guesses", ""
        )
    if arm == "image_only":
        # Preserve exact question/options and the full original image provenance.
        question = original.split("\n", 1)[0]
        start = original.find("Capture sequence is chronological")
        labels = original[start:] if start >= 0 else ""
        return question + "\n\n" + labels
    raise ValueError(arm)


def prepare(source, output, ids):
    output.mkdir(parents=True, exist_ok=False)
    cases, sections = [], []
    for qid in ids:
        rows = [json.loads(line) for line in (source / f"q{qid}.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != 1 or rows[0].get("error"):
            raise ValueError(f"q{qid}: expected one completed pilot row")
        row = rows[0]
        decisions = Path(row["debug_bundle_dir"]) / "eqa_decisions"
        iteration = sorted(decisions.glob("iter_*/meta.json"))[-1].parent
        meta = json.loads((iteration / "meta.json").read_text())
        prompt = (iteration / "prompt.txt").read_text()
        image_paths = [iteration / item["path"] for item in meta["images"]]
        if not image_paths or len(image_paths) > 4:
            raise ValueError("Expected a bounded saved image packet")
        destination = output / f"q{qid}"
        destination.mkdir()
        images = []
        for i, path in enumerate(image_paths, 1):
            data = path.read_bytes()
            target = destination / f"image_{i}.png"
            target.write_bytes(data)
            images.append({"path": str(target), "sha256": hashlib.sha256(data).hexdigest()})
        for arm in ARMS:
            (destination / f"{arm}.txt").write_text(user_text(prompt, arm))
        case = {
            "qid": qid,
            "question": meta["question"],
            "original_iteration": str(iteration),
            "images": images,
            "gold": row["gold_answer_letter"],
            "choices": row["choices"],
            "original_prediction": row["predicted_answer"],
            "original_correct": row["correct"],
        }
        cases.append(case)
        cards = "".join(f'<img width="480" src="q{qid}/image_{i}.png">' for i in range(1, len(images) + 1))
        sections.append(
            f"<h2>q{qid}: {html.escape(meta['question'])}</h2>{cards}"
            f"<details><summary>Original user context</summary><pre>{html.escape(prompt)}</pre></details>"
        )
    (output / "cases.json").write_text(json.dumps(cases, indent=2))
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Frozen context audit</title>'
        "<style>body{font:18px sans-serif}pre{white-space:pre-wrap}img{max-width:100%}</style>" + "".join(sections)
    )
    return cases


def infer(cases, output, repeats):
    import numpy as np
    import torch
    from PIL import Image

    from emet.core.parameters import get_parameters
    from emet.habitat.metrics import extract_mcq_letter
    from emet.llms.graph_eqa_vlm import _eqa_system_prompt, build_graph_eqa_vlm_clients

    parameters = get_parameters("dynav_config.yaml", robot="stretch")
    system = _eqa_system_prompt(parameters)
    (output / "original_system.txt").write_text(system)
    (output / "neutral_system.txt").write_text(NEUTRAL_SYSTEM)
    (output / "config.json").write_text(json.dumps(dict(parameters.get("eqa")), indent=2, default=str))
    _, client = build_graph_eqa_vlm_clients(parameters=parameters, device="cuda")
    with (output / "responses.jsonl").open("x") as stream:
        for case in cases:
            images = [Image.open(row["path"]).convert("RGB") for row in case["images"]]
            for repeat in range(repeats):
                arms = ARMS if repeat % 2 == 0 else ARMS[::-1]
                for arm in arms:
                    random.seed(repeat)
                    np.random.seed(repeat)
                    torch.manual_seed(repeat)
                    prompt = (output / f"q{case['qid']}" / f"{arm}.txt").read_text()
                    raw = client([prompt, *images], system_prompt=NEUTRAL_SYSTEM if arm == "neutral_system" else system)
                    try:
                        parsed, _ = json.JSONDecoder().raw_decode(str(raw).lstrip())
                    except (ValueError, TypeError):
                        parsed = {}
                    answer = str(parsed.get("answer", ""))
                    letter = extract_mcq_letter(answer, case["choices"])
                    result = {
                        "qid": case["qid"],
                        "arm": arm,
                        "repeat": repeat,
                        "raw": raw,
                        "answer": answer,
                        "predicted": letter,
                        "correct": letter == case["gold"],
                        "parsed": bool(parsed),
                        "images": case["images"],
                    }
                    stream.write(json.dumps(result) + "\n")
                    stream.flush()
                    print(
                        json.dumps(
                            {key: result[key] for key in ("qid", "arm", "repeat", "answer", "correct", "parsed")}
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+", default=[6, 12, 14, 25, 56, 65])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--infer", action="store_true")
    args = parser.parse_args()
    cases = prepare(args.source, args.output, args.ids)
    (args.output / "source.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True))
    if args.infer:
        infer(cases, args.output, args.repeats)
