"""Image-grounded QA packs, private deterministic scoring and local agent execution.

Public observations never contain answer keys, object labels or evaluator poses.
The single named-room memory case explicitly declares its metadata assistance.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
from pathlib import Path

from .recording import write_json
from .spec import fingerprint, load_suite


def readable(value: dict) -> bool:
    """Reject clipped or tiny target evidence, not merely nonzero segmentation."""
    x0, y0, x1, y1 = value.get("bbox_xyxy", [0, 0, 0, 0])
    return value.get("pixels", 0) >= 128 and min(x1 - x0 + 1, y1 - y0 + 1) >= 8 and not value.get("clipped", True)


def validate_qa_scene(suite):
    """This initial question set is valid only for its declared fixture contract."""
    scene = suite["scene"]
    if scene.get("visual_profile") != "semantic_rooms_v2":
        raise ValueError("QA requires semantic_rooms_v2")
    expected = {
        "red_cylinder": (0, "cylinder"),
        "blue_cube": (2, "box"),
        "green_marker": (1, "box"),
        "red_tray": (0, "box"),
        "blue_tray": (2, "box"),
        "green_tray": (1, "box"),
    }
    if set(scene["objects"]) != set(expected):
        raise ValueError("QA requires the six declared task objects")
    for name, (channel, shape) in expected.items():
        obj = scene["objects"][name]
        if obj["shape"] != shape or obj["color"][channel] <= max(
            c for i, c in enumerate(obj["color"][:3]) if i != channel
        ):
            raise ValueError(f"QA shape/color contract changed: {name}")
    rooms = {r["id"] for r in scene["rooms"]}
    if not {"kitchen", "living", "dining"}.issubset(rooms):
        raise ValueError("QA requires kitchen, living, and dining rooms")


def build_pack(suite: dict, output: Path):
    from PIL import Image

    from .fixture import FixtureRobot

    validate_qa_scene(suite)
    output.mkdir(parents=True, exist_ok=False)
    public = output / "public"
    public.mkdir()
    robot = FixtureRobot(suite, output)
    frames, private_frames = {}, {}
    try:

        def capture(key, room_name, required):
            room = next(r for r in suite["scene"]["rooms"] if r["id"] == room_name)
            robot.move_base_to([(room["bounds"][0] + room["bounds"][2]) / 2, 0, -math.pi / 2])
            visible = robot.visible_objects()
            bad = [name for name in required if name not in visible or not readable(visible[name])]
            if bad:
                raise ValueError(f"QA evidence unreadable: {key}: {bad}")
            fid = f"view:{len(frames)}"
            image_name = f"view_{len(frames):02d}.png"
            Image.fromarray(robot.images()["head"]).save(public / image_name)
            frames[key] = {
                "id": fid,
                "image": image_name,
                "sha256": hashlib.sha256((public / image_name).read_bytes()).hexdigest(),
            }
            private_frames[fid] = {
                "room": room_name,
                "positions": robot.positions(),
                "visible": visible,
                "camera": robot.camera_metadata()["head"],
                "robot_xyt": robot.xyt[:],
            }
            return fid

        obj = suite["scene"]["objects"]

        def room_of(name):
            x = obj[name]["position"][0]
            return next(r["id"] for r in suite["scene"]["rooms"] if r["bounds"][0] < x < r["bounds"][2])

        capture("start", "kitchen", ["red_cylinder"])
        capture("living", "living", ["blue_cube", "green_tray"])
        capture("dining", "dining", ["red_tray", "blue_tray"])
        capture("green", room_of("green_marker"), ["green_marker"])
        initial = robot.positions()["red_cylinder"]
        robot.relocate("red_cylinder", [initial[0] + 0.45, initial[1], initial[2]])
        capture("changed", "kitchen", ["red_cylinder"])
        for name, tray in [("red_cylinder", "red_tray"), ("blue_cube", "blue_tray")]:
            p = robot.positions()[tray]
            robot.relocate(name, [p[0], p[1], p[2] + 0.02])
        capture("placed", "dining", ["red_cylinder", "red_tray", "blue_cube", "blue_tray"])
    finally:
        robot.close()

    # Derive image-relative directions using actual camera right axes.
    def image_direction(view, a, b):
        matrix = private_frames[frames[view]["id"]]["camera"]["rotation_world_from_camera"]
        delta = sum(matrix[i][0] * (a[i] - b[i]) for i in range(3))
        return "right" if delta > 0 else "left" if delta < 0 else "unchanged"

    relative = image_direction("living", obj["blue_cube"]["position"], obj["green_tray"]["position"])
    moved = image_direction("start", private_frames[frames["changed"]["id"]]["positions"]["red_cylinder"], initial)
    questions, keys = [], {}

    def question(qid, category, text, views, answer, answer_format, metadata=None):
        ids = [frames[v]["id"] for v in views]
        questions.append(
            {
                "id": qid,
                "category": category,
                "question": text,
                "view_ids": ids,
                "answer_format": answer_format,
                "metadata": metadata or {},
            }
        )
        keys[qid] = {"answer": answer, "required_evidence": ids}

    question("color", "attribute", "What color is the small cylinder on the table?", ["start"], "red", "one color word")
    question("tray_count", "count", "How many trays are visible on the table?", ["dining"], 2, "integer")
    question(
        "relative_position",
        "spatial",
        "Is the blue cube to the left or right of the green tray in this image?",
        ["living"],
        relative,
        "left or right (image coordinates)",
    )
    question("empty_trays", "occupancy", "Does either tray contain a small object?", ["dining"], "no", "yes or no")
    question(
        "room_identity",
        "room_semantics",
        "Which room type best matches the furnishings: kitchen, living room, study, or dining room?",
        ["start"],
        "kitchen",
        "one of the four room types",
    )
    question(
        "observed_room",
        "multi_view_memory",
        "In which named room was the blue cube observed?",
        ["start", "living", "green"],
        "living",
        "room ID",
        {frames[v]["id"]: private_frames[frames[v]["id"]]["room"] for v in ["start", "living", "green"]},
    )
    question(
        "inventory",
        "multi_view_inventory",
        "Across these views, list the colors of the small task objects; exclude trays and furnishings. List each color once.",
        ["start", "living", "green"],
        ["blue", "green", "red"],
        "JSON array of color words",
    )
    question(
        "moved",
        "temporal",
        "The views are before then after, from the same camera pose. Did the cylinder move left, move right, or stay in place in the image?",
        ["start", "changed"],
        moved,
        "left, right, or unchanged",
    )
    question(
        "placement",
        "action_verification",
        "Is the blue cube in the blue tray and the red cylinder in the red tray?",
        ["placed"],
        "yes",
        "yes or no",
    )
    question(
        "unknown_location",
        "insufficient_evidence",
        "From this image alone, which room contains the blue cube? Answer unknown if the image does not establish its location.",
        ["start"],
        "unknown",
        "room ID or unknown",
    )
    pack = {
        "schema_version": 1,
        "profile": suite["scene"].get("visual_profile"),
        "questions": questions,
        "views": {v["id"]: v for v in frames.values()},
        "scope": "fixed observation QA; no active exploration",
        "paid_requests_enabled": False,
    }
    write_json(public / "questions.json", pack)
    write_json(
        output / "private_answers.json",
        {
            "public_fingerprint": fingerprint(pack),
            "suite_fingerprint": fingerprint(suite),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    Path(__file__).with_name("fixture.py"),
                    Path(__file__).with_name("scene_style.py"),
                ]
            },
            "keys": keys,
            "frames": private_frames,
            "assistance": "rendered procedural scenes; staged relocation/placement; named-room metadata only in observed_room",
        },
    )
    export_gallery(output, pack, keys)
    return {
        "questions": len(questions),
        "views": len(frames),
        "readability_passed": True,
        "public": str(public),
        "gallery": str(output / "review.html"),
    }


def load_public(public: Path):
    pack = json.loads((public / "questions.json").read_text())
    for view in pack["views"].values():
        path = (public / view["image"]).resolve()
        if not path.is_relative_to(public.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != view["sha256"]:
            raise ValueError("invalid QA image evidence")
    return pack


def normalize(answer):
    if isinstance(answer, str):
        return answer.strip().lower().rstrip(".")
    if isinstance(answer, list):
        if not all(isinstance(v, str) for v in answer):
            return None
        return sorted(normalize(v) for v in answer)
    if isinstance(answer, int) and not isinstance(answer, bool):
        return answer
    return None


def score_answers(root: Path, responses: dict):
    pack = load_public(root / "public")
    key = json.loads((root / "private_answers.json").read_text())
    if fingerprint(pack) != key["public_fingerprint"]:
        raise ValueError("QA questions do not match answer key")
    if not isinstance(responses, dict) or set(responses) - set(key["keys"]):
        raise ValueError("unknown question IDs or invalid response object")
    rows = []
    for qid, truth in key["keys"].items():
        response = responses.get(qid, {})
        if not isinstance(response, dict):
            response = {}
        evidence = response.get("evidence", [])
        valid_evidence = isinstance(evidence, list) and all(isinstance(v, str) for v in evidence)
        valid_evidence = (
            valid_evidence
            and set(truth["required_evidence"]).issubset(evidence)
            and set(evidence).issubset(truth["required_evidence"])
        )
        answer_ok = normalize(response.get("answer")) == normalize(truth["answer"])
        rows.append(
            {
                "id": qid,
                "answer_correct": answer_ok,
                "evidence_complete": bool(valid_evidence),
                "passed": answer_ok and bool(valid_evidence),
            }
        )
    return {
        "completed": sum(r["passed"] for r in rows),
        "total": len(rows),
        "success": all(r["passed"] for r in rows),
        "questions": rows,
        "scope": "answer and cited evidence correctness; citations alone do not prove model inspection",
    }


def export_gallery(root, pack, keys, *, responses=None, destination=None):
    sections = []
    for q in pack["questions"]:
        imgs = []
        for fid in q["view_ids"]:
            data = base64.b64encode((root / "public" / pack["views"][fid]["image"]).read_bytes()).decode()
            imgs.append(
                f'<figure><img src="data:image/png;base64,{data}" alt="{fid}"><figcaption>{fid}</figcaption></figure>'
            )
        model_result = ""
        if responses is not None:
            model_result = (
                "<p><strong>Model submission</strong></p><pre>"
                + html.escape(json.dumps(responses.get(q["id"], {})))
                + "</pre>"
            )
        sections.append(
            f'<section><h2>{html.escape(q["question"])}</h2><p>{q["category"]} · {html.escape(q["answer_format"])}</p><div class="views">'
            + "".join(imgs)
            + "</div>"
            + model_result
            + f"<details><summary>Reviewer answer key</summary><pre>{html.escape(json.dumps(keys[q['id']]))}</pre></details></section>"
        )
    (destination or root / "review.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visual QA preflight</title><style>body{font:17px system-ui;background:#edf2f6;color:#172033;margin:0}main{max-width:1450px;margin:auto;padding:24px}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.views{display:flex;flex-wrap:wrap;gap:12px}figure{flex:1;min-width:320px;margin:0}img{width:100%}summary{cursor:pointer}</style><main><h1>Visual QA preflight</h1><p>Reviewer gallery with private answers. Give agents only the public directory. Fixed observations, staged changes, no paid calls.</p>'
        + "".join(sections)
        + "</main></html>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite")
    parser.add_argument("--public")
    parser.add_argument("--model", default="qwen35-0.8B")
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.public:
        result = run_local_questions(
            Path(args.public), Path(args.out), model=args.model, question_ids=args.question, max_rounds=args.max_rounds
        )
    else:
        result = build_pack(load_suite(args.suite), Path(args.out))
    print(json.dumps(result, indent=2))


def run_question(question, pack, public, client, *, max_rounds=8, on_event=None):
    """Run one image-scoped question through the shared task agent and tools."""
    import numpy as np
    from PIL import Image

    from emet.agent.task import run_agent_task
    from emet.agent.tools import Tool

    cursor, seen, submitted, events = 0, [], {}, []

    def next_view():
        nonlocal cursor
        if cursor + 1 >= len(question["view_ids"]):
            return "No further views. Submit your answer using the observed evidence."
        cursor += 1
        return "Next view will be supplied with the next observation."

    def submit_answer(answer, evidence):
        if submitted:
            return "An answer was already submitted."
        if not isinstance(evidence, list) or any(not isinstance(v, str) or v not in seen for v in evidence):
            return "Evidence must cite only view IDs already supplied to you."
        if set(seen) != set(question["view_ids"]):
            return "Inspect the remaining supplied views before submitting."
        submitted.update(answer=answer, evidence=evidence)
        return "Answer recorded."

    tools = [
        Tool(
            "next_view",
            "Inspect the next supplied image; there is no movement or hidden scene query.",
            {"type": "object", "properties": {}},
            next_view,
            returns_info=True,
        ),
        Tool(
            "submit_answer",
            "Submit one answer and the observed view IDs that support it.",
            {
                "type": "object",
                "properties": {"answer": {}, "evidence": {"type": "array", "items": {"type": "string"}}},
                "required": ["answer", "evidence"],
            },
            submit_answer,
            returns_info=True,
        ),
    ]

    def observe():
        fid = question["view_ids"][cursor]
        if fid not in seen:
            seen.append(fid)
        state = {
            "view_id": fid,
            "view_number": cursor + 1,
            "total_views": len(question["view_ids"]),
            "seen_views": seen[:],
            "image_observation_ids": [fid],
            "answer_submitted": bool(submitted),
        }
        if question.get("metadata"):
            state["named_room"] = question["metadata"].get(fid)
        with Image.open(public / pack["views"][fid]["image"]) as im:
            image = np.asarray(im.convert("RGB"))
        return state, image

    instruction = (
        question["question"]
        + " Answer format: "
        + question["answer_format"]
        + " Use next_view with no arguments to inspect remaining supplied images, then submit_answer with answer and evidence (a list of view IDs). "
        'Cite every supplied view ID in evidence. Do not answer from unseen images. Tools: {"name":"next_view","arguments":{}} or {"name":"submit_answer","arguments":{"answer":VALUE,"evidence":["view:0"]}}.'
    )

    def emit(kind, payload):
        event = {"kind": kind, **payload, "previous_hash": events[-1]["hash"] if events else ""}
        event["hash"] = fingerprint(event)
        events.append(event)
        if on_event:
            on_event(event)

    outcome = run_agent_task(
        instruction,
        llm_client=client,
        tools=tools,
        observe=observe,
        max_rounds=max_rounds,
        max_tools_per_round=1,
        max_actions=16,
        completed=lambda: bool(submitted),
        on_event=emit,
    )
    return {"response": submitted, "outcome": outcome, "events": events, "seen_views": seen}


def run_local_questions(public: Path, output: Path, *, model="qwen35-0.8B", question_ids=(), max_rounds=8):
    """Only public question data enters this process; scoring is a separate operation."""
    import os

    import torch

    from emet.agent.task import TASK_INSTRUCTION
    from emet.llms import get_llm_client

    if model not in {"qwen35-0.8B", "qwen35-2B", "qwen35-4B"}:
        raise ValueError("QA model must be an allowlisted cached local VLM")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    pack = load_public(public)
    selected = [q for q in pack["questions"] if not question_ids or q["id"] in question_ids]
    if not selected or set(question_ids) - {q["id"] for q in selected}:
        raise ValueError("unknown question IDs")
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    client = get_llm_client(model, prompt=TASK_INSTRUCTION, device="cpu", quantization=None, max_tokens=128)
    client.model.float()
    responses = {}
    write_json(
        output / "manifest.json",
        {
            "model": model,
            "model_id": client.model.config._name_or_path,
            "model_revision": getattr(client.model.config, "_commit_hash", None),
            "public_fingerprint": fingerprint(pack),
            "question_ids": [q["id"] for q in selected],
            "paid_cost_usd": 0,
            "mode": "shared_task_agent_rgb_qa",
            "max_rounds": max_rounds,
            "source_sha256": {
                str(p.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    Path(__file__).resolve().parents[2] / "agent/task.py",
                    Path(__file__).resolve().parents[2] / "agent/loop.py",
                ]
            },
        },
    )
    manifest = json.loads((output / "manifest.json").read_text())
    for relative, expected_hash in manifest["source_sha256"].items():
        content = (Path(__file__).resolve().parents[2] / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError("QA source changed while preparing the run")
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    for q in selected:
        with (output / (q["id"] + ".events.jsonl")).open("x") as log:

            def record(event):
                log.write(json.dumps(event) + "\n")
                log.flush()

            result = run_question(q, pack, public, client, max_rounds=max_rounds, on_event=record)
        write_json(output / (q["id"] + ".json"), result)
        responses[q["id"]] = result["response"]
        write_json(output / "answers.json", responses)
    return {"submitted": sum(bool(v) for v in responses.values()), "attempted": len(selected), "output": str(output)}


if __name__ == "__main__":
    main()
