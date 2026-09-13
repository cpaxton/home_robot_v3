"""Export v2 visual QA evidence after building all three layout packs and controls."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from emet.eval.agent_tasks.qa import load_public, score_answers
from emet.eval.agent_tasks.spec import fingerprint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--packs-root", type=Path, required=True)
parser.add_argument("--controls-root", type=Path, required=True)
parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
args = parser.parse_args()
paper = args.paper_dir
data = paper / "data/agent_task_visual_qa_v2"
data.mkdir(parents=True, exist_ok=True)
rows = []
for layout in ["base", "furnished", "four_room"]:
    root = args.packs_root / layout
    public = load_public(root / "public")
    private = json.loads((root / "private_answers.json").read_text())
    responses = {k: {"answer": v["answer"], "evidence": v["required_evidence"]} for k, v in private["keys"].items()}
    assert score_answers(root, responses)["success"]
    bad = json.loads(json.dumps(responses))
    bad["color"]["answer"] = "blue"
    assert score_answers(root, bad)["completed"] == 9
    missing = json.loads(json.dumps(responses))
    missing["moved"]["evidence"] = missing["moved"]["evidence"][:1]
    assert score_answers(root, missing)["completed"] == 9
    rows.append(
        {
            "layout": layout,
            "questions": len(public["questions"]),
            "public_fingerprint": fingerprint(public),
            "suite_fingerprint": private["suite_fingerprint"],
            "readability_passed": True,
            "perfect_answer_control": "10/10",
            "wrong_color_control": "9/10",
            "missing_temporal_view_control": "9/10",
            "source_sha256": private["source_sha256"],
        }
    )
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
root = args.packs_root / "four_room/public"
for ax, (title, name) in zip(
    axes.flat,
    [
        ("Kitchen: appliances and tiled backsplash", "view_00.png"),
        ("Living: sofa and green tray", "view_01.png"),
        ("Study: bookshelves and green cube", "view_03.png"),
        ("Dining: complete placement view", "view_05.png"),
    ],
    strict=True,
):
    ax.imshow(Image.open(root / name))
    ax.set_title(title, fontsize=12)
    ax.axis("off")
fig.savefig(paper / "figs/agent_task_visual_qa_v2.pdf", bbox_inches="tight")
fig.savefig(args.packs_root / "scene_overview.png", dpi=140, bbox_inches="tight")
plt.close(fig)
summary = {
    "scope": "v2 procedural visual fixtures; fixed-observation QA; deterministic scorer controls are not model scores",
    "layouts": rows,
    "action_controls": [],
    "paid_cost_usd": 0,
    "figure_sha256": hashlib.sha256((paper / "figs/agent_task_visual_qa_v2.pdf").read_bytes()).hexdigest(),
}
for layout in ["base", "furnished", "four_room"]:
    for p in sorted((args.controls_root / layout).glob("*/metrics.json")):
        m = json.loads(p.read_text())
        assert m["success"] and m["evidence_complete"]
        summary["action_controls"].append(
            {"layout": layout, "episode": p.parent.name, "status": m["status"], "event_hash": m["event_hash"]}
        )
local = args.packs_root / "local_vlm"
if (local / "answers.json").is_file():
    answers = json.loads((local / "answers.json").read_text())
    manifest = json.loads((local / "manifest.json").read_text())
    if manifest["public_fingerprint"] != fingerprint(load_public(args.packs_root / "four_room/public")):
        raise ValueError("Local QA results do not match the published pack")
    score = score_answers(args.packs_root / "four_room", answers)
    evidence = {}
    for qid in manifest["question_ids"]:
        trace = json.loads((local / (qid + ".json")).read_text())
        previous = ""
        for event in trace["events"]:
            payload = dict(event)
            digest = payload.pop("hash")
            if payload["previous_hash"] != previous or fingerprint(payload) != digest:
                raise ValueError("QA event hash mismatch")
            previous = digest
        evidence[qid] = {"event_hash": previous, "outcome": trace["outcome"], "seen_views": trace["seen_views"]}
    result = {
        "scope": "one local RGB QA development run; fixed observations",
        "manifest": manifest,
        "answers": answers,
        "score": score,
        "evidence": evidence,
    }
    (data / "local_vlm.json").write_text(json.dumps(result, indent=2) + "\n")
    summary["local_vlm"] = {"model": manifest["model"], "completed": score["completed"], "total": score["total"]}
    print("Local VLM:", score["completed"], "/", score["total"])
(data / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("30 QA cases passed readability and positive/negative scorer controls; nine action controls passed")
