"""Assemble a workspace pilot from existing simulator demonstration receipts."""

import argparse
import json
from pathlib import Path

import numpy as np

from emet.rl.manipulation.observation_layout import history_layout
from emet.rl.manipulation.profiles import MARS

history = history_layout(MARS, retention_context=True)[0]


def record(identity, hand, base):
    return {"id": identity, "hand": hand.tolist(), "base": base.tolist()}


def from_history(identity, array, mode):
    current = array[:, -1]
    if mode == "base":
        base = current[:, history["base_goal"]]
        hand = current[:, history["hand_pose"]]
    else:
        active = current[:, history["hand_mask"]].ravel() > 0.5
        current = current[active]
        if not len(current):
            return
        base = current[:, history["base_pose"]]
        hand = current[:, history["hand_goal"]]
    return record(identity, hand, base)


def archived_videos(video, name):
    report = json.loads((video / name / "report.json").read_text())
    rows = []
    for row in report["rows"]:
        if not row["success"] or not row["stop_confirmed"] or row["case"]["split"] != "train":
            continue
        with np.load(video / name / row["recording"], allow_pickle=False) as data:
            rows.append(record(row["case"]["id"], data["planned_hand"], data["planned_base"]))
    return rows


def span_bank(folder, collection):
    report = json.loads((folder / "report.json").read_text())
    with np.load(folder / "training.npz", allow_pickle=False) as data:
        array = data["history"]
    return [from_history(e["id"], array[e["start"] : e["stop"]], collection) for e in report["training_episodes"]]


def phase_bank(recovery, mode):
    job = recovery / "overnight-distill-v1"
    report = json.loads((job / f"{mode}-collection.json").read_text())
    with np.load(job / "data" / f"{mode}.npz", allow_pickle=False) as f:
        array = f["history"]
    offset = 0
    data = []
    for row in report["train"]:
        end = offset + row["samples"]
        if end > offset:
            data.append(from_history(row["id"], array[offset:end], mode))
        offset = end
    assert offset == len(array)
    return data


def preplace_bank(recovery):
    job = recovery / "loaded-diversity-v1/handoffs"
    data = []
    for side in ("left", "right"):
        report = json.loads((job / f"handoffs-{side}.json").read_text())
        for row in report["cases"]:
            if row["success"]:
                with np.load(job / row["dataset"], allow_pickle=False) as f:
                    data.append(from_history(row["id"], f["history"][f["stage"] == "preplace"], "preplace"))
    return data


def spec(frame, episodes, edges, description, eligible=None):
    episodes = [e for e in episodes if e]
    assert episodes and len({e["id"] for e in episodes}) == len(episodes)
    return {"frame": frame, "episodes": episodes, "edges": edges, "source_scope": description} | (
        {} if eligible is None else {"eligible_bins": eligible}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True, help="Root containing eval and recovery archives")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recovery = args.logs / "recovery/expert-progression-v1"
    video = args.logs / "eval/curriculum-videos-v1"
    coupled = span_bank(recovery / "coupled-robustness-data-v1", "coupled")
    base_edges = np.linspace(-0.2, 0.2, 9)
    radius = 0.20
    centers = (base_edges[1:] + base_edges[:-1]) / 2
    eligible = [
        i * len(centers) + j
        for i, x in enumerate(centers)
        for j, y in enumerate(centers)
        if x * x + y * y <= radius * radius
    ]
    arm_edges = [
        np.linspace(-0.35, 0.45, 9).tolist(),
        np.linspace(-0.4, 0.4, 9).tolist(),
        np.linspace(0, 0.6, 7).tolist(),
    ]
    small_edges = [np.linspace(-0.08, 0.08, 5).tolist()] * 3
    families = {
        "base": spec(
            "base_start",
            archived_videos(video, "base"),
            [base_edges.tolist()] * 2,
            "archived teacher-admitted base curriculum references in a 20 cm analysis disk",
            eligible,
        ),
        "arm": spec(
            "body_hand",
            [row for stage in (0, 1, 2) for row in archived_videos(video, f"arm_splines_l{stage}")],
            arm_edges,
            "archived teacher successes across three arm spline levels, a coverage pilot",
        ),
        "coupled": spec("body_hand", coupled, arm_edges, "saved admitted robustness targets at recorded moving ticks"),
        "coupled_task_hand": spec(
            "start_base_hand",
            coupled,
            arm_edges,
            "same coupled bank, hand trajectory expressed in the initial base frame",
        ),
        "coupled_base_visited": spec(
            "base_start",
            coupled,
            [base_edges.tolist()] * 2,
            "measured base footprint from the coupled policy inputs",
            eligible,
        ),
        "carry": spec(
            "initial_hand",
            span_bank(recovery / "loaded-diversity-v1", "carry"),
            small_edges,
            "saved admitted targets at recorded moving and recovery ticks",
        ),
        "unloaded_tamp": spec(
            "body_hand", phase_bank(recovery, "tamp"), arm_edges, "teacher grasp-pose prefix inputs from training"
        ),
        "lift": spec(
            "body_hand", phase_bank(recovery, "lift"), arm_edges, "teacher supported lift inputs from training"
        ),
        "preplace": spec(
            "body_hand",
            preplace_bank(recovery),
            arm_edges,
            "successful full-task teacher preplace inputs from training",
        ),
    }
    args.output.write_text(json.dumps({"families": families}, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
