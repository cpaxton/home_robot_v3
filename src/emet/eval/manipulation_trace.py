# Copyright (c) 2026 Chris Paxton. All rights reserved.
"""Private MuJoCo manipulation evidence; never included in policy observations.

The simulator writes a 10 Hz pose/contact trace. Scoring is offline and requires
sustained physical evidence, not tool returns or object displacement alone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


class ManipulationTrace:
    def __init__(self, model, config: dict, output: Path):
        self.config = config
        self.target = int(model.body(config["object_body"]).id)
        self.support = int(model.body(config["support_body"]).id)
        self.ee = int(model.body(config["ee_body"]).id)
        grippers = {int(model.body(name).id) for name in config["gripper_bodies"]}
        if not grippers or self.target == self.support:
            raise ValueError("Trace requires distinct object/support and explicit gripper bodies")

        def subtree(roots):
            found = set(roots)
            for body in range(1, model.nbody):
                if int(model.body_parentid[body]) in found:
                    found.add(body)
            return found

        self.target_ids = subtree({self.target})
        self.support_ids = subtree({self.support})
        self.gripper_ids = subtree(grippers)
        if self.target_ids & (self.support_ids | self.gripper_ids):
            raise ValueError("Evaluator object, support and gripper subtrees must not overlap")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.stream = output.open("x")
        self.stream.write(json.dumps({"schema": 1, "config": config, "sample_period_s": 0.1}) + "\n")
        self.stream.flush()
        self.next_time = 0.0

    def record(self, model, data):
        if data.time < self.next_time:
            return
        self.next_time = float(data.time) + 0.1
        contacts = []
        touched = set()
        for contact in data.contact[: data.ncon]:
            if contact.dist > 0:
                continue
            a, b = (int(model.geom_bodyid[g]) for g in contact.geom)
            if a in self.target_ids and b not in self.target_ids:
                touched.add(b)
            elif b in self.target_ids and a not in self.target_ids:
                touched.add(a)
            else:
                continue
            contacts.append([model.body(a).name, model.body(b).name])
        obj = data.body(self.target)
        ee = data.body(self.ee)
        ee_rot = ee.xmat.reshape(3, 3)
        row = {
            "sim_time": float(data.time),
            "wall_time": time.time(),
            "object_pos": obj.xpos.tolist(),
            "object_rot": obj.xmat.tolist(),
            "ee_pos": ee.xpos.tolist(),
            "ee_rot": ee.xmat.tolist(),
            "relative_pos": (ee_rot.T @ (obj.xpos - ee.xpos)).tolist(),
            "relative_rot": (ee_rot.T @ obj.xmat.reshape(3, 3)).ravel().tolist(),
            "gripper_contact": bool(touched & self.gripper_ids),
            "support_contact": bool(touched & self.support_ids),
            "other_contact": bool(touched - self.gripper_ids - self.support_ids),
            "contacts": contacts,
            "qpos": data.qpos.tolist(),
            # Preserve measured dynamics for local checkpoint diagnostics;
            # differencing 10 Hz positions is not a reliable initial velocity.
            # This remains sampled evidence, not an exact command replay log.
            "qvel": data.qvel.tolist(),
            "act": data.act.tolist(),
            "qacc_warmstart": data.qacc_warmstart.tolist(),
            "ctrl": data.ctrl.tolist(),
        }
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


class ManipulationSequenceTrace:
    """Private, synchronized per-object traces for ordered distinct-object tasks.

    Reuse the single-object recorder/scorer rather than inferring success from
    agent tool calls. The manifest never enters the observation stream.
    """

    def __init__(self, model, config: dict, output: Path):
        steps = config["steps"]
        if not isinstance(steps, list) or len(steps) < 2:
            raise ValueError("A sequence requires at least two declared steps")
        objects = [step["object_body"] for step in steps]
        if len(set(objects)) != len(objects):
            raise ValueError("Sequence scoring currently requires distinct target objects")
        common = {key: config[key] for key in ("ee_body", "gripper_bodies")}
        self.traces = []
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive manifest creation prevents silently overwriting old evidence.
        with output.open("x") as manifest:
            try:
                paths = []
                for index, step in enumerate(steps):
                    path = output.with_name(f"{output.stem}.step-{index}.jsonl")
                    self.traces.append(ManipulationTrace(model, {**step, **common}, path))
                    paths.append(path.name)
                manifest.write(json.dumps({"schema": 2, "config": config, "traces": paths}) + "\n")
            except Exception:
                self.close()
                raise

    def record(self, model, data):
        for trace in self.traces:
            trace.record(model, data)

    def close(self):
        for trace in self.traces:
            trace.close()


def create_trace(model, config: dict, output: Path):
    recorder = ManipulationSequenceTrace if "steps" in config else ManipulationTrace
    return recorder(model, config, output)


def score_trace(rows: list[dict]) -> dict:
    """Conservative single pick/place acceptance; incomplete evidence never passes.

    Frozen thresholds: 5 cm lift, 1 s hold/release dwell, 2 cm translation and
    0.1 rad orientation stability, with no sampling gap exceeding 0.25 s.
    """
    result = {"physical_pick_success": False, "physical_place_success": False, "verified": False}
    if len(rows) < 2:
        return {**result, "reason": "missing trace"}
    times = np.asarray([r["sim_time"] for r in rows])
    fields = ("object_pos", "object_rot", "relative_pos", "relative_rot")
    if (
        not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or any(not np.isfinite(np.asarray([r[key] for r in rows])).all() for key in fields)
    ):
        return {**result, "reason": "invalid trace"}

    def stable(window, relative):
        pos_key, rot_key = ("relative_pos", "relative_rot") if relative else ("object_pos", "object_rot")
        positions = np.asarray([r[pos_key] for r in window])
        rotations = np.asarray([r[rot_key] for r in window]).reshape(-1, 3, 3)
        angle = np.arccos(np.clip((np.einsum("nij,ij->n", rotations, rotations[0]) - 1) / 2, -1, 1))
        return bool(np.max(np.linalg.norm(positions - positions[0], axis=1)) <= 0.02 and np.max(angle) <= 0.1)

    window = []
    initial_z = None
    pick_time = None
    for row in rows:
        if initial_z is None:
            # MJCF objects may start above their support and settle under
            # gravity. Measure lift from a stable supported baseline, not spawn.
            if (
                row["gripper_contact"]
                or not (row["support_contact"] or row["other_contact"])
                or (window and row["sim_time"] - window[-1]["sim_time"] > 0.25)
            ):
                window = []
            if row["gripper_contact"] or not (row["support_contact"] or row["other_contact"]):
                continue
            window.append(row)
            while len(window) > 1 and row["sim_time"] - window[1]["sim_time"] >= 1.0:
                window.pop(0)
            if row["sim_time"] - window[0]["sim_time"] >= 1.0 and stable(window, False):
                initial_z = float(np.median([r["object_pos"][2] for r in window]))
                window = []
            continue
        picking = pick_time is None
        valid = (
            row["gripper_contact"]
            and not row["support_contact"]
            and not row["other_contact"]
            and row["object_pos"][2] >= initial_z + 0.05
            if picking
            else row["support_contact"] and not row["gripper_contact"] and not row["other_contact"]
        )
        if not valid or (window and row["sim_time"] - window[-1]["sim_time"] > 0.25):
            window = []
            if not picking:
                result["physical_place_success"] = False
                result.pop("place_time", None)
                result.pop("first_place_time", None)
        if not valid:
            continue
        window.append(row)
        # Keep the shortest available window spanning at least one second.
        while len(window) > 1 and row["sim_time"] - window[1]["sim_time"] >= 1.0:
            window.pop(0)
        if row["sim_time"] - window[0]["sim_time"] >= 1.0 and stable(window, picking):
            if picking:
                pick_time = row["sim_time"]
                result.update(physical_pick_success=True, pick_time=pick_time, pick_start_time=window[0]["sim_time"])
                window = []
            else:
                result.update(physical_place_success=True, place_time=row["sim_time"])
                result.setdefault("first_place_time", row["sim_time"])
        elif not picking:
            result["physical_place_success"] = False
            result.pop("place_time", None)
            result.pop("first_place_time", None)
    result.update(
        verified=initial_z is not None,
        reason="scored physical trace" if initial_z is not None else "no settled baseline",
    )
    return result


def score_sequence(traces: list[list[dict]]) -> dict:
    """Require ordered verified releases and all earlier placements still intact.

    All targets must be recorded on the same simulation timeline. Reusing a
    target for another placement is deliberately unsupported, not implicitly
    counted twice. Single-step physical thresholds remain unchanged.
    """
    failure = {"verified": False, "physical_pick_success": False, "physical_place_success": False, "ordered": False}
    if len(traces) < 2 or not traces[0]:
        return {**failure, "reason": "missing sequence traces"}
    times = [row["sim_time"] for row in traces[0]]
    if any([row["sim_time"] for row in rows] != times for rows in traces[1:]):
        return {**failure, "reason": "sequence traces have different timelines"}
    results = [score_trace(rows) for rows in traces]
    placed = all(result["physical_place_success"] for result in results)
    ordered = placed and all(
        current["pick_start_time"] >= previous["first_place_time"]
        for previous, current in zip(results[:-1], results[1:], strict=True)
    )
    return {
        "verified": all(result["verified"] for result in results),
        "physical_pick_success": all(result["physical_pick_success"] for result in results),
        "physical_place_success": bool(placed and ordered),
        "ordered": bool(ordered),
        "steps": results,
        "reason": "scored ordered physical traces" if ordered else "incomplete or out-of-order physical subgoals",
    }


def score_recording(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if records[0].get("schema") == 1:
        return score_trace(records[1:])
    if records[0].get("schema") != 2 or len(records) != 1:
        raise ValueError("unsupported trace schema")
    header = records[0]
    steps = header["config"]["steps"]
    if len(steps) != len(header["traces"]) or len({step["object_body"] for step in steps}) != len(steps):
        raise ValueError("Sequence manifest requires distinct targets and one trace per step")
    traces = []
    for step, filename in zip(steps, header["traces"], strict=True):
        if Path(filename).name != filename:
            raise ValueError("Sequence traces must be sibling files")
        rows = [json.loads(line) for line in (path.parent / filename).read_text().splitlines()]
        expected_config = {**step, **{key: header["config"][key] for key in ("ee_body", "gripper_bodies")}}
        if rows[0].get("schema") != 1 or rows[0]["config"] != expected_config:
            raise ValueError("Trace does not match its declared physical subgoal")
        traces.append(rows[1:])
    return score_sequence(traces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score_recording(args.trace)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        result = {
            "verified": False,
            "physical_pick_success": False,
            "physical_place_success": False,
            "reason": str(exc),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["verified"] and result["physical_place_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
