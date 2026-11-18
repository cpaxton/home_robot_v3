# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Archive compact stage outcomes and physical parity without stepping physics."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith(".gz") else data)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


PHYSICAL_FIELDS = (
    "success",
    "stop_confirmed",
    "control_ticks",
    "decisions",
    "returns",
    "commands",
    "inference_counts",
    "object_translation_m",
    "object_rotation_rad",
    "initial_state_sha256",
    "model_fingerprint",
    "compiled_robot_identity",
    "initialization_failed",
    "safety_flags",
)


def summarize(run, baseline, output):
    run, baseline, output = Path(run), Path(baseline), Path(output)
    report = read(run / "results.json")
    if report["complete"] is not True or sha(run / "sources.json") != report["source_manifest_sha256"]:
        raise ValueError("complete source-bound teacher batch required")
    old = read(baseline)["teacher"]["tamp"]
    current = read(run / "pregrasp.json")
    for primitive, stage in report["stages"].items():
        if sha(run / (primitive + ".json")) != stage["rows_file_sha256"]:
            raise ValueError("stage row hash mismatch")
    parity = []
    prefixes = []
    for i, (a, b) in enumerate(zip(old, current, strict=True), 1):
        if a["id"] != b["id"]:
            raise ValueError("baseline case order changed")
        parity.append(
            {
                "case_id": b["id"],
                "changed_physical_fields": [k for k in PHYSICAL_FIELDS if a[k] != b[k]],
                "archived_code": a["code"],
                "current_code": b["code"],
            }
        )
        paths = [run / name / f"episode-{i:04d}.json.gz" for name in ("pregrasp", "grasp_pose")]
        pregrasp, grasp = map(read, paths)
        if pregrasp["case"] != grasp["case"] or pregrasp["case"]["id"] != b["id"]:
            raise ValueError("paired trace case mismatch")
        handoff = next((c["sim_time"] for c in grasp["commands"] if c.get("phase") == "grasp"), None)
        # The public command names the handoff; the servo's persistent phase
        # field can remain pregrasp during this unloaded continuation.
        prefix = [t for t in grasp["ticks"] if handoff is None or t["time"] <= handoff + 1e-9]

        def scrub(ticks):
            return [{k: v for k, v in t.items() if k != "policy_inference_s"} for t in ticks]

        before, after = scrub(pregrasp["ticks"][: len(prefix)]), scrub(prefix)
        prefixes.append(
            {
                "case_id": b["id"],
                "handoff_sim_time": handoff,
                "compared_ticks": len(prefix),
                "prefix_equal": before == after,
                "pregrasp_prefix_sha256": digest(before),
                "grasp_prefix_sha256": digest(after),
                "trace_sha256": {p.parent.name: sha(p) for p in paths},
                "grasp_stop_confirmed": grasp["stop_confirmed"],
                "closure_commands": sum(c.get("gripper") == "close" for c in grasp["commands"]),
                "loaded_commands": sum(
                    c.get("phase") in {"lift", "preplace", "place", "retreat"} for c in grasp["commands"]
                ),
            }
        )
    archive = {
        **report,
        "raw_results_sha256": sha(run / "results.json"),
        "archived_baseline_sha256": sha(baseline),
        "readout_script_sha256": sha(__file__),
        "parity_fields": list(PHYSICAL_FIELDS),
        "baseline_physical_parity": parity,
        "paired_prefixes": prefixes,
        "all_pregrasp_physical_fields_unchanged": all(not r["changed_physical_fields"] for r in parity),
        "all_paired_prefixes_equal": all(r["prefix_equal"] for r in prefixes),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(json.dumps(archive, indent=2, allow_nan=False) + "\n")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run, args.baseline_summary, args.output)
    print(json.dumps({k: result[k] for k in ("all_pregrasp_physical_fields_unchanged", "all_paired_prefixes_equal")}))
