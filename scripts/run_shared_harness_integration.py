#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Serial exact-source non-RL stack pilot; invoke through managed emet jobs."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from run_shared_room_pairs import assert_frozen_source, git

from emet.utils.process_tree import popen_session, terminate_process_tree

QUESTIONS = (2, 6, 12, 14, 15, 16, 25, 28, 31, 56, 65, 68)
ROOMS = ("default_table_s0_distinct_recep", "molmo_rby1_ithor_s2_idx0")
DEFERRED_ROOMS = {
    "robocasa_pp_s1_v2": "Optional 2.8 cm sugar-cube stress case; RoboCasa acceptance replacement pending"
}
TAMP_INSTRUCTION = "move red cylinder to cylinder goal; move blue cube to cube goal"


def eqa_score(root):
    """Missing/error records are infrastructure failures, never incorrect answers."""
    correct = 0
    for qid in QUESTIONS:
        rows = [json.loads(line) for line in (root / "eqa" / f"q{qid}.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != 1 or type(rows[0].get("correct")) is not bool or rows[0].get("error"):
            raise ValueError(f"Invalid EQA result q{qid}")
        correct += rows[0]["correct"]
    return correct


def tamp_fixture(root, output, port):
    """Preserved two-goal geometry; change initial XML pose only, before startup."""
    assets = root / "src/emet/assets/robot"
    fixture = output / "fixture"
    fixture.mkdir(parents=True)
    for name in ("scene.xml", "stretch.xml", "docking_station.xml", "scene_environment.xml"):
        tree = ET.parse(assets / name)
        for compiler in tree.findall("compiler"):
            compiler.set("assetdir", str(assets / "assets"))
        if name == "scene_environment.xml":
            tree.find(".//body[@name='object2']").set("pos", "-.02 -.70 .6")
        tree.write(fixture / name)
    config = fixture / "sim.yaml"
    config.write_text(
        f"kind: default_mujoco\nrobot: stretch\nheadless: true\nno_cameras: true\nseed: 0\nport_offset: {port}\nscene_path: {fixture / 'scene.xml'}\n"
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--habitat-bin", required=True)
    parser.add_argument("--tamp-base-url", required=True)
    args = parser.parse_args()
    roots = {"main": args.main.resolve(), "candidate": args.candidate.resolve()}
    driver = Path(__file__).resolve().parent
    revisions = {arm: git(root, "rev-parse", "HEAD") for arm, root in roots.items()}
    for arm, root in roots.items():
        assert_frozen_source(root, revisions[arm])
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    manifest = {
        "sources": revisions,
        "roots": {k: str(v) for k, v in roots.items()},
        "questions": QUESTIONS,
        "rooms": ROOMS,
        "deferred_room_cases": DEFERRED_ROOMS,
        "seed": 0,
        "decision_policies": {"main": "legacy", "candidate": "model_final"},
        "tamp_model": "Qwen/Qwen3-VL-8B-Instruct",
        "tamp_base_url": args.tamp_base_url,
        "scope": "non-RL integration; native-policy comparison, not single-factor ablation; TAMP privileged stationary",
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    control_provenance = roots["main"] / "configs/eval/harness_control_provenance.json"
    if control_provenance.is_file():
        manifest["main_control_provenance"] = json.loads(control_provenance.read_text())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    phases = {}

    def run(name, command, root, extra=None, timeout=10800):
        for arm, source in roots.items():
            assert_frozen_source(source, revisions[arm])
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("EMET_EQA_") or key in {"EMET_CONFIG", "EMET_VL_ENDPOINT", "EMET_SIM_EVAL_CONFIG"}:
                env.pop(key)
        env.update(
            EMET_UV_RUN="1",
            HF_HUB_OFFLINE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            MUJOCO_GL="egl",
            EMET_ALLOW_SDPA_ATTN="1",
            PYTHONPATH=f"{root}/src:{root}/packages/emet_habitat",
            EMET_SIM_NAV_TELEPORT="0",
            EMET_MOLMOSPACES_NAV_TELEPORT="0",
        )
        env.update(extra or {})
        print(f"START {name}", flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            try:
                proc = popen_session(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate_process_tree(proc)
                rc = 124
        phases[name] = {"exit_code": rc, "command": list(map(str, command))}
        (args.output / "progress.json").write_text(json.dumps(phases, indent=2))
        print(f"END {name} exit={rc}", flush=True)

    candidate = roots["candidate"]
    tests = [
        "src/test/llms/test_episode_vlm_conversation.py",
        "src/test/eval/test_eqa_decision_trace.py",
        "src/test/eval/test_agentic_answer_resolution.py",
        "src/test/memory/test_assessment_view_history.py",
        "src/test/controller/test_validated_relative_motion.py",
        "src/test/controller/test_measured_gaze_feedback.py",
        "src/test/eval/test_visual_presence_consumers.py",
        "src/test/mapping/test_floor_evidence.py",
        "src/test/mapping/test_navigation_map_contract.py",
        "src/test/eval/test_embodied_qa_datasets.py",
        "src/test/controller/task/tamp/test_api_contract.py",
        "src/test/controller/task/tamp/test_native_adapter.py",
    ]
    changed_tests = git(
        candidate,
        "diff",
        "--name-only",
        "--diff-filter=AM",
        revisions["main"],
        revisions["candidate"],
        "--",
        "src/test",
    ).splitlines()
    tests = sorted(
        set(
            tests
            + [path for path in changed_tests if path.endswith(".py") and Path(path).name.startswith("test_")]
            + [
                "src/test/agent/test_floor_recovery.py",
                "src/test/agent/test_tool_outcome.py",
                "src/test/agent/test_dispatch_tool_calls.py",
                "src/test/agent/test_call_llm.py",
                "src/test/agent/test_shared_vlm_conversation.py",
                "src/test/memory/test_view_selection.py",
                "src/test/controller/task/tamp/test_agent_bridge.py",
                "src/test/controller/task/tamp/test_transfer_provider.py",
            ]
        )
    )
    manifest["contract_test_paths"] = tests
    manifest["coverage_matrix"] = {
        "216": ["model_final EQA12", "answer authority/abstention/scorer contract tests"],
        "217": ["EQA and OVMM episode dialogue", "CLI TAMP", "history rollback/transport/schema/audit tests"],
        "220": ["OVMM physical motion and floor recovery", "motion receipts/budgets/visual consumers/gaze tests"],
        "221": ["metadata adapter tests; not benchmark runtime qualification"],
        "223": ["floor-enabled S0", "negative/drop/raised/unknown/immutable-map contract tests"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    run(
        "contracts",
        [sys.executable, "-m", "pytest", "-q", "--junitxml=" + str(args.output / "contracts.xml"), *tests],
        candidate,
    )
    for arm, root in roots.items():
        run(
            f"eqa-{arm}",
            ["bash", str(driver / "run_extracted_acceptance.sh"), "eqa"],
            root,
            {
                "SOURCE_DIR": str(root),
                "SOURCE_SHA": revisions[arm],
                "OUT_DIR": str(args.output / f"eqa-{arm}"),
                "HABITAT_BIN": args.habitat_bin,
                "EMET_PY": sys.executable,
                "DEV_SEED": "0",
                "DEV_EQA_IDS": " ".join(map(str, QUESTIONS)),
                "EMET_EQA_AGENTIC_DECISION_POLICY": manifest["decision_policies"][arm],
            },
        )
    room = [
        sys.executable,
        str(driver / "run_shared_room_pairs.py"),
        "--episodes-file",
        str(candidate / "configs/ovmm/find_phase_episodes.yaml"),
        "--main",
        str(roots["main"]),
        "--candidate",
        str(candidate),
        "--grounding-backend",
        "vlm",
        "--seed",
        "0",
    ]
    run(
        "rooms",
        [
            *room,
            "--candidate-decision-policy",
            "model_final",
            "--output",
            str(args.output / "rooms"),
            "--episodes",
            *ROOMS,
        ],
        candidate,
    )
    run(
        "floor-enabled",
        [
            *room,
            "--arms",
            "candidate",
            "--candidate-decision-policy",
            "model_final",
            "--floor-reference-z",
            "0",
            "--output",
            str(args.output / "floor-enabled"),
            "--episodes",
            ROOMS[0],
        ],
        candidate,
    )
    for index, (arm, root) in enumerate(roots.items()):
        config = tamp_fixture(root, args.output / f"tamp-{arm}", 1860 + 10 * index)
        task_config = root / "configs/eval/native_stretch_transfer_debug.json"
        regions_path = args.output / f"tamp-{arm}" / "regions.json"
        regions_path.write_text(
            json.dumps([region["bounds"] for region in json.loads(task_config.read_text())["regions"]])
        )
        regions = str(regions_path)
        run(
            f"tamp-scripted-{arm}",
            [
                sys.executable,
                "scripts/eval_native_table_contact.py",
                "--output",
                str(args.output / f"tamp-scripted-{arm}"),
                "--sequence",
                "object2",
                "object1",
                "--placement-regions",
                regions,
                "--separation-axis",
                "y",
                "--separation",
                "0.15",
            ],
            root,
            timeout=1800,
        )
        run(
            f"tamp-{arm}",
            [
                sys.executable,
                "scripts/eval_native_chat_session.py",
                "--output",
                str(args.output / f"tamp-{arm}"),
                "--live-sim",
                str(config),
                "--base-url",
                args.tamp_base_url,
                "--remote-model",
                manifest["tamp_model"],
                "--placement-regions",
                regions,
                "--native-task-config",
                str(task_config),
                "--task-instruction",
                TAMP_INSTRUCTION,
                "--reject-first",
            ],
            root,
            timeout=1800,
        )
    try:
        scores = {arm: eqa_score(args.output / f"eqa-{arm}") for arm in roots}
        eqa_ok = scores["candidate"] >= scores["main"]
    except (OSError, ValueError, KeyError) as error:
        scores, eqa_ok = {"infrastructure_error": str(error)}, False
    try:
        rows = [
            json.loads(line) for line in (args.output / "rooms/results.jsonl").read_text().splitlines() if line.strip()
        ]
        expected = {(arm, episode) for arm in roots for episode in ROOMS}
        room_ok = (
            len(rows) == len(expected)
            and {(r["arm"], r["episode"]) for r in rows} == expected
            and all(r["complete"] is True and (r["arm"] == "main" or r["success"] is True) for r in rows)
        )
        suites = ET.parse(args.output / "contracts.xml").getroot().iter("testsuite")
        contract_counts = [suite.attrib for suite in suites]
        contracts_ok = (
            bool(contract_counts)
            and sum(int(s["tests"]) for s in contract_counts) > 0
            and all(int(s.get(key, 0)) == 0 for s in contract_counts for key in ("skipped", "failures", "errors"))
        )
    except (OSError, ValueError, KeyError, ET.ParseError):
        room_ok, contracts_ok = False, False
    passed = (
        all(phase["exit_code"] == 0 for name, phase in phases.items() if name != "rooms")
        and eqa_ok
        and room_ok
        and contracts_ok
    )
    (args.output / "gate.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "deferred_room_cases": DEFERRED_ROOMS,
                "eqa_scores": scores,
                "eqa_non_regression": eqa_ok,
                "candidate_room_success_and_all_complete": room_ok,
                "contracts_no_skips": contracts_ok,
                "phases": phases,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
