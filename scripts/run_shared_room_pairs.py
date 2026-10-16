# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Serial, frozen main/candidate room find gate. Keep failures and original logs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def grounding_assets(roots, arms, backend):
    """Require local weights: acceptance must not download on the grounding path."""
    if backend != "vlm":
        return {}
    assets = {}
    for arm in arms:
        checkpoint = roots[arm] / "sam2.1_hiera_small.pt"
        if not checkpoint.is_file():
            raise RuntimeError(
                f"Missing SAM2 checkpoint: {checkpoint}. Link verified local weights "
                "before running VLM/SAM2 acceptance; no in-episode downloads."
            )
        with checkpoint.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        assets[arm] = {"path": str(checkpoint.resolve()), "sha256": digest.hexdigest()}
    return assets


def run(args):
    roots = {"main": args.main.resolve(), "candidate": args.candidate.resolve()}
    assets = grounding_assets(roots, args.arms, args.grounding_backend)
    revisions = {name: git(root, "rev-parse", "HEAD") for name, root in roots.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "roots": {k: str(v) for k, v in roots.items()},
        "revisions": revisions,
        "episodes": args.episodes,
        "seed": args.seed,
        "scan_steps": args.scan_steps,
        "arms": args.arms,
        "grounding_backend": args.grounding_backend,
        "grounding_assets": assets,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "find only, physical navigation, local Qwen3-VL int4; not manipulation acceptance",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    all_complete, all_success = True, True
    for index, episode in enumerate(args.episodes):
        for arm in args.arms if index % 2 == 0 else list(reversed(args.arms)):
            root = roots[arm]
            if git(root, "rev-parse", "HEAD") != revisions[arm] or git(root, "diff", "HEAD", "--stat"):
                raise RuntimeError(f"Frozen source changed: {root}")
            case = args.output / f"{episode}-{arm}"
            case.mkdir()
            config = case / "config.yaml"
            config.write_text(
                f"extends: {root}/configs/emet/default.yaml\n"
                + "mapping:\n  query_driven_memory: true\n  debug_perfect_sensor_depth: true\n"
                "  find_phase_nav_step_timeout_s: 30.0\n  motion_planner:\n    obstacle_map_mode: physical\n"
                f"  query_memory:\n    grounding_backend: {args.grounding_backend}\n"
                + (
                    "    region_strategy: depth_candidates\n    mask_backend: yoloe_sam2\n"
                    "    surface_presentation: context\n    recover_proposals_with_vlm: true\n"
                    if args.grounding_backend == "vlm"
                    else ""
                )
                + "  eqa:\n    agentic_verify: true\n    collect_agentic_trace: true\n"
                "    vl_family: qwen3_vl\n    vl_hf_model_id: Qwen/Qwen3-VL-8B-Instruct\n    vl_quantization: int4\n"
            )
            env = os.environ.copy()
            for key in list(env):
                if key.startswith("EMET_EQA_") or key in {
                    "EMET_VL_ENDPOINT",
                    "EMET_SIM_EVAL_CONFIG",
                    "EMET_OPENAI_BASE_URL",
                    "EMET_LLM_HOST",
                    "EMET_CALIBAN_HOST",
                }:
                    env.pop(key)
            env.update(
                {
                    "EMET_CONFIG": str(config),
                    "PYTHONPATH": f"{root}/src:{root}/packages/emet_habitat",
                    "EMET_UV_RUN": "1",
                    "EMET_ALLOW_SDPA_ATTN": "1",
                    "MUJOCO_GL": "egl",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "EMET_SIM_NAV_TELEPORT": "0",
                    "EMET_MOLMOSPACES_NAV_TELEPORT": "0",
                }
            )
            command = [
                "timeout",
                "--signal=TERM",
                "--kill-after=20s",
                "1500",
                sys.executable,
                "-m",
                "emet.cli",
                "ovmm",
                "find",
                "--episodes",
                "configs/ovmm/find_phase_episodes.yaml",
                "--backend",
                "lazy_graph",
                "--query-driven-memory",
                "--episode-id",
                episode,
                "--seed",
                str(args.seed),
                "--mapping-rotate-steps",
                str(args.scan_steps),
                "--agentic-max-rounds",
                "6",
                "--agentic-max-nav-steps",
                "3",
                "--no-scene-cache",
                "--port-offset",
                str(700 + index * 10 + (arm == "candidate") * 2),
                "--output-dir",
                str(case / "artifacts"),
            ]
            (case / "invocation.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "revision": revisions[arm],
                        "hashes": {
                            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [
                                config,
                                root / "configs/emet/default.yaml",
                                root / "configs/ovmm/find_phase_episodes.yaml",
                                root / "uv.lock",
                            ]
                        },
                    },
                    indent=2,
                )
            )
            print(f"START {episode} {arm} {revisions[arm]}", flush=True)
            with (case / "run.log").open("w") as log:
                result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            metrics_path = case / "artifacts" / f"{episode}_lazy_graph.json"
            metrics = None
            try:
                metrics = json.loads(metrics_path.read_text())
            except (OSError, ValueError):
                pass
            complete = (
                result.returncode == 0
                and isinstance(metrics, dict)
                and not metrics.get("error")
                and all(type(metrics.get(key)) is bool for key in ["find_object_success", "find_recep_success"])
            )
            success = bool(complete and metrics["find_object_success"] and metrics["find_recep_success"])
            row = {
                "episode": episode,
                "arm": arm,
                "exit_code": result.returncode,
                "complete": complete,
                "success": success,
                "metrics": metrics,
                "artifact_dir": str(case),
            }
            with (args.output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(f"END {episode} {arm} complete={complete} success={success}", flush=True)
            all_complete &= complete
            all_success &= success
    (args.output / "gate.json").write_text(json.dumps({"all_complete": all_complete, "all_success": all_success}))
    return 0 if all_complete and all_success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scan-steps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--arms", nargs="+", choices=("main", "candidate"), default=["main", "candidate"])
    parser.add_argument("--grounding-backend", choices=("yoloe", "vlm"), default="yoloe")
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["default_table_s0_distinct_recep", "robocasa_pp_s1", "molmo_rby1_ithor_s2_idx0"],
    )
    raise SystemExit(run(parser.parse_args()))
