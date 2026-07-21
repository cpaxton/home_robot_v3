#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Build GT/perfect-depth baseline scene maps (graph + voxel) for the cache.

Examples::

    # Build default Robocasa + Molmo scenes
    NEED_MIB=8000 ./scripts/gpu_preflight.sh --wait
    env -u PYTHONPATH uv run python scripts/build_scene_map_cache.py

    # One scene, force rebuild
    uv run python scripts/build_scene_map_cache.py \\
      --sim configs/sim/robocasa_pick_place_stretch.yaml --explore-steps 8 --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_DEFAULT_SCENES: tuple[tuple[str, int], ...] = (
    ("configs/sim/robocasa_pick_place_stretch.yaml", 8),
    ("configs/sim/molmospaces_ithor_train_stretch_0.yaml", 15),
)


def _build_one(
    sim_path: str,
    *,
    explore_steps: int,
    port_offset: int,
    cpu_only: bool,
    force: bool,
    cache_root: Path | None,
) -> Path:
    from emet.config.sim_launch_config import load_sim_launch_config_from_path
    from emet.controller.controller_dynagraph import DynagraphController
    from emet.core.parameters import get_parameters
    from emet.eval.benchmark_dynagraph import apply_dynamic_explore_backend
    from emet.eval.ovmm_find_phase import (
        resolve_find_phase_nav_step_timeout,
        run_mapping_protocol,
    )
    from emet.eval.scene_map_cache import (
        BUILD_MODE_GT,
        has_cached_map,
        scene_cache_dir,
        scene_cache_key,
        write_cache_metadata,
    )
    from emet.eval.sim_eval_session import benchmark_sim_server, connect_benchmark_robot
    from emet.memory.headless_export import export_graph_eqa_dir

    sim_cfg = load_sim_launch_config_from_path(sim_path)
    sim_cfg = replace(sim_cfg, port_offset=int(port_offset), headless=True)
    key = scene_cache_key(sim_cfg, build_mode=BUILD_MODE_GT)
    out_dir = scene_cache_dir(key, root=cache_root)

    if has_cached_map(out_dir) and not force:
        print(f"[scene-cache] skip existing {key} → {out_dir}", flush=True)
        return out_dir

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[scene-cache] building {key} explore_steps={explore_steps} → {out_dir}",
        flush=True,
    )
    t0 = time.monotonic()
    with benchmark_sim_server(sim_cfg, repo=_REPO, cpu_only=cpu_only, cwd=_REPO) as sim:
        sim_kind = sim.sim_kind
        robot = connect_benchmark_robot(sim_cfg, port_offset)
        agent = None
        try:
            parameters = apply_dynamic_explore_backend(get_parameters("dynav_config.yaml"), "dynagraph")
            parameters["encoder"] = None
            parameters["debug_perfect_sensor_depth"] = True
            parameters["find_phase_nav_step_timeout_s"] = resolve_find_phase_nav_step_timeout(
                cpu_only=cpu_only,
                sim_kind=sim_kind,
            )
            agent = DynagraphController(
                robot,
                parameters,
                save_rerun=False,
                cpu_only=cpu_only,
                use_instance_graph=True,
                use_sensor_perception=False,
            )
            agent.start()
            agent._fast_explore_lookaround = True

            n_steps = run_mapping_protocol(
                agent,
                explore_steps=int(explore_steps),
                not_rotate=False,
            )
            final_step = int(getattr(agent, "obs_count", 0) or 0)
            export_graph_eqa_dir(
                agent.graph_memory,
                agent.voxel_map,
                str(out_dir),
                title=f"Scene map cache {key}",
                robot=str(getattr(sim_cfg, "robot", "stretch")),
                final_step=final_step,
                save_voxel_pickle=True,
            )
            write_cache_metadata(
                out_dir,
                sim_cfg,
                key=key,
                build_params={
                    "sim_path": sim_path,
                    "explore_steps": int(explore_steps),
                    "mapping_steps": int(n_steps),
                    "final_step": final_step,
                    "perfect_depth": True,
                    "backend": "dynagraph",
                    "wall_s": time.monotonic() - t0,
                },
                repo_root=_REPO,
            )
        finally:
            if agent is not None:
                stop = getattr(agent, "stop", None)
                if callable(stop):
                    try:
                        stop()
                    except Exception:
                        pass
            try:
                robot.stop()
            except Exception:
                pass

    if not has_cached_map(out_dir):
        raise RuntimeError(f"build incomplete: missing manifest/voxel under {out_dir}")
    print(
        f"[scene-cache] done {key} wall_s={time.monotonic() - t0:.0f} → {out_dir}",
        flush=True,
    )
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sim",
        action="append",
        dest="sims",
        help="Sim launch YAML (repeatable). Default: Robocasa S1 + Molmo iTHOR idx0.",
    )
    ap.add_argument(
        "--explore-steps",
        type=int,
        default=None,
        help="Frontier explore steps after rotate (default per-scene or 8)",
    )
    ap.add_argument("--port-offset", type=int, default=280)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Override EMET_SCENE_MAP_CACHE_DIR / ~/.cache/emet/scene_maps",
    )
    args = ap.parse_args()

    os.environ.pop("PYTHONPATH", None)

    jobs: list[tuple[str, int]] = []
    if args.sims:
        for sim in args.sims:
            steps = int(args.explore_steps) if args.explore_steps is not None else 8
            jobs.append((sim, steps))
    else:
        for sim, default_steps in _DEFAULT_SCENES:
            steps = int(args.explore_steps) if args.explore_steps is not None else default_steps
            jobs.append((sim, steps))

    offset = int(args.port_offset)
    failed = 0
    for sim, steps in jobs:
        try:
            _build_one(
                sim,
                explore_steps=steps,
                port_offset=offset,
                cpu_only=bool(args.cpu_only),
                force=bool(args.force),
                cache_root=args.cache_root,
            )
        except Exception as exc:
            failed += 1
            print(f"[scene-cache] FAIL {sim}: {exc}", file=sys.stderr, flush=True)
        offset += 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
