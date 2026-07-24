# Dynamic exploration

Active frontier exploration (Phase 1), scripted world-change / staleness (Phase 2), and lifelong K-cycle checkpoint/fuzz/reload (Phase 3) on Stretch in Robocasa and MolmoSpaces iTHOR.

**Deep doc:** [dynamic_exploration_benchmark.md](../dynamic_exploration_benchmark.md)

## Paper references

- Section: `paper/sections/04b_dynamic_exploration.tex`
- Tables: `tab:dynamic_explore_phase1`, `tab:dynamic_explore_world_change`, `tab:dynamic_explore_lifelong`

## Primary metrics

| Phase | Aggregate CSV | Key columns |
|-------|---------------|-------------|
| P1 explore | `aggregate_dynamic_exploration.csv` | `explored_fraction`, `spatial_recall`, `eqa_accuracy`, `node_count` |
| P2 world-change | `aggregate_dynamic_exploration_world_change.csv` | `answer_correct_pre/post`, `n_stale_nodes_after_move` (nodes near old pose @ 0.75 m), `n_nodes_near_new_pos`, `recovery_steps` |
| P3 lifelong | `aggregate_dynamic_exploration_lifelong.csv` | per-cycle `eqa_accuracy`, node churn, move adaptation flags |

Phase-1 CSV `spatial_recall` / `label_recall` prefer nested `fusion.fused` then `fusion.raw` (see `flatten_eval_metrics`). Phase-1 subprocesses pass `--benchmark-harness dynamic_explore --benchmark-method {backend}` so harness EQA flags match `configs/benchmarks/dynagraph.yaml`. Before Qwen EQA, `emet run dynagraph` calls `prepare_dynagraph_vram_for_eqa`. Post-explore question banks run **answer-only** (`max_planning_steps=1`, `allow_navigation=False`) so EQA does not start another frontier chase after the VLM loads (Jul 2026 hang fix). Subprocess timeout kills the whole process group to avoid orphan GPU workers.

Metrics are **not comparable** to OVMM find-phase or SQA3D EM@1.

## Config

- `configs/benchmarks/dynamic_exploration.yaml`
- Questions: `src/emet/config/benchmarks/dynagraph_questions.yaml`
- Default output: `~/runs/emet/dynamic_exploration` (`EMET_DYNAMIC_EXPLORE_OUTPUT`)

## Smoke commands

```bash
# Config matrix only (no GPU)
uv run python scripts/eval_dynamic_exploration.py --smoke --dry-run

# Phase 1 GPU smoke (Robocasa seed0, K=3; ~60–75 min)
uv run python scripts/eval_dynamic_exploration.py --smoke \
  --output-dir ~/runs/emet/dynamic_exploration/smoke

# Phase 2 world-change (invalidates nodes near old pose before recovery explore)
uv run python scripts/eval_dynamic_exploration.py \
  --phase world-change --episode-id robocasa_seed0_world_change \
  --backend dynagraph --cpu-only

# Phase 3 lifelong
uv run python scripts/eval_dynamic_exploration.py \
  --phase lifelong --episode-id robocasa_seed0_lifelong \
  --backend dynagraph --cpu-only

# Agent FIND after world change (closed-loop memory refresh)
uv run python scripts/smoke_dynagraph_agent_world_change_find.py --cpu-only

# OVMM find backend matrix (Robocasa S1 + Molmo S2)
uv run python scripts/run_ovmm_find_backend_matrix.py --cpu-only --backends ground_truth
```

`--smoke` is equivalent to `--phase explore --env robocasa --episode-id robocasa_seed0 --backend dynagraph --explore-max-iters 3 --mapping-mode explore`.

**World-change memory update:** after `sim_set_body_pose`, the runner calls `GraphEQAMemory.invalidate_nodes_near` + `clear_eqa_working_memory` so post-move EQA / find do not reuse CONFIRMED_MEMORY at the old pose. Lifelong fuzz patches the cycle checkpoint the same way before the next reload.

## Scene map cache (skip live explore)

Prebuild each scene once (perfect-depth rotate + frontier explore); OVMM find and dynamic-explore P1 / world-change / lifelong cycle-0 load `graph.json` + `voxel_map.pkl` and skip mapping. Operator notes: [dynagraph_dynamic_memory.md](dynagraph_dynamic_memory.md).

```bash
# Build Robocasa + Molmo baselines → ~/.cache/emet/scene_maps/
NEED_MIB=8000 ./scripts/gpu_preflight.sh --wait
env -u PYTHONPATH uv run python scripts/build_scene_map_cache.py

# Consumers use the cache by default; force live mapping with:
#   EMET_USE_SCENE_MAP_CACHE=0   or   --no-scene-cache
```

## Full matrix

```bash
./scripts/run_dynamic_exploration_full.sh
# or via large queue:
./scripts/run_large_paper_eval.sh dynamic-explore
```

## Timing and debugging

- GPU Robocasa K=3: typically **60–75 min** per run (+ EQA VLM load).
- Subprocess timeout scales with explore budget (~105 min for K=3 + ~1 h EQA); override with `EMET_DYNAMIC_EXPLORE_DYNAGRAPH_TIMEOUT_S`.
- Each run writes `{export_dir}/dynagraph.log` for timeout debugging.
- Before GPU smokes: `NEED_MIB=12000 uv run emet eval wait` (avoid stacking with Habitat / other VLM jobs).
- Gate script (smoke then world-change): `nohup ./scripts/run_dynagraph_dynamic_improve_smokes.sh ~/runs/emet/dynamic_exploration/<run_id> &`

## Backend localization figure

For a quick visual of backend localization on the same Robocasa stack, see [backend_localization.md](backend_localization.md).

## Tests

```bash
uv run emet test src/test/eval/test_dynamic_exploration_config.py -q
```
