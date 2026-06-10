# DynaMem offline real-data benchmarks (plan)

**Branch:** `feature/dynamem-offline-real-benchmark`  
**Status:** SQA3D embodied harness **done** on this branch; DynaMem `emet eval-dynamem` still TBD.

**User preference:** offline **replay** benchmarks are fine when episode files already exist (no live robot / no re-scan).

## What landed on this branch (SQA3D)

| Component | Status |
|-----------|--------|
| Dataset loaders + EM@1 / localization metrics | Done |
| `emet sqa3d` / `emet eval-sqa3d` CLI | Done |
| ScanNet mesh replay (Open3D) | Done |
| ScanNet posed `.sens` RGB-D replay + mesh fallback | Done |
| Dynagraph (DynaMem voxel + GraphEQA) runner | Done (default method) |
| GPU-isolated real-VLM sweeps | Done |
| Episode JSONL: `infra_failure`, `replay_backend`, `sens_match_xy_m` | Done |
| Docs: `docs/sqa3d.md`, `docs/sqa3d_compute.md` | Done |

See [sqa3d.md](../sqa3d.md) for commands and data layout.

## Original question (DynaMem-specific)

Does `stretch_ai` `run_dynamem.py` or ok-robot provide real-data benchmarks for offline DynaMem experiments?

## Short answer (unchanged)

| Source | Replay files? | Scored harness? |
|--------|---------------|-----------------|
| **emet `logs/memory_*`** (local, gitignored) | Yes — `MemoryState` dirs (`backend: dynamem`, `point_cloud.npz`, `frames/`) + legacy `.pkl` | No `emet eval-dynamem` yet |
| **stretch_ai `dynamem_log/*.pkl`** | Same pickle schema as `read_from_pickle` | Same gap |
| **`/tmp/dynagraph_bench_smoke/*`** | Dynagraph `graph_eqa` exports + `sim_object_placements.json` | Yes — `emet eval-dynagraph` (graph GT, not DynaMem `localize_text`) |
| **SQA3D + ScanNet replay (this branch)** | ScanNet meshes + optional `.sens` | Yes — `emet sqa3d run-batch` / `run-real-sweep` (EM@1 QA, not object localization) |
| **DynaBench (paper)** | Not in repo | N/A |
| **ok-robot `.r3d`** | Not vendored | N/A |

Nothing is **checked into git** for DynaMem voxel replay (no LFS fixtures). Replay works from disk paths you already have under `logs/`.

## Replay formats (already implemented)

| Format | Load path | Rebuilds semantic map? |
|--------|-----------|------------------------|
| **MemoryState dir** (`manifest.json` + `point_cloud.npz`) | `backend.load(dir)` or `emet run dynamem --input-path DIR` | Yes — restores voxel semantic memory from `point_cloud.npz` |
| **Legacy pickle** | `SparseVoxelMap.read_from_pickle()` | Yes — replays frames into the map |
| **Dynagraph export** (`backend: graph_eqa`) | `emet eval-dynagraph --episode DIR` | Graph only; `has_point_cloud: false` — not a DynaMem voxel replay |
| **SQA3D ScanNet replay** | `emet sqa3d run-episode` | Yes — live Dynagraph mapping from replayed RGB-D |

Example local DynaMem episode (not in git): `logs/memory_2026-06-05_00-20-19/` (~27 MB, `has_point_cloud: true`, frames + detections).

Legacy pickles also exist under `logs/memory_*.pkl`.

## ok-robot real-data path

- **Loader:** `third_party/ok-robot/ok-robot-navigation/voxel_map/dataloaders/record3d.py` (`R3DSemanticDataset`, clip-fields lineage).
- **Usage:** `path_planning.py` with `dataset_path='r3d/{file}.r3d'`; docs reference `ok-robot-navigation/r3d/sample.r3d` (not present in our submodule checkout).
- **Stack:** separate conda env, ScanNet-200 class labels, OWL features — **not wired into emet** DynaMem backend.

## DynaBench (paper)

From DynaMem §evaluation (cited in `paper/sections/02_background.tex`): offline benchmark for **dynamic 3D visual grounding** — map at round *t*, query object location after objects move; 9 environments, 3 rounds each. Complements live OVMM success-rate eval on the website (70% vs OK-Robot 30%).

**Gap:** no public download URL in paper site or GitHub; likely need author release or internal Stretch logs.

## Recommended path (replay-first)

1. **`emet eval-dynamem`** on existing `logs/memory_*` or any `MemoryState` dir with `point_cloud.npz`  
   - Load map → `localize_text(query)` → compare to GT in `queries.json` (manual labels) or `sim_object_placements.json` when present.  
   - Mirror metrics from `eval-dynagraph` GT block: XY error, success @ 0.25/0.5/1.0 m.

2. **Check in one small sim fixture** (optional, for CI)  
   - Generate via short sim explore + save; store under `src/test/fixtures/episodes/dynamem_default_table/` (git-lfs if > few MB).  
   - Ship `queries.json` with known object names (e.g. red cylinder / robocasa apple).

3. **Legacy pickle replay** — same harness, entrypoint `read_from_pickle` instead of `backend.load`.

4. **SQA3D (done)** — use for situated QA EM@1 on real ScanNet RGB-D; not a substitute for object-localization benchmarks.

5. **Record3D / DynaBench** — only if we add new files; not required for replay-based eval.

## Target harness shape (mirror dynagraph)

Analogous to `emet eval-dynagraph` + `run_dynagraph_benchmark_smoke.py`:

```
emet eval-dynamem --episode MEMORY_STATE_DIR --queries queries.json -o dynamem_eval.json
```

Metrics (initial proposal):

- **Localization:** XY error (m), success @ 0.25 m / 0.5 m / 1.0 m vs GT or clicked point.
- **Dynamic (multi-round):** same metrics per round; optional map-update latency.
- **Optional:** feature-sim vs mLLM grounding mode (`-M`).

## Data to obtain

| Priority | Source | Action |
|----------|--------|--------|
| P0 | Hello Robot / DynaMem authors | Request DynaBench release or Stretch exploration logs + query JSON. |
| P1 | Record3D | Scan home/lab or fetch sample `.r3d`; add under `data/` or LFS; document tape/GT protocol from ok-robot docs. |
| P2 | Internal exports | Collect `emet run dynamem --export` from hardware when Mars/Stretch available. |
| P3 | SQA3D / ScanNet | Already integrated — download via `scripts/download_sqa3d_data.py` + `scripts/download_scannet_data.py`. |

## Out of scope (for this branch)

- Live robot OVMM success-rate replication (website eval).
- AnyGrasp manipulation scoring (GPU + license).

## Next implementation steps

1. Add `queries.json` schema (text query, optional round, GT xy or object id).
2. Implement `eval_dynamem` CLI reading `MemoryState` or pickle replay.
3. Optional: `record3d_to_frames.py` using ok-robot loader logic.
4. Smoke test on sim export first, then one Record3D scan when available.
5. Document in `docs/dynamem_benchmarks.md` once harness exists.
6. **SQA3D follow-ups:** clean GPU sweep for paper numbers; tune `sens_match_max_xy_m` if many scenes fall back to mesh.
