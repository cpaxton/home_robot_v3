# OVMM find-phase benchmark (FindObj / FindRec)

Memory ablation benchmark inspired by [OVMM](https://ovmm.github.io/) find phases. We score **FindObj** and **FindRec** localization against MuJoCo `sim_object_placements` ground truth—not full pick/place manipulation success.

## Scene tier ladder

| Tier | Sim config | Scale |
|------|------------|-------|
| **S0** | `configs/sim/default_table_stretch.yaml` | Default table, ~4 GT bodies |
| **S1** | `configs/sim/robocasa_pick_place.yaml` | Robocasa kitchen, ~20–40 bodies |
| **S2** | `configs/sim/molmospaces_ithor_train_{0,1,2}.yaml` | MolmoSpaces iTHOR multi-room (GT scan may cap) |

Episodes are listed in `configs/ovmm/find_phase_episodes.yaml`.
Paths and smoke defaults: `configs/ovmm/benchmark.yaml` (outputs under `~/runs/emet/…`, caches under `~/.cache/…`).

## Assets + smoke (multi-agent friendly)

```bash
# Verify / fetch CSVs; check HM3D scenes for habitat episodes; create ~/runs dirs
uv run python scripts/download_ovmm_benchmark_assets.py

# Full smoke: unit tests + S0 sim GT + one Habitat GT episode (~5–10 min)
uv run python scripts/smoke_ovmm_benchmark.py --cpu-only
```

Override output location: `EMET_OVMM_OUTPUT_SIM=~/runs/emet/ovmm_find_phase` (or edit `benchmark.yaml`).

## Quick start (S0)

```bash
# Unit tests (no sim)
uv run emet test src/test/memory/test_ovmm_find_phase_metrics.py -q

# S0 default table, all backends (~2–5 min with GPU)
uv run python scripts/eval_ovmm_find_phases.py \
  --tier S0 \
  --backend dynamem --backend graph_eqa --backend dynagraph \
  --cpu-only \
  --output-dir runs/ovmm_find_phase/s0
```

Integration gate (optional CI):

```bash
RUN_OVMM_FIND_TESTS=1 uv run emet test src/test/memory/test_ovmm_find_phase_integration.py -q
```

## Batch runner

```bash
uv run python scripts/eval_ovmm_find_phases.py \
  --episodes configs/ovmm/find_phase_episodes.yaml \
  --backend dynagraph \
  --tier S1 \
  --output-dir runs/ovmm_find_phase/s1_dynagraph
```

### Memory backends

| Backend | Role |
|---------|------|
| `dynamem` | Voxel semantic memory only (no graph) |
| `graph_eqa` | GraphEQA with merge/staleness off |
| `dynagraph` | Defaults (`merge_xy_m=0.45`, `staleness_horizon=256`) |
| `ground_truth` | Oracle upper bound (graph from sim GT) |

### Scaling / ablation flags

```bash
# Exploration budget (episode YAML ``explore_steps`` or dedicated episodes)
uv run python scripts/eval_ovmm_find_phases.py --episode-id molmo_ithor_s2_idx0_explore15

# Merge / staleness grid on S2
uv run python scripts/eval_ovmm_find_phases.py --tier S2 --backend dynagraph \
  --merge-xy-m 0 --staleness-horizon 0

uv run python scripts/eval_ovmm_find_phases.py --tier S2 --backend dynagraph \
  --merge-xy-m 0.75 --staleness-horizon 512
```

Outputs per run: `runs/ovmm_find_phase/<episode_id>_<backend>.json` plus `aggregate_<backends>.csv`.

### Metrics

- `find_object_success`, `find_recep_success` @ `success_radius_m`
- `find_partial_success` = mean of the two (OVMM-style 2-phase partial)
- `localization_err_obj_m`, `localization_err_recep_m`
- Scaling: `n_graph_nodes`, `n_voxel_explored_cells`, `n_voxel_explored_area_m2`, `n_placements`, `episode_wall_s`
- Optional GT diagnostics: `gt_graph_completeness`, `instance_gt_association_recall`

## Measured results (emet sim)

Target reference (real-world OVMM paper): **~70% FindObj / ~30% FindRec** partial-phase rates.
Our sim oracle and perception backends (Stretch, rotate-in-place, perfect sim depth):

| Tier | Backend | FindObj | FindRec | Partial | Notes |
|------|---------|---------|---------|---------|-------|
| S0 | ground_truth | 1.0 | 1.0 | 1.0 | `--not-rotate`, @0.30 m |
| S0 | dynagraph | target | target | target | rotate + perfect depth; err ≤0.30 m @0.30 m radius |
| S1 | ground_truth | 1.0 | 1.0 | 1.0 | `--not-rotate`, `object_gt_body: obj_main`, @0.50 m |
| S1 | dynagraph | target | target | target | rotate + perfect depth @0.50 m |

Reproduce S0 perception run:

```bash
emet install sim --no-download-assets --no-sync
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/eval_ovmm_find_phases.py \
  --episode-id default_table_s0 --backend dynagraph --cpu-only \
  --output-dir runs/ovmm_find_phase/s0_dynagraph
```

**Do not** pass `--not-rotate` for perception backends (`dynamem`, `graph_eqa`, `dynagraph`); mapping requires rotate-in-place.


| Episode | FindObj | FindRec | Partial | n_nodes | n_placements | wall_s |
|---------|---------|---------|---------|---------|--------------|--------|
| default_table_s0 | 1 | 1 | 1.0 | 5 | 5 | ~178 |
| default_table_s0_blue_cube | 1 | 1 | 1.0 | 5 | 5 | ~128 |

Artifacts: `runs/ovmm_find_phase/s0_ladder/aggregate_ground_truth.csv`

Reproduce:

```bash
uv run python scripts/eval_ovmm_find_phases.py --tier S0 \
  --backend ground_truth --not-rotate --cpu-only \
  --output-dir runs/ovmm_find_phase/s0_ladder
```

Perception backends (dynamem / graph_eqa / dynagraph) require rotate-in-place; use GPU and expect several minutes per episode:

```bash
uv run python scripts/eval_ovmm_find_phases.py --tier S0 \
  --backend dynamem --backend graph_eqa --backend dynagraph \
  --output-dir runs/ovmm_find_phase/s0_perception
```

## S1 / S2 notes

- **S1** requires sim install (`emet install sim`) and Robocasa kitchen assets.
- **S2** requires MolmoSpaces wrapper (`.venv-molmospaces`; see `docs/molmospaces.md`). Large scenes may return capped `sim_object_placements`; compare `n_placements` across Molmo indices.
- Use distinct `--port-offset` values when running parallel jobs.

## Phase 2: Habitat find-phase (HM3D proxy)

Episodes: `configs/ovmm/habitat_find_phase_episodes.yaml` (HM3D train scenes + semantic GT).
Same FindObj/FindRec metrics with **XZ** horizontal scoring (`frame: habitat_yup`).

```bash
./scripts/install_habitat.sh
uv run python scripts/download_habitat_eqa_data.py --fetch-csv --fetch-hm3d train
# Optional: HM3D semantic meshes (if scenes lack semantics)
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d-semantics train

# Batch GT smoke (3 HM3D scenes, ~7 min CPU)
uv run python scripts/eval_habitat_ovmm_find_phases.py \
  --backend ground_truth --not-rotate --cpu-only \
  --output-dir runs/ovmm_habitat/gt_batch

# Single episode
.venv-habitat/bin/emet-habitat run-ovmm-find-episode \
  --episode-id hm3d_lamp_bed_00006 --backend dynagraph --cpu-only
```

Verified GT batch: `find_partial_success=1.0`, `localization_err_*_m=0.0` on
`hm3d_lamp_bed_00006`, `00025`, `00057` (June 2026).

Full OVMM-HSSD minival (official leaderboard) is not wired yet; HM3D proxy validates the Habitat
memory → find-phase metric path before HSSD scene download.

## Paper

- Unified runbook: [paper_benchmarks.md](paper_benchmarks.md)
- Experiments plan: `paper/sections/04_experiments.tex` (`sec:ovmm_find_phase`, Table `tab:benchmark_configs`)
- Results tables: `paper/sections/05_results.tex` (`tab:ovmm_find_backend_tier`, scaling figures)
