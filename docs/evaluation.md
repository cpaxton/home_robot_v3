# Evaluation runbook

Canonical guide for paper-relevant benchmarks: Habitat HM-EQA, OVMM find-phase (Habitat + sim), and SQA3D. Use this doc for **overnight smoke**, **diagnostics artifacts** (maps, video, crops), and **figure export**.

Deep dives:

| Track | Doc |
|-------|-----|
| Habitat EQA | [habitat_eqa.md](habitat_eqa.md) → [habitat/](habitat/README.md) |
| OVMM find-phase | [ovmm_find_phase_benchmark.md](ovmm_find_phase_benchmark.md), [ovmm.md](ovmm.md) |
| SQA3D | [sqa3d.md](sqa3d.md), [sqa3d_compute.md](sqa3d_compute.md) |
| Paper tables / LaTeX | [paper_benchmarks.md](paper_benchmarks.md) |
| Dynagraph sim | [dynagraph_benchmarks.md](dynagraph_benchmarks.md) |

## Prerequisites

```bash
uv sync
./scripts/install_habitat.sh
uv run python scripts/download_habitat_eqa_data.py --fetch-csv --fetch-hm3d train
uv run python scripts/download_ovmm_benchmark_assets.py   # optional verify
uv run python scripts/download_sqa3d_data.py --fetch-annotations
# ScanNet (SQA3D): uv run python scripts/download_scannet_data.py --accept-tos --scenes-from-sqa3d --with-sens
```

## Shared diagnostics

All embodied tracks can write a **consistent episode bundle** via [`src/emet/eval/episode_diagnostics.py`](../src/emet/eval/episode_diagnostics.py):

```
~/.cache/habitat_eqa/episodes/<run_tag>/
  q0003_graph_eqa/              # HM-EQA
  ovmm_hm3d_lamp_bed_00006_dynamem/
  sqa3d_220602000049_dynagraph/
    topdown_map.png
    obstacles_2d.npy, explored_2d.npy, grid_meta.json
    trajectory.jsonl
    frames/rgb_*.png, metadata.jsonl
    episode_rgb.mp4
    floor_metrics.json
    diagnostics_manifest.json
    (track-specific: raw_eqa.txt, memory/, …)
```

**Environment variables** (see [environment_variables.md](environment_variables.md)):

| Variable | Default (smoke) | Effect |
|----------|-----------------|--------|
| `EMET_EVAL_EXPORT_MAP` | on | `topdown_map.png` |
| `EMET_EVAL_EXPORT_VIDEO` | on | `episode_rgb.mp4` |
| `EMET_EVAL_EXPORT_FRAMES` | on | RGB frame PNGs |
| `EMET_EVAL_MAP_STRIDE` | 0 | Intermediate `maps/step_NNNN.png` |
| `EMET_EVAL_EXPORT_GRAPH` | off | Full graph checkpoint (heavy) |

Habitat aliases: `HABITAT_EQA_EXPORT_MAP`, `HABITAT_EQA_EXPORT_VIDEO`, `HABITAT_EQA_EXPORT_GRAPH`.

**CLI flags** (`.venv-habitat/bin/emet-habitat`): `--export-map`, `--export-video`, `--map-stride` on `run-episode` / `run-batch`; OVMM batch adds `--run-tag`.

## Overnight smoke (all tracks)

One script runs HM-EQA, OVMM Habitat, and SQA3D (if ScanNet verify passes), then builds figures:

```bash
./scripts/run_overnight_eval_smoke.sh
# Dry layout check (no VLM):
MOCK_LLM=1 ./scripts/run_overnight_eval_smoke.sh
# Skip SQA3D if ScanNet not installed:
SKIP_SQA3D=1 ./scripts/run_overnight_eval_smoke.sh
```

**Matrix (~21 GPU episodes with real VLM):**

| Phase | Benchmark | Units | Methods |
|-------|-----------|-------|---------|
| 1 | HM-EQA | Q `3,14,17` | `graph_eqa`, `dynagraph` |
| 2 | OVMM Habitat | 3 HM3D proxy episodes | **`dynamem`**, `graph_eqa`, `dynagraph` |
| 3 | SQA3D | val Q `0–2` | `dynagraph`, `dynamem` |

Outputs:

- JSONL: `~/.cache/habitat_eqa/results/<TAG>_hmeqa_*.jsonl`
- OVMM JSON: `~/runs/emet/ovmm_habitat/<TAG>_*/`
- SQA3D: `~/runs/emet/sqa3d/<TAG>_*/`
- Bundles: `~/.cache/habitat_eqa/episodes/<TAG>_*/`
- Logs: `~/.cache/habitat_eqa/overnight/<RUN_ID>/`
- Figures: `~/runs/emet/eval_smoke/<RUN_ID>/figures/`

Post-run only:

```bash
uv run python scripts/build_eval_figure_pack.py --run-id eval_smoke_YYYYMMDD_HHMMSS
```

## Success criteria (smoke)

| Track | Metric | Minimum |
|-------|--------|---------|
| HM-EQA | MCQ accuracy (3 Qs) | **> 0%** (≥1 correct) |
| OVMM Habitat | `find_partial_success` | **> 0** on ≥1 episode × backend |
| SQA3D | EM@1 (3 Qs) | **> 0%** when phase 3 runs |
| All | Artifacts | `topdown_map.png` + `diagnostics_manifest.json` per completed episode |

If HM-EQA and OVMM metrics are all zero, `build_eval_figure_pack.py` sets `"status": "INVESTIGATE"` in `summary.json`.

## Per-track quick commands

### HM-EQA (interactive QA)

```bash
.venv-habitat/bin/emet-habitat run-batch \
  --method dynagraph --question-ids 3,14,17 \
  --export-map --export-video --resume \
  --output ~/.cache/habitat_eqa/results/smoke_dynagraph.jsonl
```

### OVMM find-phase (Habitat) — includes **dynamem**

```bash
.venv-habitat/bin/emet-habitat run-ovmm-find-batch \
  --backend dynamem --run-tag smoke_ovmm \
  --export-map --export-video \
  --output-dir ~/runs/emet/ovmm_habitat/smoke_dynamem
```

### SQA3D

```bash
uv run emet sqa3d run-real-sweep --split val --question-start 0 --question-end 2 \
  --method dynagraph --replay-mode sens --no-download \
  --export-root ~/.cache/habitat_eqa/episodes/smoke_sqa3d
uv run emet sqa3d plot-results -p ~/runs/emet/sqa3d/dynagraph_val_q0-2.jsonl -o /tmp/sqa3d_figs
```

## Method coverage

| Backend | HM-EQA | OVMM Habitat | SQA3D |
|---------|--------|--------------|-------|
| `graph_eqa` | yes | yes | — |
| `dynagraph` | yes | yes | yes |
| `dynamem` | follow-up PR | **yes** | yes |

## Paper outputs

Figures for early results: `~/runs/emet/eval_smoke/<run_id>/figures/topdown_map_grid.png`, `summary.csv`. Copy into `paper/figures/eval_smoke/` and cite in `paper/sections/05_results.tex`.
