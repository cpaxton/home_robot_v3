# Paper benchmarks — run and maintain

Operator guide for benchmarks referenced in `paper/sections/04_experiments.tex` and `paper/sections/05_results.tex`.

**LaTeX:** `./paper/build.sh` from repo root.  
**Results go in:** `paper/sections/05_results.tex` (tables/figures) — not committed automatically from sweeps.

## Benchmark map

| Track | Task | Primary metric | Config | Batch command | Aggregate → paper |
|-------|------|----------------|--------|---------------|-------------------|
| **OVMM find-phase (sim)** | Localize object + receptacle | Find partial success @ $r$ | `configs/ovmm/benchmark.yaml` | `scripts/eval_ovmm_find_phases.py` | `aggregate_<backends>.csv` in output dir |
| **OVMM find-phase (Habitat)** | Same, HM3D proxy | Same | `configs/ovmm/benchmark.yaml` | `scripts/eval_habitat_ovmm_find_phases.py` | per-run JSON under `~/runs/emet/ovmm_habitat` |
| **SQA3D** | Situated open QA | EM@1 | `configs/sqa3d/benchmark.yaml` | `emet sqa3d run-real-sweep` | `scripts/aggregate_sqa3d_sweep.py` → `aggregate_sqa3d.csv` |
| **GT object finding** | Sim oracle localization | XY error, recall @ $r$ | episode exports | `emet run dynagraph --ground-truth` + `scripts/eval_dynagraph_ground_truth.py` | manual / `emet eval-dynagraph` |
| **Dynagraph sim** | Explore + fusion + EQA | spatial/label recall, graph size | question bank yaml | `emet eval-dynagraph`, `run_dynagraph_benchmark_smoke.py` | [dynagraph_benchmarks.md](dynagraph_benchmarks.md) |
| **Habitat EQA** | HM-EQA / OpenEQA | MC accuracy, steps | Habitat install | `.venv-habitat/bin/emet-habitat` | separate branch (`feature/habitat-eqa-harness`) |

Deep dives: [ovmm_find_phase_benchmark.md](ovmm_find_phase_benchmark.md), [sqa3d.md](sqa3d.md), [sqa3d_compute.md](sqa3d_compute.md), [dynagraph_benchmarks.md](dynagraph_benchmarks.md), [habitat/README.md](habitat/README.md).

## Shared memory backends

Defined in `src/emet/eval/memory_backends.py`:

| Backend | OVMM sim | SQA3D | Meaning |
|---------|----------|-------|---------|
| `dynamem` | ✓ | ✓ | Voxel semantic map + EQA |
| `graph_eqa` | ✓ | — | GraphEQA, merge/staleness off |
| `dynagraph` | ✓ | ✓ (default) | Voxel + graph, default merge/staleness |
| `ground_truth` | ✓ (oracle) | — | Graph from sim GT placements |

Use the **same backend names** in sweep commands and paper tables.

## Output directories (default)

| Variable | Default | Used by |
|----------|---------|---------|
| `EMET_OVMM_OUTPUT_SIM` | `~/runs/emet/ovmm_find_phase` | `eval_ovmm_find_phases.py` |
| `EMET_OVMM_OUTPUT_HABITAT` | `~/runs/emet/ovmm_habitat` | `eval_habitat_ovmm_find_phases.py` |
| `EMET_SQA3D_OUTPUT` | `~/runs/emet/sqa3d` | `emet sqa3d run-real-sweep`, `aggregate_sqa3d_sweep.py` |

Caches (not under `runs/`): `SQA3D_DATA_DIR`, `SCANNET_ROOT`, `HABITAT_EQA_DATA_DIR`, `HM3D_DATA_PATH` — see [environment_variables.md](environment_variables.md).

---

## Quick smoke (before a paper sweep)

```bash
# Unit tests (no GPU, fast)
uv run emet test src/test/memory/test_ovmm_find_phase_metrics.py -q
uv run emet test src/test/benchmarks/sqa3d/ -q

# OVMM: unit + one sim GT + one Habitat GT (~5–10 min)
uv run python scripts/smoke_ovmm_benchmark.py --cpu-only

# SQA3D: mock-LLM episode (needs ScanNet mesh or skips)
uv run emet sqa3d run-episode --mock-llm --question-id 220602000000
```

---

## OVMM find-phase (Emet sim)

**Paper:** `sec:ovmm_find_phase`, Table `tab:ovmm_find_backend_tier`.

```bash
# Assets + dirs
uv run python scripts/download_ovmm_benchmark_assets.py

# S0 ladder (all backends)
uv run python scripts/eval_ovmm_find_phases.py \
  --tier S0 \
  --backend dynamem --backend graph_eqa --backend dynagraph --backend ground_truth \
  --cpu-only \
  --output-dir ~/runs/emet/ovmm_find_phase/s0_paper

# CSV written automatically: aggregate_dynamem-graph_eqa-dynagraph-ground_truth.csv
```

**Perception backends** (`dynamem`, `graph_eqa`, `dynagraph`): use GPU, **do not** pass `--not-rotate`.  
**Oracle** (`ground_truth`): may use `--not-rotate --cpu-only`.

Scale to S1/S2: `--tier S1` or `--tier S2`; see episode yaml for Molmo indices.

---

## OVMM find-phase (Habitat proxy)

**Paper:** Phase-2 row in `tab:envs`; Habitat table in “Other planned results”.

```bash
./scripts/install_habitat.sh
uv run python scripts/download_habitat_eqa_data.py --fetch-csv --fetch-hm3d train

uv run python scripts/eval_habitat_ovmm_find_phases.py \
  --backend ground_truth --not-rotate --cpu-only \
  --output-dir ~/runs/emet/ovmm_habitat/gt_batch
```

Full OVMM-HSSD minival is not wired yet; HM3D proxy validates the Habitat memory → find-phase path.

---

## SQA3D (ScanNet replay)

**Paper:** `sec:sqa3d_benchmark`, Table `tab:sqa3d_backend_replay`.

```bash
# Data
uv run python scripts/download_sqa3d_data.py --fetch-annotations
uv run python scripts/download_scannet_data.py --accept-tos --scenes-from-sqa3d --split val --limit 10
# Real RGB-D (large):
uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00 --with-sens

# Paper dev sweep (defaults from configs/sqa3d/benchmark.yaml: val 0:30, sens, isolated)
uv run emet sqa3d run-real-sweep --no-download --replay-mode sens

# Or GPU preflight wrapper:
./scripts/run_sqa3d_gpu_sweep.sh --split val --question-start 0 --question-end 30 --replay-mode sens

# Score + figures
uv run emet eval-sqa3d -p ~/runs/emet/sqa3d/dynagraph_val_q0-30.jsonl
uv run emet sqa3d plot-results -p ~/runs/emet/sqa3d/dynagraph_val_q0-30.jsonl -o paper/figures/sqa3d_val30

# Aggregate for LaTeX table fill
uv run python scripts/aggregate_sqa3d_sweep.py --input-dir ~/runs/emet/sqa3d
```

Episode JSONL fields for diagnostics: `infra_failure`, `replay_backend`, `sens_match_xy_m`, `em`, `planning_steps`.

Compare `dynamem` vs `dynagraph`:

```bash
uv run emet sqa3d run-real-sweep --method dynamem --output-dir ~/runs/emet/sqa3d/dynamem_val30 --no-download
uv run emet sqa3d run-real-sweep --method dynagraph --output-dir ~/runs/emet/sqa3d/dynagraph_val30 --no-download
uv run python scripts/aggregate_sqa3d_sweep.py \
  ~/runs/emet/sqa3d/dynamem_val30/*.jsonl ~/runs/emet/sqa3d/dynagraph_val30/*.jsonl \
  --output-dir ~/runs/emet/sqa3d
```

---

## GT-supervised object finding (Emet sim)

**Paper:** `sec:gt_experiments`.

```bash
# Terminal 1
uv run emet serve mujoco --scene robocasa --headless --port-offset 0

# Terminal 2
uv run emet run dynagraph --ground-truth --export runs/robocasa_gt --port-offset 0
uv run python scripts/eval_dynagraph_ground_truth.py --run-live --port-offset 0 --output metrics.json
```

Perception alignment: `emet run dynagraph --compare-to-gt --export runs/<id>`.

---

## Maintaining paper numbers

### 1. Run sweeps → CSV/JSON

- OVMM: `aggregate_*.csv` next to per-episode JSON in the output dir.
- SQA3D: `aggregate_sqa3d.csv` via `aggregate_sqa3d_sweep.py`.

### 2. Copy into LaTeX

Edit `paper/sections/05_results.tex`:

| Table label | Source |
|-------------|--------|
| `tab:ovmm_find_backend_tier` | `find_partial_success` column from OVMM aggregate CSV, by tier |
| `tab:sqa3d_backend_replay` | `em@1` from `aggregate_sqa3d.csv`, rows grouped by `method` × `replay_backend` |
| Habitat / HM-EQA | TBD on habitat branch |

Replace `--` placeholders; keep caption disclaimers (not comparable to official leaderboards where noted).

### 3. Figures

- SQA3D: `emet sqa3d plot-results` → `paper/figures/` (add `\includegraphics` in `05_results.tex` when ready).
- OVMM scaling: Fig. `fig:ovmm_scaling_success`, `fig:ovmm_scaling_nodes` — export from aggregate CSV or plotting notebook.

### 4. Keep docs ↔ code aligned

When changing CLI or defaults:

1. `emet <cmd> --help` and [cli.md](cli.md)
2. Topic doc (`sqa3d.md`, `ovmm_find_phase_benchmark.md`, …)
3. `paper/sections/04_experiments.tex` protocol bullets
4. `configs/*/benchmark.yaml` smoke/sweep defaults
5. [environment_variables.md](environment_variables.md) for new `EMET_*` toggles

### 5. Regression checks

```bash
uv run emet test src/test/memory/test_ovmm_find_phase_metrics.py \
  src/test/benchmarks/sqa3d/test_aggregate.py \
  src/test/benchmarks/sqa3d/test_benchmark_config.py -q
```

Optional integration (GPU/sim):

```bash
RUN_OVMM_FIND_TESTS=1 uv run emet test src/test/memory/test_ovmm_find_phase_integration.py -q
RUN_SQA3D_SCANNET_TESTS=1 uv run emet test src/test/benchmarks/sqa3d/test_scannet_embodied_smoke.py -q
```

### 6. Branch notes

| Benchmark | Typical branch |
|-----------|----------------|
| OVMM sim + Habitat proxy | `main` |
| SQA3D ScanNet replay | `feature/dynamem-offline-real-benchmark` |
| Habitat HM-EQA full harness | `feature/habitat-eqa-harness` |

Merge feature branches before final paper numbers; run smokes on the integration branch.

---

## Experiment section checklist (LaTeX)

`paper/sections/04_experiments.tex` should stay in sync with this doc:

- [ ] Goals list matches active tracks (Habitat EQA, SQA3D, OVMM, GT finding, dynamic episodes)
- [ ] Table `tab:envs` lists correct entrypoint scripts
- [ ] Each `sec:*_benchmark` protocol matches `--help` on the cited commands
- [ ] Metrics subsection states which metrics are **not** cross-comparable
- [ ] `references.bib` contains cites used in experiments (`sqa3d2023`, `ovmm2023`, …)

## See also

- [paper/README.md](../paper/README.md) — LaTeX build
- [TESTING.md](TESTING.md) — project-wide test conventions
