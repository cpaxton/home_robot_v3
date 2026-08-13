# Paper benchmarks — run and maintain

> **Start here:** [experiments/README.md](experiments/README.md) for the master experiment index (matrix, smokes, example figures). This file is the detailed operator runbook.

Operator guide for benchmarks referenced in `paper/sections/04_experiments.tex` and `paper/sections/05_results.tex`.

**Unified eval runbook (overnight smoke, diagnostics, figures):** [evaluation.md](evaluation.md)

**LaTeX:** `./paper/build.sh` from repo root (uses local `latexmk` or Docker `texlive/texlive:latest`).
**Results go in:** `paper/sections/05_results.tex` (tables/figures) — not committed automatically from sweeps.

## Benchmark map

| Track | Task | Primary metric | Config | Batch command | Aggregate → paper |
|-------|------|----------------|--------|---------------|-------------------|
| **OVMM find-phase (sim)** | Localize object + receptacle | Find partial success @ $r$ | `configs/ovmm/benchmark.yaml` | `emet ovmm find` | `aggregate_<backends>.csv` in output dir |
| **OVMM full (sim)** | Find + pick + place | `ovmm_full_success` / four phase rates | `configs/ovmm/benchmark.yaml` | `emet ovmm full` | `~/runs/emet/ovmm_full` |
| **OVMM multi-env sweep** | Robocasa + Molmo find/full | FindObj/FindRec + full rates | `configs/ovmm/sweeps/molmo_robocasa.yaml` | `emet ovmm sweep --preset molmo-robocasa` | `OUT/rates.json` |
| **OVMM find-phase (Habitat)** | Same, HM3D proxy | Same | `configs/ovmm/benchmark.yaml` | `scripts/eval_habitat_ovmm_find_phases.py` | per-run JSON under `~/runs/emet/ovmm_habitat` |
| **SQA3D** | Situated open QA | EM@1 | `configs/sqa3d/benchmark.yaml` | `emet sqa3d run-real-sweep`, `run_sqa3d_sharded_sweep.sh` | `scripts/aggregate_sqa3d_sweep.py` → `aggregate_sqa3d.csv` |
| **Large paper queue** | All tracks above | per-track | — | `scripts/run_large_paper_eval.sh` | per-track CSV under `~/runs/emet/…` |
| **GT object finding** | Sim oracle localization | XY error, recall @ $r$ | episode exports | `emet run dynagraph --ground-truth` + `scripts/eval_dynagraph_ground_truth.py` | manual / `emet eval-dynagraph` |
| **Dynagraph sim** | Explore + fusion + EQA | spatial/label recall, graph size | question bank yaml | `emet eval-dynagraph`, `run_dynagraph_benchmark_smoke.py` | [dynagraph_benchmarks.md](dynagraph_benchmarks.md) |
| **Dynamic exploration** | Frontier explore + world-change + lifelong cycles | coverage, EQA, staleness, churn | `configs/benchmarks/dynamic_exploration.yaml` | `scripts/eval_dynamic_exploration.py` | `aggregate_dynamic_exploration.csv` |
| **Backend localization figure** | Single-scene GT vs prediction | XY error, @0.5 m hit | Robocasa seed 0 | `scripts/smoke_backend_localization_figure.py` | PNG + `smoke_results.json` |
| **Habitat EQA** | HM-EQA / OpenEQA | MC accuracy, steps | Habitat install | `.venv-habitat/bin/emet-habitat` | [habitat_eqa_results.md](experiments/habitat_eqa_results.md) |
| **RoboVista** | Offline robot-centric MCQ-VQA (static images) | MC accuracy (overall + domain) | HF `sy-xie/robovista` | `emet robovista run-batch` | not comparable to HM-EQA |

Deep dives: [experiments/README.md](experiments/README.md) (index), [ovmm_find_phase_benchmark.md](ovmm_find_phase_benchmark.md), [ovmm_full_benchmark.md](ovmm_full_benchmark.md), [dynamic_exploration_benchmark.md](dynamic_exploration_benchmark.md), [experiments/backend_localization.md](experiments/backend_localization.md), [sqa3d.md](sqa3d.md), [sqa3d_compute.md](sqa3d_compute.md), [dynagraph_benchmarks.md](dynagraph_benchmarks.md), [habitat/README.md](habitat/README.md).

## Shared memory backends

Defined in `src/emet/eval/memory_backends.py`:

| Backend | OVMM sim | SQA3D | Meaning |
|---------|----------|-------|---------|
| `dynamem` | ✓ | ✓ | Voxel semantic map + EQA |
| `static_graph` | ✓ | — | Object graph, merge/staleness off (GraphEQA-inspired baseline; legacy alias `graph_eqa`) |
| `dynagraph` | ✓ | ✓ (default) | Voxel + graph, default merge/staleness |
| `ground_truth` | ✓ (oracle) | — | Graph from sim GT placements |

Use the **same backend names** in sweep commands and paper tables. CLI Choices still accept `graph_eqa` → normalize to `static_graph` (one warning per process).

**Detector vs semantics (methods):** YoloE/OWL at low `detection.confidence_threshold` supplies **high-recall instance/graph proposals**. Semantic answers come from the **VLM** and **memory backends**. You *can* raise the mapping threshold if eval shows net benefit; treat that as a recall/precision trade-off on OVMM/graph tasks. Chat-only tightening uses `describe_confidence_threshold` (separate knob). See [dynamem.md](dynamem.md), [graph_eqa.md](graph_eqa.md) (implementation package), [AGENT_RUN.md](AGENT_RUN.md).

## Dynagraph profiles (shared config)

Merge/staleness and short-episode caps are defined once in [`configs/benchmarks/dynagraph.yaml`](../configs/benchmarks/dynagraph.yaml) and applied via [`src/emet/eval/benchmark_dynagraph.py`](../src/emet/eval/benchmark_dynagraph.py). Base defaults also live in [`dynav_config.yaml`](../src/emet/config/dynav_config.yaml) (`dynagraph_merge_xy_m: 0.45`, `dynagraph_staleness_horizon: 256`) so `get_parameters` and `emet run dynagraph` agree.

| Profile | merge (m) | staleness | Used by |
|---------|-----------|-----------|---------|
| `interactive` | 0.45 | 256 | `emet run dynagraph`, `emet run agent` (default), dynav YAML default |
| `eqa` | 0.45 | 256 + nav cap 48 | SQA3D tuned (`dynagraph`) |
| `unified_eqa` | 0.45 | 256 + nav cap 48 | **HM-EQA Dynagraph default** (Habitat + shared EQA); same merge as interactive |
| `find_phase` | 0.15 | 256 | OVMM find-phase (`dynagraph`, `ground_truth`) |
| `static_graph` | 0 | 0 | GraphEQA-inspired comparison row (HM-EQA / OVMM / dynamic-explore `static_graph`; legacy profile name `graph_eqa_baseline`) |
| `smoke` | 0 | 0 | CI / mock-LLM only — **not** a paper method default |

**Merge policy:** Dynagraph’s product and paper *method* rows use merge (0.45 m interactive/EQA, 0.15 m find-phase). True zero-merge is reserved for (1) GraphEQA-inspired baseline parity (`static_graph`) and (2) fast CI smoke. Do not report HM-EQA Dynagraph numbers under `smoke` / zero-merge — that disables the instance-memory behavior the system is built around.

**HM-EQA methods:** `--method static_graph` → profile `static_graph` + `GraphEQAController`; `--method dynagraph` → `unified_eqa` + tuned extras + `DynagraphController`. Legacy `--method graph_eqa` aliases `static_graph`. See [experiments/README.md § HM-EQA baselines](experiments/README.md#hm-eqa-baselines).

**Interactive agent exploration** (same Dynagraph memory as paper harnesses):

```bash
# Live Discord/terminal agent (default --memory-backend dynagraph); add --eqa for VL captions
uv run emet run agent --robot stretch --config configs/agent_stretch_discord.yaml --eqa --rerun

# Scored HM-EQA via the shared episode function (identical to emet-habitat; no chat router)
uv run emet run agent --eqa-eval --habitat-question-id 17 --eqa-eval-mock-llm \
  --extra-instruction "Answer with a single letter A–D."

# Scripted skill smoke (no Discord)
uv run emet run agent --robot stretch --start-sim --no-discord \
  -c "describe the scene" -c "explore" -c "show me the map"
```

Benchmark tracks that exercise agent-style exploration (frontier / rotate) already default to **dynagraph** profiles — see one-liners below and [simulation_testing_plan.md](simulation_testing_plan.md). Prefer `uv run emet eval kill-stale` / `wait` before GPU VLM evals; do not stack overnight Habitat with Robocasa dynagraph in one session.

**Harness blocks** (`harness:` in the same YAML) set per-benchmark controller flags (`memory_summary`, `mcq_debias`, `explore_when_uncovered`, `use_instance_graph`, …) via `apply_dynagraph_harness()`. Tuned HM-EQA defaults (`habitat_eqa` / dynagraph): `memory_summary=true`, `mcq_debias=false`, `explore_when_uncovered=conservative` (prefer Habitat/voxel frontiers **while uncovered**; not a weaker picker than `on`).

| Harness | Profile | Dynagraph EQA extras |
|---------|---------|----------------------|
| `habitat_eqa` | `unified_eqa` | memory on, debias off, conservative explore |
| `habitat_ovmm_find` | `find_phase` | EQA off |
| `ovmm_find_phase` | `find_phase` | Robocasa / Molmo search |
| `sqa3d` | `eqa` | memory/debias off (open QA) |
| `dynamic_explore` | `interactive` | full dynagraph extras |

**HM-EQA recovery gate (merge-on):** Before claiming Dynagraph accuracy recovery, run [`scripts/run_q17_merge_on_gate.sh`](../scripts/run_q17_merge_on_gate.sh) (≥2/3 Q17 correct under default `unified_eqa`, no explore CLI override). Episode JSONL / manifests include a **harness fingerprint** (`harness` dict: `git_commit`, `dynagraph_merge_xy_m`, `fallback_spatial_merge_xy_m`, `profile`, `explore_when_uncovered`, …) — always cite it when quoting accuracy. Explore-off (`--explore-when-uncovered off`) is an **ablation only**, not the HM-EQA Dynagraph default. Re-baseline after a green gate: [`scripts/run_merge_on_hmeqa_baseline.sh`](../scripts/run_merge_on_hmeqa_baseline.sh) (holdout8 + smoke 3,14,17).

Tuning / paper battery: [`scripts/run_dynagraph_tuning_matrix.sh`](../scripts/run_dynagraph_tuning_matrix.sh), [`scripts/run_dynagraph_tuned_paper_battery.sh`](../scripts/run_dynagraph_tuned_paper_battery.sh). **Representative cross-benchmark sample:** [`scripts/run_representative_benchmark_sample.sh`](../scripts/run_representative_benchmark_sample.sh) + [`build_representative_results_tables.py`](../scripts/build_representative_results_tables.py) → [representative_sample_results.md](experiments/representative_sample_results.md). Figures: `render_paper_map_figures.py`, `render_graph_retrieval_panel.py`, `render_dynagraph_3d_figure.py`, `build_eval_figure_pack.py --render-retrieval-panels`.

**Task-specific (documented, not unified):** EQA prompts (`prompt_variant`: `sqa3d` vs default), and controller flags (`use_instance_graph`, `manipulation_only`) differ between OVMM localization and SQA3D open QA — see `harness:` in the YAML above.

CLI overrides: OVMM `--merge-xy-m` / `--staleness-horizon`; Habitat `emet-habitat` `--no-mcq-debias`, `--memory-summary`, `--explore-when-uncovered {off,on,conservative}`; env `EMET_DYNAGRAPH_*` (see [environment_variables.md](environment_variables.md)).

## Output directories (default)

| Variable | Default | Used by |
|----------|---------|---------|
| `EMET_OVMM_OUTPUT_SIM` | `~/runs/emet/ovmm_find_phase` | `emet ovmm find` / `eval_ovmm_find_phases.py` |
| `EMET_OVMM_OUTPUT_HABITAT` | `~/runs/emet/ovmm_habitat` | `eval_habitat_ovmm_find_phases.py` |
| `EMET_SQA3D_OUTPUT` | `~/runs/emet/sqa3d` | `emet sqa3d run-real-sweep`, `aggregate_sqa3d_sweep.py` |
| `EMET_DYNAMIC_EXPLORE_OUTPUT` | `~/runs/emet/dynamic_exploration` | `scripts/eval_dynamic_exploration.py` |
| `EMET_SCENE_MAP_CACHE_DIR` | `~/.cache/emet/scene_maps` | Prebuilt graph+voxel baselines (`build_scene_map_cache.py`) |

Caches (not under `runs/`): `SQA3D_DATA_DIR`, `SCANNET_ROOT`, `HABITAT_EQA_DATA_DIR`, `HM3D_DATA_PATH`, scene maps under `EMET_SCENE_MAP_CACHE_DIR` — see [environment_variables.md](environment_variables.md).

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
**CLI:** `emet ovmm find` (compat: `scripts/eval_ovmm_find_phases.py`).

```bash
# Assets + dirs
uv run python scripts/download_ovmm_benchmark_assets.py

# S0 ladder (all backends)
uv run emet ovmm find \
  --episodes configs/ovmm/find_phase_episodes.yaml \
  --tier S0 \
  --backend dynamem --backend static_graph --backend dynagraph --backend ground_truth \
  --output-dir ~/runs/emet/ovmm_find_phase/backend_matrix
# CSV written automatically: aggregate_dynamem-static_graph-dynagraph-ground_truth.csv
# (legacy runs may still use graph_eqa in the CSV name)
```

**Perception backends** (`dynamem`, `static_graph`, `dynagraph`): use GPU, **do not** pass `--not-rotate`.
**Oracle** (`ground_truth`): may use `--not-rotate --cpu-only`.

Scale to S1/S2: `--tier S1` or `--tier S2`; see episode yaml for Molmo indices.

### Multi-env sweep (Robocasa + MolmoSpaces)

Paper path for real kitchen / iTHOR rates (no `default_table`). Preset: `configs/ovmm/sweeps/molmo_robocasa.yaml`.

```bash
uv run emet ovmm sweep --preset molmo-robocasa --backend dynagraph --via-jobs
uv run emet ovmm rates --out ~/runs/emet/ovmm_molmo_robocasa/<DATE>
uv run emet ovmm status --out ~/runs/emet/ovmm_molmo_robocasa/<DATE>
```

---

## OVMM full (find + pick + place)

**Paper:** four-phase extension of find-phase (not yet a dedicated results table).
**Doc:** [ovmm_full_benchmark.md](ovmm_full_benchmark.md).
**CLI:** `emet ovmm full` (compat: `scripts/eval_ovmm_full.py`).

```bash
uv run emet test src/test/memory/test_ovmm_full_metrics.py -q

# Oracle manip smoke (fast)
uv run emet ovmm full \
  --episodes configs/ovmm/full_episodes.yaml \
  --episode-id default_table_s0_distinct_recep \
  --backend ground_truth --not-rotate --cpu-only \
  --manip-mode oracle \
  --output-dir ~/runs/emet/ovmm_full/smoke

# Sim E2E (uses ZMQ sim_set_body_pose for pick/place)
uv run emet ovmm full \
  --episodes configs/ovmm/full_episodes.yaml \
  --episode-id robocasa_pp_s1 \
  --backend dynagraph --manip-mode sim --cpu-only

# MolmoSpaces + rby1
uv run emet ovmm full \
  --episodes configs/ovmm/full_episodes.yaml \
  --episode-id molmo_ithor_rby1_s2_bowl_pp \
  --backend ground_truth --manip-mode sim --not-rotate --cpu-only
```

Shared sim body teleport: `sim_set_body_pose` (also used by Phase~2 dynamic exploration world-change). Robosuite (rby1) advertises the same capability as Stretch MuJoCo.

---

## Dynamic exploration (Emet sim)

**Paper:** Section~\ref{sec:dynamic_exploration}, Tables `tab:dynamic_explore_phase1`, `tab:dynamic_explore_world_change`.
**Doc:** [dynamic_exploration_benchmark.md](dynamic_exploration_benchmark.md) · [experiments/dynagraph_dynamic_memory.md](experiments/dynagraph_dynamic_memory.md).

```bash
# Optional: prebuild scene maps so P1 / OVMM skip live explore (once per scene)
env -u PYTHONPATH uv run python scripts/build_scene_map_cache.py

# Dry-run full Phase 1 matrix
uv run python scripts/eval_dynamic_exploration.py --dry-run

# Phase 1 smoke (Robocasa seed0, K=3; uses scene cache when present)
uv run python scripts/eval_dynamic_exploration.py \
  --phase explore --episode-id robocasa_seed0 \
  --backend dynagraph --explore-max-iters 3 --mapping-mode explore \
  --cpu-only --output-dir ~/runs/emet/dynamic_exploration/smoke

# Rotate-only contrast row
uv run python scripts/eval_dynamic_exploration.py \
  --phase explore --episode-id robocasa_seed0 \
  --mapping-mode rotate_only --backend dynagraph --cpu-only

# Phase 2 world-change
uv run python scripts/eval_dynamic_exploration.py \
  --phase world-change --cpu-only --resume

# Lifelong K-cycle checkpoint/fuzz/reload (Robocasa + Molmo iTHOR)
uv run python scripts/eval_dynamic_exploration.py \
  --phase lifelong --resume

# Full overnight matrix
./scripts/run_dynamic_exploration_full.sh
```

Aggregates: `aggregate_dynamic_exploration.csv`, `aggregate_dynamic_exploration_world_change.csv`, `aggregate_dynamic_exploration_lifelong.csv` under `EMET_DYNAMIC_EXPLORE_OUTPUT` (default `~/runs/emet/dynamic_exploration`).

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

# Or GPU preflight wrappers:
uv run emet eval kill-stale
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

## Large paper eval queue

Orchestrator: `scripts/run_large_paper_eval.sh` (SQA3D → OVMM find → dynamic exploration; multi-GPU SQA3D via `SQA3D_GPUS`; `--resume` where supported).

```bash
# Full queue: SQA3D val+test (dynamem+dynagraph) → OVMM find replicates → dynamic exploration
./scripts/run_large_paper_eval.sh

# One phase
./scripts/run_large_paper_eval.sh sqa3d-val
./scripts/run_large_paper_eval.sh ovmm
./scripts/run_large_paper_eval.sh dynamic-explore

# Skip phases
SKIP_OVMM=1 ./scripts/run_large_paper_eval.sh
SKIP_DYNAMIC_EXPLORE=1 ./scripts/run_large_paper_eval.sh

# Run OVMM on CPU while SQA3D holds the GPU (overlap manually in two terminals)
OVMM_CPU_ONLY=1 ./scripts/run_large_paper_eval.sh ovmm
```

Logs: `~/runs/emet/large_eval/<phase>.log`. Outputs: `EMET_SQA3D_OUTPUT`, stamped `ovmm_find_phase/large_<ts>`, stamped `dynamic_exploration/large_<ts>` on `phase=all`.

### Why SQA3D dominates

Default `--isolate-episodes` spawns a **fresh Python process per question**, reloading the VLM (~1–2 min overhead) plus ScanNet sim + planning. That is ~2–4 min × **13.5k** questions × **2 methods** × **2 splits** ≈ **3–4 weeks** on one GPU.

### Estimated wall-clock (resume on)

| Phase | Runs | 1 GPU isolate | 4 GPU isolate |
|-------|------|---------------|---------------|
| SQA3D val (both methods) | ~6.5k | ~10–22 days | ~2.5–6 days |
| SQA3D test (both methods) | ~7k | ~10–22 days | ~2.5–6 days |
| OVMM find replicates | 180 | ~0.5–1.5 days (CPU) | same |
| Dynamic explore | 50 | ~1–2 days | same |

| Config | Total queue |
|--------|-------------|
| Default (`phase=all`, 1 GPU) | **~22–38 days** |
| `SQA3D_GPUS=0,1,2,3` | **~6–11 days** |
| 4 GPU + `SQA3D_NO_ISOLATE=1` | **~3–6 days** (watch VRAM) |
| `SKIP_SQA3D_TEST=1` + 4 GPU | **~4–7 days** (val-only paper draft) |
| `SQA3D_METHODS=dynagraph` + 4 GPU + val only | **~2–4 days** (fastest credible table) |

### Speed knobs

```bash
# Best throughput (4 GPUs, full val+test, both methods)
SQA3D_GPUS=0,1,2,3 ./scripts/run_large_paper_eval.sh

# Or shard one sweep directly:
./scripts/run_sqa3d_sharded_sweep.sh --split val --method dynagraph --all --gpus 0,1,2,3

# Faster but riskier: keep VLM loaded in-process (2–3× per GPU; may OOM after hours)
SQA3D_NO_ISOLATE=1 SQA3D_GPUS=0,1,2,3 ./scripts/run_large_paper_eval.sh sqa3d-val

# Paper iteration: val dynagraph only
SQA3D_METHODS=dynagraph SKIP_SQA3D_TEST=1 SKIP_OVMM=1 SKIP_DYNAMIC_EXPLORE=1 \
  SQA3D_GPUS=0,1,2,3 ./scripts/run_large_paper_eval.sh sqa3d-val

# Overlap: terminal A = SQA3D on GPU; terminal B = OVMM on CPU
OVMM_CPU_ONLY=1 ./scripts/run_large_paper_eval.sh ovmm
```

Resume skips finished JSONL lines. Sharded sweeps write per-shard JSONL + `*_merged.csv` via `aggregate_sqa3d_sweep.py`.

---

## Habitat EQA (HM-EQA)

**Full tables, JSONL tags, gap analysis, planned sweeps:** [experiments/habitat_eqa_results.md](experiments/habitat_eqa_results.md)

| Comparison | Our best | Prior art | Gap |
|------------|----------|-----------|-----|
| Full 113 Q | 41.6% (gemma-3-4b, static_graph / logged as graph_eqa) | 63.5–67.0% (GraphEQA + API VLM) | ~−22 pp |
| Matched slice | dynagraph +3 pp vs static_graph (bal-32 @ 3B) | — | Dynagraph helps |
| Post-fix hold-out | 50% (8 Q, Qwen3-VL-8B) | 51.7% Explore-EQA (113 Q) | not comparable (small n) |

**Next headline run (IN PROGRESS):** full 113 with `Qwen/Qwen3-VL-8B-Instruct` + July 2026 nav stack + June 2026 fix stack — job `hmeqa-paper113-d1` (2026-08-13), OUT `~/runs/emet/hmeqa_paper113/20260813_104004` (`run-batch --debug-run-tag` fix + `EMET_ALLOW_SDPA_ATTN=1`).

Install: `./scripts/install_habitat.sh`. Entrypoint: `.venv-habitat/bin/emet-habitat`. Parity: `paper/sections/appendix/05_habitat_eqa_parity.tex`.

---

## RoboVista (offline MCQ-VQA)

Static robot-centric VQA from HuggingFace [`sy-xie/robovista`](https://huggingface.co/datasets/sy-xie/robovista) (474 expert MCQs, A–E, six domains). Uses the same local VLMs as Habitat/SQA3D but **no navigation / graph memory**. Scores are **not comparable** to HM-EQA.

```bash
uv run emet robovista info
uv run emet robovista run-batch --mock-llm --max-questions 5
uv run emet robovista run-batch --domain domestic --eqa-vl-family qwen3_vl --device cuda
```

Outputs: `~/runs/emet/robovista/<run_id>/predictions.jsonl` + `summary.json` (overall + by domain / ability_type). Not part of the seven-track sim smoke battery.

---

## Maintaining paper numbers

### 1. Run sweeps → CSV/JSON

- OVMM: `aggregate_*.csv` next to per-episode JSON in the output dir.
- SQA3D: `aggregate_sqa3d.csv` via `aggregate_sqa3d_sweep.py`.

### 2. Copy into LaTeX

Edit `paper/sections/05_results.tex`:

| Table label | Source |
|-------------|--------|
| `tab:ovmm_find_backend_tier` | OVMM aggregate CSV: `find_object_success`, `find_recep_success`, `find_partial_success` by tier (FindObj typically easier than FindRec) |
| `tab:sqa3d_backend_replay` | `em@1` from `aggregate_sqa3d.csv`, rows grouped by `method` × `replay_backend` |
| `tab:dynamic_explore_phase1` | `aggregate_dynamic_exploration.csv` (explored_fraction, eqa_accuracy, …) |
| `tab:dynamic_explore_world_change` | `aggregate_dynamic_exploration_world_change.csv` |
| `tab:dynamic_explore_lifelong` | `aggregate_dynamic_exploration_lifelong.csv` (per-cycle eqa_accuracy, node counts, moves adapted/stale) |
| Habitat / HM-EQA | `tab:hmeqa_vs_prior`, `tab:hmeqa_preliminary`, Appendix `tab:habitat_interim_results` | **[experiments/habitat_eqa_results.md](experiments/habitat_eqa_results.md)** (JSONL under `~/.cache/habitat_eqa/results/`) |

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
| SQA3D ScanNet replay | `main` (was `feature/dynamem-offline-real-benchmark`) |
| Habitat HM-EQA harness | `main` (merged); eval/diagnostics on feature branches e.g. `feature/eval-tools` |

Run smokes on the integration branch before copying numbers into `paper/sections/05_results.tex`.

### Pre-sweep checklist (Habitat)

Before a headline HM-EQA sweep:

1. `uv run emet eval kill-stale` then `NEED_MIB=12000 uv run emet eval wait`
2. Tag JSONL output with stack version (e.g. `_postfix_nav202607`) — see [habitat_eqa_results.md](experiments/habitat_eqa_results.md)
3. `uv run python scripts/download_habitat_eqa_data.py --report-hmeqa-semantics` for semantics coverage audit
4. Movement smoke: `.venv-habitat/bin/emet-habitat run-episode --mock-llm --mock-llm-explore --question-id 3 --max-planning-steps 5`
5. Merge-on recovery gate: `./scripts/run_q17_merge_on_gate.sh` (≥2/3) and cite harness fingerprints when reporting accuracy

---

## Experiment section checklist (LaTeX)

`paper/sections/04_experiments.tex` should stay in sync with this doc:

- [x] Goals list matches active tracks (Habitat EQA + OpenEQA milestone, SQA3D, OVMM, dynamic exploration, GT finding) — updated 2026-07
- [x] Habitat HM-EQA interim prose + planned sweeps match [experiments/habitat_eqa_results.md](experiments/habitat_eqa_results.md) — full 113 numbers still pending
- [ ] `scripts/run_large_paper_eval.sh` phases and `SQA3D_GPUS` / skip env vars match `docs/environment_variables.md`
- [ ] Table `tab:envs` lists correct entrypoint scripts
- [ ] Each `sec:*_benchmark` protocol matches `--help` on the cited commands
- [ ] Metrics subsection states which metrics are **not** cross-comparable
- [ ] `references.bib` contains cites used in experiments (`sqa3d2023`, `ovmm2023`, …)

## See also

- [experiments/README.md](experiments/README.md) — master experiment index
- [experiments/habitat_eqa_results.md](experiments/habitat_eqa_results.md) — HM-EQA vs prior art + planned sweeps
- [paper/README.md](../paper/README.md) — LaTeX build
- [TESTING.md](TESTING.md) — project-wide test conventions
