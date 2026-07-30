# Paper experiments — master index

**Start here** to run and reproduce Dynagraph paper benchmarks.
Detailed commands / LaTeX mapping: [paper_benchmarks.md](../paper_benchmarks.md).
GPU preflight + overnight: [evaluation.md](../evaluation.md). Repo doc map: [README § Documentation map](../../README.md#documentation-map).

| Need | Go to |
|------|--------|
| Experiment matrix + smokes | this page |
| Full operator runbook | [paper_benchmarks.md](../paper_benchmarks.md) |
| HM-EQA vs GraphEQA paper numbers | [habitat_eqa_results.md](habitat_eqa_results.md) |
| HM-EQA how-to (CLI flags) | [habitat/usage.md](../habitat/usage.md) · [habitat_eqa.md](habitat_eqa.md) |
| Agentic HM-EQA approach | [agentic_qwen_context.md](agentic_qwen_context.md#approach-current) |
| Profiles / merge policy | [`configs/benchmarks/dynagraph.yaml`](../../configs/benchmarks/dynagraph.yaml) |
| LaTeX | `paper/sections/04_experiments.tex`, `05_results.tex` |

## HM-EQA baselines

Two **internal** methods on the same Habitat harness (`configs/benchmarks/dynagraph.yaml` → `harness.habitat_eqa`):

| `--method` | Profile | Controller | Role |
|------------|---------|------------|------|
| `graph_eqa` | `graph_eqa_baseline` (merge=0, staleness=0; extras off) | `GraphEQAController` | GraphEQA **reimplementation** row |
| `dynagraph` | `unified_eqa` (0.45 m / 256) + memory on, debias off, explore **conservative**, SigLIP | `DynagraphController` | **Our method** |

External prior art (GraphEQA paper 63.5–67% with API VLMs + full HM3D semantics) is **citation only** — not the same stack. Ablations (`explore=off`, MCQ debias on, agentic H2H) must be labeled as such, not as method defaults.

### Run HM-EQA (dogfood CLI)

```bash
# Preflight (one GPU job at a time)
uv run emet eval recover --need-mib 12000
uv run emet habitat safe-start   # wait until jobs status = done

# Mock smoke (no VLM)
uv run emet habitat run-episode --question-id 0 --method graph_eqa --mock-llm
uv run emet habitat run-episode --question-id 0 --method dynagraph --mock-llm

# Classic vs agentic (Dynagraph H2H; explore-off ablation by script default)
uv run emet hmeqa overnight
# pause:  uv run emet jobs cancel JOB_ID
# resume: uv run emet hmeqa overnight --base ~/runs/emet/hmeqa_overnight_… --job-name hmeqa-overnight
uv run emet hmeqa h2h --out ~/runs/emet/hmeqa_probe --ids 15,68,105,17

# Full GraphEQA-paper 113 (both methods; prefer emet jobs)
uv run emet jobs run --name hmeqa-paper113 --need-mib 12000 -- \
  ./scripts/run_hmeqa_paper113_h2h.sh
# Or single method:
# .venv-habitat/bin/emet-habitat run-batch --method graph_eqa --paper-subset --resume ...
```

Results JSONL: `~/.cache/habitat_eqa/results/`. Numbers + planned sweeps: [habitat_eqa_results.md](habitat_eqa_results.md).

## Experiment matrix

| Track | Paper ref | Doc | Smoke command | Default output | Example figures |
|-------|-----------|-----|---------------|----------------|-----------------|
| **Backend localization** | qualitative / appendix | [backend_localization.md](backend_localization.md) | `uv run python scripts/smoke_backend_localization_figure.py --quick` | `~/runs/emet/backend_localization_smoke/` | `backend_localization_topdown.png`, `backend_localization_metrics_bar.png` |
| **Dynamic exploration P1** | `tab:dynamic_explore_phase1` | [dynamic_exploration.md](dynamic_exploration.md) · [dynagraph_dynamic_memory.md](dynagraph_dynamic_memory.md) | `uv run python scripts/eval_dynamic_exploration.py --smoke --dry-run` | `~/runs/emet/dynamic_exploration/` | Rerun map from export dir |
| **Dynamic exploration P2** | `tab:dynamic_explore_world_change` | [dynamic_exploration.md](dynamic_exploration.md) · [dynagraph_dynamic_memory.md](dynagraph_dynamic_memory.md) | `--phase world-change --cpu-only` | same | — |
| **Dynamic exploration P3** | `tab:dynamic_explore_lifelong` | [dynamic_exploration.md](dynamic_exploration.md) · [dynagraph_dynamic_memory.md](dynagraph_dynamic_memory.md) | `--phase lifelong --cpu-only` | same | — |
| **OVMM find-phase** | `tab:ovmm_find_backend_tier` | [ovmm_find_phase.md](ovmm_find_phase.md) · [dynagraph_dynamic_memory.md](dynagraph_dynamic_memory.md) | `uv run python scripts/run_ovmm_find_backend_matrix.py --cpu-only --backends ground_truth` | `~/runs/emet/ovmm_find_phase/` | scaling placeholders in `05_results.tex` |
| **OVMM full** | four-phase extension | [ovmm_full.md](ovmm_full.md) | `eval_ovmm_full.py --backend ground_truth --cpu-only` | `~/runs/emet/ovmm_full/` | — |
| **SQA3D** | `tab:sqa3d_backend_replay` | [sqa3d.md](sqa3d.md) | `uv run emet sqa3d run-episode --mock-llm --question-id 220602000000` | `~/runs/emet/sqa3d/` | `emet sqa3d plot-results` → `paper/figures/sqa3d_*` |
| **GT object finding** | `sec:gt_experiments` | [gt_object_finding.md](gt_object_finding.md) | `emet run dynagraph --ground-truth` + eval script | `runs/<export>/` | — |
| **Innate Mars** | appendix | [innate_mars.md](innate_mars.md) | `emet run dynagraph --robot innate_mars --ground-truth` | `/tmp/mars_*` | — |
| **Habitat EQA** | `tab:hmeqa_vs_prior`, Appendix | [habitat_eqa.md](habitat_eqa.md) · **[results](habitat_eqa_results.md)** · [agentic scale](agentic_scale.md) | `emet habitat` / `emet hmeqa` (above) | `~/.cache/habitat_eqa/results/` | topdown from debug bundles |
| **Representative sample (all tracks)** | — | [representative_sample_results.md](representative_sample_results.md) | `./scripts/run_representative_benchmark_sample.sh` | `~/runs/emet/representative_sample/` | ablation chart, maps, retrieval panels |
| **Cross-track smoke** | — | **[cross_track_smoke.md](cross_track_smoke.md)** | `./scripts/run_overnight_cross_track_smoke.sh` | `~/runs/emet/overnight_cross_track/` | validate before multi-day sweeps |
| **Simulation smoke (7-track)** | — | **[simulation_testing_plan.md](../simulation_testing_plan.md)** | `./scripts/run_simulation_smoke_battery.sh` | `~/runs/emet/simulation_smoke/` | paper-facing sequential battery |
| **Large paper queue** | all tracks | [large_eval_queue.md](large_eval_queue.md) | `./scripts/run_large_paper_eval.sh` | `~/runs/emet/<track>/` | — |

Shared backend names: `dynamem`, `graph_eqa`, `dynagraph`, `ground_truth` — see [paper_benchmarks.md § Shared memory backends](../paper_benchmarks.md#shared-memory-backends).

## Update from main

On a feature branch, stay current with shared `main` and refresh the env:

```bash
git pull origin main
uv sync
```

Paper experiment docs and CLI flags track `main`; after pulling, use the same `emet stream` / `emet capture` / `emet run dynamem|dynagraph` commands on sim **and** hardware ([zmq_obs.md](../zmq_obs.md), [innate_mars.md](innate_mars.md) § Sim vs hardware visualization). **Known issues:** [known_issues.md](../known_issues.md).

## Before any sweep

```bash
# Unit tests (no GPU, fast)
uv run emet test src/test/config/ -q
uv run emet test src/test/memory/test_ovmm_find_phase_metrics.py -q
uv run emet test src/test/benchmarks/sqa3d/ -q
uv run emet test src/test/eval/ -q

# OVMM: unit + one sim GT + one Habitat GT (~5–10 min)
uv run python scripts/smoke_ovmm_benchmark.py --cpu-only

# SQA3D: mock-LLM episode (needs ScanNet mesh or skips)
uv run emet sqa3d run-episode --mock-llm --question-id 220602000000

# Dynamic exploration: config matrix only (no GPU)
uv run python scripts/eval_dynamic_exploration.py --smoke --dry-run

# Backend localization figure (GPU; one job at a time)
uv run python scripts/smoke_backend_localization_figure.py --quick
```

## GPU hygiene

Run **one GPU-heavy job at a time** (dynamic exploration, backend localization, SQA3D real sweeps, Habitat HM-EQA with VLM). Parallel VLM loads can OOM (~15 GB each) or wedge the NVIDIA driver when the same GPU drives the desktop.

**Shared preflight** ([`emet eval`](../cli.md), [`scripts/gpu_preflight.sh`](../../scripts/gpu_preflight.sh)):

```bash
uv run emet eval recover --need-mib 12000         # preferred: status + diagnose + wait
uv run emet eval kill-stale                       # only if no intentional job is live
NEED_MIB=12000 uv run emet eval wait
```

Long GPU runs: **`uv run emet jobs run --name … -- CMD`** (not bare `nohup` when avoidable). Overnight orchestrators:

| Script | Purpose |
|--------|---------|
| [`run_overnight_cross_track_smoke.sh`](../../scripts/run_overnight_cross_track_smoke.sh) | Five-track smoke + safe no-sim pytest (`RUN_DEEP_EVAL=0` default) |
| [`run_overnight_eval_smoke.sh`](../../scripts/run_overnight_eval_smoke.sh) | HM-EQA + OVMM + SQA3D diagnostics matrix |
| [`run_overnight_habitat_eval.sh`](../../scripts/run_overnight_habitat_eval.sh) | Multi-phase HM-EQA subsets with `--resume` |
| [`run_hmeqa_paper113_h2h.sh`](../../scripts/run_hmeqa_paper113_h2h.sh) | Full 113 graph_eqa + dynagraph |
| [`run_sqa3d_gpu_sweep.sh`](../../scripts/run_sqa3d_gpu_sweep.sh) | SQA3D slice with VRAM preflight |

For backend localization, prefer `--explore-steps 0` or `--quick` (light mapping protocol) unless you need OVMM-identical rotate+explore mapping (`--full-protocol`, much slower).

**Cursor / agent:** launch via `emet jobs`; after agent death use `uv run emet status tail` — see [cross_track_smoke.md § Cursor sessions](cross_track_smoke.md#cursor--long-agent-sessions).

## Example figures

Figures are generated locally under `~/runs/emet/` — not committed to git by default. See [examples/README.md](examples/README.md) for paths and copy instructions.

| Figure | Regenerate | Validated example path |
|--------|------------|------------------------|
| Backend top-down map | `smoke_backend_localization_figure.py --quick` | `~/runs/emet/backend_localization_figures_20260617_final/backend_localization_topdown.png` |
| Backend metrics bar | same script | `~/runs/emet/backend_localization_figures_20260617_final/backend_localization_metrics_bar.png` |
| SQA3D diagnostics | `emet sqa3d plot-results -p <jsonl> -o paper/figures/sqa3d_val30` | `paper/figures/sqa3d_*` (after sweep) |

## Maintaining paper numbers

1. Run sweeps → per-track CSV/JSON under `~/runs/emet/` (HM-EQA under `~/.cache/habitat_eqa/results/`).
2. Copy aggregates into `paper/sections/05_results.tex` (table labels in [paper_benchmarks.md § Maintaining paper numbers](../paper_benchmarks.md#maintaining-paper-numbers)).
3. Build paper: `./paper/build.sh` from repo root.

## Output directory env vars

| Variable | Default |
|----------|---------|
| `EMET_OVMM_OUTPUT_SIM` | `~/runs/emet/ovmm_find_phase` |
| `EMET_OVMM_OUTPUT_FULL` | `~/runs/emet/ovmm_full` |
| `EMET_OVMM_OUTPUT_HABITAT` | `~/runs/emet/ovmm_habitat` |
| `EMET_SQA3D_OUTPUT` | `~/runs/emet/sqa3d` |
| `EMET_DYNAMIC_EXPLORE_OUTPUT` | `~/runs/emet/dynamic_exploration` |

Full list: [environment_variables.md](../environment_variables.md).

## Self-review notes

Last checked: **2026-07-28** on `main` (HM-EQA `graph_eqa` → `graph_eqa_baseline`; Dynagraph → `unified_eqa`).

- **Baselines:** HM-EQA methods must not share merge settings; see § HM-EQA baselines above.
- **Tests:** `src/test/eval/test_benchmark_dynagraph.py`, `test_dynagraph_harness.py` — `uv run emet test src/test/eval/ -q`.
- **Eval diagnostics:** unified `eval:` config — [evaluation.md](../evaluation.md), [emet_config.md](../emet_config.md).
- **GPU smoke:** full Phase 1 dynamic explore ~60–75 min; Habitat full 113 ≈ multi-night on one RTX 4090; run sequentially per [evaluation.md](../evaluation.md).
