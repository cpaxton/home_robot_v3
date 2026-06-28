# Paper experiments — master index

Start here to run and reproduce benchmarks for the Dynagraph CoRL paper.

- **Detailed operator runbook:** [paper_benchmarks.md](../paper_benchmarks.md) (full commands, LaTeX table mapping, large-queue timing).
- **LaTeX plan:** `paper/sections/04_experiments.tex`
- **Results tables:** `paper/sections/05_results.tex` (fill manually from sweep CSV/JSON)

## Experiment matrix

| Track | Paper ref | Doc | Smoke command | Default output | Example figures |
|-------|-----------|-----|---------------|----------------|-----------------|
| **Backend localization** | qualitative / appendix | [backend_localization.md](backend_localization.md) | `uv run python scripts/smoke_backend_localization_figure.py --quick` | `~/runs/emet/backend_localization_smoke/` | `backend_localization_topdown.png`, `backend_localization_metrics_bar.png` |
| **Dynamic exploration P1** | `tab:dynamic_explore_phase1` | [dynamic_exploration.md](dynamic_exploration.md) | `uv run python scripts/eval_dynamic_exploration.py --smoke --dry-run` | `~/runs/emet/dynamic_exploration/` | Rerun map from export dir |
| **Dynamic exploration P2** | `tab:dynamic_explore_world_change` | [dynamic_exploration.md](dynamic_exploration.md) | `--phase world-change --cpu-only` | same | — |
| **Dynamic exploration P3** | `tab:dynamic_explore_lifelong` | [dynamic_exploration.md](dynamic_exploration.md) | `--phase lifelong --cpu-only` | same | — |
| **OVMM find-phase** | `tab:ovmm_find_backend_tier` | [ovmm_find_phase.md](ovmm_find_phase.md) | `uv run python scripts/smoke_ovmm_benchmark.py --cpu-only` | `~/runs/emet/ovmm_find_phase/` | scaling placeholders in `05_results.tex` |
| **OVMM full** | four-phase extension | [ovmm_full.md](ovmm_full.md) | `eval_ovmm_full.py --backend ground_truth --cpu-only` | `~/runs/emet/ovmm_full/` | — |
| **SQA3D** | `tab:sqa3d_backend_replay` | [sqa3d.md](sqa3d.md) | `uv run emet sqa3d run-episode --mock-llm --question-id 220602000000` | `~/runs/emet/sqa3d/` | `emet sqa3d plot-results` → `paper/figures/sqa3d_*` |
| **GT object finding** | `sec:gt_experiments` | [gt_object_finding.md](gt_object_finding.md) | `emet run dynagraph --ground-truth` + eval script | `runs/<export>/` | — |
| **Innate Mars** | appendix | [innate_mars.md](innate_mars.md) | `emet run dynagraph --robot innate_mars --ground-truth` | `/tmp/mars_*` | — |
| **Habitat EQA** | `tab:hmeqa_vs_prior`, Appendix | [habitat_eqa.md](habitat_eqa.md) · **[results](habitat_eqa_results.md)** | `.venv-habitat/bin/emet-habitat` | `~/.cache/habitat_eqa/results/` | topdown from debug bundles |
| **Cross-track smoke** | — | **[cross_track_smoke.md](cross_track_smoke.md)** | per-track commands | `~/runs/emet/` | validate before multi-day sweeps |
| **Large paper queue** | all tracks | [large_eval_queue.md](large_eval_queue.md) | `./scripts/run_large_paper_eval.sh` | `~/runs/emet/<track>/` | — |

Shared backend names: `dynamem`, `graph_eqa`, `dynagraph`, `ground_truth` — see [paper_benchmarks.md § Shared memory backends](../paper_benchmarks.md#shared-memory-backends).

## Update from main

On a feature branch, stay current with shared `main` and refresh the env:

```bash
git pull origin main
uv sync
```

Paper experiment docs and CLI flags track `main`; after pulling, use the same `emet stream` / `emet capture` / `emet run dynamem|dynagraph` commands on sim **and** hardware ([zmq_obs.md](zmq_obs.md), [innate_mars.md](innate_mars.md) § Sim vs hardware visualization). **Known issues:** [known_issues.md](../known_issues.md).

## Before any sweep

```bash
# Unit tests (no GPU, fast)
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

Run **one GPU-heavy job at a time** (dynamic exploration, backend localization, SQA3D real sweeps). Parallel VLM loads can OOM (~15 GB each). Kill stale `emet run dynagraph` / `emet sqa3d` processes before starting a new sweep.

For backend localization, prefer `--explore-steps 0` or `--quick` (light mapping protocol) unless you need OVMM-identical rotate+explore mapping (`--full-protocol`, much slower).

## Example figures

Figures are generated locally under `~/runs/emet/` — not committed to git by default. See [examples/README.md](examples/README.md) for paths and copy instructions.

| Figure | Regenerate | Validated example path |
|--------|------------|------------------------|
| Backend top-down map | `smoke_backend_localization_figure.py --quick` | `~/runs/emet/backend_localization_figures_20260617_final/backend_localization_topdown.png` |
| Backend metrics bar | same script | `~/runs/emet/backend_localization_figures_20260617_final/backend_localization_metrics_bar.png` |
| SQA3D diagnostics | `emet sqa3d plot-results -p <jsonl> -o paper/figures/sqa3d_val30` | `paper/figures/sqa3d_*` (after sweep) |

## Maintaining paper numbers

1. Run sweeps → per-track CSV/JSON under `~/runs/emet/`.
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

## Self-review notes (feature branch)

Last checked on `feature/dynamic-exploration-eval` after merge with `origin/main`:

- **Tests:** `src/test/eval/` (24), `src/test/simulation/test_robocasa_gen.py` (5) — pass.
- **Uncommitted harness work:** `--smoke` flag, scaled `EMET_DYNAMIC_EXPLORE_DYNAGRAPH_TIMEOUT_S`, Robocasa seed determinism, `scripts/smoke_backend_localization_figure.py`.
- **GPU smoke:** full Phase 1 dynamic explore ~60–75 min; run sequentially; check `dynagraph.log` in export dir on timeout.
