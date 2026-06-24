# Dynagraph navigation and exploration benchmarks

Scripts under `src/test/app/` validate GT object navigation and frontier exploration across sim environments.

## Quick run

```bash
# Default MuJoCo table (GT nav to red cylinder + 3 explore steps)
EMET_SIM_NAV_TELEPORT=1 uv run python src/test/app/run_dynagraph_nav_benchmark.py --default

# XLeRobot on default table + MolmoSpaces iTHOR (DynaMem explore + Dynagraph GT nav)
EMET_SIM_NAV_TELEPORT=1 uv run python src/test/app/run_dynagraph_nav_benchmark.py --robot xlerobot --dynamem --default --molmo

# All tiers (default + Robocasa + MolmoSpaces + DynaMem baseline)
EMET_SIM_NAV_TELEPORT=1 uv run python src/test/app/run_dynagraph_nav_benchmark.py --all
```

Reports are written to `/tmp/dynagraph_nav_bench/nav_benchmark_report.json` (override with `DYNAGRAPH_NAV_BENCH_BASE`).

## Pytest gate

```bash
RUN_DYNAGRAPH_NAV_BENCHMARK=1 uv run emet test src/test/app/test_dynagraph_nav_benchmark.py -v
```

## What is checked

| Tier | Nav query (GT) | Explore |
|------|----------------|---------|
| `default_table_gt_nav_explore` | graph localize → `red cylinder` | `run_exploration` ×3, frontier nodes |
| `robocasa_gt_nav_explore` | `sink_main` / sink | same |
| `molmospaces_gt_nav_explore` | sink (from placements; iTHOR train index 0 kitchen) | same |
| `dynamem_explore_baseline` | — | DynaMem without graph memory |

Pass criteria: graph localization succeeds, base moves closer to GT XY (or within ~35 cm with teleport), and at least one frontier excursion succeeds with positive explored area.

## Related

- `src/test/app/run_dynagraph_benchmark_smoke.py` — export / floor-metrics smoke tiers
- `src/emet/config/benchmarks/dynagraph_questions.yaml` — EQA question bank for `emet eval-dynagraph`
