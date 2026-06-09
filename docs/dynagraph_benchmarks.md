# Dynagraph benchmarks

Experiment suite aligned with the paper introduction: **explore**, **remember**, **update**, and **open-ended EQA** across default table, Robocasa, and MolmoSpaces.

## Quick commands

| Goal | Command |
|------|---------|
| CI smoke (unit, no sim) | `uv run emet test src/test/app/test_dynagraph_benchmark_smoke.py -v` |
| Full smoke (sim, slow) | `RUN_DYNAGRAPH_BENCHMARK_SMOKE=1 uv run emet test src/test/app/test_dynagraph_benchmark_smoke.py -k test_benchmark_smoke` |
| Benchmark harness (tiers) | `uv run python src/test/app/run_dynagraph_benchmark_smoke.py --default` |
| Unified episode eval | `uv run emet eval-dynagraph --episode /tmp/export -o dynagraph_eval.json` |
| Question bank EQA | `uv run emet run dynagraph --export /tmp/ep --question-file src/emet/config/benchmarks/dynagraph_questions.yaml --question-env default_table` |
| Fusion A/B | `./scripts/run_dynagraph_fusion_ab.sh innate_mars 0 20` |
| Full matrix | [docs/experiments/innate_mars.md](experiments/innate_mars.md) |
| MolmoSpaces benchmark | `uv run python scripts/run_dynagraph_molmo_benchmark.py` |
| Staleness / disappearance (unit) | `uv run emet test src/test/memory/test_dynagraph_staleness_disappearance.py -v` |

## Metrics (`emet eval-dynagraph`)

One JSON per episode export:

| Section | Metrics |
|---------|---------|
| `explore` | `explored_area_m2`, `explored_fraction`, cell counts |
| `graph` | `node_count`, `edge_count`, `viewpoint_count` |
| `fusion` | `spatial_recall`, `label_recall`, `duplication_penalty` (from detections JSONL or `detections_*.json`) |
| `gt` | completeness, association recall, localization error (when `sim_object_placements.json` present) |
| `eqa` | accuracy from `eqa_results.json` + question bank token checks |

Reference thresholds and latest measured runs: [`src/test/fixtures/baselines/dynagraph_eval_reference.json`](../src/test/fixtures/baselines/dynagraph_eval_reference.json).

### Measured results (2026-06-05)

| Experiment | spatial_recall | label_recall | graph nodes | explored_fraction | Notes |
|------------|----------------|--------------|-------------|-------------------|-------|
| Default table GT export | — | — | **3** | — | `gt_graph_completeness=1.0`, 14.15 m² explored |
| innate_mars calibration JSONL | **1.00** | 0.20 | — | — | Geometry strong; taxonomy diagnostic low |
| stretch calibration JSONL | **0.60** | 0.00 | — | — | Viewpoint / facing wrong scene |
| innate_mars cal dynagraph export | 1.00 | 0.20 | **66** | **0.46** | 48-step calibration capture + explore |
| innate_mars explore smoke (5 iters) | — | — | **0** | 0.11 | Detections logged; merged nodes still 0 (gap) |

Reproduce:

```bash
uv run emet eval-calibration --gt /tmp/emet_fusion_tune/innate_mars/gt.json --frames /tmp/emet_fusion_tune/innate_mars/frames.jsonl
uv run emet eval-dynagraph --episode /tmp/dynagraph_bench_smoke/default_table
uv run emet eval-dynagraph --episode /tmp/emet_fusion_tune/innate_mars/cal
```

## Question bank

[`src/emet/config/benchmarks/dynagraph_questions.yaml`](../src/emet/config/benchmarks/dynagraph_questions.yaml) defines per-environment questions with `expected_tokens` (substring match on real LLM answers). Requires API keys configured in dynav YAML for nightly runs.

## Fusion A/B

Compare duplicate-node suppression without changing explore seed:

- **ON:** `default_graph_object_fusion.yaml`
- **OFF:** `graph_object_fusion_ab_off.yaml` via `--graph-fusion-config`

Report: `node_count` ratio, `spatial_recall` delta, optional EQA accuracy from paired runs.

## Paper-scale matrix (manual / nightly)

1. Multi-robot Robocasa: `run_dynagraph_multi_robot_e2e.py` + `eval-dynagraph` per robot export
2. Multi-seed: repeat with `--seed 0,1,2`
3. MolmoSpaces sweep: `run_dynagraph_molmo_benchmark.py` with varying `--molmospaces-index`
4. EQA: `--question-file` + `--export`; score with `emet eval-dynagraph --questions …`

## SQA3D (situated ScanNet QA)

Open-ended situated QA on ScanNet scenes — EM@1 scoring, not multiple choice.

| Goal | Command |
|------|---------|
| Download annotations | `uv run python scripts/download_sqa3d_data.py --fetch-annotations` |
| Download ScanNet mesh | `uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00` |
| Score predictions | `uv run emet eval-sqa3d -p preds.jsonl --split val` |
| Paper figures | `uv run emet sqa3d plot-results -p preds.jsonl -o figs/` |
| Docs | [sqa3d.md](sqa3d.md) |

## See also

- [sqa3d.md](sqa3d.md) — SQA3D loaders and EM@1 eval
- [TESTING.md](TESTING.md) — test index
- [dynagraph.md](dynagraph.md) — CLI and fusion calibration
- [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md) — floor parity harness
