# OVMM find-phase

FindObj / FindRec localization on Emet sim tiers S0–S2 and Habitat HM3D proxy episodes.

**Deep doc:** [ovmm_find_phase_benchmark.md](../ovmm_find_phase_benchmark.md)

## Paper reference

- Section: `sec:ovmm_find_phase`
- Table: `tab:ovmm_find_backend_tier`
- Metric: find partial success @ radius $r$

## Backends

`dynamem`, `graph_eqa`, `dynagraph`, `ground_truth` (sim oracle). Perception backends use GPU and **must not** pass `--not-rotate`. Oracle may use `--not-rotate --cpu-only`.

## Config and output

- `configs/ovmm/benchmark.yaml`
- Default output: `~/runs/emet/ovmm_find_phase` (`EMET_OVMM_OUTPUT_SIM`)
- Aggregate: `aggregate_<backends>.csv` in output dir

## Smoke

```bash
uv run python scripts/smoke_ovmm_benchmark.py --cpu-only
```

## Paper sweep (S0 ladder)

```bash
uv run python scripts/download_ovmm_benchmark_assets.py

uv run python scripts/eval_ovmm_find_phases.py \
  --tier S0 \
  --backend dynamem --backend graph_eqa --backend dynagraph --backend ground_truth \
  --cpu-only \
  --output-dir ~/runs/emet/ovmm_find_phase/s0_paper
```

Scale to S1/S2 with `--tier S1` or `--tier S2`.

## Habitat proxy

```bash
uv run python scripts/eval_habitat_ovmm_find_phases.py \
  --backend ground_truth --not-rotate --cpu-only \
  --output-dir ~/runs/emet/ovmm_habitat/gt_batch
```

## Related

- [backend_localization.md](backend_localization.md) — single-scene figure smoke
- [ovmm_full.md](ovmm_full.md) — find + pick + place extension
