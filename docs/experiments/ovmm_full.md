# OVMM full (find + pick + place)

Four-phase extension of OVMM find-phase: localize object and receptacle, then pick and place in sim.

**Deep doc:** [ovmm_full_benchmark.md](../ovmm_full_benchmark.md)

## Paper reference

Four-phase extension (no dedicated results table yet in `05_results.tex`).

## Primary metrics

`ovmm_full_success`, per-phase success rates (find obj, find rec, pick, place).

## Config and output

- `configs/ovmm/benchmark.yaml`
- Default output: `~/runs/emet/ovmm_full` (`EMET_OVMM_OUTPUT_FULL`)

## Smoke

```bash
uv run emet test src/test/memory/test_ovmm_full_metrics.py -q

# Oracle manip smoke (fast)
uv run python scripts/eval_ovmm_full.py \
  --episode-id default_table_s0_distinct_recep \
  --backend ground_truth --not-rotate --cpu-only \
  --manip-mode oracle \
  --output-dir ~/runs/emet/ovmm_full/smoke
```

## Sim E2E

```bash
uv run python scripts/eval_ovmm_full.py \
  --episode-id robocasa_pp_s1 \
  --backend dynagraph --manip-mode sim --cpu-only

# MolmoSpaces + rby1 teleport manip
uv run python scripts/eval_ovmm_full.py \
  --episode-id molmo_ithor_rby1_s2_bowl_pp \
  --backend ground_truth --manip-mode sim --not-rotate --cpu-only \
  --output-dir ~/runs/emet/ovmm_full/molmo_rby1_smoke
```

Uses ZMQ `sim_set_body_pose` for pick/place (same API as dynamic exploration world-change). Robosuite (rby1) and Stretch MuJoCo both advertise the capability.

Uses ZMQ `sim_set_body_pose` for pick/place (same API as dynamic exploration world-change).

## Related

- [ovmm_find_phase.md](ovmm_find_phase.md) — localization-only benchmark
