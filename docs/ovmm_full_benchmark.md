# Full OVMM benchmark (FindObj + Pick + FindRec + Place)

Extends the [find-phase harness](ovmm_find_phase_benchmark.md) with **Pick** and **Place** phases aligned with the [OVMM](https://ovmm.github.io/) task structure. Scoring uses MuJoCo `sim_object_placements` GT deltas (not official HSSD minival yet).

## Phases

| Phase | Metric | Find-phase equivalent |
|-------|--------|------------------------|
| FindObj | `find_object_success` | same |
| Pick | `pick_success` | GT: object left start recep or moved ≥ threshold |
| FindRec | `find_recep_success` | same |
| Place | `place_success` | GT: object within radius of goal recep **and closer than at manip start** (when `placements_before` is set) |
| Aggregate | `ovmm_full_partial` | mean of active phases (2 or 4) |
| Full task | `ovmm_full_success` | AND of all four (when manip enabled) |

## Manip modes

| Mode | Behavior |
|------|----------|
| `skip` | Find phases only (same as `eval_ovmm_find_phases.py`) |
| `oracle` (default) | Pick/place success copied from find success (harness smoke / upper bound) |
| `sim` | MuJoCo freejoint teleport via ZMQ `sim_set_body_pose` (sim E2E) |
| `attempt` | AnyGrasp pick/place on real robot; **auto uses sim teleport when `is_simulation`** |

`attempt` needs a working AnyGrasp socket on real hardware. In sim, `attempt` and `sim` both use body teleport (no AnyGrasp).

## Quick start

```bash
uv run emet test src/test/memory/test_ovmm_full_metrics.py -q

# S0 distinct recep, oracle manip (fast GT smoke)
uv run python scripts/eval_ovmm_full.py \
  --episode-id default_table_s0_distinct_recep \
  --backend ground_truth \
  --not-rotate --cpu-only \
  --manip-mode oracle \
  --output-dir ~/runs/emet/ovmm_full/smoke

# Sim E2E (find + sim pick/place; GPU for perception mapping)
uv run python scripts/eval_ovmm_full.py \
  --episode-id default_table_s0_distinct_recep \
  --backend dynagraph \
  --manip-mode attempt \
  --output-dir ~/runs/emet/ovmm_full/e2e
```

Episodes: `configs/ovmm/full_episodes.yaml`. Outputs default to `~/runs/emet/ovmm_full` (`EMET_OVMM_OUTPUT_FULL` or `configs/ovmm/benchmark.yaml`).

CI / local smoke (find-phase + full oracle + unit tests):

```bash
uv run python scripts/smoke_ovmm_benchmark.py --skip-habitat
```

## Relation to find-phase

- Shared runner: `run_episode_find_phase()` with `FindPhaseRunConfig.manip_mode != "skip"`.
- Fair-default flags (`use_sensor_perception`, `prefer_voxel`, timing split) apply unchanged.
- Habitat full OVMM (HSSD minival) is not wired; use Habitat find-phase proxy for memory-only ablations.

See also: [paper_benchmarks.md](paper_benchmarks.md).
