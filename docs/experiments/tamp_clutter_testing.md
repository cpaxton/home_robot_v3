# TAMP GT+MCTS test battery

Deterministic smoke battery for the TAMP clutter stack — **ground-truth positions + the
MCTS planner only, no AI models** (no VLM / LLM / detection). Runs on MolmoSpaces iTHOR
across a (robot × scene) matrix.

## Entrypoint

```bash
# Fast (new robot, two environments): nori on iTHOR scenes 0 and 1
uv run python scripts/eval_tamp_clutter.py --test-battery \
  --battery-robots nori --battery-scenes 0,1

# Full matrix (all latch robots × more scenes)
uv run python scripts/eval_tamp_clutter.py --test-battery \
  --battery-robots nori,innate_mars,rby1 --battery-scenes 0,1,2

# Dry run (matrix only, no sim)
uv run python scripts/eval_tamp_clutter.py --test-battery --dry-run

# GPU-hygiene: queue behind the MIB gate (waits for the free 12 GB), then poll:
NEED_MIB=12000 uv run emet jobs run --name tamp-gt-battery --need-mib 12000 -- \
  uv run python scripts/eval_tamp_clutter.py --test-battery --battery-robots nori --battery-scenes 0,1
uv run emet jobs status JOB_ID
```

Result: per-run JSON + `battery_summary.json` under `EMET_TAMP_CLUTTER_OUTPUT`
(default `~/runs/emet/tamp_clutter`). Exit code 0 iff every test passes.

## Tests (per robot × scene)

| Test | Mode | Clutter | Pass condition |
|------|------|---------|----------------|
| `pickplace` | cleanup | 1 object | relocated to the bin (`n_relocated >= 1`) |
| `declutter` | cleanup | 3 objects | all relocated (`n_relocated == 3`) |
| `navblocked` | nav_goal | 8 (tight closed ring @ 0.5 m) | GT probe says **blocked** and, after clearing, the landmark is reached |
| `navclear` | nav_goal | 0 | no clutter; navigate straight to the landmark (`goal_reached`) |

Every step is sim-GT: placements from `sim_object_placements` (staticness/category from the
scene `*_physics_metadata.json`), tasks assigned by `plan_pick_place_mcts`, executed by
`execute_task_plan` / `plan_clear_clutter`. The battery defaults to the **`sim` (teleport)
oracle** so it validates the TAMP chain (what to move, where, in what order) independent of
per-robot arm reach; kinematic `latch` is a separate per-robot experiment via
`--manip-mode latch`. **Finding:** the Nori A3 model's arm bottoms out at z≈0.29 m, so it
cannot reach true-floor objects (z≈0.02) with `latch` — use the oracle or raise the drop
height for `latch` on Nori. No memory backend, no VLM worker.

This battery is the **blocked-route protocol** (tight ring vs pure nav). The 200-episode
YAML is a live-scatter **template**; scored nav_goal rows there use the same 8@0.5 m
tight ring and drop `skipped_invalid` when the probe is not blocked. See
[tamp_clutter.md](tamp_clutter.md).

**Validated 2026-08-28:** 24/24 episodes pass across nori, innate_mars, and rby1 × iTHOR
scenes 0–1 (pickplace / declutter / navblocked / navclear).

## Coverage

- **New robot**: `nori` (bimanual A3) is the default first battery; add `innate_mars`,
  `rby1` via `--battery-robots`.
- **Different environments**: `--battery-scenes` selects iTHOR FloorPlan indices (0, 1, …).

## Units (no sim)

```bash
uv run emet test src/test/eval/test_tamp_clutter_config.py -q   # incl. battery matrix builder
```

See [tamp_clutter.md](tamp_clutter.md) for the full benchmark and `docs/robots/nori.md`
for the Nori backend.
