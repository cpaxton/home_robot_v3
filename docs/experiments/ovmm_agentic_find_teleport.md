# OVMM agentic find (teleport dynagraph)

Branch: `feat/tamp-ovmm-perf`

North star: raise **FindObj** and **FindRec** under teleport dynagraph — but
**routine gates must stay fast**. Stretch Realsense head sweeps made the old
3-episode slice take ~3.5 h (0/3 success). Default experiments now use **rby1**.

## Default gate (rby1, no head sweep)

| Profile | Episodes | Rounds | Target wall |
|---------|----------|--------|-------------|
| `smoke` (default) | `default_table_rby1_s0_distinct_recep` | 4 | ~5–15 min |
| `slice` | + `robocasa_rby1_pp_s1` | 4 | ~15–40 min |
| `stretch-legacy` | old Stretch S0/S1/S2 trio | 8 | hours — overnight only |

```bash
# Routine (preferred):
uv run emet jobs run --name ovmm-find-rby1-smoke --need-mib 8000 --gpu-exclusive -- \
  ./scripts/run_ovmm_find_recep_slice.sh

# Broader rby1:
PROFILE=slice uv run emet jobs run --name ovmm-find-rby1-slice --need-mib 8000 --gpu-exclusive -- \
  ./scripts/run_ovmm_find_recep_slice.sh

uv run python scripts/summarize_ovmm_agentic_traces.py --out-dir OUT
```

Controller: `DynamemController.look_around` skips head pans for non-Stretch robots
(rby1 / GenericZmqClient). Override with `EMET_FORCE_HEAD_SWEEP=1`.

## Historical Stretch baseline (do not use for iteration)

Pre-PR #110/#111 teleport: ~**1/9 FindObj**, **0/9 FindRec**.
2026-08-23 Stretch recep slice (`recep_slice_20260823_172713`): **0/3 FindObj, 0/3 FindRec**
in ~3.5 h (S0 ~10 min, RoboCasa ~101 min, Molmo ~109 min). Cancelled mid-postfix.

## Code changes (this branch)

| Area | Change |
|------|--------|
| Hypothesis recall | `_recall_nav_hypotheses()` + OVMM GT placement seeds |
| find_recep routing | Nearby investigate bias; skip `_prefer_explore` when recep card is close |
| Trace meta | richer OVMM `trace_meta` |
| Router observability | `nav_outcome` in Recent actions |
| **Efficiency** | rby1 episodes + skip head sweep on non-Stretch + `--mapping-rotate-steps 4` (do **not** `--not-rotate`; table scan must map the workspace) |
| Close-look map | Occupancy-aligned min camera range + aimed flag; investigate **stays** on a place card until aimed-close or **escapes** when unreachable / attempts exhausted |

## Acceptance (rby1 smoke / slice)

| Step | Gate |
|------|------|
| `PROFILE=smoke` | Completes &lt; 20 min; FindObj or FindRec ≥ 1 on the S0 episode |
| `PROFILE=slice` | FindRec ≥ 1/2 on rby1 S0+S1 |

Stretch 9-episode matrix is **not** the default validation ladder.
