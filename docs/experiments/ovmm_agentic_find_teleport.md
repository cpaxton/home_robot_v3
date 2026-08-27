# OVMM agentic find (teleport dynagraph)

Branch: `feat/tamp-ovmm-perf`

North star: raise **FindObj** and **FindRec** under teleport dynagraph — but
**routine gates must stay fast**. Stretch Realsense head sweeps made the old
3-episode slice take ~3.5 h (0/3 success). Default experiments now use **rby1**.

## Default gate (rby1, no head sweep)

| Profile | Episodes | Rounds | Target wall |
|---------|----------|--------|-------------|
| `oneshot` | `default_table_rby1_s0_distinct_recep` | mapping + voxel localize only | ~1–3 min |
| `verify` | Robocasa rby1 PickPlace | GT drive-up + YOLOE/SigLIP on current RGB | ~5–15 min |
| `smoke` (default) | same S0 table, agentic | 6 | ~5–15 min |
| `slice` | + `robocasa_rby1_pp_s1` | 6 | ~15–40 min |
| `stretch-legacy` | old Stretch S0/S1/S2 trio | 8 | hours — overnight only |

**Iterate here, not on Stretch `molmo-robocasa-find8`.** The table episode backs up and looks at the workspace (`_prepare_default_table_rby1_mapping_view`) so red cylinder / blue cube are in view after a 4-step spin. This morning’s rby1 smoke: FindObj **1/1** in **106 s** (voxel XYZ). Stretch kitchen find8 is ~3 h/episode with head sweeps and `goal_recep: cab` (VLM looks for the word “cab”). PickPlace `obj_main` is often a sugar cube — `emet ovmm probe-verify` skips that and aims from spawn at a jar/bottle/bowl-class body (`emet.eval.ovmm_probe_targets`). Offline dumps: `emet ovmm probe-map`.

Saved maps (no new sim): `~/.cache/emet/scene_maps/<key>/voxel_map.pkl` + `graph.json`. Probe without MuJoCo:

```bash
uv run emet ovmm probe-map --list
uv run emet ovmm probe-map --cache-key robocasa_pickplacecountertocabinet_s1_l1_seed0_stretch_gt
# SigLIP on the pickle (GPU, still no sim):
uv run emet jobs run --name ovmm-probe-map-voxel --need-mib 8000 --gpu-exclusive -- \
  uv run emet ovmm probe-map --voxel --cache-key robocasa_pickplacecountertocabinet_s1_l1_seed0_stretch_gt
```

The July Robocasa L1 Stretch GT cache has **cabinet/counter** nodes and **zero `jar` labels** — graph FindObj cannot succeed until mapping actually names the object. Live `category_matches` is a substring, so query `cab` still hits `cabinet`.

```bash
# Fastest GPU inner loop (spin + voxel localize, no 8-round VLM wander):
PROFILE=oneshot uv run emet jobs run --name ovmm-find-rby1-oneshot --need-mib 8000 --gpu-exclusive -- \
  ./scripts/run_ovmm_find_recep_slice.sh

# Live kitchen: teleport to GT jar/cabinet and verify on the head camera (no AgenticEQA):
PROFILE=verify uv run emet jobs run --name ovmm-probe-verify-rby1 --need-mib 8000 --gpu-exclusive -- \
  ./scripts/run_ovmm_find_recep_slice.sh
# or: uv run emet ovmm probe-verify --out OUT

# Routine agentic gate:
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
| Hypothesis recall | `_recall_nav_hypotheses()`; voxel `localize_text` cards from the **question**, not episode YAML / GT placement seeds |
| find_recep routing | Nearby investigate bias; skip `_prefer_explore` when recep card is close |
| Trace meta | richer OVMM `trace_meta` |
| Router observability | `nav_outcome` in Recent actions |
| **Efficiency** | rby1 episodes + skip head sweep on non-Stretch + `--mapping-rotate-steps 4` (do **not** `--not-rotate`; table scan must map the workspace) |
| Close-look map | Occupancy-aligned min camera range + aimed flag; investigate **stays** on a place card until aimed-close or **escapes** when unreachable / attempts exhausted |
| Voxel localize | `localize_text` (YoloE / cosine) is an investigate card (`source=voxel`) and the FindObj/FindRec coordinate; camera pose is not scored |

## Acceptance (rby1 smoke / slice)

| Step | Gate |
|------|------|
| `PROFILE=smoke` | Completes &lt; 20 min; FindObj or FindRec ≥ 1 on the S0 episode |
| `PROFILE=slice` | FindRec ≥ 1/2 on rby1 S0+S1 |

Stretch 9-episode matrix is **not** the default validation ladder.
