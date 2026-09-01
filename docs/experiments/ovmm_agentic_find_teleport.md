# OVMM agentic find (teleport dynagraph)

Branch: `feat/ovmm-fast-iterate`

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
| `stretch` | Stretch table S0, **no head pan** | 6 | rby1-smoke-like; known-good look_front camera |
| `stretch-kitchen` | Stretch `robocasa_pp_s1`, **no head pan** | 6 | kitchen camera check without 40–80s sweeps |
| `stretch-legacy` | old Stretch S0/S1/S2 trio **with** pans | 8 | hours — overnight only |

**Iterate here, not on Stretch `molmo-robocasa-find8`.** The table episode backs up and looks at the workspace (`_prepare_default_table_rby1_mapping_view`) so red cylinder / blue cube are in view after a 4-step spin. This morning’s rby1 smoke: FindObj **1/1** in **106 s** (voxel XYZ). Stretch kitchen find8 is ~3 h/episode with head sweeps and `goal_recep: cab` (VLM looks for the word “cab”). The 4-pan `look_around` is optional: `EMET_SKIP_HEAD_SWEEP=1` (kept by `PROFILE=stretch` / `stretch-kitchen`) captures one frame at the current pose instead. PickPlace `obj_main` is often a sugar cube — `emet ovmm probe-verify` skips that and aims from spawn at a jar/bottle/bowl-class body (`emet.eval.ovmm_probe_targets`). Offline dumps: `emet ovmm probe-map`.

Saved maps (no new sim): `~/.cache/emet/scene_maps/<key>/voxel_map.pkl` + `graph.json`. Probe without MuJoCo:

```bash
uv run emet ovmm probe-map --list
uv run emet ovmm probe-map --cache-key robocasa_pickplacecountertocabinet_s1_l1_seed0_stretch_gt
# SigLIP on the pickle (CUDA if available; still no sim). `--cpu-only` forces CPU:
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

# Stretch, no 4-pan look_around (look_front camera; faster than find8):
PROFILE=stretch uv run emet jobs run --name ovmm-find-stretch-nosweep --need-mib 8000 --gpu-exclusive -- \
  ./scripts/run_ovmm_find_recep_slice.sh
PROFILE=stretch-kitchen uv run emet jobs run --name ovmm-find-stretch-kitchen-nosweep --need-mib 8000 --gpu-exclusive -- \
  ./scripts/run_ovmm_find_recep_slice.sh

uv run python scripts/summarize_ovmm_agentic_traces.py --out-dir OUT
```

Controller: `look_around` reads `mapping.look_around_head_sweep` from the
`robots.<id>` overlay in [`configs/emet/default.yaml`](../../configs/emet/default.yaml)
(Stretch and rby1 default **false** = single look_front capture). Env:
`EMET_SKIP_HEAD_SWEEP=1` still skips; `EMET_FORCE_HEAD_SWEEP=1` pans
(`PROFILE=stretch-legacy`). `--set mapping.look_around_head_sweep=true` for a
one-off Realsense coverage sweep.

## Historical Stretch baseline (do not use for iteration)

Pre-PR #110/#111 teleport: ~**1/9 FindObj**, **0/9 FindRec**.
2026-08-23 Stretch recep slice (`recep_slice_20260823_172713`): **0/3 FindObj, 0/3 FindRec**
in ~3.5 h (S0 ~10 min, RoboCasa ~101 min, Molmo ~109 min). Cancelled mid-postfix.

## Code changes (this branch)

| Area | Change |
|------|--------|
| **Mapping** | `run_mapping_protocol` with `mapping_max_nav_steps>0` uses `AgenticEQAExecutor` `mode=explore` (coverage, `question=None`, no object `toward`); `S0` `mapping_max_nav_steps=0` stays rotate-only. (`explore_steps` is a deprecated alias.) Frontier picks are uncovered-first / VLM among frontier RGBs — object-biased mapping is wrong for random OVMM placement. After hop-until-arrival this budget is completed journeys, not leftover A* chunks (kitchen rby1 is 8). |
| Arrival capture | `_tool_explore_frontier` after nav: face frontier (`target_theta`), `look_ahead` (tilt 0) then `capture_and_update` at arrival; no pre-move `look_around` sweep and not `look_front` −30°. Hop-until-arrival in `navigate_to_target_pose` finishes 27-wp kitchen paths. |
| Hypothesis recall | `_recall_nav_hypotheses()`; voxel `localize_text` cards from the **question**, not episode YAML / GT placement seeds. Voxel proposal beats camera-pose-at-feet graph view (`CAMERA_POSE_PLACE` redirect). |
| Find voxel-first | `localize_text` on finished voxel map, then `investigate` that XYZ; do not pin-hunt object while mapping. If no voxel hit, one `explore_frontier` rather than 150 wall nodes. |
| find_recep routing | Nearby investigate bias; skip `_prefer_explore` when recep card is close |
| Trace meta | richer OVMM `trace_meta` (mapping `n_explore` / wall time in episode JSON) |
| Router observability | `nav_outcome` in Recent actions |
| **SigLIP per-phase re-attach** | `_do_submit_answer` releases the voxel encoder for Qwen; the OVMM harness re-attaches it (`re_attach_siglip_encoder`) before **each** find phase so FindRec can still `localize_text` the finished map (before: second phase silently returned nothing — `siglip-seed sim=None`). HM-EQA keeps released-SigLIP behavior. |
| **One-shot proposals** | A voxel proposal (`obs_id < 0`) is blocked after one real nav attempt (`_hypothesis_nav_blocked`): a close ABSENT is decisive, so the router/fallback cannot re-chase the same wall XYZ. The loop re-localizes from the grown map. |
| **Unpin on ABSENT** | A close ABSENT on a **voxel proposal handle** (`obs_id < 0`) also removes the retrieval pin (`unpin_localize_xyz`) so a disproven point is never scored by the `pinned_xyz_from_phrases` fallback. A close ABSENT on a nearby graph view still retracts the claim at that obs but does **not** unpin. |
| **Explore no-progress block** | A nav that moved < 0.10 m blocks that frontier XY (`_habitat_recent_goals`/`_blocked_goals`) so the next pick rotates to a different frontier or falls to multi-goal explore (fixes the kitchen re-pick-the-same-frontier stall). |
| **Efficiency** | rby1 episodes, or Stretch with `EMET_SKIP_HEAD_SWEEP=1`, + `--mapping-rotate-steps 4` (do **not** `--not-rotate`; table scan must map the workspace) |
| Close-look map | Occupancy-aligned min camera range + aimed flag; investigate **stays** on a place card until aimed-close or **escapes** when unreachable / attempts exhausted |
| Voxel localize | `localize_text` (YoloE / cosine) is an investigate card (`source=voxel`) and the FindObj/FindRec coordinate; camera pose is not scored |

## Acceptance (rby1 smoke / slice)

| Step | Gate |
|------|------|
| `PROFILE=smoke` | Completes < 20 min; FindObj ≥ 1 on the S0 episode (measured: **FindObj 4/4**, FindRec 2/4, both voxel err 0.0 when found) |
| `PROFILE=slice` | FindRec ≥ 1/2 on rby1 S0+S1 |
| Mixed gate | `scripts/run_habitat_ovmm_joint_gate.sh` — HM-EQA countclock stays at **gateAB 7/15** while OVMM S0 gains the fixes. Both phases always run; the process exits non-zero if a requested phase failed (`emet jobs` marks the gate red). |

Stretch 9-episode matrix is **not** the default validation ladder.
