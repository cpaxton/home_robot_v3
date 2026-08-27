# OVMM find: voxel localize + close-map (2026-08-26)

**Status:** First slice landed (investigate cards + score from `localize_text`). Stretch
S0 GPU oneshot still the empirical gate.

## What used to work

`test_red_cylinder_detected_in_sim[stretch]` (and the Hello Robot `stretch_ai`
`SparseVoxelMap.localize_text` path it comes from) is the existence proof:

1. Map the default table (`rotate_in_place` / table scan).
2. Call **`voxel_map.localize_text("red cylinder")`** — YoloE `compute_obj_coord`
   if the detector hits, else a **0.21 SigLIP cosine** gate on the voxel cloud.
3. Navigate to that XYZ.

Pytest still allows **0.55 m** atol. OVMM FindObj/FindRec uses **0.3 m**. Stretch
oneshot on 2026-08-26 localized both objects via voxel but collapsed to one table
blob (~0.37–0.47 m) when query variants expanded `"red cylinder"` → `"cylinder"`.
rby1 dynamem table-scan the night before **hit the blue cube at 0.0 m** (`yoloe_hit`)
and missed the cylinder (cosine ~0.11, no detector).

The original repo (`hello-robot/stretch_ai`) is the same algorithm: detector first,
cosine second. A lot of wrapping changed (GraphEQA agentic loop, close-map,
rby1 ZED pitch). The **localize_text contract** did not.

## What was broken (2026-08-26 slice)

Default `emet ovmm find --backend dynagraph` never called `localize_text` for the
score. It ran HM-EQA `AgenticEQAExecutor`, then `xyz_from_verified_obs` which
fell through to **camera pose**. The VLM sometimes saw the cube; scoring used
`(-0.78, 0.47, 0)` → 1.23 m. Graph completeness stayed 0.0 (YoloE `tv`/`box`).

Close-map stay never ran on the table: after mapping, verify required a fresh
APPROACH obs, then the loop **explored frontiers** off the workspace. rby1
explore is a single capture (no head sweep).

`s0_parity` (pytest harness match) incorrectly applied the **interactive 0.45 m
merge** to agentic dynagraph S0.

## Product path (this is the tool)

```
mapping spin
    │ DynaMem voxels + CloseDistanceMap stamps
    ▼
localize_text(phrase)     ← the tool Stretch already had
    │ XYZ + (optional) YoloE
    ▼
investigate card source=voxel
    │ close-map STAY on that XY until aimed ≤ 0.55 m
    ▼
VLM assess is a check, not the coordinate
    ▼
score GT vs voxel XYZ (never camera pose)
```

Do **not** add a new EQA tool name (traces). The existing `investigate(obs_id)`
card is the agent-facing handle. `inspect_graph` refresh re-runs `localize_text`.

CHAT already uses `get_voxel_map().localize_text` in `emet.agent.loop` /
`face_toward`. EQA_EPISODE now gets the same point as a place card.

## Landed in this change

1. `emet.mapping.voxel_localize.localize_text_xyz` — shared helper.
2. Agentic recall **prepends** a `source=voxel` investigate card when
   `localize_text` hits, even if the graph already has junk nodes.
3. OVMM agentic scoring: phrase-matched **graph node**, else **voxel XYZ**.
   Camera pose is never a FindObj/FindRec coordinate. Unverified + voxel hit
   still scores (the loop may still go look).
4. Agentic S0 no longer switches to interactive 0.45 m merge / harness.
5. Per-episode query PNG dir (no sticky `setdefault` across episodes).

## Remaining plan

| Priority | Work | Why |
|----------|------|-----|
| P0 GPU | `emet ovmm find --backend dynamem` on Stretch S0 `default_table_s0_distinct_recep` and rby1 S0. Phrase-only oneshot. Compare to pytest 0.55 m vs OVMM 0.3 m. | Prove dynamem still localizes after mapping. |
| P0 GPU | Agentic dynagraph S0 after this slice. Expect FindRec if voxel hits the cube; FindObj if cylinder cosine/YoloE recovers. | End-to-end of voxel card + close-map + score. |
| P1 | If Stretch voxel still collapses red/blue to one peak: **do not** expand `"red cylinder"` to `"cylinder"` on S0 (already `phrase_only` on oneshot). Check YoloE vocab / confidence on the two GT bodies. | Instance separation. |
| P1 | Locate questions: allow VLM assess on the **mapping view** (drop “must APPROACH first”) *or* restore table mapping pose before the first investigate. | Stops frontier wander into sky/floor RGB. |
| P2 | When close-map `resolved` at voxel XY, treat that as localize success even if Qwen says unknown. | Geometry over EQA letter. |
| P2 | Dump query PNGs and fail the episode loudly if `vlm_assess` text looks like gradient/sky. | Catch camera-off-table. |
| P3 | stretch_ai diff of `localize_with_feature_similarity` cosine 0.21 / YoloE `compute_obj_coord` vs current `voxel_dynamem.py` — only if P0 GPU misses. | True regression vs Hello Robot. |

## What not to do

- Do not rename EQA tools (`investigate`, `explore_frontier`, …).
- Do not make `--oneshot-localize` the product path; oneshot is the **mapping
  ablation**. Agentic should **call** voxel, not skip the loop.
- Do not stack Habitat/HM-EQA on the same GPU night as this S0 gate.
- Do not treat VLM “The tv is at (x,y,z)” as a pose — that string is graph
  dumping, which is why we score `localize_text` instead.

## Commands

```bash
# CPU
uv run emet test --no-sim src/test/mapping/test_voxel_localize.py \
  src/test/memory/test_ovmm_agentic_find.py \
  src/test/memory/test_ovmm_agentic_routing.py \
  src/test/eval/test_benchmark_dynagraph.py src/test/agent/test_skill_packs.py -q

# GPU (jobs, exclusive) — dynamem oneshot S0
uv run emet jobs run --name ovmm-dynamem-s0 --need-mib 8000 --gpu-exclusive \
  -d "Stretch+rby1 S0 dynamem oneshot after voxel-card wiring" -- \
  uv run emet ovmm find --episodes configs/ovmm/find_phase_episodes.yaml \
  --backend dynamem --oneshot-localize \
  --episode-id default_table_s0_distinct_recep \
  --episode-id default_table_rby1_s0_distinct_recep \
  --output-dir ~/runs/emet/ovmm_find_phase/dynamem_s0_YYYYMMDD
```
