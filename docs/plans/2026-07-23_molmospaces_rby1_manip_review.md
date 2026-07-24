# Branch review: MolmoSpaces rby1 agent manip (teleport → motion planning)

**Branch:** `feature/molmospaces-rby1-agent-manip`
**Date:** 2026-07-23

## What this branch delivers

| Piece | Status |
|-------|--------|
| `sim_set_body_pose` on robosuite/rby1 ZMQ | Done |
| Molmo freejoint-ancestor teleport for `*_1_1_0` mesh children | Done |
| Agent / OVMM / scripted `pick_place` via teleport | Done (proxy, not physics) |
| Molmo GT placement scan uncapped | Done (`n_placements=124` on FloorPlan1) |
| Episode `molmo_ithor_rby1_s2_bowl_pp` | Done; measured `ovmm_full_success=true` |
| Scripted no-LLM tool calls | `scripts/scripted_sim_pick_place.py` |

## Self-review findings (fixed or accepted)

1. **Stretch visual-servo regression (fixed):** teleport ran whenever `sim_set_body_pose` was advertised, including Stretch+`--visual-servo`. Now gated by `prefer_sim_teleport_manip(..., visual_servo=…)`.
2. **False pickup success (fixed):** `sim_teleport_pickup` / `place` optionally verify session GT pose after ZMQ (~5 cm).
3. **Stale held body (fixed):** clear `_last_sim_picked_body` on failed pickup.
4. **Teleport ≠ simulation (accepted for v1):** documented; arm IK smoke added as next step.
5. **Nav target vs GT category match (accepted):** teleport still resolves by category substring; OVMM episodes set `object_gt_body`. Improve later with nearest-to-`point` disambiguation.
6. **Uncapped Molmo scan (accepted):** larger session payloads; needed for apple/etc.

## Motion planning status (2026-07-24)

1. **Done:** MuJoCo position IK, joint streaming, `sim_attach_body`, voxel-map arm RRT, agent `manip_mode` teleport|kinematic.
2. **Blockers fixed:** pick_place success via `_last_exec_ok`; GT manip without nav point; kinematic→teleport fallback when caps missing.
3. **Later:** CuRobo RPC from `.venv-molmospaces`; nearest-to-`point` GT disambiguation.
