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

## Minimal sim ladder (2026-07-26)

Logs: `~/runs/emet/manip_smoke/20260726/`.

| Step | Result | Notes |
|------|--------|-------|
| Offline pytest (motion + `test_agent_manip_tool_sequence`) | PASS | |
| Table teleport `scripted_sim_pick_place` | PASS | `displacement_m≈0.10` (`table_teleport_p190.log`) |
| Table kinematic IK+RRT+attach | PASS | `success=True`, `displacement_m≈0.14`, grasp_err≈0.033, place_err≈0.05 (`table_kinematic_atomic_p223.log`) |
| Agent tool-calls path (kinematic) | PASS | same script + `--tool-calls-json`; unit test green |
| Molmo iTHOR bowl→microwave kinematic | FAIL | `pregrasp_ik_failed` grasp_err≈0.60; approach goal vs `base_after` disagree (`molmo_kinematic_bowl_microwave_p230.log`) |

### Fixes that unblocked table kinematic

1. **Galaxea PD:** torso/arm `<general>` actuators need `biastype="affine"` so `biasprm` tracks joint position (was raw torque → arms collapsed).
2. **Atomic snap+attach:** bundle `sim_set_body_pose` + `sim_attach_body` in one ZMQ action; the 0.25s meta-action sleep was letting gravity drop the freejoint between separate commands.
3. **Home posture:** skip wheel/steer actuators; loop settle after nav teleport.
4. **Approach standoff:** `+0.55` m so base teleport does not embed in the table.

### Next

- Fix Molmo approach / nav-world vs MuJoCo base frame so pregrasp IK is in reach.
- Keep Molmo/OVMM/TAMP rungs for a follow-up night (not required for this minimal ladder).
