# Expanded mobile transfer failure catalogue

The seven-layout provided-map Stretch cohort finished **2/7**. These runs used
native robot/object state and an external simulated RGBD scanner. Right clutter
and the far control passed for the cube. Use those two passes as replay controls
alongside the five failures. The layout declaration is
[`mobile_clutter_layout_20261118.json`](../../configs/eval/mobile_clutter_layout_20261118.json).

| Replay case | First blocking phase and evidence | Saved inputs and next comparison |
|---|---|---|
| `cube_clutter_left` | Gripper-opening preflight rejected the starting payload volume as unknown. Zero IK calls, 0.12 s of planning. Public code `no_plan_within_budget`. | Original source RGBD and grid. Compare the same starting payload volume with a second calibrated view and check that the public result names the evidence rejection. |
| `cube_clutter_front` | Same starting-volume rejection. Zero IK calls, 0.11 s of planning. Public code `no_plan_within_budget`. | Original source RGBD and grid. Compare with left clutter and the far control to isolate which returns leave the volume unknown. |
| `cylinder_clutter_right` | Initial search stopped on `stale_observation`. Scan/map 24.34 s, search 274.69 s, 18 IK calls. Some candidates also hit the distractor. | Source RGBD, grid, and candidate trace. Profile arm paths and loaded base sweeps separately. Compare timing and collision decisions on frozen replay inputs. |
| `cylinder_clutter_left` | Initial search ended with `support_geometry_unknown`. Scan/map 24.00 s, search 275.03 s, 30 IK calls. The trace also records finger contact with the distractor at a rejected destination. | Source RGBD, grid, and candidate trace. Replay the support query to separate insufficient coverage from evidence age and candidate collision. The elapsed time alone does not establish the cause. |
| `cube_clutter_back` | Grasp and carry completed. Release correction paths rejected the measured start. The map logged unknown volume around the held payload. Public receipt is partial with grasp complete. | `release-states/release-00.npz` and destination RGBD/grid. Reproduce the first rejected check on private data, then compare choosing a different destination before descent. |

All artifacts are relative to `.tamp-runs/expanded-clutter-20261118/`.
Start with `lane-0.json` through `lane-2.json` for the cohort, then open the
case's `result.json` for the public receipt, `events.jsonl` for planner
rejections, and `worker.log` for map witnesses. Each case includes its fixture
and config. Scans are under `source/` and, where reached, `destination/`.
These historical runs predate integration-state checkpoint capture. The
cube-back release snapshot supports a geometry replay. The source cases need
a rerun with checkpoint capture to reproduce the full physics state. All seven
were scripted tool trials and have no saved live agent session.

Use the [hard-case index](../../configs/eval/mobile_transfer_hard_cases_20261118.json)
to pick a case by ID and compare against its frozen baseline. The index test
checks unique IDs and coverage of the current layout suite. Three near-side
layouts removed from the current suite remain in the index as source-side
diagnostics, with their original fixture overrides.
Replay inputs are local run artifacts. The index test runs without them.

Keep future replay tests focused on one decision. For cube back, load the
release snapshot and destination grid and assert the first rejected check.
Pair it with a passing release control. For search performance, compare time
to a certified plan in a separate benchmark with the same inputs and unchanged
collision decisions. The historical source scans need matching simulator
checkpoints before they can serve as full source-search replay fixtures.

The cube-back saved-state replay rendered a new scan from the same viewpoint
and reproduced the unknown held-payload volume. The original release search
spent 39 IK calls. Checking the common held start after the opening-in-place
attempt keeps the same rejection and skips the adjustment search. The 57 release
tests pass. This change targets wasted work. A different certified destination
or additional qualified depth evidence is still needed to complete cube back.
Artifacts are under `.tamp-runs/cube-back-replay/`.

New mobile runs link unsuccessful tool receipts from `result.json` via
`failure_bundles`. Open the corresponding `tool-calls/*/failure.json` for
the before/after simulator checkpoints, capture identities and map generations,
and bounded map rejection witnesses. Event and command ranges are zero-based
and half-open. The event range indexes JSONL rows and the command range indexes
the `commands` array in the final result. These bundles describe returned tool
failures. An interrupted process may leave only the before checkpoint and
events recorded up to the interruption.

## Failure bundle smoke

Job `20261118_174912_bc4427` replayed cube front and produced
`failure-bundle-smoke/tool-calls/01-plan_pick_place/failure.json` under
`.tamp-runs/` (SHA256
`2f9247a4aedc189b7742996ad2b9733ee38d3e1c8bb3a5ba813b6900887c8444`).
Both checkpoint files and the referenced source scan were present. Event
rows `[2, 4)` contain the gripper-opening rejection, the map witness records
`unknown_swept_volume`, and the command range `[2, 2)` confirms the failed
tool issued no commands. The public receipt still reports
`no_plan_within_budget`. This validates the diagnostic bundle for a returned
preflight failure.

## Pre-descent placement pilot

Job `20261118_193547_b44a88` reran cube back using its original layout.
The planner completed source finalization, grasp and loaded carry, then
checked release before the first placement arm command. Seven of 11
destinations failed the retreat start check and four failed pose IK. The tool
returned `partial / no_collision_free_placement` at `place_refinement` with
`completed_ops=["grasp"]` and `recovery=inspect`. The trace contains no
preplace, place, or release commands. The destination map reported unknown
payload volume and the retreat events listed no native contact pairs. The
exact failing fall sample still needs a targeted replay.

The failure bundle is at
`.tamp-runs/pre-descent-cube-back/retry/cube_clutter_back/tool-calls/02-execute_pick_place_plan/failure.json`.
The separate boundary check passed and the pilot regression batch passed
209 tests. The first simulator attempt failed to load the missing software
renderer. The retry used a temporary extracted OSMesa package. The pilot code
is preserved locally at `8c8d03d3` on `experiment/tamp-pre-descent-admission`.
This run is a safe stop, and cube back remains an open transfer failure.

The #250 closeout reran release and mobile case configuration tests on the
published implementation. Job `20261118_224127_6fcb93` passed all 68 tests
with zero failures, errors, or skips.
