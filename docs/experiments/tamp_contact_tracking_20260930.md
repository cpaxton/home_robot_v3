# RBY1 contact and tracking diagnosis (2026-09-30)

The large r7 errors were caused by invalid base approaches interacting with the
benchmark's stationary pose reset. A separate free-space offset also came from
that base-hold implementation. Commands arrived at the server unchanged.

## Evidence

Source trial: `joint-tracking-r7/tamp_v2_rby1_scene00_cleanup_0`, source `525ec399`,
job `20260929_152521_a077e1` (completed). Its measured errors against Cartesian
pregrasp targets were 2.408 m, 1.799 m and 0.0539 m. All three were correctly
rejected at the unchanged 0.035 m tolerance.

The diagnostic reconstructs the merged scene, initial object poses, recorded base
approaches and joint targets. It ramps each command for two simulation seconds
and holds until eight seconds. It does not reproduce the original asynchronous
command timing, prior object attempts or full measured initial robot state.
Each row is one deterministic reconstruction, not a reliability estimate.

| Attempt | Base/wall penetration | Old hold, contacts on | Old hold, contacts off | Solver hold, contacts off |
| --- | ---: | ---: | ---: | ---: |
| Potato | 6.90 cm | 2.383 m | 46.11 mm | 2.03 mm |
| Apple | 27.21 cm | 1.168 m | 57.88 mm | 3.51 mm |
| Kettle | none detected | 34.10 mm | 34.10 mm | 3.49 mm |

Errors in this table compare actual EE position with FK of commanded joints;
they are distinct from error against the Cartesian IK target. The sole detected
robot/environment contact pair in the original reconstructed failures was
`wall_bf73ef61aba158e368b1f9b3764c478c_1_1_0` / `base_link`.
Peak articulated constraint torques were approximately 1,105 and 20,039 Nm in
the two intersecting cases; actuator torques saturated at their declared limits.

A fixed-base model isolates arm dynamics without disabling gravity or changing
actuator limits. All three targets settle within 0.005 rad. Using solver-based
stationary support in the floating-base model reproduces that test result.
For the clear kettle approach in the furnished scene, contacts remain enabled:
EE tracking error is 3.49 mm and Cartesian target error is 28.83 mm, within the
unchanged 35 mm tolerance. This is a pregrasp reconstruction, not a full pickup.

## Repairs and regression suite

- Simulator oracle teleports now check robot/scene penetration on private data
  before moving the live base or attached objects. Wall-intersecting endpoints
  fail with `teleport_endpoint_collision`; shallow contacts within the existing
  1 mm collision tolerance are not treated as deep penetration. This checks an
  endpoint, not the full route, and does not certify carried-object clearance.
- RBY1 declares an initially inactive world/base weld. The server uses it for
  stationary oracle support instead of repeatedly resetting a floating base.
  The physics solver now accounts for the support force within each step.
  It disables the weld during navigation and passive-support mode. Physical
  execution guards still forbid stationary pose assistance. Other robot models
  without this declared constraint retain their existing behavior.
- Arm planning starts at measured joint positions. A previous command may seed
  IK but cannot replace a measured start after contact prevented motion.
- `test_contact_tracking.py` adds 14 cases: free and blocked slider dynamics,
  failure to arrive under contact, recovery after retracting, rejection of a
  path through an obstacle, free/planar teleport rejection without live-state
  mutation, shallow contact acceptance, server rejection before actuation,
  measured planning starts, and all three RBY1 commands with fixed and server
  supported bases. The support tests also verify deactivation for navigation
  and passive support.

**102 tests pass** across the contact suite, controller arrival/frame tests,
simulator navigation, physical-motion contracts, and shared agent-tool tests.
Ruff and whitespace checks pass.

## Artifacts and reproduction

Durable artifacts are under
`~/runs/emet/tamp-validated-fixtures-20260928/contact-diagnosis-r8/`:
`results.json`, `weld_results.json`, compiled models, source logs, diagnostic
scripts, and `figures/tamp_contact_tracking.{pdf,svg,png}` with a caption.
The figure explicitly labels contact-disabled runs as diagnostic ablations.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python scripts/diagnose_tamp_tracking.py \
  --scene <merged-scene.xml> --initial <case.initial.json> --log <process.log> \
  --base-z 0.3009 --base-support weld --output <output.json>
```

The welded diagnostic scene adds only the inactive support constraint to the
original r7 robot include; simulation code activates it during the diagnostic.
Use `--base-support pose_reset` and the original scene to reconstruct the old
hold. The scripts record parameters, input hashes and compiled-model hashes.
The original first diagnostic predates compiled-model export; its original
scene is archived separately as `legacy_model.mjb`.

## Remaining acceptance gates

Accurate tracking does not make an intersecting base pose valid. Even where the
new support improves tracking inside a wall, the endpoint guard must reject it.
The task grounder currently proposes a fixed +Y standoff; it must search and
validate alternative approaches before a full task improvement is established.
The new robot model and stricter oracle navigation require a new benchmark
version; do not overwrite or reuse the original scores as evidence of success.

RBY1's declared collision geometry does not cover its visual-only arm links.
The synthetic blocked-arm regression tests the control/acceptance contract,
not full robot arm collision coverage. Full TAMP acceptance still requires
appropriate arm geometry, swept paths, payload checks, a complete shared-agent
rollout, and fresh full-task trials. No physical or learned-agent success gain
is claimed by this diagnostic.

## Live replay after repair

Job `20260930_231839_57fd7c`, immutable source `4b512ffe`, replayed original
scene00 cleanup slot 0 on the original r3 harness. Outputs are in
`contact-replay-r8/` under the same durable root. It completed with **0/1 full
tasks**. Both wall-intersecting approaches now fail before movement, with the
same 6.90 cm / 27.21 cm penetration evidence. The kettle advances through
pregrasp (30.75 mm measured error) and grasp (34.88 mm), then rejects lift at
36.07 mm versus the unchanged 35 mm limit.

FK of the three commanded targets has residuals 29.23, 28.59 and 29.58 mm.
The lift plan therefore left only about 5.4 mm for execution error. Follow-up
`a2394a2b` tightens the internal IK tolerance to `min(10 mm, arrival_tol / 4)`
while preserving measured acceptance. The 40-test affected regression slice
passes. Paired job `20260930_232300_561b86`, source `6665220e`, is testing this
change in `ik-margin-r9/`; it is terminal with **1/3 objects relocated and
0/1 full tasks**. The kettle completed approach, grasp, lift and placement with
measured arrival checks. Potato and apple failed only at their rejected
wall-intersecting approaches. This is kinematic-latch control success for one
object, not a physical grasp or full-agent success.

A static audit of sixteen 55 cm standoff directions found seven clear and
position-IK-reachable potato approaches, nine apple approaches, and eleven
kettle approaches. These are possibilities for alternative approach search,
not executed or fully collision-certified solutions. The audit and source are
archived as `contact-replay-r8/alternative_approaches.json` and
`audit_alternative_approaches.py`. Neither replay changes historical scores or
establishes renewed fixture admission under the stricter controller.
